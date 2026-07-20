"""
LLM verification validator plugin.

Adapted from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391). The paper's central idea is that verification should
produce a *continuous* score by taking the expectation over the distribution
of scoring-token logits, rather than asking an LM judge for a single discrete
label. That probabilistic formulation lets verification scale along three
axes that each improve accuracy:

    1. score granularity  - finer scoring-token rubrics separate good/bad
                            solutions better than a binary yes/no label.
    2. repeated evaluation - averaging several stochastic reads cuts variance.
    3. criteria decomposition - scoring independent rubrics (syntax, schema,
                            intent) and combining them cuts complexity.

This module keeps that core mechanism at full fidelity and surfaces it through
Querybook's existing query-validation plugin slot as a ``BaseQueryValidator``.

Adapted (auxiliary) components, per the Mode 2 port:

    * The paper's fine-tuned verifier LLM is replaced by Querybook's already
      configured OpenAI chat model (``ChatOpenAI`` with ``logprobs=True``).
      No training and no new model artifacts are required.
    * The paper's bespoke benchmark suite (Terminal-Bench, SWE-Bench, ...) is
      out of scope; evaluation belongs in a downstream PR. Results are returned
      through the standard ``QueryValidationResult`` channel.
    * The paper's Claude Code extension and RL dense-feedback application are
      out of scope; Querybook hosts neither surface.

The scoring core is deliberately pure and dependency-free so it can be unit
tested directly. The LLM call is isolated behind a ``LogProbScorer`` callable
that returns a token -> probability distribution for the first generated
token; production builds one from OpenAI logprobs, tests inject a fake.
"""

import json
import math
import os
from typing import Any, Callable, Dict, List, Optional

from lib.query_analysis.validation.base_query_validator import (
    BaseQueryValidator,
    QueryValidationResult,
    QueryValidationResultObjectType,
    QueryValidationSeverity,
)

# A scorer maps a prompt to the probability distribution (token -> prob) of the
# first generated token. Production wraps ChatOpenAI(logprobs=True); tests pass
# a deterministic fake. Probabilities need only be proportional.
LogProbScorer = Callable[[str], Dict[str, float]]

# A granularity rubric maps each allowed scoring token (lower-cased) to a
# numeric grade in [0, 1]. Scoring against increasingly fine rubrics is the
# paper's "score granularity" scaling axis: it yields better separation between
# positive and negative solutions than a single binary label.
GranularityRubric = Dict[str, float]

DEFAULT_RUBRICS: List[GranularityRubric] = [
    {"yes": 1.0, "true": 1.0, "no": 0.0, "false": 0.0},
    {"low": 0.0, "medium": 0.5, "high": 1.0},
    {"1": 0.0, "2": 0.25, "3": 0.5, "4": 0.75, "5": 1.0},
]


class Criterion:
    """One decomposed verification criterion (syntax, schema, intent, ...)."""

    def __init__(self, name: str, prompt: str, weight: float = 1.0) -> None:
        self.name = name
        self.prompt = prompt
        self.weight = weight


# Default decomposition. Criteria decomposition is the paper's third scaling
# axis: judging independent facets and combining them beats one monolithic
# "is this correct?" question.
DEFAULT_CRITERIA: List[Criterion] = [
    Criterion(
        "syntax",
        "Is this {dialect} SQL syntactically and structurally well-formed? "
        "Answer with a single token.",
    ),
    Criterion(
        "schema",
        "Do the tables and columns referenced in this {dialect} SQL look "
        "plausibly named given the schema? Answer with a single token.",
    ),
    Criterion(
        "intent",
        "Does this {dialect} SQL correctly satisfy the user's stated intent? "
        "Answer with a single token.",
    ),
]


def softmax_logprobs(logprobs: Dict[str, float]) -> Dict[str, float]:
    """Convert log-probabilities to a normalized probability distribution.

    OpenAI returns log-probs for a subset of top tokens; renormalizing that
    subset is exactly the expectation the paper computes over scoring-token
    logits.
    """
    if not logprobs:
        return {}
    highest = max(logprobs.values())
    exps = {token: math.exp(value - highest) for token, value in logprobs.items()}
    total = sum(exps.values())
    if total <= 0:
        return {}
    return {token: value / total for token, value in exps.items()}


def expected_grade(
    token_probs: Dict[str, float], rubric: GranularityRubric
) -> Optional[float]:
    """Expected grade over a rubric = sum(grade_token * prob_token) / total.

    This is the paper's expected scoring-token score in [0, 1]. Returns None
    when none of the rubric's tokens were considered (genuinely uncertain).
    """
    mass = {t: p for t, p in token_probs.items() if t in rubric}
    total = sum(mass.values())
    if total <= 0:
        return None
    return sum(rubric[t] * p for t, p in mass.items()) / total


def score_criterion(
    scorer: LogProbScorer,
    prompt: str,
    rubrics: Optional[List[GranularityRubric]] = None,
    n_repeats: int = 1,
) -> float:
    """Score one criterion as a continuous value in [0, 1].

    Aggregates across granularity rubrics (axis 1) and repeated evaluations
    (axis 2). Each rubric contributes its expected grade; repeating the
    stochastic read ``n_repeats`` times and averaging reduces variance.
    """
    rubrics = rubrics if rubrics is not None else DEFAULT_RUBRICS
    n_repeats = max(1, n_repeats)
    grades: List[float] = []
    for rubric in rubrics:
        repeats: List[float] = []
        for _ in range(n_repeats):
            grade = expected_grade(scorer(prompt) or {}, rubric)
            if grade is not None:
                repeats.append(grade)
        if repeats:
            grades.append(sum(repeats) / len(repeats))
    if not grades:
        # No rubric token was ever emitted: treat as maximally uncertain rather
        # than silently passing or failing the query.
        return 0.5
    return sum(grades) / len(grades)


def verify_sql(
    scorer: LogProbScorer,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    criteria: Optional[List[Criterion]] = None,
    rubrics: Optional[List[GranularityRubric]] = None,
    n_repeats: int = 1,
) -> Dict[str, Any]:
    """Run the full LLM-as-a-Verifier pipeline over a SQL query.

    Returns ``{"score": float, "criteria": {name: score}}``. The intent
    criterion is skipped when no natural-language intent is available, since it
    cannot be judged without one.
    """
    context = context or {}
    criteria = criteria if criteria is not None else DEFAULT_CRITERIA
    breakdown: Dict[str, float] = {}
    weighted: List[tuple] = []
    for criterion in criteria:
        if criterion.name == "intent" and not context.get("question"):
            continue
        prompt = _build_prompt(criterion, query, context)
        score = score_criterion(scorer, prompt, rubrics, n_repeats)
        breakdown[criterion.name] = score
        weighted.append((score, criterion.weight))
    total_weight = sum(weight for _, weight in weighted)
    if total_weight <= 0:
        overall = 0.5
    else:
        overall = sum(score * weight for score, weight in weighted) / total_weight
    return {"score": overall, "criteria": breakdown}


def _build_prompt(criterion: Criterion, query: str, context: Dict[str, Any]) -> str:
    dialect = context.get("dialect") or "SQL"
    body = criterion.prompt.format(dialect=dialect)
    question = context.get("question")
    schemas = context.get("table_schemas")
    parts = [body, "SQL:", query.strip()]
    if question:
        parts.extend(["User intent:", str(question)])
    if schemas:
        parts.extend(["Available schema:", json.dumps(schemas, default=str)])
    parts.append("Verdict token:")
    return "\n".join(parts)


class LLMVerificationValidator(BaseQueryValidator):
    """Query validator that scores SQL correctness continuously via an LLM.

    Registered through the ``query_validation_plugin`` slot and invoked by the
    existing validation endpoint (``get_validator_by_name``). Emits a GENERAL
    ``QueryValidationResult`` carrying the continuous score and per-criterion
    breakdown; escalates to a WARNING when confidence is low.
    """

    # Below this continuous score the validator emits an additional WARNING so
    # likely-incorrect generated SQL is flagged for review.
    DEFAULT_WARNING_THRESHOLD = 0.5

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config or {})
        # Injection seam for tests; lazily built otherwise.
        self._scorer: Optional[LogProbScorer] = config.get("scorer") if config else None
        self._warning_threshold = (
            (config or {}).get("warning_threshold", self.DEFAULT_WARNING_THRESHOLD)
            if config
            else self.DEFAULT_WARNING_THRESHOLD
        )
        self._n_repeats = int((config or {}).get("n_repeats", 1)) if config else 1

    def set_scorer(self, scorer: Optional[LogProbScorer]) -> None:
        """Override the LLM scorer (used by tests, or to share a configured client)."""
        self._scorer = scorer

    def languages(self) -> List[str]:
        # The verifier is dialect-agnostic: the prompt carries the dialect and
        # the backing LLM handles it. Restricting languages would hide it from
        # engines it could legitimately score.
        return ["presto", "trino", "hive", "spark", "snowflake", "bigquery", "mysql"]

    def _get_scorer(self) -> Optional[LogProbScorer]:
        if self._scorer is not None:
            return self._scorer
        self._scorer = make_openai_logprob_scorer(self._config)
        return self._scorer

    def validate(
        self,
        query: str,
        uid: int,
        engine_id: int,
        **kwargs,
    ) -> List[QueryValidationResult]:
        scorer = self._get_scorer()
        if scorer is None:
            # Never break the validation flow when no LLM is configured.
            return [
                QueryValidationResult(
                    0,
                    0,
                    QueryValidationSeverity.INFO,
                    "LLM verification skipped (no verifier model configured).",
                    obj_type=QueryValidationResultObjectType.GENERAL,
                )
            ]

        context = {
            "dialect": kwargs.get("dialect") or kwargs.get("language") or "SQL",
            "question": kwargs.get("question"),
            "table_schemas": kwargs.get("table_schemas"),
        }
        result = verify_sql(
            scorer,
            query=query,
            context=context,
            n_repeats=self._n_repeats,
        )
        return _results_from_score(result, self._warning_threshold)


def _results_from_score(
    result: Dict[str, Any], warning_threshold: float
) -> List[QueryValidationResult]:
    score = result["score"]
    breakdown = result.get("criteria", {})
    detail = " ".join(
        f"{name}={value:.2f}" for name, value in sorted(breakdown.items())
    )
    info = QueryValidationResult(
        0,
        0,
        QueryValidationSeverity.INFO,
        f"LLM verification score: {score:.2f}. {detail}".strip(),
        obj_type=QueryValidationResultObjectType.GENERAL,
        suggestion=json.dumps({"score": round(score, 4), "criteria": breakdown}),
    )
    if score < warning_threshold:
        info_list: List[QueryValidationResult] = [
            QueryValidationResult(
                0,
                0,
                QueryValidationSeverity.WARNING,
                "Low LLM verification confidence; consider reviewing the generated SQL.",
                obj_type=QueryValidationResultObjectType.GENERAL,
                suggestion=json.dumps(
                    {"score": round(score, 4), "criteria": breakdown}
                ),
            )
        ]
        return info_list + [info]
    return [info]


def parse_openai_logprobs(response: Any) -> Dict[str, float]:
    """Extract a token -> probability distribution from a langchain OpenAI reply."""
    metadata = getattr(response, "response_metadata", None) or {}
    logprobs = metadata.get("logprobs") or []
    if not logprobs:
        return {}
    first = logprobs[0] if isinstance(logprobs, list) else {}
    candidates = (first or {}).get("top_logprobs") or []
    raw = {}
    for candidate in candidates:
        token = candidate.get("token") if isinstance(candidate, dict) else None
        logprob = candidate.get("logprob") if isinstance(candidate, dict) else None
        if token is None or logprob is None:
            continue
        token = token.strip().lower()
        if token:
            raw[token] = float(logprob)
    if not raw:
        token = (first or {}).get("token")
        return {str(token).lower(): 1.0} if token else {}
    return softmax_logprobs(raw)


def make_openai_logprob_scorer(
    config: Optional[Dict[str, Any]] = None,
) -> Optional[LogProbScorer]:
    """Build a production scorer from OpenAI logprobs, or None if unconfigured.

    Returns None (so the validator degrades gracefully) when the OpenAI client
    or API key is unavailable, rather than raising at import time.
    """
    config = config or {}
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    model = config.get("model_name") or os.environ.get(
        "LLM_VERIFIER_MODEL", "gpt-4o-mini"
    )
    llm = ChatOpenAI(
        model=model,
        temperature=config.get("temperature", 0.0),
        logprobs=True,
        top_logprobs=int(config.get("top_logprobs", 20)),
        max_tokens=1,
        api_key=api_key,
    )

    def _score(prompt: str) -> Dict[str, float]:
        response = llm.invoke(prompt)
        return parse_openai_logprobs(response)

    return _score
