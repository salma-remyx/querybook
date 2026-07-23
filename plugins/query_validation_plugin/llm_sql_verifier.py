"""LLM-as-a-Verifier validator for generated SQL.

Wires the continuous, logprob-based scoring from ``verifier_score`` (adapted
from "LLM-as-a-Verifier: A General-Purpose Verification Framework",
arXiv:2607.05391) into Querybook's ``query_validation_plugin`` extension
point. It scores LLM-generated SQL (Text2SQL output / inline autocomplete)
along several rubric criteria and surfaces a low-confidence query as an
advisory ``QueryValidationResult``.

Adapted port (Mode 2). The paper's *core mechanism* -- taking the expected
score over the distribution of scoring-token logits, then scaling it through
granularity / repeated evaluation / criteria decomposition -- is implemented
verbatim in ``verifier_score``. The *auxiliary* components are target-native:

  * **No fine-tuned verifier model.** Any chat LLM that returns logprobs is
    used through an injectable provider; the default builds an
    OpenAI-compatible client from the standard ``OPENAI_*`` environment.
  * **No bespoke benchmark suite.** The paper's Terminal-Bench / SWE-Bench /
    RoboRewardBench evaluation harness is out of scope -- evaluation belongs
    in a downstream PR.

Intentionally scoped out (would need a different call site): the paper's
cost-efficient *ranking* algorithm for choosing among several candidate
solutions, which requires the validator to see multiple candidates at once --
the ``BaseQueryValidator.validate`` contract only receives a single query.
"""

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from lib.query_analysis.validation.base_query_validator import (
    BaseQueryValidator,
    QueryValidationResult,
    QueryValidationResultObjectType,
    QueryValidationSeverity,
)

from .verifier_score import (
    DEFAULT_GRANULAR_SCALE,
    TokenScale,
    aggregate_criteria,
    aggregate_repeats,
    expectation_score,
)

LOG = logging.getLogger(__name__)

# A provider maps (judge prompt, candidate scoring tokens) -> {token: logprob}.
# It is the only piece of I/O in the validator, which makes it trivial to swap
# (e.g. for a different vendor) or to fake in tests.
LogProbProvider = Callable[[str, List[str]], Mapping[str, float]]

# Each criterion is judged independently from the query text alone (criteria
# decomposition axis). They are chosen so no external schema / intent context
# is required -- ``BaseQueryValidator.validate`` only receives the query.
DEFAULT_CRITERIA = ("validity", "completeness", "soundness")

_CRITERION_PROMPTS: Dict[str, str] = {
    "validity": (
        "Is the following SQL syntactically valid? Judge only grammar/syntax.\n"
        "SQL:\n{query}\n"
        "1 = clearly invalid, 5 = clearly valid."
    ),
    "completeness": (
        "Is the following SQL a complete, runnable statement rather than a "
        "dangling fragment?\nSQL:\n{query}\n"
        "1 = incomplete fragment, 5 = fully executable statement."
    ),
    "soundness": (
        "Is the following SQL logically coherent and free of obvious errors "
        "(e.g. nonsense column refs, contradictory clauses)?\nSQL:\n{query}\n"
        "1 = clearly unsound, 5 = clearly sound."
    ),
}

# Broad default so an operator can opt any engine into this verifier via
# ``feature_params.validator``. It only runs on engines explicitly configured
# to use it, so a wide language list does not trigger unwanted LLM calls.
DEFAULT_LANGUAGES = (
    "presto",
    "trino",
    "hive",
    "sparksql",
    "mysql",
    "postgres",
    "snowflake",
    "bigquery",
    "druid",
    "sqlite",
    "clickhouse",
    "mssql",
    "redshift",
    "sql",
)


def _default_logprob_provider(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> LogProbProvider:
    """Build an OpenAI-compatible provider that returns scoring-token logprobs.

    Lazily imports ``openai`` so the validator module imports cleanly even when
    no LLM client is configured (e.g. in unit tests with an injected provider).
    """
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - exercised only without openai
        raise RuntimeError(
            "The default LLM-as-a-Verifier provider needs the 'openai' package, "
            "or pass 'logprob_provider' in the validator config."
        ) from exc

    client = openai.OpenAI()

    def provider(prompt: str, scoring_tokens: List[str]) -> Mapping[str, float]:
        system = (
            "You are a strict SQL verification judge. Reply with exactly one "
            "token from this set and nothing else: " + ", ".join(scoring_tokens)
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1,
            temperature=temperature,
            logprobs=True,
            top_logprobs=min(20, max(len(scoring_tokens) + 5, 5)),
        )
        logprobs_content = response.choices[0].logprobs.content
        if not logprobs_content:
            return {}
        top = logprobs_content[0].top_logprobs or []
        # Normalise tokens (strip whitespace/case) so they line up with the
        # scoring scale, which keeps expectation_score matching robust.
        return {item.token.strip().lower(): item.logprob for item in top}

    return provider


class LLMSQLVerifierValidator(BaseQueryValidator):
    """Score SQL with an LLM-as-a-Verifier continuous confidence score.

    Config keys (all optional, passed via ``config``):

        logprob_provider: ``LogProbProvider`` used to fetch scoring-token
            logprobs. If omitted a default OpenAI-compatible provider is built
            lazily on first use.
        model_name / temperature: forwarded to the default provider.
        scale: scoring-token -> value mapping (default 5-level granularity).
        criteria: rubric dimensions scored independently (decomposition axis).
        repeats: draws per criterion, averaged (repeated-evaluation axis).
        fail_threshold: combined score below this emits an advisory result.
        severity: ``QueryValidationSeverity`` of the advisory result.
        languages: languages this validator applies to.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, config or {})
        cfg = self._config
        self._scale: TokenScale = dict(cfg.get("scale") or DEFAULT_GRANULAR_SCALE)
        self._criteria: Tuple[str, ...] = tuple(cfg.get("criteria") or DEFAULT_CRITERIA)
        self._repeats: int = max(1, int(cfg.get("repeats", 1)))
        self._fail_threshold: float = float(cfg.get("fail_threshold", 0.5))
        self._severity = cfg.get("severity", QueryValidationSeverity.INFO)
        self._languages: List[str] = list(cfg.get("languages") or DEFAULT_LANGUAGES)
        # Lazily constructed if not injected.
        self._provider: Optional[LogProbProvider] = cfg.get("logprob_provider")

    def _get_provider(self) -> LogProbProvider:
        if self._provider is None:
            self._provider = _default_logprob_provider(
                model_name=self._config.get("model_name", "gpt-4o-mini"),
                temperature=float(self._config.get("temperature", 0.0)),
            )
        return self._provider

    def languages(self) -> List[str]:
        return list(self._languages)

    def score(self, query: str) -> Tuple[float, Dict[str, float]]:
        """Return ``(overall_score, per_criterion_scores)`` for ``query``.

        For each rubric criterion the judge prompt is evaluated ``repeats``
        times (repeated-evaluation axis), each draw reduced to a continuous
        score via ``expectation_score``; the per-criterion scores are then
        combined (criteria-decomposition axis).
        """
        provider = self._get_provider()
        tokens = list(self._scale.keys())
        per_criterion: Dict[str, float] = {}
        for criterion in self._criteria:
            template = _CRITERION_PROMPTS.get(
                criterion, _CRITERION_PROMPTS["soundness"]
            )
            prompt = template.format(query=query)
            draws: List[float] = []
            for _ in range(self._repeats):
                logprobs = provider(prompt, tokens)
                draws.append(expectation_score(self._scale, logprobs))
            per_criterion[criterion] = aggregate_repeats(draws)
        return aggregate_criteria(per_criterion), per_criterion

    def validate(
        self,
        query: str,
        uid: int,
        engine_id: int,
        **kwargs,
    ) -> List[QueryValidationResult]:
        # The verifier is an advisory, probabilistic judge: never let a provider
        # failure (e.g. missing API key, network error) break /query/validate/.
        try:
            overall, per_criterion = self.score(query)
        except Exception as exc:  # noqa: BLE001 - fail open, log and skip
            LOG.warning("LLM SQL verifier skipped a query: %s", exc)
            return []

        if overall >= self._fail_threshold:
            return []

        breakdown = ", ".join(f"{c}={s:.2f}" for c, s in per_criterion.items())
        return [
            QueryValidationResult(
                0,
                0,
                self._severity,
                f"LLM verifier low confidence (score {overall:.2f}): {breakdown}",
                QueryValidationResultObjectType.GENERAL,
            )
        ]
