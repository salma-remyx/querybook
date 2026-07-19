"""SQL verification via LLM-as-a-Verifier continuous scoring.

Adapted from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391). The paper's core mechanism scores a candidate solution by
taking the *expectation over the distribution of scoring-token logits* rather
than trusting a single discrete judge token. That yields a continuous,
calibrated score and unlocks the three scaling axes the paper identifies:
(1) score granularity, (2) repeated evaluation, and (3) criteria decomposition.

This module ports that core mechanism onto Querybook's existing
ChatOpenAI / langchain path: the model is prompted to emit a 0-9 score digit
as its first token, the digit's logprob distribution is read back from the
response, and its expected value is taken as the score. The paper's separate
benchmark suite is intentionally out of scope (evaluation belongs downstream).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage

from .prompts.sql_verify_prompt import SQL_VERIFY_PROMPT

# How many top scoring-token logprobs to request per call.
SQL_VERIFIER_TOP_LOGPROBS = 10

# Scoring vocabulary: digits 0-9 mapped to a value in [0, 1]. More tokens mean
# finer granularity (the paper's scaling axis #1).
DigitVocab = Dict[str, float]


def default_scoring_vocab() -> DigitVocab:
    return {str(d): d / 9.0 for d in range(10)}


# (name, description, weight) -- criteria decomposition (paper axis #3).
Criterion = Tuple[str, str, float]
DEFAULT_VERIFIER_CRITERIA: List[Criterion] = [
    (
        "syntax",
        "Is the SQL syntactically valid and well-formed for the dialect?",
        1.0,
    ),
    (
        "schema",
        "Does the SQL reference only tables and columns that exist in the "
        "provided schema, with correct names and types?",
        1.0,
    ),
    (
        "intent",
        "Does the SQL correctly answer the user's question or complete the "
        "intended query?",
        1.0,
    ),
]


@dataclass
class SQLVerificationResult:
    """Continuous verification score for a generated SQL query."""

    score: float
    per_criterion: Dict[str, float] = field(default_factory=dict)
    samples: Dict[str, List[float]] = field(default_factory=dict)
    method: str = "logprobs"
    n_samples: int = 1

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "per_criterion": dict(self.per_criterion),
            "method": self.method,
            "n_samples": self.n_samples,
        }


class SQLVerifier:
    """Score generated SQL with continuous LLM-as-a-Verifier scoring.

    The verifier prompts the model to emit a 0-9 score digit as the first
    token, then -- instead of trusting that digit -- reads back the logprob
    distribution over scoring tokens and takes its expected value. Criteria
    are scored independently and averaged (decomposition); each criterion may
    be sampled repeatedly and averaged (variance reduction).
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        criteria: Optional[Sequence[Criterion]] = None,
        top_logprobs: int = SQL_VERIFIER_TOP_LOGPROBS,
        vocab: Optional[DigitVocab] = None,
    ):
        self._llm = llm
        self._criteria: List[Criterion] = (
            list(criteria) if criteria else list(DEFAULT_VERIFIER_CRITERIA)
        )
        self._top_logprobs = top_logprobs
        self._vocab: DigitVocab = vocab or default_scoring_vocab()

    # -- public API ---------------------------------------------------------

    def verify(
        self,
        dialect: str,
        question: str,
        table_schemas,
        generated_query: str,
        n_samples: int = 1,
    ) -> SQLVerificationResult:
        """Verify a single generated query, returning a continuous score."""
        n_samples = max(1, int(n_samples))
        total_weight = sum(w for _, _, w in self._criteria) or 1.0
        per_criterion: Dict[str, float] = {}
        samples: Dict[str, List[float]] = {}
        method = "logprobs"
        for name, desc, weight in self._criteria:
            trial_scores: List[float] = []
            for _ in range(n_samples):
                score, used = self._score_criterion(
                    dialect=dialect,
                    question=question,
                    table_schemas=table_schemas,
                    generated_query=generated_query,
                    name=name,
                    desc=desc,
                )
                trial_scores.append(score)
                if used != "logprobs":
                    method = used
            per_criterion[name] = sum(trial_scores) / len(trial_scores)
            samples[name] = trial_scores
        overall = (
            sum(per_criterion[name] * w for name, _, w in self._criteria) / total_weight
        )
        return SQLVerificationResult(
            score=overall,
            per_criterion=per_criterion,
            samples=samples,
            method=method,
            n_samples=n_samples,
        )

    def rank(
        self,
        dialect: str,
        question: str,
        table_schemas,
        candidates: Sequence[str],
        n_samples: int = 1,
        budget: Optional[int] = None,
    ) -> List[Tuple[str, SQLVerificationResult]]:
        """Rank candidate SQL queries best-first by verification score.

        With no ``budget`` every candidate is scored fully and sorted. With a
        finite ``budget`` (max candidate evaluations) promising candidates are
        refined via sequential halving -- the paper's cost-efficient selection
        -- so the best candidate is found with fewer total evaluations.
        Returns the ranked candidates that survived the (optional) halving.
        """
        candidates = list(candidates)
        if not candidates:
            return []
        if budget is None or budget >= len(candidates):
            scored = [
                (
                    c,
                    self.verify(
                        dialect, question, table_schemas, c, n_samples=n_samples
                    ),
                )
                for c in candidates
            ]
            scored.sort(key=lambda cr: cr[1].score, reverse=True)
            return scored
        return self._rank_sequential_halving(
            dialect, question, table_schemas, candidates, budget
        )

    # -- internals ----------------------------------------------------------

    def _rank_sequential_halving(
        self, dialect, question, table_schemas, candidates, budget
    ) -> List[Tuple[str, SQLVerificationResult]]:
        survivors = list(candidates)
        results: Dict[str, SQLVerificationResult] = {}
        remaining = budget
        while len(survivors) > 1 and remaining > 0:
            round_score: Dict[str, float] = {}
            for candidate in list(survivors):
                result = self.verify(dialect, question, table_schemas, candidate)
                results[candidate] = result
                round_score[candidate] = result.score
                remaining -= 1
                if remaining <= 0:
                    break
            survivors.sort(key=lambda c: round_score.get(c, 0.0), reverse=True)
            survivors = survivors[: max(1, len(survivors) // 2)]
        ranked = [
            (
                c,
                results.get(c) or self.verify(dialect, question, table_schemas, c),
            )
            for c in survivors
        ]
        ranked.sort(key=lambda cr: cr[1].score, reverse=True)
        return ranked

    def _score_criterion(
        self, dialect, question, table_schemas, generated_query, name, desc
    ) -> Tuple[float, str]:
        prompt = SQL_VERIFY_PROMPT.format(
            dialect=dialect,
            criterion=name,
            criterion_description=desc,
            question=question,
            table_schemas=table_schemas,
            generated_query=generated_query,
        )
        message = self._invoke(prompt)
        return self._expected_score(message)

    def _invoke(self, prompt: str) -> AIMessage:
        # Enable scoring-token logprobs for this call when the backend supports
        # it (ChatOpenAI forwards bound kwargs to the API).
        try:
            bound = self._llm.bind(logprobs=True, top_logprobs=self._top_logprobs)
        except Exception:
            bound = self._llm
        return bound.invoke(prompt)

    def _expected_score(self, message: AIMessage) -> Tuple[float, str]:
        content = self._extract_logprob_content(message)
        if content:
            hits = self._find_scoring_position(content)
            if hits:
                return self._expected_from_hits(hits), "logprobs"
        return self._fallback_discrete_score(message), "discrete_fallback"

    @staticmethod
    def _extract_logprob_content(message: AIMessage) -> Optional[List[dict]]:
        meta = getattr(message, "response_metadata", None) or {}
        logprobs = meta.get("logprobs")
        if not logprobs:
            extra = getattr(message, "additional_kwargs", None) or {}
            logprobs = extra.get("logprobs")
        if isinstance(logprobs, dict):
            return logprobs.get("content")
        return None

    def _find_scoring_position(self, content) -> List[Tuple[str, float]]:
        """First response position whose tokens include a scoring digit."""
        for entry in content:
            if not entry:
                continue
            hits: List[Tuple[str, float]] = []
            emitted = entry.get("token")
            if emitted is not None:
                hits.append((emitted, float(entry.get("logprob", 0.0))))
            for top in entry.get("top_logprobs") or []:
                hits.append((top["token"], float(top["logprob"])))
            hits = [
                (tok, lp)
                for tok, lp in hits
                if self._normalize_token(tok) in self._vocab
            ]
            if hits:
                return hits
        return []

    def _expected_from_hits(self, hits: Sequence[Tuple[str, float]]) -> float:
        # Softmax over the scoring-vocab support present at this position.
        token_lp: Dict[str, float] = {}
        for tok, lp in hits:
            key = self._normalize_token(tok)
            token_lp[key] = max(token_lp.get(key, -1e9), lp)
        keys = list(token_lp)
        lps = [token_lp[k] for k in keys]
        m = max(lps)
        exps = [math.exp(lp - m) for lp in lps]
        total = sum(exps) or 1.0
        return sum(self._vocab[keys[i]] * exps[i] for i in range(len(keys))) / total

    def _fallback_discrete_score(self, message: AIMessage) -> float:
        content = message.content
        text = content if isinstance(content, str) else str(content or "")
        for char in text.strip():
            digit = char.strip()
            if digit in self._vocab:
                return self._vocab[digit]
        return 0.5

    @staticmethod
    def _normalize_token(token: str) -> str:
        return (token or "").strip()
