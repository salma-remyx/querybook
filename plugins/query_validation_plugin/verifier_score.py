"""Continuous verification scoring.

Implements the *core scoring mechanism* of "LLM-as-a-Verifier: A
General-Purpose Verification Framework" (arXiv:2607.05391).

A standard LM judge prompts a model for a *discrete* verdict and keeps only
the argmax token, discarding calibration. LLM-as-a-Verifier instead reads the
model's logprobs over a set of *scoring tokens* and computes the expected
score over that distribution, yielding a **continuous** score. The paper shows
this exposes three scaling axes that each improve verification accuracy:

  1. **score granularity** -- a finer scoring-token set separates positive
     from negative candidates more sharply;
  2. **repeated evaluation** -- averaging several stochastic draws reduces
     score variance;
  3. **criteria decomposition** -- scoring several rubric dimensions and
     combining them reduces per-decision complexity.

This module is deliberately free of Querybook / LLM-client dependencies so the
math can be unit-tested in isolation. The provider that fetches raw logprobs
and the validator that turns scores into ``QueryValidationResult`` objects
live in ``llm_sql_verifier``.
"""

import math
from typing import Dict, List, Mapping, Sequence

# A scoring token mapped to the numeric value it represents, e.g.
# {"yes": 1.0, "no": 0.0} (binary) or {"1": 0.0, ..., "5": 1.0} (granular).
TokenScale = Mapping[str, float]
# logprob of each candidate scoring token, as returned by the model.
LogProbs = Mapping[str, float]


def normalize_logprobs(logprobs: LogProbs) -> Dict[str, float]:
    """Renormalize candidate-token logprobs into a probability distribution.

    A language model spreads probability across its entire vocabulary, so the
    logprobs of a small candidate set do not sum to one. Softmax over the
    candidate logits recovers "probability mass among the choices" -- exactly
    the distribution LLM-as-a-Verifier takes its expectation over.
    """
    if not logprobs:
        return {}
    max_lp = max(logprobs.values())
    exps = {token: math.exp(lp - max_lp) for token, lp in logprobs.items()}
    total = sum(exps.values())
    if total <= 0:
        # Every candidate underflowed; fall back to a uniform distribution so
        # the caller still gets a well-formed expectation (maximally uncertain).
        count = len(logprobs)
        return {token: 1.0 / count for token in logprobs}
    return {token: mass / total for token, mass in exps.items()}


def expectation_score(scale: TokenScale, logprobs: LogProbs) -> float:
    """Continuous score = ``E[value]`` over the scoring-token distribution.

    This is the paper's central operator. A discrete judge would return
    ``argmax(value)`` and lose calibration; the expectation keeps the full
    distribution, so as the granularity of ``scale`` grows the scores of
    positive and negative candidates separate more cleanly.
    """
    candidates = {token: lp for token, lp in logprobs.items() if token in scale}
    if not candidates:
        # No usable signal from the requested scale: return a neutral score.
        return 0.5
    probabilities = normalize_logprobs(candidates)
    return sum(probabilities[token] * scale[token] for token in probabilities)


def aggregate_repeats(scores: Sequence[float]) -> float:
    """Repeated-evaluation axis: average several draws to cut variance."""
    values = list(scores)
    if not values:
        return 0.5
    return sum(values) / len(values)


def aggregate_criteria(criterion_scores: Mapping[str, float]) -> float:
    """Criteria-decomposition axis: combine rubric dimensions (mean)."""
    values = list(criterion_scores.values())
    if not values:
        return 0.5
    return sum(values) / len(values)


# Binary "yes/no" scale -- the coarsest granularity a judge can use.
BINARY_SCALE: TokenScale = {"no": 0.0, "yes": 1.0}

# Five-level granular scale. Finer granularity than BINARY_SCALE, so for the
# same logprob distribution it yields more separated, better-calibrated scores.
DEFAULT_GRANULAR_SCALE: TokenScale = {
    "1": 0.0,
    "2": 0.25,
    "3": 0.5,
    "4": 0.75,
    "5": 1.0,
}

ALL_SCALES: List[TokenScale] = [BINARY_SCALE, DEFAULT_GRANULAR_SCALE]
