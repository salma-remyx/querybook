"""Question-complexity estimation for Text2SQL prompt building.

Adapted from the E3 framework (Estimate, Execute, Expand) in
"Do AI Agents Know When a Task Is Simple? Toward Complexity-Aware
Reasoning and Execution" (https://arxiv.org/abs/2607.13034v1).

The paper's core insight is *task-aware execution-scope estimation*: judge how
much context a task truly needs and commit only the minimum-sufficient scope
before executing, rather than always reaching for maximum context first and
trimming reactively. Querybook's Text2SQL prompt builder historically did the
latter -- it always rendered full table schemas (every column description, data
element, and statistic) and only fell back to the slimmed schema when the
prompt blew past the model's context window.

This module implements the E3 **Estimate** step for that builder: a
parameter-free classifier that scores the relational complexity implied by a
natural-language question and maps it to the minimum-sufficient schema depth.
The paper's learned / LLM-based complexity judge is replaced by this
keyword-scoring proxy (a parameter-free estimator standing in for the paper's
auxiliary learned estimator), which keeps the Estimate step free of any extra
model call -- the same property that makes E3 leaner than the baselines.

Classification is deliberately conservative on the side of correctness: a
question only qualifies for the slimmed schema when it shows no aggregation,
join, or nesting signal at all. Misclassifying a complex question as simple
only drops column *descriptions* (column names and types are still present in
the slimmed view), while misclassifying a simple question as complex merely
keeps the existing full-schema behavior -- so neither error regresses query
generation.
"""

import re
from enum import Enum
from typing import Iterable


class QueryComplexity(Enum):
    """Relational complexity tiers implied by a natural-language question.

    Mirrors the breakdown suggested for an E3 Estimate step (simple lookup vs.
    aggregation vs. join/nesting). ``MODERATE`` and ``COMPLEX`` both warrant the
    full schema; the distinction is retained because it is meaningful for
    logging and future depth tiers.
    """

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class SchemaDepth(Enum):
    """How much of a table schema to render into a Text2SQL prompt."""

    SLIMMED = "slimmed"
    FULL = "full"


# Keyword signals used by the parameter-free estimator. Each phrase is matched
# on word boundaries so that, e.g., "count" does not fire inside "account" or
# "discount". These approximate -- without a learned model -- the relational
# footprint (projection vs. aggregation vs. join vs. nesting) a question implies.
_AGGREGATION_SIGNALS = frozenset(
    {
        "aggregate",
        "aggregation",
        "average",
        "avg",
        "count",
        "counts",
        "distinct",
        "group by",
        "grouped",
        "how many",
        "max",
        "mean",
        "median",
        "min",
        "number of",
        "per",
        "percentile",
        "percentage",
        "proportion",
        "quantile",
        "ratio",
        "standard deviation",
        "stddev",
        "sum",
        "total",
        "totals",
        "unique",
        "variance",
    }
)

_JOIN_SIGNALS = frozenset(
    {
        "across",
        "along with",
        "combine",
        "combined",
        "corresponding",
        "correlate",
        "correlation",
        "intersect",
        "join",
        "joined",
        "joins",
        "look up",
        "lookup",
        "merged",
        "relate",
        "related",
        "relating",
        "relationship",
        "together with",
        "union",
    }
)

_NESTING_SIGNALS = frozenset(
    {
        "cumulative",
        "for each",
        "hierarchical",
        "hierarchy",
        "nested",
        "over",
        "partition",
        "rank",
        "ranking",
        "recursive",
        "row number",
        "running",
        "subquery",
        "window",
    }
)

# Weight each family by how much relational machinery it tends to require:
# a single aggregation is cheap, a join is heavier, and nesting/subqueries are
# heaviest. Thresholds then carve the score into the three tiers.
_AGGREGATION_WEIGHT = 1
_JOIN_WEIGHT = 2
_NESTING_WEIGHT = 3

_SIMPLE_MAX_SCORE = 0
_MODERATE_MAX_SCORE = 2


def _count_signal_hits(text: str, signals: Iterable[str]) -> int:
    """Count non-overlapping, word-boundary occurrences of any signal phrase."""
    total = 0
    for signal in signals:
        total += len(re.findall(r"\b" + re.escape(signal) + r"\b", text))
    return total


def estimate_query_complexity(question: str) -> QueryComplexity:
    """Estimate the relational complexity implied by a Text2SQL question.

    This is the E3 **Estimate** step realized as a parameter-free proxy: it
    scores how much context the question likely needs *before* any schema is
    rendered, so the prompt builder can commit the minimum-sufficient schema
    depth instead of maximum-context-first.

    Args:
        question: The natural-language question to classify.

    Returns:
        A :class:`QueryComplexity` tier. An empty question defaults to
        ``MODERATE`` so an uninformative prompt still gets the full schema.
    """
    if not question or not question.strip():
        return QueryComplexity.MODERATE

    normalized = " ".join(question.lower().split())

    aggregation = _count_signal_hits(normalized, _AGGREGATION_SIGNALS)
    joins = _count_signal_hits(normalized, _JOIN_SIGNALS)
    nesting = _count_signal_hits(normalized, _NESTING_SIGNALS)

    score = (
        _AGGREGATION_WEIGHT * aggregation
        + _JOIN_WEIGHT * joins
        + _NESTING_WEIGHT * nesting
    )

    if score <= _SIMPLE_MAX_SCORE:
        return QueryComplexity.SIMPLE
    if score <= _MODERATE_MAX_SCORE:
        return QueryComplexity.MODERATE
    return QueryComplexity.COMPLEX


def select_schema_depth(complexity: QueryComplexity) -> SchemaDepth:
    """Map a question's complexity to the minimum-sufficient schema depth.

    Simple lookups/projections get the slimmed schema (name + type only); any
    aggregation, join, or nesting keeps the full schema so column descriptions,
    data elements, and statistics remain available to the model.

    Args:
        complexity: The tier returned by :func:`estimate_query_complexity`.

    Returns:
        ``SchemaDepth.SLIMMED`` for simple questions, ``SchemaDepth.FULL``
        otherwise.
    """
    if complexity is QueryComplexity.SIMPLE:
        return SchemaDepth.SLIMMED
    return SchemaDepth.FULL
