"""Multi-sample critiquing for text-to-SQL candidate selection.

This adapts the core idea of MSc-SQL (Putta et al., 2024, "Multi-Sample
Critiquing for Text-to-SQL", arXiv:2410.12916): instead of trusting a single
NL->SQL generation, sample several candidates and keep the one a *critique*
judges best.

The paper trains a dedicated critique model. We do not ship a trained model
here -- the critique is a parameter-free, metadata-grounded scorer that
approximates the same signal using the table schema Querybook already gathers
for the prompt. It rewards candidates whose referenced identifiers actually
exist in the provided schema and whose referenced columns overlap the question,
and it penalises unparsable or refusal ("explanation"-only) outputs. In the
contribution-mode terms: the multi-sample-then-select mechanism is faithful,
while the learned critic is replaced by a schema-overlap proxy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import sqlglot
from sqlglot import exp

# Common English / SQL words that carry no signal for question-column overlap.
_STOP_WORDS = frozenset("""
    a an the of to in on at and or not is are was were be been being as by for
    with from into where select group order limit join left right inner outer
    count sum avg min max distinct how many list show get find give me i we
    this that these those per each all every who which what when why how do does
    """.split())


@dataclass(frozen=True)
class CandidateScore:
    """Score breakdown for a single candidate SQL.

    ``total`` is the value selection maximises; the sub-scores exist so callers
    (and tests) can reason about *why* one candidate beat another.
    """

    total: float
    validity: float
    schema_grounding: float
    question_relevance: float
    reason: str = ""


def _known_identifiers(
    table_schemas: Iterable[Optional[dict[str, Any]]],
) -> tuple[set[str], set[str]]:
    """Collect the tables and columns the prompt actually exposed.

    Returns lower-cased sets so the grounding check is case-insensitive. Both
    the fully qualified ``schema.table`` and the bare ``table`` token are kept,
    so a candidate that references either form counts as grounded.
    """
    tables: set[str] = set()
    columns: set[str] = set()
    for schema in table_schemas or []:
        if not schema:
            continue
        name = schema.get("table_name") or ""
        if name:
            name = name.lower()
            tables.add(name)
            if "." in name:
                tables.add(name.rsplit(".", 1)[-1])
        for column in schema.get("columns") or []:
            col_name = (column or {}).get("name")
            if col_name:
                columns.add(col_name.lower())
    return tables, columns


def _referenced_identifiers(parsed: exp.Expression) -> tuple[set[str], set[str]]:
    """Return the (tables, columns) a parsed SQL expression references."""
    tables = {table.name.lower() for table in parsed.find_all(exp.Table)}
    columns = {column.name.lower() for column in parsed.find_all(exp.Column)}
    return tables, columns


def _question_tokens(question: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9_]+", (question or "").lower())
    return {token for token in tokens if token not in _STOP_WORDS and len(token) > 1}


def _coverage(
    ref_tables: set[str],
    known_tables: set[str],
    ref_columns: set[str],
    known_columns: set[str],
) -> float:
    """Mean fraction of referenced identifiers that exist in the schema.

    Tables and columns are weighted equally; a candidate that references no
    identifiers at all scores 0 (it never bound to a provided table).
    """
    components = []
    if ref_tables:
        components.append(len(ref_tables & known_tables) / len(ref_tables))
    if ref_columns:
        components.append(len(ref_columns & known_columns) / len(ref_columns))
    if not components:
        return 0.0
    return sum(components) / len(components)


def score_sql_candidate(
    candidate: Any,
    *,
    table_schemas: Iterable[Optional[dict[str, Any]]],
    question: str,
    dialect: Optional[str] = None,
) -> CandidateScore:
    """Score one NL->SQL candidate against the schema metadata.

    ``candidate`` is the JSON the model returns (``{"query": ..., "explanation": ...}``);
    a bare SQL string is also accepted. Higher ``total`` is better. The total
    blends three parameter-free signals:

    * ``validity`` (weight 0.4): the candidate produced a non-empty, parsable
      SQL statement rather than a refusal explanation.
    * ``schema_grounding`` (weight 0.4): of the tables/columns the SQL
      references, the fraction that exist in the provided schema -- rewards
      real identifiers, penalises hallucinated ones.
    * ``question_relevance`` (weight 0.2): overlap between question keywords
      and the columns the SQL touches, a cheap proxy for "answers the ask".
    """
    if isinstance(candidate, dict):
        query = (candidate.get("query") or "").strip()
    else:
        query = (candidate or "").strip()

    if not query:
        return CandidateScore(0.0, 0.0, 0.0, 0.0, reason="empty query")

    try:
        parsed = sqlglot.parse_one(query, read=dialect or None)
    except Exception:
        # Unparsable SQL cannot be executed, so it cannot be the best pick.
        return CandidateScore(0.0, 0.0, 0.0, 0.0, reason="unparsable")

    known_tables, known_columns = _known_identifiers(table_schemas)
    ref_tables, ref_columns = _referenced_identifiers(parsed)
    grounding = _coverage(ref_tables, known_tables, ref_columns, known_columns)

    question_tokens = _question_tokens(question)
    relevance = (
        len(ref_columns & question_tokens) / len(question_tokens)
        if question_tokens
        else 0.0
    )

    total = 0.4 * 1.0 + 0.4 * grounding + 0.2 * relevance
    return CandidateScore(
        total=total,
        validity=1.0,
        schema_grounding=grounding,
        question_relevance=relevance,
        reason="ok",
    )


def select_best_sql_candidate(
    candidates: Iterable[Any],
    *,
    table_schemas: Iterable[Optional[dict[str, Any]]],
    question: str,
    dialect: Optional[str] = None,
) -> Optional[Any]:
    """Pick the highest-scoring candidate.

    Returns the original candidate object unchanged so the caller can stream it
    as-is. Ties break toward validity, then schema grounding. Returns ``None``
    only when ``candidates`` is empty.
    """
    best = None
    best_score: Optional[CandidateScore] = None
    for candidate in candidates:
        score = score_sql_candidate(
            candidate,
            table_schemas=table_schemas,
            question=question,
            dialect=dialect,
        )
        if best_score is None or _is_better(score, best_score):
            best = candidate
            best_score = score
    return best


def critique_and_select(
    sample: Any,
    *,
    sample_count: int,
    table_schemas: Iterable[Optional[dict[str, Any]]],
    question: str,
    dialect: Optional[str] = None,
    on_sample_error: Any = None,
) -> Optional[Any]:
    """Run multi-sample critiquing and return the best candidate.

    This is the MSc-SQL inference-time loop: draw ``sample_count`` candidates,
    then keep the one the critique scores highest. ``sample`` is a zero-arg
    callable returning one parsed candidate (the JSON the model returns); it is
    invoked up to ``sample_count`` times. A sampling exception is swallowed and
    reported through ``on_sample_error`` (if given) so a single bad draw does
    not sink the whole batch. Returns the selected candidate, or ``None`` when
    every draw failed.
    """
    candidates = []
    for _ in range(sample_count):
        try:
            candidates.append(sample())
        except Exception as exc:  # one bad sample shouldn't sink the batch
            if on_sample_error is not None:
                on_sample_error(exc)

    if not candidates:
        return None

    return select_best_sql_candidate(
        candidates,
        table_schemas=table_schemas,
        question=question,
        dialect=dialect,
    )


def _is_better(a: CandidateScore, b: CandidateScore) -> bool:
    return (a.total, a.validity, a.schema_grounding) > (
        b.total,
        b.validity,
        b.schema_grounding,
    )
