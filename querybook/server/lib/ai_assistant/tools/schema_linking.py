"""Soft schema linking for text-to-SQL.

Adapted from the Soft_Schema_linker in MAG-SQL (Cui et al., 2024,
https://arxiv.org/abs/2408.07930): given a natural-language question and a
set of table schemas, a model selects the columns relevant to the question
and each table's schema is pruned to those columns before the text-to-SQL
prompt is built. Focusing the generator on question-relevant columns is the
paper's core schema-linking contribution and improves accuracy on databases
with many columns.

This is an adapted implementation rather than a full port:

- The paper's separate entity-extraction agent is folded into the single
  column-selection prompt below (one model call instead of a dedicated
  extractor).
- The paper's multi-agent pipeline and iterative sub-SQL refinement are not
  reproduced here; only the schema-linking step is ported.
- The caller keeps the repo's existing token-based ``get_slimmed_table_schemas``
  as a final fallback when selection fails or the prompt is still too long.

Evaluation against the BIRD benchmark belongs in a downstream change.
"""

import json
from typing import Callable, Optional

from ..prompts.schema_linking_prompt import SCHEMA_LINKING_PROMPT
from .table_schema import TableSchema


def format_schema_linking_prompt(question: str, table_schemas) -> str:
    """Render the schema-linking prompt for a question and table schemas."""
    return SCHEMA_LINKING_PROMPT.format(question=question, table_schemas=table_schemas)


def parse_column_selection(raw) -> dict[str, list[str]]:
    """Normalize a model's schema-linking response into {table: [columns]}.

    Accepts either a pre-parsed dict (e.g. the output of ``JsonOutputParser``)
    or a raw JSON string. Returns an empty dict on any malformed input so a
    caller can safely fall back to the full schema.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return {}

    if not isinstance(raw, dict):
        return {}

    selection: dict[str, list[str]] = {}
    for table_name, columns in raw.items():
        if not isinstance(table_name, str):
            continue

        if isinstance(columns, str):
            columns = [columns]

        if not isinstance(columns, (list, tuple)):
            continue

        cleaned = [str(column).strip() for column in columns if str(column).strip()]
        if cleaned:
            selection[table_name] = cleaned

    return selection


def prune_table_schemas_by_relevance(
    table_schemas: list[Optional[TableSchema]],
    selection: dict[str, list[str]],
) -> list[Optional[TableSchema]]:
    """Return copies of ``table_schemas`` keeping only selected columns.

    For any table whose name appears in ``selection``, only the listed
    columns are kept; the remaining columns are pruned. Tables absent from
    ``selection`` (including failed/empty selections) are returned unchanged
    so a missing selection never empties a table's schema.
    """
    if not selection:
        return list(table_schemas or [])

    pruned: list[Optional[TableSchema]] = []
    for schema in table_schemas or []:
        if not schema:
            pruned.append(schema)
            continue

        wanted = selection.get(schema.get("table_name"))
        if not wanted:
            pruned.append(schema)
            continue

        wanted_names = {str(column) for column in wanted}
        kept_columns = [
            column
            for column in (schema.get("columns") or [])
            if str(column.get("name")) in wanted_names
        ]

        linked: TableSchema = dict(schema)
        linked["columns"] = kept_columns
        pruned.append(linked)

    return pruned


def link_and_prune_schemas(
    *,
    invoke: Callable[[str], object],
    question: str,
    table_schemas: list[Optional[TableSchema]],
) -> list[Optional[TableSchema]]:
    """Run soft schema linking end-to-end.

    ``invoke`` is a callable that takes the formatted prompt string and
    returns the model's response (a JSON object or a JSON string). This
    function deliberately does NOT catch exceptions: callers wrap it so they
    can fall back to the original schema on failure.
    """
    prompt = format_schema_linking_prompt(question, table_schemas)
    selection = parse_column_selection(invoke(prompt))
    return prune_table_schemas_by_relevance(table_schemas, selection)
