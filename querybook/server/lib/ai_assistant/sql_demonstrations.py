"""Selective in-domain (question -> SQL) demonstrations for text-to-SQL.

Adapted from "Selective Demonstrations for Cross-domain Text-to-SQL"
(Wang et al., 2023; arXiv:2310.06302). The paper's central finding is that
*question similarity* is the dominant signal when choosing in-context
demonstrations for text-to-SQL: retrieving past (question -> SQL) pairs whose
question resembles the user's question substantially lifts accuracy -- even
without in-domain annotations, and without needing the demonstration's schema
or gold SQL to drive selection.

Target-native adaptation (Mode 2 -- auxiliary components substituted):

  * The paper's sparse BM25 retriever over BIRD/Spider training questions is
    replaced by the repo's existing vector-store query search
    (``logic.vector_store.search_query``), which performs dense
    question-similarity ranking over indexed ``query_cell`` documents. Each
    cell's ``title`` is a human-authored description of the query (the natural
    "question") and ``query_text`` is the SQL -- a faithful, higher-recall
    realization of the paper's question-similarity selection.
  * Demonstrations are scoped to the tables the user is already querying so
    they share the target schema (in-domain). This follows the team's
    schema-grounding direction rather than the paper's cross-domain split.
  * The paper's execution-accuracy eval harness on Spider/BIRD is intentionally
    out of scope -- evaluation belongs in a downstream PR.

The module is additive: every public function degrades gracefully (returns
``""`` or ``[]``) when the vector store is unconfigured or returns nothing, so
callers fall back to the existing zero-shot text-to-SQL prompt unchanged.
"""

from typing import Callable, List, Optional

# How many (question -> SQL) demonstration pairs to retrieve before budgeting.
DEFAULT_SQL_DEMONSTRATION_COUNT = 3
# Cap the demonstration section at this fraction of the usable token window so
# demonstrations never crowd out schema context.
DEMONSTRATION_TOKEN_BUDGET_FRACTION = 0.25


def select_sql_demonstrations(
    question: str,
    tables: Optional[List[str]] = None,
    k: int = DEFAULT_SQL_DEMONSTRATION_COUNT,
) -> List[dict]:
    """Return up to ``k`` in-domain (title, query_text) demos for ``question``.

    Retrieval mirrors the paper's question-similarity selection: the vector
    store ranks indexed SQL queries by semantic similarity to ``question``,
    scoped to ``tables`` so demonstrations share the target schema. Returns
    ``[]`` when the vector store is unconfigured or nothing matches, so the
    caller falls back to the zero-shot prompt unchanged.
    """
    if not question or k <= 0:
        return []

    # Lazy import to avoid a module-load cycle: logic.vector_store imports the
    # ai_assistant package, which imports this module's neighbors.
    from lib.vector_store import get_vector_store
    from logic.vector_store import search_query

    if not get_vector_store():
        return []

    filters = [["full_table_name", list(tables)]] if tables else None
    try:
        response = search_query(keywords=question, filters=filters, limit=k)
    except Exception:
        return []

    demos: List[dict] = []
    seen = set()
    for cell in response.get("results", []):
        title = (cell.get("title") or "").strip()
        query_text = (cell.get("query_text") or "").strip()
        if not title or title.lower() == "untitled" or not query_text:
            continue
        if query_text in seen:
            continue
        seen.add(query_text)
        demos.append({"title": title, "query_text": query_text})
        if len(demos) >= k:
            break
    return demos


def format_sql_demonstrations(demonstrations: List[dict]) -> str:
    """Render demonstrations as a prompt section, or ``""`` when there are none.

    The empty case returns ``""`` (rather than a bare header) so the
    surrounding prompt template collapses to the original zero-shot form.
    """
    if not demonstrations:
        return ""

    blocks = [
        "Question: {title}\nSQL: {query_text}".format(**demo) for demo in demonstrations
    ]
    return (
        "===Demonstrations\n"
        "The following example questions and their SQL are for reference only; "
        "adapt them to the question and tables above.\n\n"
        "{joined}\n"
    ).format(joined="\n\n".join(blocks))


def select_and_format_sql_demonstrations(
    question: str,
    tables: List[str],
    token_counter: Callable[[str], int],
    token_budget: int,
    k: int = DEFAULT_SQL_DEMONSTRATION_COUNT,
) -> str:
    """Retrieve demonstrations and trim them to fit ``token_budget``.

    ``token_counter`` maps a string to its token count and is supplied by the
    assistant so the same tokenizer the LLM uses is applied. Demonstrations
    that would push the section past the budget are skipped in rank order; if
    none fit, ``""`` is returned and the caller stays zero-shot.
    """
    if token_budget <= 0:
        return ""

    demos = select_sql_demonstrations(question, tables=tables, k=k)
    if not demos:
        return ""

    kept: List[dict] = []
    for demo in demos:
        candidate = format_sql_demonstrations(kept + [demo])
        if token_counter(candidate) <= token_budget:
            kept.append(demo)
    return format_sql_demonstrations(kept)
