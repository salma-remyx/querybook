"""Tests for selective text-to-SQL demonstrations.

Covers the new ``sql_demonstrations`` capability and its wiring into the
existing ``BaseAIAssistant`` text-to-SQL prompt path (the integration
surface). The vector-store and Elasticsearch lookups are mocked so the tests
run without a configured backend.
"""

from unittest import TestCase
from unittest.mock import patch

from lib.ai_assistant.base_ai_assistant import BaseAIAssistant
from lib.ai_assistant.prompts.text_to_sql_prompt import TEXT_TO_SQL_PROMPT
from lib.ai_assistant.sql_demonstrations import (
    format_sql_demonstrations,
    select_and_format_sql_demonstrations,
    select_sql_demonstrations,
)


class _ConcreteAssistant(BaseAIAssistant):
    """Minimal concrete assistant so BaseAIAssistant helpers can be exercised."""

    name = "concrete"

    def _get_token_count(self, ai_command, prompt):
        return len(prompt)

    def _get_context_length_by_model(self, model_name):
        return 100_000

    def _get_llm(self, ai_command, prompt_length):
        return None


def _make_assistant():
    assistant = _ConcreteAssistant()
    assistant.set_config(
        {
            "default": {"model_args": {"model_name": "stub"}, "reserved_tokens": 0},
        }
    )
    return assistant


DEMO_CELL = {
    "id": 1,
    "title": "Total revenue by region",
    "query_text": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
    "full_table_name": ["db.sales"],
}


def _patch_vector_store(results):
    """Patch the lazy vector-store imports used by select_sql_demonstrations."""
    get_vs = patch("lib.vector_store.get_vector_store", return_value=object())
    search = patch(
        "logic.vector_store.search_query",
        return_value={"count": len(results), "results": results},
    )
    get_vs.start()
    search.start()
    return (get_vs, search)


class FormatSqlDemonstrationsTestCase(TestCase):
    def test_renders_question_sql_pairs(self):
        rendered = format_sql_demonstrations([DEMO_CELL])
        self.assertIn("Question: Total revenue by region", rendered)
        self.assertIn(
            "SQL: SELECT region, SUM(revenue) FROM sales GROUP BY region", rendered
        )
        self.assertIn("===Demonstrations", rendered)

    def test_empty_returns_blank(self):
        self.assertEqual(format_sql_demonstrations([]), "")


class SelectSqlDemonstrationsTestCase(TestCase):
    def tearDown(self):
        patch.stopall()

    def test_retrieves_in_domain_demos(self):
        _patch_vector_store([DEMO_CELL])
        demos = select_sql_demonstrations("revenue by region", tables=["db.sales"])
        self.assertEqual(len(demos), 1)
        self.assertEqual(demos[0]["title"], "Total revenue by region")

    def test_skips_untitled_and_duplicates(self):
        untitled = dict(DEMO_CELL, title="Untitled")
        _patch_vector_store([untitled, dict(DEMO_CELL), DEMO_CELL])
        demos = select_sql_demonstrations("revenue", tables=["db.sales"])
        self.assertEqual(len(demos), 1)

    def test_no_question_returns_empty(self):
        self.assertEqual(select_sql_demonstrations("", tables=["db.sales"]), [])

    def test_unconfigured_vector_store_returns_empty(self):
        with patch("lib.vector_store.get_vector_store", return_value=None):
            self.assertEqual(
                select_sql_demonstrations("revenue", tables=["db.sales"]), []
            )


class SelectAndFormatTestCase(TestCase):
    def tearDown(self):
        patch.stopall()

    def test_zero_budget_returns_blank(self):
        _patch_vector_store([DEMO_CELL])
        self.assertEqual(
            select_and_format_sql_demonstrations(
                "revenue", ["db.sales"], lambda text: 1, token_budget=0
            ),
            "",
        )

    def test_renders_when_budget_allows(self):
        _patch_vector_store([DEMO_CELL])
        rendered = select_and_format_sql_demonstrations(
            "revenue", ["db.sales"], lambda text: len(text), token_budget=10_000
        )
        self.assertIn("===Demonstrations", rendered)
        self.assertIn("Total revenue by region", rendered)

    def test_skips_demos_that_exceed_budget(self):
        _patch_vector_store([DEMO_CELL])
        # Budget large enough for the header but not the example body.
        rendered = select_and_format_sql_demonstrations(
            "revenue", ["db.sales"], lambda text: len(text), token_budget=40
        )
        self.assertEqual(rendered, "")


class TextToSqlPromptIntegrationTestCase(TestCase):
    """Exercises the wiring edits in the existing prompt + assistant modules."""

    def test_prompt_contains_demonstrations_slot(self):
        block = format_sql_demonstrations([DEMO_CELL])
        rendered = TEXT_TO_SQL_PROMPT.format(
            dialect="hive",
            question="show me revenue by region",
            table_schemas="CREATE TABLE sales ...",
            original_query="",
            demonstrations=block,
        )
        self.assertIn("Question: Total revenue by region", rendered)

    def test_zero_shot_when_no_demonstrations(self):
        rendered = TEXT_TO_SQL_PROMPT.format(
            dialect="hive",
            question="show me revenue by region",
            table_schemas="CREATE TABLE sales ...",
            original_query="",
            demonstrations="",
        )
        self.assertNotIn("===Demonstrations", rendered)

    def test_get_text_to_sql_demonstrations_wires_vector_store(self):
        assistant = _make_assistant()
        with patch("lib.vector_store.get_vector_store", return_value=object()), patch(
            "logic.vector_store.search_query",
            return_value={"count": 1, "results": [DEMO_CELL]},
        ):
            demonstrations = assistant._get_text_to_sql_demonstrations(
                question="revenue by region", tables=["db.sales"]
            )
        self.assertIn("Total revenue by region", demonstrations)

    def test_get_text_to_sql_demonstrations_falls_back_when_unconfigured(self):
        assistant = _make_assistant()
        with patch("lib.vector_store.get_vector_store", return_value=None):
            self.assertEqual(
                assistant._get_text_to_sql_demonstrations(
                    question="revenue by region", tables=["db.sales"]
                ),
                "",
            )

    def test_text_to_sql_prompt_passes_demonstrations_through(self):
        assistant = _make_assistant()
        block = format_sql_demonstrations([DEMO_CELL])
        prompt = assistant._get_text_to_sql_prompt(
            dialect="hive",
            question="revenue by region",
            table_schemas="CREATE TABLE sales ...",
            original_query="",
            demonstrations=block,
        )
        self.assertIn("Question: Total revenue by region", prompt)
