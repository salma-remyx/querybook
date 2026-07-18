from unittest import TestCase

from langchain_core.language_models.base import BaseLanguageModel

from lib.ai_assistant.base_ai_assistant import BaseAIAssistant
from lib.ai_assistant.query_complexity import (
    QueryComplexity,
    SchemaDepth,
    estimate_query_complexity,
    select_schema_depth,
)


class _StubAssistant(BaseAIAssistant):
    """Minimal concrete assistant so prompt-building logic can run without a
    real model or config. Token count is approximated by prompt length and the
    context window is set large so the reactive (Expand) trim does not fire."""

    def __init__(self, context_length: int = 100_000):
        self.set_config({"default": {"model_args": {"model_name": "stub"}}})
        self._context_length = context_length

    @property
    def name(self) -> str:
        return "stub"

    def _get_token_count(self, ai_command: str, prompt: str) -> int:
        return len(prompt)

    def _get_context_length_by_model(self, model_name: str) -> int:
        return self._context_length

    def _get_llm(self, ai_command: str, prompt_length: int) -> BaseLanguageModel:
        return None  # not exercised by prompt building


# A single-table schema in the shape produced by tools.table_schema. The full
# representation carries description/partition fields the slimmed view drops.
_TABLE_SCHEMAS = [
    {
        "table_name": "dw.users",
        "table_description": "User accounts",
        "latest_partitions": ["ds=20240101"],
        "column_info": None,
        "tags": [{"type": "TABLE_TIER", "name": "gold"}],
        "columns": [
            {
                "name": "id",
                "type": "bigint",
                "description": "User id",
                "data_element": None,
                "statistics": None,
            },
            {
                "name": "email",
                "type": "string",
                "description": "User email",
                "data_element": None,
                "statistics": None,
            },
        ],
    }
]


class TextToSqlPromptDepthTestCase(TestCase):
    """Integration: the Estimate step in _get_text_to_sql_prompt renders the
    slimmed schema for simple questions and the full schema otherwise."""

    def test_simple_question_uses_slimmed_schema(self):
        assistant = _StubAssistant()
        prompt = assistant._get_text_to_sql_prompt(
            dialect="hive",
            question="Show the name and email of every user",
            table_schemas=_TABLE_SCHEMAS,
            original_query=None,
        )
        # slimmed-only markers are present ...
        self.assertIn("properties", prompt)
        self.assertIn("table_tier", prompt)
        # ... while full-schema-only fields are dropped.
        self.assertNotIn("latest_partitions", prompt)
        self.assertNotIn("User email", prompt)

    def test_complex_question_uses_full_schema(self):
        assistant = _StubAssistant()
        prompt = assistant._get_text_to_sql_prompt(
            dialect="hive",
            question="Join users with orders and count the number of orders per user",
            table_schemas=_TABLE_SCHEMAS,
            original_query=None,
        )
        # full-schema-only fields are retained
        self.assertIn("latest_partitions", prompt)
        self.assertIn("User email", prompt)

    def test_simple_question_over_limit_stays_slimmed(self):
        # A tiny context window forces a complex question past the budget; the
        # Expand fallback must still slim it rather than overflow.
        assistant = _StubAssistant(context_length=1)
        prompt = assistant._get_text_to_sql_prompt(
            dialect="hive",
            question="Join users with orders and count the number of orders per user",
            table_schemas=_TABLE_SCHEMAS,
            original_query=None,
        )
        self.assertIn("properties", prompt)
        self.assertNotIn("latest_partitions", prompt)


class EstimateQueryComplexityTestCase(TestCase):
    def test_classifies_simple_projection(self):
        self.assertEqual(
            estimate_query_complexity("Show the name and email of every user"),
            QueryComplexity.SIMPLE,
        )

    def test_classifies_aggregation_as_moderate(self):
        self.assertEqual(
            estimate_query_complexity("How many users signed up last month"),
            QueryComplexity.MODERATE,
        )

    def test_classifies_join_with_aggregation_as_complex(self):
        self.assertEqual(
            estimate_query_complexity(
                "Join users with orders and count the number of orders per user"
            ),
            QueryComplexity.COMPLEX,
        )

    def test_does_not_match_count_inside_substrings(self):
        # "count" must not fire inside "account" / "discount"
        self.assertEqual(
            estimate_query_complexity(
                "Show the account balance and discount code for a user"
            ),
            QueryComplexity.SIMPLE,
        )

    def test_empty_question_defaults_to_moderate(self):
        self.assertEqual(estimate_query_complexity(""), QueryComplexity.MODERATE)
        self.assertEqual(estimate_query_complexity("   "), QueryComplexity.MODERATE)


class SelectSchemaDepthTestCase(TestCase):
    def test_simple_maps_to_slimmed(self):
        self.assertEqual(
            select_schema_depth(QueryComplexity.SIMPLE), SchemaDepth.SLIMMED
        )

    def test_non_simple_maps_to_full(self):
        self.assertEqual(
            select_schema_depth(QueryComplexity.MODERATE), SchemaDepth.FULL
        )
        self.assertEqual(select_schema_depth(QueryComplexity.COMPLEX), SchemaDepth.FULL)
