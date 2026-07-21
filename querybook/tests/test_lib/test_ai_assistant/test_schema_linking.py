from langchain_core.runnables import Runnable

from const.ai_assistant import AICommandType
from lib.ai_assistant.base_ai_assistant import BaseAIAssistant
from lib.ai_assistant.tools.schema_linking import (
    link_and_prune_schemas,
    parse_column_selection,
    prune_table_schemas_by_relevance,
)


def _column(name, type="STRING"):
    return {
        "name": name,
        "type": type,
        "description": None,
        "data_element": None,
        "statistics": None,
    }


TABLE_SCHEMAS = [
    {
        "table_name": "marketing.spend",
        "table_description": "Marketing spend per campaign.",
        "latest_partitions": None,
        "column_info": None,
        "tags": [],
        "columns": [
            _column("campaign_name"),
            _column("impressions", "BIGINT"),
            _column("billing_address"),
            _column("internal_audit_hash"),
        ],
    }
]


class _FakeLlm(Runnable):
    """Minimal Runnable returning a canned response, for testing chains."""

    def __init__(self, response):
        super().__init__()
        self.response = response

    def invoke(self, input, config=None, **kwargs):
        return self.response


class _StubAssistant(BaseAIAssistant):
    """Concrete BaseAIAssistant that returns a canned schema-linking response."""

    def __init__(self, llm_response):
        super().__init__()
        self._llm_response = llm_response
        self.set_config(
            {
                "default": {
                    "model_args": {"model_name": "stub"},
                    "reserved_tokens": 0,
                },
                AICommandType.TEXT_TO_SQL.value: {"model_args": {"model_name": "stub"}},
            }
        )

    @property
    def name(self) -> str:
        return "stub"

    def _get_token_count(self, ai_command, prompt):
        return 1

    def _get_context_length_by_model(self, model_name):
        return 1_000_000

    def _get_llm(self, ai_command, prompt_length):
        return _FakeLlm(self._llm_response)


def test_parse_column_selection_accepts_dict_and_string():
    assert parse_column_selection({"a.b": ["x", "y"]}) == {"a.b": ["x", "y"]}
    assert parse_column_selection('{"a.b": ["x"]}') == {"a.b": ["x"]}
    # malformed inputs collapse to an empty selection, never raise
    assert parse_column_selection("not json") == {}
    assert parse_column_selection(None) == {}
    assert parse_column_selection({"a.b": "single"}) == {"a.b": ["single"]}


def test_prune_keeps_only_selected_columns_and_preserves_unselected_tables():
    selection = {"marketing.spend": ["campaign_name", "impressions"]}

    pruned = prune_table_schemas_by_relevance(TABLE_SCHEMAS, selection)
    assert len(pruned) == 1

    kept = {col["name"] for col in pruned[0]["columns"]}
    assert kept == {"campaign_name", "impressions"}
    # original schema is not mutated
    original = {col["name"] for col in TABLE_SCHEMAS[0]["columns"]}
    assert "billing_address" in original

    # an empty selection returns every table unchanged
    assert prune_table_schemas_by_relevance(TABLE_SCHEMAS, {}) == TABLE_SCHEMAS


def test_get_text_to_sql_prompt_prunes_to_relevant_columns():
    # The model selects the marketing-relevant columns for the question; the
    # prompt built by the (existing) call site must contain only those columns.
    assistant = _StubAssistant(
        llm_response='{"marketing.spend": ["campaign_name", "impressions"]}'
    )

    prompt = assistant._get_text_to_sql_prompt(
        dialect="presto",
        question="What is the marketing spend by campaign?",
        table_schemas=TABLE_SCHEMAS,
        original_query=None,
    )

    assert "campaign_name" in prompt
    assert "impressions" in prompt
    assert "billing_address" not in prompt
    assert "internal_audit_hash" not in prompt


def test_soft_linking_falls_back_to_full_schema_on_error():
    class _BoomLlm(Runnable):
        def invoke(self, input, config=None, **kwargs):
            raise RuntimeError("model unavailable")

    class _FailingAssistant(_StubAssistant):
        def _get_llm(self, ai_command, prompt_length):
            return _BoomLlm()

    assistant = _FailingAssistant(llm_response=None)

    result = assistant._soft_link_table_schemas(
        question="anything", table_schemas=TABLE_SCHEMAS
    )

    # every original column is retained when linking fails
    assert result == TABLE_SCHEMAS


def test_link_and_prune_schemas_end_to_end():
    def invoke(prompt):
        assert "marketing spend" in prompt
        return '{"marketing.spend": ["impressions"]}'

    pruned = link_and_prune_schemas(
        invoke=invoke,
        question="Show marketing spend.",
        table_schemas=TABLE_SCHEMAS,
    )

    assert [col["name"] for col in pruned[0]["columns"]] == ["impressions"]
