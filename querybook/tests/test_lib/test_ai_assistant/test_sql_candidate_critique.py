"""Integration tests for the multi-sample text-to-SQL critique path.

These exercise the wiring added to ``BaseAIAssistant`` (the existing call-site
module) -- not just the scorer in isolation -- by driving
``_run_multisample_sql_critique`` with a fake model and a fake socket and
asserting which candidate the critique forwards to the user.
"""

import json
from typing import Any
from unittest import TestCase

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.runnables import RunnableLambda

from lib.ai_assistant.base_ai_assistant import BaseAIAssistant


class _FakeSocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send_data(self, data: Any) -> None:
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True


class _MinimalAssistant(BaseAIAssistant):
    """Concrete subclass so we can exercise inherited text-to-sql wiring."""

    @property
    def name(self) -> str:
        return "minimal"

    def _get_token_count(self, ai_command: str, prompt: str) -> int:
        return 0

    def _get_context_length_by_model(self, model_name: str) -> int:
        return 8000

    def _get_llm(self, ai_command: str, prompt_length: int) -> BaseLanguageModel:
        raise AssertionError("the multi-sample path takes its llm directly")


def _candidate(query: str = "") -> str:
    """A model response in the JSON shape the text_to_sql prompt requests."""
    return json.dumps({"query": query, "explanation": ""})


def _llm_returning(outputs):
    """A langchain Runnable that emits the given JSON strings in order."""
    iterator = iter(outputs)

    def _emit(_prompt_text: str) -> str:
        try:
            return next(iterator)
        except StopIteration:
            return outputs[-1]

    return RunnableLambda(_emit)


class MultiSampleCritiqueTestCase(TestCase):
    grounded = "SELECT user_id FROM analytics.users " "WHERE signup_date > '2024-01-01'"
    hallucinated = "SELECT bogus_column FROM made_up_table"

    table_schemas = [
        {
            "table_name": "analytics.users",
            "columns": [{"name": "user_id"}, {"name": "signup_date"}],
        }
    ]

    def _run(self, llm, *, sample_count):
        assistant = _MinimalAssistant()
        socket = _FakeSocket()
        assistant._run_multisample_sql_critique(
            socket=socket,
            llm=llm,
            prompt_text="(prompt text is irrelevant to the critique)",
            sample_count=sample_count,
            table_schemas=self.table_schemas,
            question="user_id of users who signed up after 2024-01-01",
            dialect="mysql",
        )
        return socket

    def test_selects_the_schema_grounded_candidate(self):
        # The hallucinated candidate is offered first; the critique must still
        # pick the grounded one regardless of ordering.
        socket = self._run(
            _llm_returning([_candidate(self.hallucinated), _candidate(self.grounded)]),
            sample_count=2,
        )

        self.assertTrue(socket.closed)
        self.assertEqual(len(socket.sent), 1)
        self.assertEqual(socket.sent[0]["query"], self.grounded)

    def test_sends_an_apology_when_every_sample_fails(self):
        def _always_raise(_prompt_text: str) -> str:
            raise ValueError("model is down")

        socket = self._run(RunnableLambda(_always_raise), sample_count=3)

        self.assertTrue(socket.closed)
        self.assertEqual(len(socket.sent), 1)
        self.assertIn("candidates", socket.sent[0]["explanation"])
