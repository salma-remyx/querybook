from unittest import TestCase

from langchain_core.messages import AIMessage

from lib.ai_assistant.base_ai_assistant import BaseAIAssistant
from lib.ai_assistant.sql_verifier import (
    DEFAULT_VERIFIER_CRITERIA,
    SQLVerificationResult,
    SQLVerifier,
)


def _logprob_message(top_tokens):
    """Build an AIMessage whose response_metadata mimics OpenAI logprobs.

    ``top_tokens`` is a list of (token, logprob) pairs for the first generated
    position. The emitted token is the highest-probability one.
    """
    ordered = sorted(top_tokens, key=lambda pair: pair[1], reverse=True)
    emitted, emitted_lp = ordered[0]
    content = [
        {
            "token": emitted,
            "logprob": emitted_lp,
            "top_logprobs": [
                {"token": token, "logprob": lp} for token, lp in top_tokens
            ],
        }
    ]
    return AIMessage(
        content=emitted, response_metadata={"logprobs": {"content": content}}
    )


class _ScriptedLLM:
    """Duck-typed LLM that maps each prompt to a canned response."""

    def __init__(self, respond):
        self._respond = respond

    def bind(self, **kwargs):
        return self

    def invoke(self, prompt):
        return self._respond(prompt)


class _ConcreteAssistant(BaseAIAssistant):
    """Minimal concrete assistant so BaseAIAssistant methods can be exercised."""

    @property
    def name(self) -> str:
        return "concrete-test"

    def _get_token_count(self, ai_command, prompt):
        return len(prompt)

    def _get_context_length_by_model(self, model_name):
        return 999_999

    def _get_llm(self, ai_command, prompt_length):
        raise AssertionError("an llm should be injected, not built")


class ExpectedScoreFromLogprobsTestCase(TestCase):
    def test_confident_high_score(self):
        verifier = SQLVerifier(
            llm=_ScriptedLLM(lambda prompt: _logprob_message([("9", 0.0)]))
        )
        result = verifier.verify(
            "presto", "count rows", "t(a int)", "SELECT count(*) FROM t"
        )
        self.assertAlmostEqual(result.score, 1.0, places=6)
        self.assertEqual(result.method, "logprobs")

    def test_confident_low_score(self):
        verifier = SQLVerifier(
            llm=_ScriptedLLM(lambda prompt: _logprob_message([("0", 0.0)]))
        )
        result = verifier.verify("presto", "count rows", "t(a int)", "DROP TABLE t")
        self.assertAlmostEqual(result.score, 0.0, places=6)

    def test_mixed_distribution_is_continuous(self):
        # equal mass on digits 4 and 5 -> EV = (4/9 + 5/9) / 2 = 0.5
        verifier = SQLVerifier(
            llm=_ScriptedLLM(lambda prompt: _logprob_message([("4", 0.0), ("5", 0.0)]))
        )
        result = verifier.verify("presto", "q", "t(a int)", "SELECT * FROM t")
        self.assertAlmostEqual(result.score, 0.5, places=6)
        # continuous: not equal to either discrete digit value
        self.assertNotAlmostEqual(result.score, 4 / 9.0, places=6)
        self.assertNotAlmostEqual(result.score, 5 / 9.0, places=6)

    def test_discrete_fallback_without_logprobs(self):
        verifier = SQLVerifier(
            llm=_ScriptedLLM(
                lambda prompt: AIMessage(content="3\nreasonable", response_metadata={})
            )
        )
        result = verifier.verify("presto", "q", "t(a int)", "SELECT 1")
        self.assertAlmostEqual(result.score, 3 / 9.0, places=6)
        self.assertEqual(result.method, "discrete_fallback")


class CriteriaDecompositionTestCase(TestCase):
    def test_per_criterion_breakdown(self):
        # syntax -> 9, schema -> 9, intent -> 0  => overall = (1 + 1 + 0) / 3
        def respond(prompt):
            if "intent" in prompt:
                return _logprob_message([("0", 0.0)])
            return _logprob_message([("9", 0.0)])

        verifier = SQLVerifier(llm=_ScriptedLLM(respond))
        result = verifier.verify("presto", "q", "t(a int)", "SELECT * FROM missing")
        self.assertEqual(set(result.per_criterion), {"syntax", "schema", "intent"})
        self.assertAlmostEqual(result.per_criterion["intent"], 0.0, places=6)
        self.assertAlmostEqual(result.per_criterion["syntax"], 1.0, places=6)
        self.assertAlmostEqual(result.score, 2 / 3.0, places=6)

    def test_repeated_evaluation_averages(self):
        calls = {"i": 0}

        def respond(prompt):
            calls["i"] += 1
            # alternate 0 and 9 -> mean = 0.5 regardless of order
            digit = "0" if calls["i"] % 2 else "9"
            return _logprob_message([(digit, 0.0)])

        verifier = SQLVerifier(llm=_ScriptedLLM(respond))
        result = verifier.verify("presto", "q", "t(a int)", "SELECT 1", n_samples=2)
        for scores in result.samples.values():
            self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(result.score, 0.5, places=6)


class RankCandidatesTestCase(TestCase):
    def test_orders_best_first(self):
        good = "SELECT count(*) FROM t"
        bad = "SELECT * FROM nonexistent"

        def respond(prompt):
            digit = "9" if good in prompt else "0"
            return _logprob_message([(digit, 0.0)])

        verifier = SQLVerifier(llm=_ScriptedLLM(respond))
        ranked = verifier.rank("presto", "q", "t(a int)", [bad, good])
        self.assertEqual([candidate for candidate, _ in ranked], [good, bad])
        self.assertGreater(ranked[0][1].score, ranked[1][1].score)

    def test_sequential_halving_respects_budget(self):
        calls = {"n": 0}
        good = "SELECT 1"

        def respond(prompt):
            calls["n"] += 1
            digit = "9" if good in prompt else "0"
            return _logprob_message([(digit, 0.0)])

        verifier = SQLVerifier(llm=_ScriptedLLM(respond))
        # good is first so it survives every halving round; budget is tight.
        candidates = [good] + [f"q{i}" for i in range(7)]
        ranked = verifier.rank("presto", "q", "t(a int)", candidates, budget=4)
        self.assertEqual(ranked[0][0], good)
        full_evaluations = len(candidates) * len(DEFAULT_VERIFIER_CRITERIA)
        self.assertLess(calls["n"], full_evaluations)


class AssistantWiringTestCase(TestCase):
    """Integration: the call-site method on the existing BaseAIAssistant."""

    def test_verify_generated_sql_via_assistant(self):
        message = _logprob_message([("9", 0.0)])
        assistant = _ConcreteAssistant()
        assistant.set_config({"default": {"model_args": {"model_name": "gpt-4o"}}})
        result = assistant.verify_generated_sql(
            dialect="presto",
            table_schemas="CREATE TABLE t (a int)",
            question="How many rows?",
            generated_query="SELECT count(*) FROM t",
            llm=_ScriptedLLM(lambda prompt: message),
        )
        self.assertIsInstance(result, SQLVerificationResult)
        self.assertAlmostEqual(result.score, 1.0, places=6)
        self.assertEqual(
            set(result.per_criterion),
            {criterion[0] for criterion in DEFAULT_VERIFIER_CRITERIA},
        )

    def test_assistant_builds_llm_when_none_injected(self):
        # llm=None routes through _get_llm; here it raises, and @catch_error
        # surfaces it, proving the production wiring path is taken.
        assistant = _ConcreteAssistant()
        assistant.set_config({"default": {"model_args": {"model_name": "gpt-4o"}}})
        with self.assertRaises(Exception):
            assistant.verify_generated_sql(
                dialect="presto",
                table_schemas="t(a int)",
                question="q",
                generated_query="SELECT 1",
            )
