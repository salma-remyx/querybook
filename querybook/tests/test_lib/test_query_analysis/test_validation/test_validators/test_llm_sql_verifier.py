import os
import sys
from typing import List, Mapping
from unittest import TestCase

# The plugins package is not on sys.path when tests run outside the container,
# so make it importable here. Inside the container PYTHONPATH already includes
# it, so this is a guarded no-op there. Putting it on the path lets us import
# ``query_validation_plugin`` and exercise the registration edit in its __init__.
_PLUGINS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
        "plugins",
    )
)
if _PLUGINS_DIR not in sys.path:
    sys.path.insert(0, _PLUGINS_DIR)

# Imported from a NON-NEW (existing) module: the validator base contract.
from lib.query_analysis.validation.base_query_validator import (  # noqa: E402
    BaseQueryValidator,
    QueryValidationResultObjectType,
    QueryValidationSeverity,
)
from query_validation_plugin import (  # noqa: E402
    ALL_PLUGIN_QUERY_VALIDATORS_BY_NAME,
)
from query_validation_plugin.llm_sql_verifier import (  # noqa: E402
    LLMSQLVerifierValidator,
)
from query_validation_plugin.verifier_score import (  # noqa: E402
    BINARY_SCALE,
    DEFAULT_GRANULAR_SCALE,
    aggregate_criteria,
    aggregate_repeats,
    expectation_score,
    normalize_logprobs,
)


def _provider_returning(logprobs: Mapping[str, float]):
    """A fake LogProbProvider that always returns the given logprobs."""

    def _provider(_prompt: str, _tokens: List[str]) -> Mapping[str, float]:
        return logprobs

    return _provider


class VerifierScoreTestCase(TestCase):
    """Unit tests for the pure, paper-core scoring math."""

    def test_normalize_logprobs_is_a_distribution(self):
        probs = normalize_logprobs({"1": -3.0, "2": -1.0, "3": -0.5})
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertGreater(probs["3"], probs["2"])
        self.assertGreater(probs["2"], probs["1"])

    def test_expectation_is_continuous_and_in_range(self):
        good = expectation_score(
            DEFAULT_GRANULAR_SCALE,
            {"1": -10, "2": -9, "3": -8, "4": -1, "5": -0.1},
        )
        bad = expectation_score(
            DEFAULT_GRANULAR_SCALE,
            {"1": -0.1, "2": -1, "3": -8, "4": -9, "5": -10},
        )
        self.assertGreater(good, 0.8)
        self.assertLess(bad, 0.2)
        self.assertGreater(good, bad)

    def test_granularity_separates_more_than_binary(self):
        # The same positive-leaning signal, expressed on each scale. The granular
        # scale yields a strictly interior continuous score (not a 0/1 argmax).
        granular = expectation_score(
            DEFAULT_GRANULAR_SCALE,
            {"1": -2.0, "2": -1.5, "3": -1.0, "4": -0.7, "5": -0.5},
        )
        binary = expectation_score(BINARY_SCALE, {"no": -1.0, "yes": -0.5})
        self.assertGreater(granular, binary)
        self.assertLess(granular, 1.0)

    def test_aggregates_and_no_signal(self):
        self.assertAlmostEqual(aggregate_repeats([0.2, 0.4, 0.6]), 0.4)
        self.assertAlmostEqual(aggregate_criteria({"a": 0.0, "b": 1.0}), 0.5)
        # No candidate tokens on the scale -> neutral, well-defined score.
        self.assertEqual(expectation_score(DEFAULT_GRANULAR_SCALE, {"?": -1.0}), 0.5)


class LLMSQLVerifierWiringTestCase(TestCase):
    """Exercises the registration edit in plugins/query_validation_plugin/__init__."""

    def test_validator_is_registered_by_name(self):
        # This is the integration assertion: the plugin package, when imported,
        # must surface the validator through the same dict the server merges into
        # ALL_QUERY_VALIDATORS_BY_NAME (see all_validators.py).
        self.assertIn("LLMSQLVerifierValidator", ALL_PLUGIN_QUERY_VALIDATORS_BY_NAME)
        validator = ALL_PLUGIN_QUERY_VALIDATORS_BY_NAME["LLMSQLVerifierValidator"]
        self.assertIsInstance(validator, LLMSQLVerifierValidator)
        self.assertIsInstance(validator, BaseQueryValidator)
        # languages() must cover at least one real dialect so an engine can opt in.
        self.assertIn("presto", validator.languages())


class LLMSQLVerifierBehaviorTestCase(TestCase):
    """Asserts the integrated validate() behavior via the registered instance."""

    def _registered_validator_with(self, logprobs) -> LLMSQLVerifierValidator:
        validator = ALL_PLUGIN_QUERY_VALIDATORS_BY_NAME["LLMSQLVerifierValidator"]
        # Inject the fake provider on the live, registered instance (the wiring
        # edit registers it with no provider; production builds a default lazily).
        validator._provider = _provider_returning(logprobs)
        return validator

    def test_high_confidence_query_produces_no_result(self):
        validator = self._registered_validator_with(
            {"1": -10, "2": -9, "3": -8, "4": -1, "5": -0.1}
        )
        results = validator.validate("SELECT id FROM users WHERE id = 1", 0, 0)
        self.assertEqual(results, [])

    def test_low_confidence_query_produces_advisory_result(self):
        validator = self._registered_validator_with(
            {"1": -0.1, "2": -1, "3": -8, "4": -9, "5": -10}
        )
        results = validator.validate("SELECT FROM WHERE", 0, 0)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.severity, QueryValidationSeverity.INFO)
        self.assertEqual(result.type, QueryValidationResultObjectType.GENERAL)
        # The continuous score and per-criterion breakdown appear in the message.
        self.assertIn("score", result.message)

    def test_provider_failure_fails_open(self):
        def _raising(_prompt, _tokens):
            raise RuntimeError("api down")

        validator = ALL_PLUGIN_QUERY_VALIDATORS_BY_NAME["LLMSQLVerifierValidator"]
        validator._provider = _raising
        # An advisory verifier must never break /query/validate/ on provider error.
        self.assertEqual(validator.validate("SELECT 1", 0, 0), [])


class LLMSQLVerifierConfigTestCase(TestCase):
    """Validates config-driven knobs through a freshly built instance."""

    def test_threshold_and_criteria_are_configurable(self):
        validator = LLMSQLVerifierValidator(
            "t",
            {
                "logprob_provider": _provider_returning(
                    {"1": -0.1, "2": -1, "3": -8, "4": -9, "5": -10}
                ),
                "fail_threshold": 0.9,
                "criteria": ("validity",),
            },
        )
        # With a 0.1-ish score and a 0.9 threshold, the advisory fires.
        results = validator.validate("nonsense", 0, 0)
        self.assertEqual(len(results), 1)
