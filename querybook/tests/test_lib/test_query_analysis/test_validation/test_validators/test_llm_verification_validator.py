from typing import Dict, List
from unittest import TestCase

from lib.query_analysis.validation.all_validators import (
    ALL_QUERY_VALIDATORS_BY_NAME,
    get_validator_by_name,
)
from lib.query_analysis.validation.base_query_validator import (
    QueryValidationResult,
    QueryValidationResultObjectType,
    QueryValidationSeverity,
)

from query_validation_plugin.llm_verification_validator import (
    DEFAULT_RUBRICS,
    LLMVerificationValidator,
    expected_grade,
    score_criterion,
    softmax_logprobs,
    verify_sql,
)


class FakeScorer:
    """Deterministic stand-in for the OpenAI logprob scorer.

    Returns a token -> probability distribution per call. ``call_log`` records
    how many times the scorer was invoked so repeated-evaluation can be
    asserted. ``rotate`` cycles a list of distributions to emulate a stochastic
    model across repeated evaluations.
    """

    def __init__(
        self, distribution: Dict[str, float], rotate: List[Dict[str, float]] = None
    ):
        self._distribution = distribution
        self._rotate = rotate
        self._idx = 0
        self.call_log: List[str] = []

    def __call__(self, prompt: str) -> Dict[str, float]:
        self.call_log.append(prompt)
        if self._rotate:
            dist = self._rotate[self._idx % len(self._rotate)]
            self._idx += 1
            return dict(dist)
        return dict(self._distribution)


class SoftmaxLogprobsTestCase(TestCase):
    def test_normalizes_to_probability_distribution(self):
        probs = softmax_logprobs({"yes": -0.4, "no": -1.6})
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        # higher logprob -> higher probability
        self.assertGreater(probs["yes"], probs["no"])

    def test_empty_input_returns_empty(self):
        self.assertEqual(softmax_logprobs({}), {})


class ExpectedGradeTestCase(TestCase):
    def test_binary_rubric_expected_value(self):
        # 80% yes, 20% no -> expected grade 0.8
        dist = {"yes": 0.8, "no": 0.2}
        binary = {"yes": 1.0, "no": 0.0}
        self.assertAlmostEqual(expected_grade(dist, binary), 0.8, places=6)

    def test_returns_none_when_no_rubric_token_present(self):
        self.assertIsNone(expected_grade({"maybe": 1.0}, {"yes": 1.0, "no": 0.0}))


class ScoreCriterionTestCase(TestCase):
    def test_averages_granularity_rubrics(self):
        # Model is confidently positive across binary/ternary/quinary rubrics.
        scorer = FakeScorer({"yes": 1.0, "high": 1.0, "5": 1.0})
        score = score_criterion(scorer, "prompt", DEFAULT_RUBRICS, n_repeats=1)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_repeated_evaluation_averages_and_reduces_variance(self):
        # Two reads: one fully positive, one fully negative -> expect 0.5.
        scorer = FakeScorer(
            distribution={"yes": 1.0},
            rotate=[
                {"yes": 1.0, "high": 1.0, "5": 1.0},
                {"no": 1.0, "low": 1.0, "1": 1.0},
            ],
        )
        score = score_criterion(scorer, "prompt", DEFAULT_RUBRICS, n_repeats=2)
        self.assertAlmostEqual(score, 0.5, places=6)
        # One call per rubric per repeat: 3 rubrics * 2 repeats == 6 calls.
        self.assertEqual(len(scorer.call_log), len(DEFAULT_RUBRICS) * 2)

    def test_uncertain_when_no_rubric_token_ever_emitted(self):
        scorer = FakeScorer({"maybe": 1.0})
        self.assertAlmostEqual(
            score_criterion(scorer, "prompt", DEFAULT_RUBRICS, n_repeats=1), 0.5
        )


class VerifySqlTestCase(TestCase):
    def test_intent_skipped_without_question(self):
        scorer = FakeScorer({"yes": 1.0, "high": 1.0, "5": 1.0})
        result = verify_sql(scorer, "SELECT 1", context={"dialect": "presto"})
        self.assertNotIn("intent", result["criteria"])
        self.assertIn("syntax", result["criteria"])
        self.assertIn("schema", result["criteria"])
        self.assertAlmostEqual(result["score"], 1.0, places=6)

    def test_intent_included_with_question(self):
        scorer = FakeScorer({"yes": 1.0, "high": 1.0, "5": 1.0})
        result = verify_sql(
            scorer,
            "SELECT 1",
            context={"dialect": "presto", "question": "count rows"},
        )
        self.assertIn("intent", result["criteria"])


class LLMVerificationValidatorTestCase(TestCase):
    def _make_validator(self, scorer) -> LLMVerificationValidator:
        validator = LLMVerificationValidator(LLMVerificationValidator.__name__)
        validator.set_scorer(scorer)
        return validator

    def test_languages_is_dialect_agnostic(self):
        validator = LLMVerificationValidator(LLMVerificationValidator.__name__)
        self.assertIn("presto", validator.languages())

    def test_validate_emits_continuous_score_result(self):
        validator = self._make_validator(
            FakeScorer({"yes": 1.0, "high": 1.0, "5": 1.0})
        )
        results = validator.validate("SELECT * FROM users", uid=1, engine_id=1)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.severity, QueryValidationSeverity.INFO)
        self.assertEqual(result.type, QueryValidationResultObjectType.GENERAL)
        self.assertIn("score", result.message.lower())

    def test_validate_warns_on_low_confidence(self):
        validator = self._make_validator(FakeScorer({"no": 1.0, "low": 1.0, "1": 1.0}))
        results = validator.validate("SELECT FROM", uid=1, engine_id=1)
        severities = [r.severity for r in results]
        self.assertIn(QueryValidationSeverity.WARNING, severities)


class PluginWiringTestCase(TestCase):
    """Integration: the plugin is discovered and merged by the existing
    all_validators module, and is invokable through get_validator_by_name.
    """

    def test_registered_through_plugin_slot(self):
        # all_validators.py merges ALL_PLUGIN_QUERY_VALIDATORS_BY_NAME into the
        # active map; an entry here proves the plugin __init__ was loaded.
        self.assertIn("LLMVerificationValidator", ALL_QUERY_VALIDATORS_BY_NAME)

    def test_invokable_via_get_validator_by_name(self):
        validator = get_validator_by_name("LLMVerificationValidator")
        # Stored value must be an instance (the validation endpoint calls
        # .validate_with_templated_vars / .validate directly on it).
        self.assertIsInstance(validator, LLMVerificationValidator)
        validator.set_scorer(FakeScorer({"yes": 1.0, "high": 1.0, "5": 1.0}))
        results = validator.validate("SELECT 1", uid=1, engine_id=1)
        self.assertTrue(all(isinstance(r, QueryValidationResult) for r in results))
