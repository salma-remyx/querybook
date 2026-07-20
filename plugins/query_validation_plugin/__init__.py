from .llm_verification_validator import LLMVerificationValidator

# Values are validator INSTANCES: all_validators.py merges this dict with the
# built-in {name: instance} map, and get_validator_by_name() calls
# .validate_with_templated_vars(...) directly on the stored value.
ALL_PLUGIN_QUERY_VALIDATORS_BY_NAME = {
    LLMVerificationValidator.__name__: LLMVerificationValidator(
        LLMVerificationValidator.__name__
    ),
}
