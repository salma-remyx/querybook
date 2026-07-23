from .llm_sql_verifier import LLMSQLVerifierValidator

# Register the LLM-as-a-Verifier SQL validator (advisory, logprob-based
# continuous scoring of generated SQL). Opt in per engine via
# feature_params.validator = "LLMSQLVerifierValidator".
_verifier = LLMSQLVerifierValidator(LLMSQLVerifierValidator.__name__)
ALL_PLUGIN_QUERY_VALIDATORS_BY_NAME = {LLMSQLVerifierValidator.__name__: _verifier}
