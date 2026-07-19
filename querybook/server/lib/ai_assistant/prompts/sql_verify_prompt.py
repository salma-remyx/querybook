from langchain.prompts import PromptTemplate


# Each verification call scores the generated SQL on exactly ONE criterion
# (criteria decomposition). The model is asked to emit the score digit as the
# very first token so the caller can read back its logprob distribution and
# take the expected value (continuous scoring) instead of trusting the digit.
prompt_template = """You are a {dialect} SQL verification expert. You are scoring a generated SQL query on exactly one criterion.

===Criterion
{criterion}: {criterion_description}

===User Intent
{question}

===Available Table Schemas
{table_schemas}

===Generated SQL to Verify
{generated_query}

===Instructions
Score the generated SQL on the criterion above using a single digit from 0 (worst) to 9 (best).
Output the score digit as the VERY FIRST token of your response, then a newline, then a one-sentence justification.

===Response Format
<digit>
<one-sentence justification>
"""

SQL_VERIFY_PROMPT = PromptTemplate.from_template(prompt_template)
