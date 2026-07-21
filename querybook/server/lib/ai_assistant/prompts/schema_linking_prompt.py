from langchain.prompts import PromptTemplate


prompt_template = """
You are a data scientist that performs schema linking for SQL query tasks.

For each table below, select ONLY the columns that are needed to answer the question. Drop columns that are clearly irrelevant to the question, such as surrogate primary keys, audit timestamps, or unrelated metrics. Keep every column that could plausibly be referenced by the question. If you are unsure about a column, keep it.

===Response Guidelines
- Response should be a valid JSON object mapping each table name to a JSON array of selected column names, which can be parsed by Python json.loads().
- Only include tables that appear in the context.
- Column names must exactly match the column names provided.
- If no columns of a table are relevant, map that table to an empty array.

===Tables
{table_schemas}

===Question
{question}
"""

SCHEMA_LINKING_PROMPT = PromptTemplate.from_template(prompt_template)
