"""
Prompt construction for SQL generation.
"""

from typing import Dict, List


def build_prompt(schema: Dict, examples: List[Dict], user_question: str) -> str:
    """
    Build final prompt with Spider-friendly canonical SQL constraints.
    """
    prompt_parts = []

    prompt_parts.append(
        "You are a SQL expert.\n"
        "Generate exactly one SQLite SELECT query for the user question.\n"
        "Output SQL only.\n"
        "Style constraints for evaluation:\n"
        "1) Keep SQL minimal and canonical.\n"
        "2) Do not add trailing semicolon.\n"
        "3) Do not add aliases unless required for disambiguation.\n"
        "4) Do not add extra selected columns, ORDER BY, GROUP BY, LIMIT, or conditions unless required.\n"
        "5) Prefer the simplest correct SQL.\n"
        "6) Use only table and column names that appear in the provided schema.\n"
        "7) Do not invent identifiers (forbidden: guessed table/column names).\n"
        "8) IMPORTANT: Use the column description (after '--') to identify the correct column, "
        "but ALWAYS use the exact column identifier (before the type) in your SQL, never the description text. "
        "9) For 'youngest/oldest/highest/lowest' queries, use ORDER BY + LIMIT 1 instead of subqueries with MIN/MAX.\n"
        "10) For set operations (both X and Y), use INTERSECT instead of AND/OR.\n"
        "11) For exclusion (not in X), use EXCEPT or NOT IN as appropriate.\n"
        "12) Only add WHERE conditions explicitly mentioned in the question.\n"
        "13) For counting related items per group, use JOIN + GROUP BY + COUNT(*), not correlated subqueries.\n"
        "14) IMPORTANT: If a table or column name contains spaces or special characters, "
        "wrap it in backticks (e.g., `Life Expectancy`, `Government Form`).\n"
    )

    prompt_parts.append("# Database Schema\n")
    prompt_parts.append(format_schema(schema))
    prompt_parts.append("\n")

    if examples:
        prompt_parts.append("# Example Questions and SQL Queries\n")
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Question: {example['question']}")
            prompt_parts.append(f"SQL: {example['sql']}")
            prompt_parts.append("")

    prompt_parts.append("# Your Task\n")
    prompt_parts.append(f"Question: {user_question}")
    prompt_parts.append("SQL: ")

    return "\n".join(prompt_parts)


def _quote_if_needed(name: str) -> str:
    """对含空格或特殊字符的标识符加反引号"""
    if " " in name or "-" in name:
        return f"`{name}`"
    return name


def format_schema(schema: Dict) -> str:
    lines = []
    for table in schema["tables"]:
        table_name = _quote_if_needed(table["name"])
        lines.append(f"Table: {table_name}")
        for col in table["columns"]:
            col_name = _quote_if_needed(col['name'])
            desc = col.get("description", "")
            if desc and desc.lower() != col_name.lower().replace("`", ""):
                col_info = f"  - {col_name} ({col['type']}) -- {desc}"
            else:
                col_info = f"  - {col_name} ({col['type']})"
            if col.get("primary_key"):
                col_info += " [PK]"
            lines.append(col_info)
        lines.append("")

    if schema.get("foreign_keys"):
        lines.append("Foreign Keys:")
        for fk in schema["foreign_keys"]:
            lines.append(
                f"  - {fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
            )

    return "\n".join(lines)


def build_prompt_with_cot(schema: Dict, examples: List[Dict], user_question: str) -> str:
    prompt = build_prompt(schema, examples, user_question)
    cot_instruction = """
Before generating the SQL, think step by step:
1. Identify the relevant tables
2. Determine the JOIN conditions
3. Specify the WHERE conditions
4. Add aggregations only if needed

Now generate the SQL query:
"""
    return prompt.replace("SQL: ", cot_instruction + "SQL: ")
