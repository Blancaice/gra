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
        "Output SQL only.\n\n"
        "Guidelines:\n"
        "1) Use only table and column names from the provided schema. "
        "Use the column description (after '--') to identify the correct column, "
        "but write the exact column identifier in your SQL.\n"
        "2) Only add WHERE conditions, JOINs, GROUP BY, ORDER BY that the question requires. "
        "Prefer the simplest correct SQL.\n"
        "3) When the question asks for entities satisfying multiple conditions on DIFFERENT rows "
        "of the same table, consider whether INTERSECT or EXCEPT is needed instead of AND/OR in WHERE.\n"
        "4) For 'youngest/oldest/highest/lowest' queries, prefer ORDER BY + LIMIT 1.\n"
        "5) Before choosing GROUP BY columns, ask yourself: should rows be grouped by identity (ID) "
        "or by a descriptive attribute?\n"
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

#已弃用
def build_prompt_with_cot(schema: Dict, examples: List[Dict], user_question: str) -> str:
    prompt = build_prompt(schema, examples, user_question)
    cot_instruction = """
Before generating the SQL, think step by step:
1. What is the question asking for? Identify the target columns to SELECT.
2. Which tables contain the needed data? List them.
3. How do these tables connect? Determine JOIN conditions using foreign keys.
4. What filtering is needed? Identify WHERE conditions from the question.
5. Does the question need aggregation (COUNT, SUM, AVG, MAX, MIN)? If so, what GROUP BY is needed?
6. Does the question need ordering or limiting results (top/bottom/highest/lowest)?
7. Does the question involve set operations (both X and Y → INTERSECT, not in X → EXCEPT)?
8. Double-check: are all table and column names from the schema above?

Now generate the SQL query:
"""
    return prompt.replace("SQL: ", cot_instruction + "SQL: ")
