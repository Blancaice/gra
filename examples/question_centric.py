"""
Question-centric example generation module.
"""

from typing import Dict, List
import json
import re

from src.examples.schema_centric import classify_sql_complexity


def generate_from_question(
    question: str,
    schema: Dict,
    llm_client,
    num_examples: int = 8,
    db_path: str = None,
    max_retries: int = 1,
) -> List[Dict]:
    """
    Generate question-centric training examples.
    Note: by design this function does not filter invalid SQL examples.
    """
    schema_str = format_schema_for_prompt(schema)

    base_prompt = f"""You are an expert SQL developer. Given the database schema and user question below, generate {num_examples} training examples.

Database Schema:
{schema_str}

User Question: "{question}"

Requirements for generated examples:
1. Each example must be semantically related to the user question
2. Examples should demonstrate complex SQL structures:
   - JOIN operations (INNER JOIN, LEFT JOIN, etc.)
   - Nested subqueries (in WHERE, FROM, or SELECT clauses)
   - Aggregation functions (COUNT, SUM, AVG, MAX, MIN)
   - HAVING clauses for filtering aggregated results
   - GROUP BY with multiple columns
3. Use only table and column names from the provided schema
4. Provide step-by-step reasoning for each SQL query

Return ONLY a valid JSON array with this exact format:
[
  {{
    "question": "Natural language question",
    "sql": "SELECT ... FROM ...",
    "reasoning": "Step 1: ... Step 2: ... Step 3: ..."
  }}
]

Generate {num_examples} diverse examples covering different complexity levels."""

    collected: List[Dict] = []
    prompt = base_prompt

    for _ in range(max(0, max_retries) + 1):
        response = llm_client.generate(prompt, max_tokens=3000)
        parsed = parse_examples_response(response)
        if parsed:
            collected.extend(parsed)

        if len(collected) >= num_examples:
            break

        prompt = (
            base_prompt
            + "\n\nThe previous output had too few items. Regenerate with full count.\n"
            + f"Return at least {num_examples} examples in JSON array."
        )

    return collected[:num_examples]


def format_schema_for_prompt(schema: Dict) -> str:
    lines = []

    for table in schema.get("tables", []):
        table_name = table["name"]
        col_parts = []
        for col in table.get("columns", []):
            desc = col.get("description", "")
            if desc and desc.replace(" ", "_").lower() != col["name"].lower():
                col_parts.append(f"{col['name']}: {desc} ({col['type']})")
            else:
                col_parts.append(f"{col['name']} ({col['type']})")
        columns = ", ".join(col_parts)
        lines.append(f"Table {table_name}: {columns}")

    if schema.get("foreign_keys"):
        lines.append("\nForeign Keys:")
        for fk in schema["foreign_keys"]:
            lines.append(
                f"  {fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
            )

    return "\n".join(lines)


def parse_examples_response(response: str) -> List[Dict]:
    json_match = re.search(r"\[[\s\S]*\]", response)
    if not json_match:
        return []

    try:
        examples = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    valid_examples = []
    for ex in examples:
        if isinstance(ex, dict) and "question" in ex and "sql" in ex:
            token_count = len(ex.get("question", "")) // 4 + len(ex.get("sql", "")) // 4 + 20
            valid_examples.append(
                {
                    "question": ex["question"],
                    "sql": ex["sql"],
                    "reasoning": ex.get("reasoning", ""),
                    "source": "question_centric",
                    "token_count": token_count,
                    "complexity_tag": classify_sql_complexity(ex.get("sql", "")),
                }
            )
    return valid_examples
