"""
Agent-based example generation for text-to-SQL data construction.
"""

import json
import re
from typing import Dict, List, Optional


def _format_schema(schema: Dict) -> str:
    lines = []
    for table in schema.get("tables", []):
        col_parts = []
        for col in table.get("columns", []):
            col_name = col.get("name", "")
            col_type = col.get("type", "")
            col_parts.append(f"{col_name} ({col_type})" if col_type else col_name)
        lines.append(f"Table {table.get('name', '')}: {', '.join(col_parts)}")

    if schema.get("foreign_keys"):
        lines.append("Foreign Keys:")
        for fk in schema["foreign_keys"]:
            lines.append(
                f"{fk.get('from_table')}.{fk.get('from_column')} -> "
                f"{fk.get('to_table')}.{fk.get('to_column')}"
            )
    return "\n".join(lines)


def _extract_json_array(text: str) -> List[Dict]:
    if not text:
        return []
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def classify_sql_complexity(sql: str) -> str:
    s = (sql or "").upper()
    has_set_ops = any(op in s for op in (" UNION ", " INTERSECT ", " EXCEPT "))
    has_sub = s.count("SELECT") > 1
    join_cnt = s.count("JOIN")
    has_group = "GROUP BY" in s
    has_having = "HAVING" in s
    has_order = "ORDER BY" in s

    if has_set_ops or (has_sub and join_cnt > 0) or (has_sub and has_group and has_having):
        return "extra"
    if has_sub or join_cnt >= 2 or (has_group and has_having):
        return "hard"
    if join_cnt >= 1 or has_group or has_order:
        return "medium"
    return "easy"


def _normalize_examples(items: List[Dict], source: str, fallback_sqls: Optional[List[str]] = None) -> List[Dict]:
    valid = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        question = (item.get("question") or "").strip()
        sql = (item.get("sql") or "").strip()
        if fallback_sqls and idx < len(fallback_sqls):
            sql = (fallback_sqls[idx] or "").strip() or sql
        if not question or not sql:
            continue

        token_count = len(question) // 4 + len(sql) // 4 + 20
        valid.append(
            {
                "question": question,
                "sql": sql,
                "reasoning": (item.get("reasoning") or "").strip(),
                "source": source,
                "token_count": token_count,
                "complexity_tag": classify_sql_complexity(sql),
            }
        )
    return valid


class QuestionCentricAgent:
    """Generate examples semantically centered on user question."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def run(self, question: str, schema: Dict, num_examples: int) -> List[Dict]:
        schema_str = _format_schema(schema)
        prompt = f"""You are a Question-Centric SQL Agent.
Your objective is to create {num_examples} SQL training examples semantically close to the user question.

Database Schema:
{schema_str}

User Question:
{question}

Constraints:
1. Use only tables/columns in schema.
2. Cover mixed complexity: at least one easy/medium and at least one hard (join/subquery/having/set op).
3. Keep SQL executable and realistic for this schema.
4. Return JSON array only:
[{{"question":"...","sql":"...","reasoning":"..."}}]
"""
        response = self.llm_client.generate(prompt, max_tokens=3000)
        return _normalize_examples(_extract_json_array(response), source="question_centric")


class SchemaCentricAgent:
    """Generate examples centered on schema coverage and join paths."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def run(self, schema: Dict, num_examples: int, sql_templates: Optional[List[str]] = None) -> List[Dict]:
        schema_str = _format_schema(schema)
        templates_block = ""
        if sql_templates:
            numbered = "\n".join([f"{i + 1}. {sql}" for i, sql in enumerate(sql_templates)])
            templates_block = (
                "Seed SQL templates (preserve each template SQL exactly if you use it):\n"
                f"{numbered}\n"
            )

        prompt = f"""You are a Schema-Centric SQL Agent.
Generate {num_examples} examples emphasizing schema coverage.

Database Schema:
{schema_str}

{templates_block}
Constraints:
1. Maximize coverage of different tables and foreign-key relations.
2. Include diverse SQL structures (join/group/having/subquery/set-op where appropriate).
3. Use only valid schema identifiers.
4. Return JSON array only:
[{{"question":"...","sql":"...","reasoning":"..."}}]
"""
        response = self.llm_client.generate(prompt, max_tokens=3200)
        parsed = _extract_json_array(response)
        return _normalize_examples(parsed, source="schema_centric", fallback_sqls=sql_templates)


class ASTRefinerAgent:
    """Refine generated examples for stronger AST-structure diversity."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def run(self, schema: Dict, examples: List[Dict], target_size: Optional[int] = None) -> List[Dict]:
        if not examples:
            return []

        schema_str = _format_schema(schema)
        keep_n = target_size if isinstance(target_size, int) and target_size > 0 else len(examples)
        payload = json.dumps(
            [
                {
                    "question": e.get("question", ""),
                    "sql": e.get("sql", ""),
                    "reasoning": e.get("reasoning", ""),
                }
                for e in examples[:keep_n]
            ],
            ensure_ascii=False,
            indent=2,
        )

        prompt = f"""You are an AST-Refiner Agent for SQL examples.
Given schema and seed examples, improve structure diversity while keeping semantics close.

Database Schema:
{schema_str}

Seed Examples:
{payload}

Refinement goals:
1. Keep each example question-SQL pair semantically aligned.
2. Improve AST diversity across set operations, nested subqueries, join depth, and having/group combinations.
3. Keep SQL executable and schema-valid.
4. Return exactly {keep_n} items as JSON array only:
[{{"question":"...","sql":"...","reasoning":"..."}}]
"""
        response = self.llm_client.generate(prompt, max_tokens=3600)
        refined = _normalize_examples(_extract_json_array(response), source="ast_agent")
        return refined if refined else examples[:keep_n]
