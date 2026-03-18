"""
Question-guided semantic example generation.

根据用户问题 + pruned schema，让 LLM 生成语义相近的 question-SQL 示例对，
作为 few-shot 候选进入 scoring 流程。
"""

import json
import re
from typing import Dict, List

from src.examples.example_validator import validate_generated_examples
from src.examples.schema_centric import classify_sql_complexity


def generate_semantic_examples(
    question: str,
    pruned_schema: Dict,
    llm_client,
    db_path: str = None,
    num_examples: int = 3,
) -> List[Dict]:
    """
    用 LLM 根据用户问题语义 + schema 生成相近的 question-SQL 示例。

    Returns:
        标准格式的示例列表，经过 schema/执行验证。
    """
    if not pruned_schema.get("tables"):
        return []

    schema_str = _format_schema_compact(pruned_schema)

    prompt = f"""You are a SQL expert. Given the database schema and a user question, generate {num_examples} similar but different question-SQL pairs that exercise the same tables and similar query patterns.

Database Schema:
{schema_str}

User Question: {question}

Requirements:
1. Each generated question should be semantically related to the user question (similar topic, similar tables, similar complexity).
2. SQL must be valid SQLite and use ONLY tables/columns from the schema above.
3. Cover slightly different query patterns (e.g., different aggregations, filters, or joins).
4. Do NOT generate the exact same question as the user question.
5. Return ONLY a JSON array, no explanation:
[{{"question":"...","sql":"...","reasoning":"..."}}]
"""

    try:
        response = llm_client.generate(prompt, max_tokens=2000)
        parsed = _parse_examples(response)

        valid_examples, _ = validate_generated_examples(
            parsed,
            schema=pruned_schema,
            db_path=db_path,
            keep_complex_on_exec_fail=False,
            reject_empty_result=True,
            sample_limit=5,
        )
        return valid_examples if valid_examples else parsed[:num_examples]
    except Exception as exc:
        print(f"  Semantic example generation failed: {exc}")
        return []


def _parse_examples(response: str) -> List[Dict]:
    """从 LLM 响应中解析示例 JSON 数组。"""
    json_match = re.search(r"\[[\s\S]*\]", response or "")
    if not json_match:
        return []

    try:
        items = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    results = []
    for item in items:
        if not isinstance(item, dict) or "question" not in item or "sql" not in item:
            continue
        q = (item.get("question") or "").strip()
        sql = (item.get("sql") or "").strip()
        if not q or not sql:
            continue
        token_count = len(q) // 4 + len(sql) // 4 + 20
        results.append({
            "question": q,
            "sql": sql,
            "reasoning": (item.get("reasoning") or "").strip(),
            "source": "semantic",
            "token_count": token_count,
            "complexity_tag": classify_sql_complexity(sql),
        })
    return results


def _format_schema_compact(schema: Dict) -> str:
    """紧凑格式的 schema 描述。"""
    lines = []
    for table in schema.get("tables", []):
        cols = []
        for c in table.get("columns", []):
            pk = " [PK]" if c.get("primary_key") else ""
            desc = c.get("description", "")
            if desc and desc.lower() != c["name"].lower():
                cols.append(f"{c['name']} ({c['type']}{pk}) -- {desc}")
            else:
                cols.append(f"{c['name']} ({c['type']}{pk})")
        lines.append(f"{table['name']}: {', '.join(cols)}")

    if schema.get("foreign_keys"):
        lines.append("Foreign Keys:")
        for fk in schema["foreign_keys"]:
            lines.append(
                f"  {fk['from_table']}.{fk['from_column']} -> "
                f"{fk['to_table']}.{fk['to_column']}"
            )
    return "\n".join(lines)
