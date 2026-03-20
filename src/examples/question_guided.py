"""
Question-guided column mapping.

让 LLM 分析用户问题中的关键概念对应 schema 中的哪些表和列，
输出映射分析文本，拼入最终 SQL 生成 prompt 辅助消歧。
"""

import json
import re
from typing import Dict, List, Optional


def generate_column_mapping(
    question: str,
    pruned_schema: Dict,
    llm_client,
) -> str:
    """
    让 LLM 分析问题关键词与 schema 列的映射关系。
    返回映射分析文本，供最终 prompt 使用。
    """
    if not pruned_schema.get("tables"):
        return ""

    schema_str = _format_schema_compact(pruned_schema)

    prompt = f"""Given the database schema and user question below, identify which tables and columns are needed to answer the question.

For each key concept in the question, map it to the exact column name from the schema. If multiple columns have similar names, explain which one is correct and why.

Be concise. Output only the mapping, one per line, format:
"<concept>" -> <table>.<column>

## Schema
{schema_str}

## Question
{question}
"""

    try:
        response = llm_client.generate(prompt, max_tokens=300)
        return (response or "").strip()
    except Exception as exc:
        print(f"  Column mapping failed: {exc}")
        return ""


def _format_schema_compact(schema: Dict) -> str:
    """紧凑格式的 schema 描述。"""
    lines = []
    for table in schema.get("tables", []):
        cols = []
        for c in table.get("columns", []):
            pk = " [PK]" if c.get("primary_key") else ""
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
