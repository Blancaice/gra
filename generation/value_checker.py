"""
值对齐检查：当 SQL 返回空结果时，查询实际列值并让 LLM 修正
"""

import re
import sqlite3
from typing import Dict, List, Optional, Tuple


def extract_where_conditions(sql: str) -> List[Tuple[str, str, str]]:
    """
    提取 WHERE 子句中的等值条件（包括子查询中的）
    Returns: [(table, column, value), ...]
    """
    conditions = []
    # 匹配所有 column = 'value' 或 table.column = 'value'
    pattern = r"(?:(\w+)\.)?(\w+)\s*=\s*['\"]([^'\"]+)['\"]"
    for match in re.finditer(pattern, sql, re.IGNORECASE):
        table = match.group(1) or ""
        column = match.group(2)
        value = match.group(3)
        conditions.append((table, column, value))
    return conditions


def get_actual_values(db_path: str, table: str, column: str, limit: int = 10) -> List[str]:
    """
    查询列的实际取值
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT {column} FROM {table} LIMIT {limit}")
        results = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
        conn.close()
        return results
    except Exception:
        return []


def check_empty_result_and_repair(
    sql: str,
    user_question: str,
    schema: Dict,
    db_path: str,
    llm_client,
    evaluation_mode: str = "general",
    min_complexity: str = "high",
) -> Optional[str]:
    """
    零结果自检：如果 SQL 返回空，检查 WHERE 条件中的值是否存在，不存在则修复

    Args:
        min_complexity: 最小复杂度要求 ("low", "medium", "high")，默认只对 high 触发

    Returns: 修复后的 SQL，如果不需要修复则返回 None
    """
    from src.scoring.query_analyzer import analyze_question

    # 复杂度检查
    if min_complexity != "low":
        query_profile = analyze_question(user_question)
        complexity_order = {"low": 0, "medium": 1, "high": 2}
        question_level = complexity_order.get(query_profile.get("complexity_level", "low"), 0)
        required_level = complexity_order.get(min_complexity, 2)

        if question_level < required_level:
            return None
    # 1. 执行 SQL
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()

        if results and results != [(0,)] and results != [(None,)]:
            return None
    except Exception:
        return None

    # 2. 空结果，提取 WHERE 条件
    conditions = extract_where_conditions(sql)
    if not conditions:
        return None

    # 3. 查询实际值并构建反馈
    feedback_parts = []
    for table, column, value in conditions:
        target_table = table if table else _infer_table_from_schema(column, schema)
        if not target_table:
            continue

        actual_values = get_actual_values(db_path, target_table, column)
        if actual_values and value not in actual_values:
            feedback_parts.append(
                f"Column '{target_table}.{column}' does not contain '{value}'. "
                f"Actual values: {actual_values}"
            )

    if not feedback_parts:
        return None

    print(f"  [Value Check] Detected mismatches: {len(feedback_parts)} condition(s)")

    # 4. 让 LLM 修复
    from src.generation.prompt_builder import format_schema

    feedback = "\n".join(feedback_parts)
    repair_prompt = f"""The SQL query returned 0 results. This might be due to value mismatch:

{feedback}

Original question: {user_question}

Schema:
{format_schema(schema)}

Original SQL:
{sql}

If the issue is value mismatch, fix the SQL by choosing the most appropriate value from the available values above.
If the SQL logic is correct and should return 0, return the original SQL unchanged.
Return ONLY the SQL query without explanation."""

    response = llm_client.generate(repair_prompt, max_tokens=512, temperature=0.0)
    from src.generation.generator import extract_sql_from_response
    return extract_sql_from_response(response, evaluation_mode=evaluation_mode)


def _infer_table_from_schema(column: str, schema: Dict) -> Optional[str]:
    """
    从 schema 中推断列所属的表
    """
    for table in schema.get("tables", []):
        for col in table.get("columns", []):
            if col["name"].lower() == column.lower():
                return table["name"]
    return None
