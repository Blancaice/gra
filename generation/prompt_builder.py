"""
Prompt construction for SQL generation.
"""

from typing import Dict, List


def build_prompt(schema: Dict, examples: List[Dict], user_question: str,
                 schema_hints: Dict = None, structure_hints: Dict = None,
                 db_path: str = None, include_values: bool = False) -> str:
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
        "but ALWAYS use the exact column identifier (before the type) in your SQL, never the description text.\n"
        "9) For 'youngest/oldest/highest/lowest' queries, prefer MAX/MIN aggregation over ORDER BY + LIMIT when possible.\n"
        "10) For counting, use COUNT(*) unless the question specifically asks to count a particular column.\n"
        "11) For multiple OR conditions on the same column (e.g., year = 1948 OR year = 1949), write them as separate OR clauses, not IN.\n"
        "12) For GROUP BY, group by the column the question asks about (usually a name/description column, not an ID).\n"
        "13) CRITICAL: Only add WHERE conditions explicitly mentioned in the question. Do NOT add:\n"
        "    - NULL checks (IS NULL, IS NOT NULL) unless the question mentions 'null', 'missing', 'empty'\n"
        "    - Date filters unless the question specifies a date range\n"
        "    - Status/state filters unless explicitly asked\n"
        "14) CRITICAL: Do NOT fabricate literal values in WHERE clauses. If the question uses pronouns like 'this type',"
        " it refers to values that exist in the data, not literal strings. You must JOIN tables to resolve such references.\n"
        "15) For exclusion (not in X), use NOT IN with a subquery.\n"
        "16) For set operations (both X and Y), use INTERSECT only when combining two complete SELECT queries.\n"
        "17) CRITICAL: If a table or column name contains spaces, special characters, or is a SQL keyword, "
        "you MUST wrap it in backticks. Check the schema above for names with spaces.\n"
    )

    prompt_parts.append("# Database Schema\n")
    prompt_parts.append(format_schema(schema, db_path=db_path, include_values=include_values))
    prompt_parts.append("\n")

    if schema_hints:
        from src.generation.schema_linker import format_schema_hints
        prompt_parts.append("# Schema Analysis\n")
        prompt_parts.append(format_schema_hints(schema_hints))
        prompt_parts.append("\n")

    # 结构预测提示：告诉 LLM 这个问题可能需要什么 SQL 结构
    if structure_hints:
        prompt_parts.append("# Predicted SQL Structure (use as guidance, not as strict rules)\n")
        prompt_parts.append(format_structure_hints(structure_hints))
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


def format_structure_hints(hints: Dict) -> str:
    """将结构预测格式化为 prompt 中的自然语言提示。"""
    lines = []
    # hints 格式: {"has_join": 0.85, "has_subquery": 0.12, ...}
    # 只输出高置信度的预测（>0.6），避免噪声
    feature_desc = {
        "has_aggregation": "This question likely requires aggregation (COUNT/SUM/AVG/MAX/MIN).",
        "has_join": "This question likely requires JOIN across multiple tables.",
        "has_where": "This question likely requires WHERE conditions.",
        "has_subquery": "This question likely requires a subquery.",
        "has_groupby": "This question likely requires GROUP BY.",
        "has_orderby": "This question likely requires ORDER BY.",
        "has_having": "This question likely requires HAVING.",
        "has_limit": "This question likely requires LIMIT.",
        "has_set_op": "This question likely requires set operations (UNION/INTERSECT/EXCEPT).",
        "has_distinct": "This question likely requires DISTINCT.",
    }
    for feat, proba in hints.items():
        if proba > 0.6 and feat in feature_desc:
            lines.append(f"- {feature_desc[feat]} (confidence: {proba:.0%})")

    return "\n".join(lines) if lines else "No strong structural predictions."


def _quote_if_needed(name: str) -> str:
    """对含空格或特殊字符的标识符加反引号"""
    if " " in name or "-" in name:
        return f"`{name}`"
    return name


def format_schema(schema: Dict, db_path: str = None, include_values: bool = False) -> str:
    """
    格式化 schema 为 prompt 文本。

    Args:
        schema: schema 字典
        db_path: 数据库路径（用于值检索）
        include_values: 是否包含枚举列的候选值
    """
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

            # 值检索：枚举类型列附加候选值
            if include_values and db_path and col['type'].upper() in ('TEXT', 'VARCHAR'):
                values = _fetch_column_values(db_path, table["name"], col["name"], limit=10)
                if values:
                    col_info += f" [values: {', '.join(values)}]"

            lines.append(col_info)
        lines.append("")

    if schema.get("foreign_keys"):
        lines.append("Foreign Keys:")
        for fk in schema["foreign_keys"]:
            lines.append(
                f"  - {fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
            )

    return "\n".join(lines)


def _fetch_column_values(db_path: str, table: str, column: str, limit: int = 20) -> list:
    """从数据库中获取列的 distinct 值（用于值检索）。"""
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # 使用参数化查询防注入（表名/列名需要手动验证）
        query = f"SELECT DISTINCT `{column}` FROM `{table}` WHERE `{column}` IS NOT NULL LIMIT {limit}"
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return [str(row[0]) for row in rows if row[0]]
    except Exception:
        return []


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
