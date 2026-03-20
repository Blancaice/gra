"""
Schema Linking 模块：用 LLM 精确识别问题所需的表、列、JOIN 路径。
在向量检索粗筛之后，用一次 LLM 调用做精选。
"""

import json
import re
from typing import Dict, Optional

from src.generation.prompt_builder import format_schema


# 复杂查询标志词
_COMPLEX_PATTERNS = (
    "both", "not in", "except", "more than average", "more than the",
    "less than average", "less than the", "at least", "at most",
    "who also", "that also", "in common", "but not", "neither",
)
# 介词连词（出现 >= 2 次暗示多步推理）
_PREPOSITIONS = ("of", "by", "for", "in", "with")


def is_complex_question(question: str) -> bool:
    """启发式判断问题是否为复杂查询，满足任意一条即返回 True。"""
    q = (question or "").lower()
    tokens = re.findall(r"\w+", q)

    # 规则 1：包含复杂查询标志词
    if any(p in q for p in _COMPLEX_PATTERNS):
        return True

    # 规则 2：介词出现 >= 2 种
    prep_count = sum(1 for p in _PREPOSITIONS if f" {p} " in f" {q} ")
    if prep_count >= 2:
        return True

    # 规则 3：问题长度 > 20 词
    if len(tokens) > 20:
        return True

    return False


def link_schema(
    question: str,
    pruned_schema: Dict,
    llm_client,
    max_tokens: int = 512,
) -> Optional[Dict]:
    """
    用 LLM 分析问题与 schema 的对应关系。

    Args:
        question: 用户自然语言问题
        pruned_schema: 向量检索后的裁剪 schema
        llm_client: LLM 客户端
        max_tokens: 最大生成 token 数

    Returns:
        schema linking hints dict，解析失败返回 None
    """
    schema_str = format_schema(pruned_schema)
    prompt = _build_linking_prompt(question, schema_str)

    try:
        response = llm_client.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        hints = _parse_linking_response(response)
        if hints:
            hints = _validate_hints(hints, pruned_schema)
        return hints
    except Exception as exc:
        print(f"  Schema linking failed: {exc}")
        return None


def _build_linking_prompt(question: str, schema_str: str) -> str:
    return f"""You are a database schema analyst. Given a question and database schema, identify exactly which tables, columns, and joins are needed to write the SQL query.

Database Schema:
{schema_str}

Question: {question}

Return a JSON object with these fields:
- "tables": list of table names needed (use exact names from schema)
- "columns": dict mapping each table name to a list of its needed columns
- "join_path": list of join conditions as "table1.col = table2.col" strings (empty list if single table)

Return ONLY valid JSON, no markdown, no explanation outside JSON."""


def _parse_linking_response(response: str) -> Optional[Dict]:
    """从 LLM 响应中解析 JSON。"""
    text = response.strip()
    # 去掉可能的 markdown 代码块
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试找到第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def _validate_hints(hints: Dict, pruned_schema: Dict) -> Dict:
    """验证 hints 中的表/列名是否存在于 pruned_schema，过滤无效项。"""
    valid_tables = {t["name"].lower(): t["name"] for t in pruned_schema["tables"]}
    valid_columns = {}
    for t in pruned_schema["tables"]:
        valid_columns[t["name"].lower()] = {
            c["name"].lower(): c["name"] for c in t["columns"]
        }

    # 过滤表名
    filtered_tables = []
    for tname in hints.get("tables", []):
        if tname.lower() in valid_tables:
            filtered_tables.append(valid_tables[tname.lower()])
    hints["tables"] = filtered_tables

    # 过滤列名
    filtered_columns = {}
    for tname, cols in hints.get("columns", {}).items():
        tkey = tname.lower()
        if tkey in valid_columns:
            real_tname = valid_tables[tkey]
            filtered_cols = []
            for c in cols:
                if c.lower() in valid_columns[tkey]:
                    filtered_cols.append(valid_columns[tkey][c.lower()])
            if filtered_cols:
                filtered_columns[real_tname] = filtered_cols
    hints["columns"] = filtered_columns

    return hints


def format_schema_hints(hints: Dict) -> str:
    """将 schema linking hints 格式化为 prompt 注入文本。"""
    parts = []

    tables = hints.get("tables", [])
    if tables:
        parts.append(f"Relevant tables: {', '.join(tables)}")

    columns = hints.get("columns", {})
    if columns:
        col_parts = []
        for tname, cols in columns.items():
            col_parts.append(f"{tname}({', '.join(cols)})")
        parts.append(f"Key columns: {'; '.join(col_parts)}")

    join_path = hints.get("join_path", [])
    if join_path:
        parts.append(f"Join conditions: {'; '.join(join_path)}")

    return "\n".join(parts)
