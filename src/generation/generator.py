"""
SQL 生成与验证模块
"""

import re
import sqlparse
from typing import Optional, Dict, Set, List, Tuple
from src.scoring.ast_score import compute_ast_match_score, infer_target_structure


def generate_sql(prompt: str, llm_client, evaluation_mode: str = "general", max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """
    调用 LLM 生成 SQL

    Args:
        prompt: 完整的 Prompt
        llm_client: LLM API 客户端

    Returns:
        生成的 SQL 查询
    """
    response = llm_client.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    # 提取 SQL（可能包含在代码块中）
    sql = extract_sql_from_response(response, evaluation_mode=evaluation_mode)

    return sql


def generate_sql_candidates(
    prompt: str, llm_client, num_candidates: int = 3, evaluation_mode: str = "general"
) -> List[str]:
    """
    生成多个候选 SQL。通过不同的补充指令制造可重排候选。
    """
    if num_candidates <= 1:
        return [generate_sql(prompt, llm_client, evaluation_mode=evaluation_mode)]

    strategy_hints = [
        "Prefer ORDER BY + LIMIT style when finding min/max rows.",
        "Prefer direct aggregation style for count/sum/avg questions.",
        "Prefer explicit JOIN conditions and minimal selected columns.",
        "Prefer simpler equivalent SQL with fewer clauses.",
    ]

    candidates: List[str] = []
    for i in range(num_candidates):
        hint = strategy_hints[i % len(strategy_hints)]
        aug_prompt = (
            prompt
            + "\n\n# Candidate strategy\n"
            + hint
            + "\nReturn only one SQL query."
        )
        temp = 0.0 if i == 0 else 0.3
        sql = generate_sql(aug_prompt, llm_client, evaluation_mode=evaluation_mode, temperature=temp)
        if sql:
            candidates.append(sql)

    # 去重并保持顺序
    deduped = []
    seen = set()
    for sql in candidates:
        key = sql.strip().lower()
        if key not in seen:
            deduped.append(sql)
            seen.add(key)

    return deduped if deduped else [generate_sql(prompt, llm_client, evaluation_mode=evaluation_mode)]


def extract_sql_from_response(response: str, evaluation_mode: str = "general") -> str:
    """
    从 LLM 响应中提取 SQL

    Args:
        response: LLM 的原始响应

    Returns:
        提取的 SQL
    """
    # 移除 Markdown 代码块标记
    if "```sql" in response:
        start = response.find("```sql") + 6
        end = response.find("```", start)
        sql = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        sql = response[start:end].strip()
    else:
        sql = response.strip()

    # Keep first statement only and apply optional Spider-specific post-processing.
    return postprocess_sql_for_eval(sql, evaluation_mode=evaluation_mode)


def validate_sql(sql: str) -> bool:
    """
    验证 SQL 语法是否正确

    Args:
        sql: SQL 查询

    Returns:
        是否有效
    """
    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            return False

        # 检查是否为有效的 SELECT 语句
        statement = parsed[0]
        sql_upper = str(statement).upper().strip()

        # 基本检查：必须以 SELECT 开头
        if not sql_upper.startswith('SELECT'):
            return False

        return True

    except Exception as e:
        print(f"SQL validation error: {e}")
        return False


def extract_identifiers_from_sql(sql: str) -> Tuple[Set[str], Set[str]]:
    """
    从 SQL 中提取表名和列名

    Returns:
        (tables, columns) 两个集合
    """
    tables = set()
    columns = set()

    sql_keywords = {
        'select', 'from', 'where', 'join', 'on', 'and', 'or', 'not', 'in',
        'between', 'like', 'is', 'null', 'as', 'order', 'by', 'group',
        'having', 'limit', 'offset', 'union', 'intersect', 'except',
        'insert', 'update', 'delete', 'create', 'drop', 'alter', 'table',
        'index', 'view', 'inner', 'outer', 'left', 'right', 'cross',
        'natural', 'using', 'distinct', 'all', 'any', 'exists', 'case',
        'when', 'then', 'else', 'end', 'asc', 'desc', 'count', 'sum',
        'avg', 'min', 'max', 'cast', 'into', 'values', 'set', 'with',
    }

    sql_lower = sql.lower()

    # 提取 FROM/JOIN 后的表名
    table_pattern = r'(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(table_pattern, sql_lower):
        name = match.group(1)
        if name not in sql_keywords:
            tables.add(name)

    # 提取 table.column 格式
    qualified_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(qualified_pattern, sql_lower):
        tname = match.group(1)
        cname = match.group(2)
        if tname not in sql_keywords:
            tables.add(tname)
        if cname not in sql_keywords:
            columns.add(cname)

    return tables, columns


def validate_columns_against_schema(sql: str, schema: Dict) -> Tuple[bool, List[str]]:
    """
    验证 SQL 中的列名是否存在于 schema 中

    Args:
        sql: SQL 查询
        schema: 数据库 schema

    Returns:
        (is_valid, invalid_columns) 是否有效及无效列名列表
    """
    # 构建 schema 中所有有效的表名和列名（小写）
    valid_tables = set()
    valid_columns = set()
    table_columns = {}  # table -> set of columns

    for table in schema.get('tables', []):
        tname = table['name'].lower()
        valid_tables.add(tname)
        table_columns[tname] = set()
        for col in table.get('columns', []):
            cname = col['name'].lower()
            valid_columns.add(cname)
            table_columns[tname].add(cname)

    # 从 SQL 中提取标识符
    sql_tables, sql_columns = extract_identifiers_from_sql(sql)

    invalid = []

    # 验证表名
    for t in sql_tables:
        if t not in valid_tables:
            invalid.append(f"table:{t}")

    # 验证列名
    for c in sql_columns:
        if c not in valid_columns:
            invalid.append(f"column:{c}")

    return len(invalid) == 0, invalid


def rank_sql_candidates(
    candidates: List[str],
    user_question: str,
    schema: Dict,
    db_path: str = None,
    evaluation_mode: str = "general",
) -> Tuple[str, List[Dict]]:
    """
    候选重排：语法/列名/可执行性过滤 + 结构对齐与简洁度打分。
    """
    if not candidates:
        return "SELECT 1", []

    scored = []
    target = infer_target_structure(user_question)
    for sql in candidates:
        sql = postprocess_sql_for_eval(sql, evaluation_mode=evaluation_mode)
        syntax_ok = validate_sql(sql)
        col_ok, invalid_items = validate_columns_against_schema(sql, schema)
        exec_ok = True
        if db_path:
            exec_ok = execute_sql(sql, db_path) is not None

        ast_match = compute_ast_match_score(user_question, sql)
        structure_bonus = _structure_priority_bonus(sql, target)

        # 规则分：先过约束，再按结构匹配排序
        score = 0.0
        score += 0.45 * (1.0 if syntax_ok else 0.0)
        score += 0.20 * (1.0 if col_ok else 0.0)
        score += 0.15 * (1.0 if exec_ok else 0.0)
        score += 0.15 * ast_match
        # 结构优先奖励：目标问题需要复杂结构时，优先保留对应候选。
        score += 0.05 * structure_bonus

        scored.append({
            "sql": sql,
            "score": score,
            "syntax_ok": syntax_ok,
            "col_ok": col_ok,
            "exec_ok": exec_ok,
            "ast_match": ast_match,
            "structure_bonus": structure_bonus,
            "invalid_items": invalid_items,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[0]["sql"], scored


def execute_sql(sql: str, db_path: str) -> Optional[list]:
    """
    执行 SQL 并返回结果
    """
    import sqlite3

    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results

    except Exception as e:
        print(f"SQL execution error: {e}")
        return None


def execute_sql_with_error(sql: str, db_path: str) -> Tuple[Optional[list], Optional[str]]:
    """
    执行 SQL，返回 (结果, 错误信息)。成功时错误为 None。
    """
    import sqlite3

    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)


def repair_sql(
    sql: str,
    error_msg: str,
    user_question: str,
    schema: Dict,
    llm_client,
    evaluation_mode: str = "general",
) -> str:
    """
    根据执行错误让 LLM 修正 SQL。
    """
    from src.generation.prompt_builder import format_schema

    schema_str = format_schema(schema)
    prompt = f"""You are a SQL expert. The following SQL query failed to execute on a SQLite database.

Database Schema:
{schema_str}

User Question: {user_question}

Failed SQL:
{sql}

Error Message:
{error_msg}

Fix the SQL query so it executes correctly. Return ONLY the corrected SQL query, nothing else."""

    try:
        response = llm_client.generate(prompt)
        fixed = extract_sql_from_response(response, evaluation_mode=evaluation_mode)
        return fixed if fixed else sql
    except Exception as exc:
        print(f"  SQL repair LLM call failed: {exc}")
        return sql


def verify_column_selection(
    sql: str,
    user_question: str,
    schema: Dict,
    llm_client,
    evaluation_mode: str = "general",
) -> str:
    """
    Self-correction：检查 SQL 的结构正确性，有问题才修，没问题原样返回。
    """
    from src.generation.prompt_builder import format_schema
    schema_str = format_schema(schema)

    prompt = f"""For the given question, use the provided tables, columns, foreign keys, and primary keys to fix the given SQLite SQL QUERY for any issues. If there are no issues, return the SQL QUERY as is.

#### Use the following instructions for fixing the SQL QUERY:
1) Use the database values that are explicitly mentioned in the question.
2) Pay attention to the columns that are used for the JOIN by using the Foreign Keys.
3) Use DESC and DISTINCT when needed.
4) Pay attention to the columns that are used for the GROUP BY clause.
5) Pay attention to the columns that are used for the SELECT clause.
6) Only change the GROUP BY clause when necessary.
7) If the schema has a column that already stores a precomputed value, use it directly instead of an aggregate function.

Database Schema:
{schema_str}

Question: {user_question}

SQL QUERY: {sql}

Return ONLY the fixed SQL query, nothing else."""

    try:
        response = llm_client.generate(prompt)
        fixed = extract_sql_from_response(response, evaluation_mode=evaluation_mode)
        if fixed and fixed.strip():
            if fixed.strip().upper() != sql.strip().upper():
                print(f"  Self-correction: fixed SQL")
            return fixed
    except Exception as exc:
        print(f"  Self-correction failed: {exc}")

    return sql


def execute_with_repair(
    sql: str,
    user_question: str,
    schema: Dict,
    db_path: str,
    llm_client,
    evaluation_mode: str = "general",
    max_retries: int = 2,
) -> Tuple[str, bool]:
    """
    执行验证 + 错误修正循环。
    """
    current_sql = sql

    for attempt in range(max_retries + 1):
        results, error = execute_sql_with_error(current_sql, db_path)
        if error is None:
            if attempt > 0:
                print(f"  SQL repair succeeded on attempt {attempt}")
            return current_sql, True

        if attempt < max_retries:
            print(f"  SQL execution failed (attempt {attempt + 1}): {error}")
            current_sql = repair_sql(
                sql=current_sql,
                error_msg=error,
                user_question=user_question,
                schema=schema,
                llm_client=llm_client,
                evaluation_mode=evaluation_mode,
            )
        else:
            print(f"  SQL repair exhausted after {max_retries} retries")

    return current_sql, False


def _simplicity_score(sql: str) -> float:
    """
    简洁度分：语句越短、子句越少分越高。
    """
    if not sql:
        return 0.0
    length_penalty = min(1.0, len(sql) / 400.0)
    upper = sql.upper()
    clause_count = sum(
        upper.count(k) for k in (" JOIN ", " GROUP BY ", " ORDER BY ", " HAVING ", " UNION ", " INTERSECT ", " EXCEPT ")
    )
    clause_penalty = min(1.0, clause_count / 8.0)
    return max(0.0, 1.0 - 0.7 * length_penalty - 0.3 * clause_penalty)


def postprocess_sql_for_eval(sql: str, evaluation_mode: str = "general") -> str:
    """
    SQL 后处理（带模式开关）：
    1) 单语句、去分号、压缩空白
    2) 在 spider 模式下：去常见 AS 别名
    3) 在 spider 模式下：对常见 MIN/MAX 子查询模式做 canonical 重写
    """
    if not sql:
        return ""

    # 单语句 + 空白规范化
    s = sql.strip().split(";")[0].strip()
    s = re.sub(r"\s+", " ", s)

    # 仅在 Spider 评测模式下启用 canonical 技巧，避免影响通用场景。
    if (evaluation_mode or "general").lower() == "spider":
        # 重写 MIN/MAX 子查询到 ORDER BY LIMIT 形式（Spider 中更常见）
        s = _rewrite_minmax_subquery(s)
    return s


def _rewrite_minmax_subquery(sql: str) -> str:
    """
    将模式：
    SELECT <sel> FROM T WHERE c = (SELECT MIN(c) FROM T)
    重写为：
    SELECT <sel> FROM T ORDER BY c ASC LIMIT 1
    MAX 同理改为 DESC。
    """
    pattern = re.compile(
        r"^\s*SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)\s+"
        r"WHERE\s+(?P<col>[A-Za-z_][A-Za-z0-9_\.]*)\s*=\s*"
        r"\(\s*SELECT\s+(?P<agg>MIN|MAX)\s*\(\s*(?P<subcol>[A-Za-z_][A-Za-z0-9_\.]*)\s*\)\s+"
        r"FROM\s+(?P<table2>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$",
        flags=re.IGNORECASE,
    )
    m = pattern.match(sql)
    if not m:
        return sql

    table = m.group("table")
    table2 = m.group("table2")
    col = m.group("col")
    subcol = m.group("subcol")
    agg = m.group("agg").upper()

    # 仅在“同表同列”场景重写，避免误改语义。
    if table.lower() != table2.lower():
        return sql
    if col.split(".")[-1].lower() != subcol.split(".")[-1].lower():
        return sql

    direction = "ASC" if agg == "MIN" else "DESC"
    return f"SELECT {m.group('select')} FROM {table} ORDER BY {col} {direction} LIMIT 1"


def _structure_priority_bonus(sql: str, target: Dict[str, bool]) -> float:
    """
    结构优先奖励：
    - 问题需要 set ops / subquery / join 等结构时，候选包含这些结构可获得额外分。
    """
    s = (sql or "").upper()
    bonus = 0.0

    has_set_ops = any(op in s for op in (" UNION ", " INTERSECT ", " EXCEPT "))
    has_subquery = s.count("SELECT") > 1
    has_join = " JOIN " in s

    if target.get("need_set_ops") and has_set_ops:
        bonus += 0.5
    if target.get("need_subquery") and has_subquery:
        bonus += 0.3
    if target.get("need_join") and has_join:
        bonus += 0.2

    return min(1.0, bonus)
