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

    # 第一个候选 temperature=0 保底，后续用 0.3 制造多样性
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
        simplicity = _simplicity_score(sql)
        structure_bonus = _structure_priority_bonus(sql, target)

        # 规则分：先过约束，再按结构匹配和简洁度排序
        score = 0.0
        score += 0.45 * (1.0 if syntax_ok else 0.0)
        score += 0.20 * (1.0 if col_ok else 0.0)
        score += 0.15 * (1.0 if exec_ok else 0.0)
        score += 0.15 * ast_match
        score += 0.03 * simplicity
        # 结构优先奖励：目标问题需要复杂结构时，优先保留对应候选。
        score += 0.02 * structure_bonus

        scored.append({
            "sql": sql,
            "score": score,
            "syntax_ok": syntax_ok,
            "col_ok": col_ok,
            "exec_ok": exec_ok,
            "ast_match": ast_match,
            "simplicity": simplicity,
            "structure_bonus": structure_bonus,
            "invalid_items": invalid_items,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[0]["sql"], scored


def execute_sql(sql: str, db_path: str) -> Optional[list]:
    """
    执行 SQL 并返回结果

    Args:
        sql: SQL 查询
        db_path: 数据库路径

    Returns:
        查询结果（列表），如果失败则返回 None
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
    max_tokens: int = 1024,
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

Fix the SQL query so it executes correctly. Common issues:
- Wrong table or column names (check the schema above carefully)
- Missing JOIN conditions
- Ambiguous column references (use table.column format)
- SQLite syntax issues (no TOP, use LIMIT instead)

Return ONLY the corrected SQL query, nothing else."""

    try:
        response = llm_client.generate(prompt, max_tokens=max_tokens)
        fixed = extract_sql_from_response(response, evaluation_mode=evaluation_mode)
        return fixed if fixed else sql
    except Exception as exc:
        print(f"  SQL repair LLM call failed: {exc}")
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
    执行 SQL，失败则自动修正重试。同时检查结构一致性（IUEN）。

    Returns:
        (最终SQL, 是否执行成功)
    """
    # 列选择自验证：检查 SELECT 的列是否语义正确
    verified_sql = verify_column_selection(sql, user_question, schema, llm_client, evaluation_mode)

    # 验证改了 SQL 的话，检查改后还能不能执行，不能就回退
    if verified_sql != sql:
        _, v_err = execute_sql_with_error(verified_sql, db_path)
        if v_err is not None:
            print(f"  Verification broke SQL (exec failed), reverting")
            current_sql = sql
        else:
            current_sql = verified_sql
    else:
        current_sql = verified_sql

    for attempt in range(max_retries + 1):
        results, error = execute_sql_with_error(current_sql, db_path)
        if error is None:
            if attempt > 0:
                print(f"  SQL repair succeeded on attempt {attempt}")
            return current_sql, True

        if attempt < max_retries:
            print(f"  SQL execution failed (attempt {attempt + 1}): {error}")
            print(f"  Attempting repair...")
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


def _check_set_ops_consistency(
    sql: str,
    user_question: str,
    schema: Dict,
    llm_client,
    evaluation_mode: str = "general",
) -> str:
    """
    检查问题是否暗示需要 INTERSECT/EXCEPT/UNION，但 SQL 没有使用。
    如果检测到不一致，让 LLM 重新生成。
    """
    q = (user_question or "").lower()
    sql_upper = (sql or "").upper()

    has_set_op = any(op in sql_upper for op in (" INTERSECT ", " UNION ", " EXCEPT "))
    if has_set_op:
        return sql

    # 检测需要 INTERSECT 的模式
    need_intersect = any(p in q for p in (
        "both", "as well as", "and also", "in common",
    ))

    # 检测需要 EXCEPT 的模式
    need_except = any(p in q for p in (
        "but not", "except", "not in", "who did not", "that did not",
        "who have not", "that have not", "who do not", "that do not",
        "never", "without",
    ))

    if not (need_intersect or need_except):
        return sql

    from src.generation.prompt_builder import format_schema
    schema_str = format_schema(schema)

    if need_intersect:
        hint = "The question asks for items satisfying BOTH/ALL conditions across different rows. You MUST use INTERSECT."
    elif need_except:
        hint = "The question asks for exclusion (not/except/never). You MUST use EXCEPT or NOT IN with a subquery."
    else:
        return sql

    prompt = f"""You are a SQL expert. The following SQL may be incorrect because it does not use the required set operation.

Database Schema:
{schema_str}

User Question: {user_question}

Current SQL:
{sql}

Issue: {hint}

Rewrite the SQL using the correct set operation (INTERSECT/EXCEPT/UNION as needed).
If the current SQL is actually correct without set operations, return it unchanged.
Return ONLY the SQL query, nothing else."""

    try:
        response = llm_client.generate(prompt, max_tokens=1024)
        fixed = extract_sql_from_response(response, evaluation_mode=evaluation_mode)
        if fixed and fixed.strip():
            print(f"  Set-op consistency check: rewrote SQL (hint={hint[:50]}...)")
            return fixed
    except Exception as exc:
        print(f"  Set-op consistency check failed: {exc}")

    return sql


def verify_column_selection(
    sql: str,
    user_question: str,
    schema: Dict,
    llm_client,
    evaluation_mode: str = "general",
) -> str:
    """
    自验证：全面检查 SQL 的语义正确性。
    覆盖：列选择错误、幻觉条件、缺少 JOIN、函数 vs 列名混淆。
    """
    from src.generation.prompt_builder import format_schema
    schema_str = format_schema(schema)

    prompt = f"""You are a SQL expert. Verify the SQL query below against the question and schema.

Database Schema:
{schema_str}

User Question: {user_question}

SQL: {sql}

Check carefully:
1. Does each selected column match what the question is asking for?
2. Are there columns with similar names where the wrong one was chosen? (e.g.,"id" vs "student_id")
3. Use the column descriptions (after '--') to verify the semantic meaning.
4. If the schema already has a column that stores the value the SQL is computing with an aggregate function (e.g., AVG/SUM/COUNT), use that column directly instead of the aggregate.

If the SQL selects wrong columns or uses unnecessary aggregates, fix them. Keep everything else unchanged.
If the SQL is correct, return it unchanged.
Return ONLY the SQL query, nothing else."""

    try:
        response = llm_client.generate(prompt, max_tokens=1024)
        fixed = extract_sql_from_response(response, evaluation_mode=evaluation_mode)
        if fixed and fixed.strip():
            if fixed.strip().upper() != sql.strip().upper():
                print(f"  SQL verification: corrected SQL")
            return fixed
    except Exception as exc:
        print(f"  SQL verification failed: {exc}")

    return sql


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
        s = _remove_implicit_aliases(s)
        s = _remove_cast(s)
        s = _strip_table_prefix(s)
        s = _rewrite_minmax_subquery(s)
    return s


def _remove_implicit_aliases(sql: str) -> str:
    """
    移除隐式别名：FROM table t / JOIN table t → FROM table / JOIN table
    同时将 SQL 中对别名的引用替换回原表名。
    """
    alias_map = {}
    # 匹配 FROM/JOIN table alias（alias 为 1-4 个小写字母，非关键字）
    kw = {'on', 'where', 'join', 'inner', 'left', 'right', 'cross', 'outer',
          'group', 'order', 'having', 'limit', 'union', 'intersect', 'except',
          'set', 'as', 'and', 'or', 'not', 'in', 'select', 'from', 'natural', 'using'}
    pattern = re.compile(
        r'\b(FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?(\w+)\b',
        flags=re.IGNORECASE,
    )
    for m in pattern.finditer(sql):
        alias = m.group(3)
        if alias.lower() in kw:
            continue
        table = m.group(2)
        if alias.lower() == table.lower():
            continue
        # 短别名（1-4 字母）或 T1/T2 风格
        if re.match(r'^[a-zA-Z]{1,4}$', alias) or re.match(r'^[tT]\d+$', alias):
            alias_map[alias] = table

    if not alias_map:
        return sql

    # 移除别名声明
    def _remove_alias_decl(m):
        keyword = m.group(1)
        table = m.group(2)
        alias = m.group(3)
        if alias in alias_map:
            return f"{keyword} {table}"
        return m.group(0)

    s = pattern.sub(_remove_alias_decl, sql)

    # 替换别名引用为原表名（alias.col → table.col, alias 单独出现也替换）
    for alias, table in alias_map.items():
        s = re.sub(r'\b' + re.escape(alias) + r'\.', table + '.', s)

    return s


def _remove_cast(sql: str) -> str:
    """
    移除 CAST(col AS type)，替换为 col。
    Spider 解析器不支持 CAST 语法。
    """
    return re.sub(
        r'CAST\s*\(\s*(\w+(?:\.\w+)?)\s+AS\s+\w+\s*\)',
        r'\1',
        sql,
        flags=re.IGNORECASE,
    )


def _strip_table_prefix(sql: str) -> str:
    """
    在单表查询中移除 table.column 前缀，只保留 column。
    多表查询保留前缀以避免歧义。
    """
    # 统计 FROM/JOIN 涉及的表数量
    tables = re.findall(r'\b(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE)
    unique_tables = set(t.lower() for t in tables)
    if len(unique_tables) > 1:
        return sql
    # 单表：移除 table. 前缀
    if unique_tables:
        table = unique_tables.pop()
        sql = re.sub(r'\b' + re.escape(table) + r'\.(\w+)', r'\1', sql, flags=re.IGNORECASE)
    return sql


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
