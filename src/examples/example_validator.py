"""
Validation helpers for generated training examples.
"""

import re
import sqlite3
from typing import Dict, List, Optional, Tuple

from src.generation.generator import validate_columns_against_schema, validate_sql


def _complexity_tag(sql: str) -> str:
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


def _detect_db_type(db_type: Optional[str], db_path: Optional[str], db_url: Optional[str]) -> str:
    if db_type:
        return db_type.lower()
    if db_url:
        if db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
            return "postgres"
        if db_url.startswith("mysql://") or db_url.startswith("mysql+pymysql://"):
            return "mysql"
    if db_path and str(db_path).lower().endswith(".sqlite"):
        return "sqlite"
    return "sqlite"


def _strip_trailing_semicolon(sql: str) -> str:
    return re.sub(r";\s*$", "", (sql or "").strip())


def _with_limit(sql: str, limit_n: int = 5) -> str:
    s = _strip_trailing_semicolon(sql)
    if re.search(r"\bLIMIT\b", s, flags=re.IGNORECASE):
        return s
    return f"{s} LIMIT {limit_n}"


def _parse_sql(sql: str, db_type: str) -> Tuple[bool, Optional[str]]:
    if not validate_sql(sql):
        return False, "invalid_syntax"

    # Prefer sqlglot parser when available.
    try:
        import sqlglot

        dialect = {"postgres": "postgres", "mysql": "mysql", "sqlite": "sqlite"}.get(db_type, "sqlite")
        sqlglot.parse_one(_strip_trailing_semicolon(sql), read=dialect)
        return True, None
    except ImportError:
        # Fallback: sqlparse-based validate_sql already passed.
        return True, None
    except Exception as exc:
        return False, f"parse_error:{exc}"


def _execute_and_check(
    sql: str,
    db_type: str,
    db_path: Optional[str] = None,
    db_url: Optional[str] = None,
    sample_limit: int = 5,
) -> Tuple[bool, Optional[str], int]:
    """
    Returns:
      (ok, reason_if_failed, row_count)
    """
    normalized = _strip_trailing_semicolon(sql)

    # SQLite path-based execution.
    if db_type == "sqlite":
        if not db_path:
            return False, "missing_db_path", 0
        try:
            conn = sqlite3.connect(db_path)
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            cur = conn.cursor()
            cur.execute(f"EXPLAIN QUERY PLAN {normalized}")
            cur.execute(_with_limit(normalized, sample_limit))
            rows = cur.fetchall()
            conn.close()
            return True, None, len(rows)
        except Exception as exc:
            return False, f"runtime_error:{exc}", 0

    # Postgres/MySQL via SQLAlchemy URL.
    if db_type in {"postgres", "mysql"}:
        if not db_url:
            return False, "missing_db_url", 0
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(text(f"EXPLAIN {normalized}"))
                rows = conn.execute(text(_with_limit(normalized, sample_limit))).fetchall()
            return True, None, len(rows)
        except Exception as exc:
            return False, f"runtime_error:{exc}", 0

    return False, f"unsupported_db_type:{db_type}", 0


def validate_generated_examples(
    examples: List[Dict],
    schema: Dict,
    db_path: str = None,
    db_type: str = None,
    db_url: str = None,
    keep_complex_on_exec_fail: bool = False,
    reject_empty_result: bool = True,
    sample_limit: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Validate generated examples by SQL syntax/schema/execution.

    Returns:
        (valid_examples, rejected_reports)
    """
    valid_examples: List[Dict] = []
    rejected: List[Dict] = []

    resolved_db_type = _detect_db_type(db_type=db_type, db_path=db_path, db_url=db_url)

    for ex in examples or []:
        sql = (ex.get("sql") or "").strip()
        if not sql:
            rejected.append({"reason": "empty_sql", "sql": sql})
            continue

        parse_ok, parse_reason = _parse_sql(sql, resolved_db_type)
        if not parse_ok:
            rejected.append({"reason": parse_reason or "invalid_syntax", "sql": sql})
            continue

        col_ok, invalid_items = validate_columns_against_schema(sql, schema)
        if not col_ok:
            rejected.append(
                {"reason": "invalid_schema_columns", "sql": sql, "details": invalid_items}
            )
            continue

        if db_path or db_url:
            exec_ok, exec_reason, row_count = _execute_and_check(
                sql=sql,
                db_type=resolved_db_type,
                db_path=db_path,
                db_url=db_url,
                sample_limit=sample_limit,
            )
            if not exec_ok:
                # Optional soft-keep for complex samples.
                tag = _complexity_tag(sql)
                if keep_complex_on_exec_fail and tag in {"hard", "extra"}:
                    ex = dict(ex)
                    ex["validation_warning"] = exec_reason or "execution_failed"
                    ex["complexity_tag"] = ex.get("complexity_tag") or tag
                    valid_examples.append(ex)
                    continue
                rejected.append({"reason": exec_reason or "execution_failed", "sql": sql})
                continue

            if reject_empty_result and row_count <= 0:
                rejected.append({"reason": "empty_result", "sql": sql})
                continue

        valid_examples.append(ex)

    return valid_examples, rejected


def summarize_rejections(rejected: List[Dict], max_items: int = 5) -> str:
    if not rejected:
        return ""
    lines: List[str] = []
    for item in rejected[:max_items]:
        reason = item.get("reason", "unknown")
        details = item.get("details")
        if details:
            lines.append(f"- {reason}: {details}")
        else:
            lines.append(f"- {reason}")
    return "\n".join(lines)
