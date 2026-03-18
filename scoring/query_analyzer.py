"""
Query analysis utilities for dynamic scoring and example allocation.
"""

import re
from typing import Dict


AGG_KEYWORDS = ("count", "sum", "avg", "max", "min", "total", "average")
ORDER_KEYWORDS = ("order by", "sort", "top", "highest", "lowest", "rank")
TIME_KEYWORDS = ("year", "month", "day", "date", "between", "before", "after")
JOIN_HINTS = ("each", "per", "with", "whose", "and their", "across")
SUBQUERY_HINTS = ("more than", "less than", "at least", "at most", "than the")


def analyze_question(question: str) -> Dict:
    """
    Build a lightweight query profile from NL question text.
    """
    q = (question or "").lower()
    tokens = re.findall(r"\w+", q)

    has_agg = _contains_any(q, AGG_KEYWORDS)
    has_order = _contains_any(q, ORDER_KEYWORDS)
    has_time = _contains_any(q, TIME_KEYWORDS)
    has_join_hint = _contains_any(q, JOIN_HINTS)
    has_subquery_hint = _contains_any(q, SUBQUERY_HINTS)

    complexity = 0
    complexity += 2 if has_agg else 0
    complexity += 1 if has_order else 0
    complexity += 1 if has_time else 0
    complexity += 2 if has_join_hint else 0
    complexity += 2 if has_subquery_hint else 0
    complexity += 1 if len(tokens) > 16 else 0

    if complexity >= 6:
        level = "high"
    elif complexity >= 3:
        level = "medium"
    else:
        level = "low"

    return {
        "has_aggregation": has_agg,
        "has_order": has_order,
        "has_time_filter": has_time,
        "needs_multi_table": has_join_hint,
        "needs_subquery": has_subquery_hint,
        "token_len": len(tokens),
        "complexity_score": complexity,
        "complexity_level": level,
    }


def dynamic_weights(profile: Dict) -> Dict:
    """
    Dynamic weight policy for final score components.
    Returns weights for: nl/sql/ast/schema/token_cost.
    """
    level = profile.get("complexity_level", "medium")

    if level == "low":
        return {
            "nl": 0.50,
            "sql": 0.35,
            "ast": 0.15,
        }
    if level == "high":
        return {
            "nl": 0.30,
            "sql": 0.35,
            "ast": 0.35,
        }
    return {
        "nl": 0.40,
        "sql": 0.35,
        "ast": 0.25,
    }


def source_ratio(profile: Dict) -> Dict:
    """
    Dynamic ratio for question-centric vs schema-centric pools.
    """
    level = profile.get("complexity_level", "medium")
    if level == "low":
        return {"question_centric": 0.7, "schema_centric": 0.3}
    if level == "high":
        return {"question_centric": 0.4, "schema_centric": 0.6}
    return {"question_centric": 0.55, "schema_centric": 0.45}


def _contains_any(text: str, keywords) -> bool:
    return any(k in text for k in keywords)
