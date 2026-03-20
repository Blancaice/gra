"""
AST-related scoring:
1) SQL structural complexity score
2) Question-to-SQL structural alignment (AST node-set Jaccard)
3) SQL-to-SQL AST similarity (normalized node multiset Jaccard)
"""

from collections import Counter
from typing import Dict, List, Set, Tuple

import sqlparse
from sqlparse.sql import (
    Function,
    Identifier,
    IdentifierList,
    Parenthesis,
    Where,
)
from sqlparse.tokens import Keyword, DML


# ============================================================
# 1. AST 解析与标准化
# ============================================================

def _parse_to_ast_nodes(sql_text: str) -> Counter:
    """
    将 SQL 解析为标准化 AST 节点多重集合。
    标准化：大写关键字、移除别名、提取结构节点类型。
    """
    nodes: Counter = Counter()
    parsed = sqlparse.parse((sql_text or "").strip())
    if not parsed:
        return nodes

    for stmt in parsed:
        _walk_ast(stmt, nodes)
    return nodes


def _walk_ast(token, nodes: Counter):
    """递归遍历 sqlparse token 树，提取标准化节点"""
    if token.ttype is DML:
        nodes[f"DML:{token.normalized.upper()}"] += 1
        return

    if token.ttype is Keyword:
        kw = token.normalized.upper()
        if kw in (
            "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "OUTER JOIN",
            "CROSS JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN",
            "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "DISTINCT",
            "UNION", "UNION ALL", "INTERSECT", "EXCEPT",
            "EXISTS", "IN", "NOT IN", "BETWEEN", "LIKE", "NOT",
            "AND", "OR", "ON", "WHERE", "AS",
        ):
            nodes[f"KW:{kw}"] += 1
        return

    if isinstance(token, Function):
        fname = token.get_name()
        if fname:
            nodes[f"FUNC:{fname.upper()}"] += 1

    if isinstance(token, Parenthesis):
        inner = str(token).upper()
        if "SELECT" in inner:
            nodes["SUBQUERY"] += 1

    if isinstance(token, Where):
        nodes["WHERE_CLAUSE"] += 1

    if hasattr(token, "tokens"):
        for child in token.tokens:
            _walk_ast(child, nodes)


def _ast_node_jaccard(nodes_a: Counter, nodes_b: Counter) -> float:
    """计算两个 AST 节点多重集合的 Jaccard 相似度"""
    if not nodes_a and not nodes_b:
        return 1.0
    if not nodes_a or not nodes_b:
        return 0.0

    all_keys = set(nodes_a.keys()) | set(nodes_b.keys())
    intersection = sum(min(nodes_a[k], nodes_b[k]) for k in all_keys)
    union = sum(max(nodes_a[k], nodes_b[k]) for k in all_keys)
    return intersection / union if union > 0 else 0.0


# ============================================================
# 2. 公开接口
# ============================================================

def compute_ast_score(example_sql: str, user_question: str = None) -> float:
    """
    复杂度分：衡量 SQL 本身结构复杂程度 [0, 1]。
    """
    features = get_sql_features(example_sql)
    return calculate_complexity_score(features)


def compute_ast_match_score(user_question: str, example_sql: str) -> float:
    """
    结构对齐分：用 AST 节点集合 Jaccard 衡量问题目标结构与 SQL 结构的对齐度。
    从问题推断目标 AST 节点，与 SQL 实际 AST 节点做 Jaccard。
    """
    target_nodes = _infer_target_ast_nodes(user_question)
    actual_nodes = _parse_to_ast_nodes(example_sql)

    if not target_nodes and not actual_nodes:
        return 0.5
    return _ast_node_jaccard(target_nodes, actual_nodes)


def compute_sql_ast_similarity(sql_a: str, sql_b: str) -> float:
    """两条 SQL 之间的 AST 节点集合 Jaccard 相似度"""
    nodes_a = _parse_to_ast_nodes(sql_a)
    nodes_b = _parse_to_ast_nodes(sql_b)
    return _ast_node_jaccard(nodes_a, nodes_b)


# ============================================================
# 3. 从问题推断目标 AST 节点
# ============================================================

def _infer_target_ast_nodes(question: str) -> Counter:
    """从问题文本推断期望的 AST 结构节点"""
    q = (question or "").lower()
    nodes: Counter = Counter()

    nodes["DML:SELECT"] = 1

    if _contains_any(q, ("count", "how many", "number of", "总数", "数量")):
        nodes["FUNC:COUNT"] += 1
    if _contains_any(q, ("sum", "total", "总和", "合计")):
        nodes["FUNC:SUM"] += 1
    if _contains_any(q, ("average", "avg", "mean", "平均")):
        nodes["FUNC:AVG"] += 1
    if _contains_any(q, ("maximum", "max", "highest", "most", "最大", "最高")):
        nodes["FUNC:MAX"] += 1
    if _contains_any(q, ("minimum", "min", "lowest", "least", "最小", "最低")):
        nodes["FUNC:MIN"] += 1

    # 聚合 -> GROUP BY
    if any(nodes.get(f"FUNC:{f}", 0) > 0 for f in ("COUNT", "SUM", "AVG", "MAX", "MIN")):
        if _contains_any(q, ("each", "every", "per", "group", "每个", "各")):
            nodes["KW:GROUP BY"] += 1

    if _contains_any(q, ("order", "sort", "rank", "排序", "排名")):
        nodes["KW:ORDER BY"] += 1
    if _contains_any(q, ("top", "first", "last", "前", "limit")):
        nodes["KW:LIMIT"] += 1
    if _contains_any(q, ("join", "with", "whose", "across", "对应", "以及")):
        nodes["KW:JOIN"] += 1
    if _contains_any(q, ("more than", "less than", "at least", "at most", "高于", "低于")):
        nodes["SUBQUERY"] += 1
    if _contains_any(q, ("union", "intersect", "except", "并集", "交集")):
        nodes["KW:UNION"] += 1

    return nodes


# ============================================================
# 4. SQL 特征提取与复杂度评分（保留原有逻辑）
# ============================================================

def get_sql_features(sql_text: str) -> Dict:
    """Parse SQL into a normalized feature dict."""
    parsed = sqlparse.parse(sql_text or "")
    if not parsed:
        return {
            "num_joins": 0, "num_aggregations": 0,
            "num_subqueries": 0, "num_where_conditions": 0,
            "has_group_by": False, "has_order_by": False, "has_limit": False,
        }

    sql_upper = str(parsed[0]).upper()
    agg_count = sum(sql_upper.count(f + "(") for f in ("COUNT", "SUM", "AVG", "MAX", "MIN"))
    return {
        "num_joins": sql_upper.count("JOIN"),
        "num_aggregations": agg_count,
        "num_subqueries": max(0, sql_upper.count("SELECT") - 1),
        "num_where_conditions": (
            sql_upper.count(" AND ") + sql_upper.count(" OR ")
            + (1 if "WHERE" in sql_upper else 0)
        ),
        "has_group_by": "GROUP BY" in sql_upper,
        "has_order_by": "ORDER BY" in sql_upper,
        "has_limit": "LIMIT" in sql_upper,
    }


def calculate_complexity_score(features: Dict) -> float:
    """Heuristic complexity score in [0, 1]."""
    def sigmoid(v, scale=2.0):
        return 1.0 / (1.0 + pow(2.718, -v / scale))

    w = {
        "num_joins": 0.3, "num_aggregations": 0.2,
        "num_subqueries": 0.25, "num_where_conditions": 0.1,
        "has_group_by": 0.05, "has_order_by": 0.05, "has_limit": 0.05,
    }
    score = 0.0
    score += w["num_joins"] * sigmoid(features["num_joins"])
    score += w["num_aggregations"] * sigmoid(features["num_aggregations"])
    score += w["num_subqueries"] * sigmoid(features["num_subqueries"])
    score += w["num_where_conditions"] * sigmoid(features["num_where_conditions"], 3.0)
    score += w["has_group_by"] * (1.0 if features["has_group_by"] else 0.0)
    score += w["has_order_by"] * (1.0 if features["has_order_by"] else 0.0)
    score += w["has_limit"] * (1.0 if features["has_limit"] else 0.0)
    return min(1.0, score)


def _contains_any(text: str, keywords) -> bool:
    return any(k in text for k in keywords)


# 兼容旧接口
def infer_target_structure(question: str) -> Dict[str, bool]:
    """从问题文本推断目标结构（兼容旧调用）"""
    q = (question or "").lower()
    return {
        "need_aggregation": _contains_any(
            q, ("count", "sum", "avg", "average", "max", "min", "总数", "平均", "最大", "最小")
        ),
        "need_order": _contains_any(
            q, ("order by", "sort", "highest", "lowest", "排名", "排序")
        ),
        "need_limit": _contains_any(
            q, ("top", "first", "last", "前", "最", "lowest", "highest")
        ),
        "need_join": _contains_any(
            q, ("join", "with", "whose", "across", "每个", "以及", "对应")
        ),
        "need_subquery": _contains_any(
            q, ("more than", "less than", "at least", "at most", "than", "高于", "低于", "至少", "至多")
        ),
        "need_set_ops": _contains_any(
            q, ("union", "intersect", "except", "并集", "交集", "差集")
        ),
    }


def extract_sql_skeleton(sql_text: str) -> Dict[str, bool]:
    """从 SQL 提取骨架结构（兼容旧调用）"""
    features = get_sql_features(sql_text)
    sql_upper = (sql_text or "").upper()
    return {
        "need_aggregation": features["num_aggregations"] > 0 or features["has_group_by"],
        "need_order": features["has_order_by"],
        "need_limit": features["has_limit"],
        "need_join": features["num_joins"] > 0,
        "need_subquery": features["num_subqueries"] > 0,
        "need_set_ops": any(op in sql_upper for op in (" UNION ", " INTERSECT ", " EXCEPT ")),
    }
