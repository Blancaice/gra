"""
Final scoring with dynamic weighting, structure match, and cost penalty.
"""

import re
from typing import Dict, List, Optional

import numpy as np

from .ast_score import compute_ast_match_score, compute_ast_score, get_sql_features
from .query_analyzer import analyze_question, dynamic_weights
from .schema_link_score import compute_schema_link_score
from .semantic_score import compute_nl_score_from_embeddings
from .sql_score import (
    compute_sql_score,
    compute_sql_score_from_embeddings,
    translate_sql_to_nl,
)


def compute_final_score(
    user_question: str,
    user_embedding,
    example: Dict,
    llm_client,
    vectorizer,
    relevant_schema: Optional[Dict] = None,
    relevant_tables: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    dynamic_enabled: bool = True,
    sql_nl_cache: Optional[Dict[str, str]] = None,
) -> float:
    """
    Compute final example score in [0, 1].
    """
    profile = analyze_question(user_question)
    weight_policy = _resolve_weight_policy(weights, profile, dynamic_enabled)

    example_q = example.get("question", "")
    example_sql = example.get("sql", "")
    example_q_emb = _to_array(example.get("question_embedding"))
    example_sql_nl_emb = _to_array(example.get("sql_nl_embedding"))
    cached_sql_nl = example.get("sql_nl")

    nl_semantic_score = compute_nl_score_from_embeddings(
        user_embedding, example_q_emb, user_question, example_q
    )
    nl_link_score = compute_schema_link_score(
        user_question=user_question,
        example_question=example_q,
        schema=relevant_schema or {},
        vectorizer=vectorizer,
    )
    # Blend semantic similarity with explicit schema-link overlap.
    nl_score = 0.5 * nl_semantic_score + 0.5 * nl_link_score

    if example_sql_nl_emb is not None:
        sql_score = compute_sql_score_from_embeddings(example_sql_nl_emb, user_embedding)
    else:
        # Cache SQL->NL translation by SQL text to reduce online LLM calls.
        if not cached_sql_nl and sql_nl_cache is not None and example_sql in sql_nl_cache:
            cached_sql_nl = sql_nl_cache[example_sql]
        if not cached_sql_nl and example_sql:
            cached_sql_nl = translate_sql_to_nl(example_sql, llm_client)
            if sql_nl_cache is not None:
                sql_nl_cache[example_sql] = cached_sql_nl
        sql_score = compute_sql_score(
            example_sql, user_question, llm_client, vectorizer, cached_sql_nl=cached_sql_nl
        )

    ast_complexity_score = compute_ast_score(example_sql)
    ast_match_score = compute_ast_match_score(user_question, example_sql)
    # AST分融合：复杂度保留，结构对齐权重更高。
    ast_score = 0.4 * ast_complexity_score + 0.6 * ast_match_score
    sql_features = get_sql_features(example_sql)
    structure_match = structure_match_score(profile, sql_features)
    schema_overlap = compute_schema_overlap(example_sql, example, relevant_tables)
    token_cost = token_cost_penalty(example)

    final_score = (
        weight_policy["nl"] * nl_score
        + weight_policy["sql"] * sql_score
        + weight_policy["ast"] * ast_score
        + weight_policy["schema"] * ((structure_match + schema_overlap) / 2.0)
        - weight_policy["token_cost"] * token_cost
    )
    return max(0.0, min(1.0, float(final_score)))


def score_all_examples(
    user_question: str,
    examples: List[Dict],
    llm_client,
    vectorizer,
    weights: List[float] = None,
    relevant_schema: Optional[Dict] = None,
    relevant_tables: Optional[List[str]] = None,
    dynamic_enabled: bool = True,
) -> List[Dict]:
    """
    Score all examples and return descending rank.
    """
    user_embedding = vectorizer.encode_question(user_question)
    sql_nl_cache = {}

    scored_examples = []
    for example in examples:
        score = compute_final_score(
            user_question=user_question,
            user_embedding=user_embedding,
            example=example,
            llm_client=llm_client,
            vectorizer=vectorizer,
            relevant_schema=relevant_schema,
            relevant_tables=relevant_tables,
            weights=weights,
            dynamic_enabled=dynamic_enabled,
            sql_nl_cache=sql_nl_cache,
        )
        item = example.copy()
        item["score"] = score
        item["nl_link_score"] = compute_schema_link_score(
            user_question=user_question,
            example_question=example.get("question", ""),
            schema=relevant_schema or {},
        )
        item["ast_match_score"] = compute_ast_match_score(
            user_question=user_question,
            example_sql=example.get("sql", ""),
        )
        scored_examples.append(item)

    scored_examples.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return scored_examples


def structure_match_score(profile: Dict, sql_features: Dict) -> float:
    """
    Match user intent structure with SQL structure.
    """
    targets = {
        "has_aggregation": 1.0 if profile.get("has_aggregation") else 0.0,
        "has_order": 1.0 if profile.get("has_order") else 0.0,
        "needs_multi_table": 1.0 if profile.get("needs_multi_table") else 0.0,
        "needs_subquery": 1.0 if profile.get("needs_subquery") else 0.0,
    }
    actuals = {
        "has_aggregation": 1.0 if sql_features.get("num_aggregations", 0) > 0 else 0.0,
        "has_order": 1.0 if sql_features.get("has_order_by") else 0.0,
        "needs_multi_table": 1.0 if sql_features.get("num_joins", 0) > 0 else 0.0,
        "needs_subquery": 1.0 if sql_features.get("num_subqueries", 0) > 0 else 0.0,
    }
    diffs = [abs(targets[k] - actuals[k]) for k in targets]
    return 1.0 - (sum(diffs) / len(diffs))


def compute_schema_overlap(
    example_sql: str,
    example: Dict,
    relevant_tables: Optional[List[str]],
) -> float:
    """
    Reward examples whose SQL tables overlap with pruned schema tables.
    """
    if not relevant_tables:
        return 0.5

    example_tables = set(example.get("tables", []))
    if not example_tables:
        example_tables = extract_tables_from_sql(example_sql)

    if not example_tables:
        return 0.0

    target = set(relevant_tables)
    inter = len(example_tables.intersection(target))
    union = len(example_tables.union(target))
    return inter / union if union > 0 else 0.0


def token_cost_penalty(example: Dict) -> float:
    """
    Normalize token usage cost to [0, 1].
    """
    token_count = int(example.get("token_count", 0))
    if token_count <= 0:
        question = example.get("question", "")
        sql = example.get("sql", "")
        token_count = max(1, len(question) // 4 + len(sql) // 4 + 20)
    return min(1.0, token_count / 800.0)


def extract_tables_from_sql(sql_text: str) -> set:
    """
    Best-effort table extractor from FROM/JOIN clauses.
    """
    if not sql_text:
        return set()
    matches = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w\.]*)", sql_text, flags=re.IGNORECASE)
    cleaned = []
    for table in matches:
        # remove aliases in dotted names like db.table -> table
        cleaned.append(table.split(".")[-1])
    return set(cleaned)


def _resolve_weight_policy(
    weights: Optional[List[float]], profile: Dict, dynamic_enabled: bool
) -> Dict:
    if not dynamic_enabled:
        static = weights or [0.4, 0.4, 0.2]
        return {
            "nl": static[0],
            "sql": static[1],
            "ast": static[2],
            "schema": 0.0,
            "token_cost": 0.0,
        }
    return dynamic_weights(profile)


def _to_array(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.array(value, dtype=float)
    return None
