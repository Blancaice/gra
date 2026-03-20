"""
Dynamic example selector with source-ratio control, MMR diversity, and token budget.
"""

from typing import Dict, List

from src.scoring.query_analyzer import source_ratio


def select_top_k_examples(
    scored_examples: List[Dict],
    k: int = 3,
    token_budget: int = 2000,
    user_profile: Dict = None,
    mmr_lambda: float = 0.75,
) -> List[Dict]:
    """
    Backward-compatible entrypoint. Now performs dynamic selection.
    """
    return select_dynamic_examples(
        scored_examples=scored_examples,
        k=k,
        token_budget=token_budget,
        user_profile=user_profile,
        mmr_lambda=mmr_lambda,
    )


def select_dynamic_examples(
    scored_examples: List[Dict],
    k: int = 3,
    token_budget: int = 2000,
    user_profile: Dict = None,
    mmr_lambda: float = 0.75,
) -> List[Dict]:
    """
    Two-phase selection:
    1) source-aware candidate allocation
    2) budgeted MMR reranking with score/token utility
    """
    if not scored_examples or k <= 0:
        return []

    ratio = source_ratio(user_profile or {})
    q_target = max(0, min(k, round(k * ratio["question_centric"])))
    s_target = max(0, k - q_target)

    q_pool = [e for e in scored_examples if e.get("source") == "question_centric"]
    s_pool = [e for e in scored_examples if e.get("source") == "schema_centric"]
    unknown_pool = [
        e for e in scored_examples if e.get("source") not in {"question_centric", "schema_centric"}
    ]

    candidates = []
    candidates.extend(q_pool[: max(q_target * 3, q_target)])
    candidates.extend(s_pool[: max(s_target * 3, s_target)])

    if len(candidates) < max(3 * k, k):
        used_ids = {id(item) for item in candidates}
        for item in scored_examples:
            if id(item) not in used_ids:
                candidates.append(item)
            if len(candidates) >= max(3 * k, k):
                break

    if not candidates:
        candidates = unknown_pool[: max(3 * k, k)]

    return _select_with_mmr_budget(candidates, k, token_budget, mmr_lambda)


def _select_with_mmr_budget(
    candidates: List[Dict], k: int, token_budget: int, mmr_lambda: float
) -> List[Dict]:
    selected = []
    total_tokens = 0
    remaining = list(candidates)

    while remaining and len(selected) < k:
        best_idx = -1
        best_utility = -1e9

        for idx, ex in enumerate(remaining):
            ex_tokens = estimate_tokens(ex)
            if total_tokens + ex_tokens > token_budget:
                continue

            base_score = float(ex.get("score", 0.0))
            novelty_penalty = 0.0
            if selected:
                novelty_penalty = max(
                    compute_sql_similarity(ex.get("sql", ""), s.get("sql", "")) for s in selected
                )

            mmr_score = mmr_lambda * base_score - (1.0 - mmr_lambda) * novelty_penalty
            utility = mmr_score / max(1, ex_tokens)

            if utility > best_utility:
                best_utility = utility
                best_idx = idx

        if best_idx < 0:
            break

        picked = remaining.pop(best_idx)
        selected.append(picked)
        total_tokens += estimate_tokens(picked)

    return selected


def estimate_tokens(example: Dict) -> int:
    """
    Approximate prompt token cost of one example.
    """
    if "token_count" in example:
        return int(example["token_count"])
    question_tokens = len(example.get("question", "")) // 4
    sql_tokens = len(example.get("sql", "")) // 4
    overhead = 20
    return question_tokens + sql_tokens + overhead


def select_with_diversity(
    scored_examples: List[Dict],
    k: int = 3,
    token_budget: int = 2000,
    diversity_threshold: float = 0.3,
) -> List[Dict]:
    """
    Compatibility helper. Uses threshold to derive MMR lambda.
    """
    mmr_lambda = min(0.9, max(0.55, 1.0 - diversity_threshold))
    return select_dynamic_examples(
        scored_examples=scored_examples,
        k=k,
        token_budget=token_budget,
        user_profile=None,
        mmr_lambda=mmr_lambda,
    )


def is_diverse(example: Dict, selected: List[Dict], threshold: float) -> bool:
    if not selected:
        return True
    for sel in selected:
        if compute_sql_similarity(example.get("sql", ""), sel.get("sql", "")) > (1 - threshold):
            return False
    return True


def compute_sql_similarity(sql1: str, sql2: str) -> float:
    words1 = set((sql1 or "").lower().split())
    words2 = set((sql2 or "").lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1.intersection(words2)) / len(words1.union(words2))
