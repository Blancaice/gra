"""
NL-to-Schema linking score for question/example alignment.
使用预计算 Embedding 匹配，无需 LLM 推理。
"""

import re
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_schema_link_score(
    user_question: str,
    example_question: str,
    schema: Dict,
    vectorizer=None,
) -> float:
    """
    Compute Jaccard overlap between schema-linked elements of user/example NL.
    If vectorizer is provided, use embedding matching; otherwise fall back to token matching.
    """
    if not schema or not schema.get("tables"):
        return 0.5

    if vectorizer is not None:
        user_links = extract_linked_schema_elements_emb(user_question, schema, vectorizer)
        example_links = extract_linked_schema_elements_emb(example_question, schema, vectorizer)
    else:
        user_links = extract_linked_schema_elements_token(user_question, schema)
        example_links = extract_linked_schema_elements_token(example_question, schema)

    if not user_links and not example_links:
        return 0.5
    if not user_links or not example_links:
        return 0.0

    inter = len(user_links & example_links)
    union = len(user_links | example_links)
    return inter / union if union > 0 else 0.0


# ---------- Embedding-based schema linking ----------

_schema_emb_cache: Dict[str, np.ndarray] = {}


def _get_schema_elements(schema: Dict) -> List[Tuple[str, str]]:
    """提取 schema 元素列表: [(id, text), ...]"""
    elements = []
    for table in schema.get("tables", []):
        tname = table.get("name", "")
        elements.append((f"table:{tname.lower()}", tname.replace("_", " ")))
        for col in table.get("columns", []):
            cname = col.get("name", "")
            desc = col.get("description", cname)
            text = f"{tname} {cname}".replace("_", " ")
            if desc and desc != cname:
                text = f"{text} {desc}"
            elements.append((f"column:{tname.lower()}.{cname.lower()}", text))
    return elements


def _encode_schema_elements(
    elements: List[Tuple[str, str]], vectorizer
) -> np.ndarray:
    """对 schema 元素批量编码，带缓存"""
    cache_key = "|".join(e[0] for e in elements)
    if cache_key in _schema_emb_cache:
        return _schema_emb_cache[cache_key]

    texts = [e[1] for e in elements]
    embs = vectorizer.model.encode(texts, convert_to_numpy=True)
    _schema_emb_cache[cache_key] = embs
    return embs


def extract_linked_schema_elements_emb(
    text: str, schema: Dict, vectorizer, threshold: float = 0.45,
) -> Set[str]:
    """
    Embedding-based schema linking:
    对问题编码，与 schema 元素逐一计算余弦相似度，超过阈值即链接。
    """
    if not text:
        return set()

    elements = _get_schema_elements(schema)
    if not elements:
        return set()

    q_emb = vectorizer.encode_question(text).reshape(1, -1)
    s_embs = _encode_schema_elements(elements, vectorizer)
    sims = cosine_similarity(q_emb, s_embs)[0]

    links = set()
    for i, sim in enumerate(sims):
        if sim >= threshold:
            links.add(elements[i][0])
    return links


# ---------- Token-based fallback ----------

def extract_linked_schema_elements_token(text: str, schema: Dict) -> Set[str]:
    """Token overlap fallback (original logic)."""
    tokens = _tokenize(text)
    if not tokens:
        return set()

    links: Set[str] = set()
    for table in schema.get("tables", []):
        tname = table.get("name", "")
        if _has_overlap(tokens, _tokenize(tname)):
            links.add(f"table:{tname.lower()}")
        for col in table.get("columns", []):
            cname = col.get("name", "")
            if _has_overlap(tokens, _tokenize(cname)):
                links.add(f"column:{tname.lower()}.{cname.lower()}")
    return links


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    raw = re.findall(r"[A-Za-z0-9_]+", text.lower())
    out = set()
    for token in raw:
        out.add(token)
        if "_" in token:
            for part in token.split("_"):
                if part:
                    out.add(part)
    return out


def _has_overlap(a: Set[str], b: Set[str]) -> bool:
    return bool(a and b and a & b)
