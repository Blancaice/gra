"""
SQL alignment scoring. Prefer cached SQL->NL artifacts from example store.
"""

from typing import Optional

from sklearn.metrics.pairwise import cosine_similarity


def compute_sql_score(
    example_sql: str,
    user_question: str,
    llm_client,
    vectorizer,
    cached_sql_nl: Optional[str] = None,
) -> float:
    """
    Score SQL alignment in [0, 1] by comparing user question with SQL-derived NL.
    """
    nl_from_sql = cached_sql_nl or translate_sql_to_nl(example_sql, llm_client)
    if not nl_from_sql:
        return 0.0

    vec1 = vectorizer.encode_question(nl_from_sql).reshape(1, -1)
    vec2 = vectorizer.encode_question(user_question).reshape(1, -1)
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return max(0.0, min(1.0, float(similarity)))


def compute_sql_score_from_embeddings(sql_nl_embedding, user_embedding) -> float:
    """
    Fast path: compute SQL alignment score from cached embeddings.
    """
    if sql_nl_embedding is None or user_embedding is None:
        return 0.0

    vec1 = sql_nl_embedding.reshape(1, -1)
    vec2 = user_embedding.reshape(1, -1)
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return max(0.0, min(1.0, float(similarity)))


def translate_sql_to_nl(sql: str, llm_client) -> str:
    """
    Translate SQL to a plain NL question.
    """
    prompt = f"""Translate the following SQL query into a natural language question:

SQL: {sql}

Return only the question, without any explanation or additional text.
"""
    try:
        response = llm_client.generate(prompt)
        return response.strip()
    except Exception as exc:
        print(f"Error translating SQL to NL: {exc}")
        return ""
