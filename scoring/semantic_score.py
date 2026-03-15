"""
Semantic alignment scoring for NL question similarity.
"""

import re

from sklearn.metrics.pairwise import cosine_similarity


def compute_nl_score(user_question: str, example_question: str, vectorizer=None) -> float:
    """
    Compute semantic score in [0, 1].
    """
    jaccard_score = compute_jaccard_similarity(user_question, example_question)
    if not vectorizer:
        return jaccard_score

    vec1 = vectorizer.encode_question(user_question).reshape(1, -1)
    vec2 = vectorizer.encode_question(example_question).reshape(1, -1)
    cosine_score = cosine_similarity(vec1, vec2)[0][0]
    return 0.4 * jaccard_score + 0.6 * float(cosine_score)


def compute_nl_score_from_embeddings(
    user_embedding,
    example_embedding,
    user_question: str,
    example_question: str,
) -> float:
    """
    Fast path with cached embeddings.
    """
    jaccard_score = compute_jaccard_similarity(user_question, example_question)
    if user_embedding is None or example_embedding is None:
        return jaccard_score

    cosine_score = cosine_similarity(
        user_embedding.reshape(1, -1), example_embedding.reshape(1, -1)
    )[0][0]
    return 0.4 * jaccard_score + 0.6 * float(cosine_score)


def compute_jaccard_similarity(text1: str, text2: str) -> float:
    words1 = set(tokenize((text1 or "").lower()))
    words2 = set(tokenize((text2 or "").lower()))
    if not words1 or not words2:
        return 0.0
    return len(words1.intersection(words2)) / len(words1.union(words2))


def tokenize(text: str) -> list:
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()
