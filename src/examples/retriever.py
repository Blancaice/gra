"""
训练集双向检索模块。
- Question-centric: FAISS 索引 + embedding 语义相似度（持久化到磁盘）
- Schema-centric: Query-guided Schema Path Sampling + LLM 实时生成
"""

import hashlib
import json
import os
from typing import Dict, List

import faiss
import numpy as np

from src.examples.schema_centric import (
    classify_sql_complexity,
    generate_from_schema,
)
from src.schema.indexer import SchemaIndexer

_train_cache: Dict[str, List[Dict]] = {}


def load_train_data(train_paths) -> List[Dict]:
    """加载并缓存训练数据，支持单文件路径或列表"""
    if isinstance(train_paths, str):
        train_paths = [train_paths]

    cache_key = "|".join(sorted(train_paths))
    if cache_key in _train_cache:
        return _train_cache[cache_key]

    all_data = []
    for path in train_paths:
        if not os.path.exists(path):
            print(f"  Warning: train file not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        all_data.extend(data)
        print(f"  Loaded {len(data)} examples from {path}")

    print(f"  Total training examples: {len(all_data)}")
    _train_cache[cache_key] = all_data
    return all_data


def _content_hash(items: List[Dict]) -> str:
    """根据数据内容生成稳定 hash，用于索引文件名"""
    sample = [d.get("question", "")[:50] for d in items[:10] + items[-10:]]
    sample.append(str(len(items)))
    return hashlib.md5(json.dumps(sample).encode()).hexdigest()[:12]


def _get_or_build_index(
    items: List[Dict],
    vectorizer,
    cache_dir: str = "data/cache",
) -> SchemaIndexer:
    """构建或从磁盘加载训练集 FAISS 索引"""
    os.makedirs(cache_dir, exist_ok=True)
    ch = _content_hash(items)
    index_path = os.path.join(cache_dir, f"train_faiss_{ch}.index")
    meta_path = os.path.join(cache_dir, f"train_meta_{ch}.json")

    indexer = SchemaIndexer(vectorizer.dimension)

    if os.path.exists(index_path) and os.path.exists(meta_path):
        print(f"  Loading FAISS index from {index_path}")
        indexer.load_index(index_path)
        with open(meta_path, encoding="utf-8") as f:
            indexer.metadata = json.load(f)
        return indexer

    print(f"  Building FAISS index for {len(items)} questions...")
    questions = [item.get("question", "") for item in items]
    embs = vectorizer.model.encode(
        questions, convert_to_numpy=True, show_progress_bar=True
    )
    metadata = [{"idx": i} for i in range(len(items))]
    indexer.build_index(embs, metadata)
    indexer.save_index(index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    print(f"  Saved FAISS index to {index_path}")
    return indexer


def _build_example(item: Dict, source: str) -> Dict:
    """将训练数据条目转为统一的示例格式"""
    question = item.get("question", "")
    sql = item.get("query", "")
    token_count = len(question) // 4 + len(sql) // 4 + 20
    return {
        "question": question,
        "sql": sql,
        "reasoning": "",
        "source": source,
        "token_count": token_count,
        "complexity_tag": classify_sql_complexity(sql),
    }


def retrieve_question_centric(
    question: str,
    vectorizer,
    train_data: List[Dict],
    db_id: str,
    top_n: int = 6,
) -> List[Dict]:
    """
    Question-centric 检索：FAISS 索引 + embedding 语义相似度。
    优先同 db_id，不足时扩展到全库。索引持久化到磁盘。
    """
    same_db = [item for item in train_data if item.get("db_id") == db_id]
    candidates = same_db if len(same_db) >= top_n else train_data

    if not candidates:
        return []

    indexer = _get_or_build_index(candidates, vectorizer)
    q_emb = vectorizer.encode_question(question)
    search_results = indexer.search(q_emb, top_k=top_n)

    results = []
    for res in search_results:
        idx = res["idx"]
        if idx < len(candidates):
            ex = _build_example(candidates[idx], "question_centric")
            ex["distance"] = res["distance"]
            results.append(ex)
    return results


def retrieve_schema_centric(
    pruned_schema: Dict,
    schema_graph: Dict,
    llm_client,
    user_question: str,
    num_paths: int = 3,
    examples_per_path: int = 2,
    db_path: str = None,
    schema_hints: Dict = None,
) -> List[Dict]:
    """
    Schema-centric: Query-guided Schema Path Sampling + LLM 实时生成。
    """
    return generate_from_schema(
        schema=pruned_schema,
        schema_graph=schema_graph,
        llm_client=llm_client,
        num_paths=num_paths,
        min_nodes=2,
        db_path=db_path,
        user_question=user_question,
        examples_per_path=examples_per_path,
        schema_hints=schema_hints,
    )
