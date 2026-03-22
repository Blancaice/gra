"""
结构感知示例检索模块。

核心创新：
- 通过 Embedding + sklearn 分类器预测 NL 问题的 SQL 结构特征（零 LLM 成本）
- 结构向量加权相似度 + 语义相似度双通道检索
- 离线预计算 train set 结构特征，在线只需分类器推理（毫秒级）
"""

import hashlib
import json
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 结构特征名称（10 维二值向量）
FEATURE_NAMES = [
    "has_aggregation", "has_join", "has_where", "has_subquery",
    "has_groupby", "has_orderby", "has_having", "has_limit",
    "has_set_op", "has_distinct",
]

# 各特征在检索时的重要性权重（基于对 SQL 结构区分度的贡献）
FEATURE_WEIGHTS = np.array([
    2.0,  # has_aggregation — 区分聚合 vs 非聚合
    3.0,  # has_join — 单表 vs 多表，影响最大
    1.0,  # has_where — 太常见，区分度低
    3.0,  # has_subquery — 嵌套查询，结构差异大
    2.0,  # has_groupby — 分组聚合
    1.5,  # has_orderby — 排序
    2.5,  # has_having — 聚合过滤，较少见
    1.5,  # has_limit — 通常配合 orderby
    3.0,  # has_set_op — 集合操作，结构完全不同
    1.0,  # has_distinct — 区分度低
], dtype=np.float32)

# Spider aggregation IDs
AGG_MAP = {0: "none", 1: "max", 2: "min", 3: "count", 4: "sum", 5: "avg"}


# ============================================================
# 1. SQL 结构特征提取（从 Spider 解析字段）
# ============================================================

def extract_structure_features(sql_parsed: Dict) -> np.ndarray:
    """
    从 Spider 的 sql 解析字段提取 10 维二值结构向量。
    """
    f = _extract_features_dict(sql_parsed)
    return np.array([
        int(f["has_aggregation"]),
        int(f["has_join"]),
        int(f["has_where"]),
        int(f["has_subquery"]),
        int(f["has_groupby"]),
        int(f["has_orderby"]),
        int(f["has_having"]),
        int(f["has_limit"]),
        int(f["has_set_op"]),
        int(f["has_distinct"]),
    ], dtype=np.float32)


def _extract_features_dict(sql_parsed: Dict) -> Dict:
    """提取结构特征字典"""
    f = {k: False for k in FEATURE_NAMES}
    if not sql_parsed:
        return f

    # SELECT
    select_clause = sql_parsed.get("select", [False, []])
    f["has_distinct"] = select_clause[0] if isinstance(select_clause[0], bool) else False
    select_items = select_clause[1] if len(select_clause) > 1 else []
    for item in select_items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            agg_id = item[0]
            if isinstance(agg_id, int) and agg_id > 0:
                f["has_aggregation"] = True

    # FROM
    from_clause = sql_parsed.get("from", {})
    table_units = from_clause.get("table_units", [])
    f["has_join"] = len(table_units) > 1
    for tu in table_units:
        if isinstance(tu, (list, tuple)) and len(tu) >= 2 and tu[0] == "sql":
            f["has_subquery"] = True

    # WHERE
    where_clause = sql_parsed.get("where", [])
    if where_clause:
        f["has_where"] = True
        _check_subquery_in_conds(where_clause, f)

    # GROUP BY / ORDER BY / HAVING / LIMIT
    f["has_groupby"] = bool(sql_parsed.get("groupBy", []))
    f["has_orderby"] = bool(sql_parsed.get("orderBy", []))
    having = sql_parsed.get("having", [])
    if having:
        f["has_having"] = True
        _check_subquery_in_conds(having, f)
    f["has_limit"] = sql_parsed.get("limit") is not None

    # Set operations
    for op in ("intersect", "union", "except"):
        if sql_parsed.get(op):
            f["has_set_op"] = True
            f["has_subquery"] = True
            break

    return f


def _check_subquery_in_conds(cond_list, f):
    for item in cond_list:
        if isinstance(item, (list, tuple)):
            for sub in item:
                if isinstance(sub, dict) and "select" in sub:
                    f["has_subquery"] = True
                    return


# ============================================================
# 2. 结构感知检索器
# ============================================================

class StructureAwareRetriever:
    """
    结构感知示例检索器。

    离线阶段：
        - 对 train set 提取语义 embedding + 结构特征向量
        - 训练 sklearn 分类器（embedding → 结构特征）
        - 构建 Faiss 语义索引

    在线阶段：
        - NL 问题 → embedding → 分类器预测结构向量 [零 LLM 成本]
        - 结构加权相似度 + 语义相似度融合排序
        - 返回 top-N 示例
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.train_data: List[Dict] = []
        self.train_embeddings: Optional[np.ndarray] = None
        self.train_structure_vectors: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.scaler: Optional[StandardScaler] = None
        self.classifiers: Dict[str, object] = {}  # per-feature classifiers
        self._ready = False

    def build(
        self,
        train_data: List[Dict],
        vectorizer,
        force_rebuild: bool = False,
    ):
        """
        离线构建：提取特征、训练分类器、构建索引。
        结果自动缓存到磁盘。
        """
        self.train_data = train_data
        n = len(train_data)
        print(f"[StructureAwareRetriever] Building with {n} training examples...")

        # --- 1. 语义 Embeddings ---
        emb_path = os.path.join(self.cache_dir, "struct_retriever_embeddings.npy")
        if os.path.exists(emb_path) and not force_rebuild:
            print("  Loading cached embeddings...")
            self.train_embeddings = np.load(emb_path)
            if len(self.train_embeddings) != n:
                force_rebuild = True

        if not os.path.exists(emb_path) or force_rebuild:
            print("  Encoding training questions...")
            questions = [item.get("question", "") for item in train_data]
            self.train_embeddings = vectorizer.model.encode(
                questions, convert_to_numpy=True, show_progress_bar=True, batch_size=64,
            )
            np.save(emb_path, self.train_embeddings)
            print(f"  Saved embeddings to {emb_path}")

        # --- 2. 结构特征向量 ---
        struct_path = os.path.join(self.cache_dir, "struct_retriever_features.npy")
        if os.path.exists(struct_path) and not force_rebuild:
            print("  Loading cached structure vectors...")
            self.train_structure_vectors = np.load(struct_path)
            if len(self.train_structure_vectors) != n:
                force_rebuild = True

        if not os.path.exists(struct_path) or force_rebuild:
            print("  Extracting structure features from SQL...")
            vecs = []
            for item in train_data:
                sql_parsed = item.get("sql", {})
                vec = extract_structure_features(sql_parsed)
                vecs.append(vec)
            self.train_structure_vectors = np.array(vecs, dtype=np.float32)
            np.save(struct_path, self.train_structure_vectors)
            print(f"  Saved structure vectors to {struct_path}")

        # --- 3. 训练分类器 ---
        clf_path = os.path.join(self.cache_dir, "struct_classifier.pkl")
        if os.path.exists(clf_path) and not force_rebuild:
            print("  Loading cached classifier...")
            with open(clf_path, "rb") as f:
                saved = pickle.load(f)
            self.scaler = saved["scaler"]
            self.classifiers = saved["classifiers"]
        else:
            print("  Training structure classifiers...")
            self._train_classifiers()
            with open(clf_path, "wb") as f:
                pickle.dump({
                    "scaler": self.scaler,
                    "classifiers": self.classifiers,
                }, f)
            print(f"  Saved classifier to {clf_path}")

        # --- 4. Faiss 语义索引 ---
        print("  Building Faiss semantic index...")
        dim = self.train_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # 内积 = 余弦相似度（已归一化）
        # 归一化 embeddings
        norms = np.linalg.norm(self.train_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = self.train_embeddings / norms
        self.faiss_index.add(normalized.astype(np.float32))

        self._ready = True
        print(f"[StructureAwareRetriever] Ready. {n} examples indexed.")

    def _train_classifiers(self):
        """训练每个结构特征的二分类器"""
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(self.train_embeddings)
        Y = self.train_structure_vectors

        self.classifiers = {}
        for i, fname in enumerate(FEATURE_NAMES):
            y = Y[:, i].astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                # 全正或全负，不需要分类器
                self.classifiers[fname] = ("constant", int(y[0]))
                continue

            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X, y)
            self.classifiers[fname] = ("model", clf)

    def predict_structure(self, question_embedding: np.ndarray) -> np.ndarray:
        """
        从 NL embedding 预测结构特征向量。

        Args:
            question_embedding: shape (dim,) 的问题 embedding

        Returns:
            shape (10,) 的二值预测向量
        """
        X = self.scaler.transform(question_embedding.reshape(1, -1))
        pred = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

        for i, fname in enumerate(FEATURE_NAMES):
            entry = self.classifiers[fname]
            if entry[0] == "constant":
                pred[i] = entry[1]
            else:
                pred[i] = entry[1].predict(X)[0]

        return pred

    def predict_structure_proba(self, question_embedding: np.ndarray) -> np.ndarray:
        """预测结构特征的概率（用于软匹配）"""
        X = self.scaler.transform(question_embedding.reshape(1, -1))
        proba = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

        for i, fname in enumerate(FEATURE_NAMES):
            entry = self.classifiers[fname]
            if entry[0] == "constant":
                proba[i] = float(entry[1])
            else:
                p = entry[1].predict_proba(X)[0]
                # predict_proba 返回 [P(0), P(1)]
                proba[i] = p[1] if len(p) > 1 else p[0]

        return proba

    def retrieve(
        self,
        question: str,
        vectorizer,
        top_n: int = 15,
        alpha: float = 0.4,
        semantic_candidates: int = 100,
    ) -> List[Dict]:
        """
        结构感知检索。

        Args:
            question: 用户 NL 问题
            vectorizer: SchemaVectorizer（用于 encoding）
            top_n: 返回的示例数量
            alpha: 结构相似度权重（1-alpha 为语义权重）
            semantic_candidates: 语义粗召回数量

        Returns:
            排序后的示例列表，每个包含 question, sql, scores 等
        """
        if not self._ready:
            raise RuntimeError("Retriever not built. Call build() first.")

        # 1. 编码问题
        q_emb = vectorizer.model.encode(question, convert_to_numpy=True)

        # 2. 预测结构特征
        pred_structure = self.predict_structure(q_emb)
        pred_proba = self.predict_structure_proba(q_emb)

        # 3. 语义粗召回（Faiss top-K）
        q_norm = q_emb / max(np.linalg.norm(q_emb), 1e-8)
        distances, indices = self.faiss_index.search(
            q_norm.reshape(1, -1).astype(np.float32),
            min(semantic_candidates, len(self.train_data)),
        )

        # 4. 对候选集计算结构相似度 + 语义相似度，融合排序
        candidates = []
        for rank, (idx, sem_score) in enumerate(zip(indices[0], distances[0])):
            if idx < 0:
                continue

            train_struct = self.train_structure_vectors[idx]

            # 结构相似度：加权 Jaccard
            struct_sim = _weighted_structure_similarity(
                pred_proba, train_struct, FEATURE_WEIGHTS
            )

            # 语义相似度：Faiss 内积（已归一化 = 余弦相似度）
            semantic_sim = float(sem_score)

            # 融合分数
            combined_score = alpha * struct_sim + (1 - alpha) * semantic_sim

            item = self.train_data[idx]
            candidates.append({
                "question": item.get("question", ""),
                "sql": item.get("query", ""),
                "db_id": item.get("db_id", ""),
                "source": "structure_aware",
                "token_count": len(item.get("question", "")) // 4 + len(item.get("query", "")) // 4 + 20,
                "structure_score": float(struct_sim),
                "semantic_score": float(semantic_sim),
                "combined_score": float(combined_score),
                "pred_structure": pred_structure.tolist(),
                "train_structure": train_struct.tolist(),
                "reasoning": "",
            })

        # 按融合分数排序
        candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        return candidates[:top_n]


# ============================================================
# 3. 结构相似度计算
# ============================================================

def _weighted_structure_similarity(
    pred_proba: np.ndarray,
    target_vec: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    加权结构相似度。使用概率软匹配而非硬二值匹配。

    对每个特征 i：
    - 如果 target[i] == 1: 贡献 = pred_proba[i] * weight[i]
    - 如果 target[i] == 0: 贡献 = (1 - pred_proba[i]) * weight[i]

    最终归一化到 [0, 1]。
    """
    agreement = np.where(
        target_vec > 0.5,
        pred_proba,           # target=1, 越接近 1 越好
        1.0 - pred_proba,     # target=0, 越接近 0 越好
    )
    weighted_sum = np.sum(agreement * weights)
    max_sum = np.sum(weights)
    return weighted_sum / max_sum if max_sum > 0 else 0.0


def compute_structure_similarity_batch(
    pred_vec: np.ndarray,
    target_matrix: np.ndarray,
    weights: np.ndarray = FEATURE_WEIGHTS,
) -> np.ndarray:
    """
    批量计算结构相似度（向量化版本，用于大规模检索）。

    Args:
        pred_vec: shape (10,) 预测向量
        target_matrix: shape (N, 10) 训练集结构矩阵
        weights: shape (10,) 特征权重

    Returns:
        shape (N,) 相似度数组
    """
    agreement = np.where(
        target_matrix > 0.5,
        pred_vec,
        1.0 - pred_vec,
    )
    weighted = agreement * weights
    scores = weighted.sum(axis=1) / weights.sum()
    return scores
