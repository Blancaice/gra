"""
Faiss 索引构建与检索模块
"""

import faiss
import numpy as np
from typing import List, Dict, Tuple


class SchemaIndexer:
    """Schema Faiss 索引器"""

    def __init__(self, dimension: int):
        """
        初始化索引

        Args:
            dimension: 向量维度
        """
        self.dimension = dimension
        self.index = None
        self.metadata = None

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        构建 Faiss 索引

        Args:
            embeddings: shape (N, dimension) 的向量矩阵
            metadata: 每个向量对应的元信息
        """
        # 使用 L2 距离的平面索引（适合小规模数据）
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        self.metadata = metadata

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        检索最相关的 Schema 元素

        Args:
            query_vector: 查询向量
            top_k: 返回前 k 个结果

        Returns:
            相关的 Schema 元素列表
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index first.")

        # 搜索
        distances, indices = self.index.search(
            query_vector.reshape(1, -1).astype('float32'),
            top_k
        )

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            result = self.metadata[idx].copy()
            result['distance'] = float(dist)
            results.append(result)

        return results

    def save_index(self, path: str):
        """保存索引到文件"""
        if self.index is not None:
            faiss.write_index(self.index, path)

    def load_index(self, path: str):
        """从文件加载索引"""
        self.index = faiss.read_index(path)
