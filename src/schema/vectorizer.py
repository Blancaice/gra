"""
Schema 向量化模块
将表名、列名转换为向量表示
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple


class SchemaVectorizer:
    """Schema 向量化器"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化向量化模型

        Args:
            model_name: 预训练模型名称
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def vectorize_schema(self, schema: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """
        将 Schema 转换为向量

        Args:
            schema: extract_schema 返回的 Schema 字典

        Returns:
            (embeddings, metadata)
            - embeddings: shape (N, dimension) 的向量矩阵
            - metadata: 每个向量对应的元信息列表
        """
        texts = []
        metadata = []

        for table in schema['tables']:
            table_name = table['name']

            # 为每个列生成描述文本
            for col in table['columns']:
                col_name = col['name']
                col_type = col['type']

                # 构造描述文本
                text = f"table: {table_name}, column: {col_name}, type: {col_type}"
                texts.append(text)

                metadata.append({
                    'type': 'column',
                    'table': table_name,
                    'column': col_name,
                    'data_type': col_type
                })

        # 批量向量化
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        return embeddings, metadata

    def encode_question(self, question: str) -> np.ndarray:
        """
        将用户问题转换为向量

        Args:
            question: 用户的自然语言问题

        Returns:
            向量表示
        """
        return self.model.encode(question, convert_to_numpy=True)
