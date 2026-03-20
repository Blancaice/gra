"""
Schema 检索与路径拓展模块
"""

from typing import List, Dict, Set
from collections import deque


def retrieve_relevant_schema(
    question: str,
    vectorizer,
    indexer,
    schema_graph: Dict,
    schema: Dict,
    top_k: int = 5
) -> Dict:
    """
    检索相关 Schema 并进行关系拓展

    Args:
        question: 用户问题
        vectorizer: SchemaVectorizer 实例
        indexer: SchemaIndexer 实例
        schema_graph: build_schema_graph 返回的图结构
        schema: 完整的 Schema 字典
        top_k: 检索的列数量

    Returns:
        裁剪后的 Schema 子集
    """
    # 1. 向量化问题
    query_vector = vectorizer.encode_question(question)

    # 2. 检索相关列
    relevant_columns = indexer.search(query_vector, top_k)

    # 3. 提取相关表
    relevant_tables = set()
    for col in relevant_columns:
        relevant_tables.add(col['table'])

    # 4. 路径拓展：添加通过外键关联的表
    expanded_tables = expand_tables(relevant_tables, schema_graph)

    # 5. 构建裁剪后的 Schema
    pruned_schema = build_pruned_schema(expanded_tables, schema)

    return pruned_schema


def expand_tables(seed_tables: Set[str], schema_graph: Dict, max_hops: int = 1) -> Set[str]:
    """
    通过外键关系拓展表集合

    Args:
        seed_tables: 初始表集合
        schema_graph: 表关系图
        max_hops: 最大跳数

    Returns:
        拓展后的表集合
    """
    expanded = set(seed_tables)
    queue = deque([(table, 0) for table in seed_tables])

    while queue:
        current_table, hops = queue.popleft()

        if hops >= max_hops:
            continue

        # 添加相邻表
        for neighbor in schema_graph.get(current_table, []):
            if neighbor not in expanded:
                expanded.add(neighbor)
                queue.append((neighbor, hops + 1))

    return expanded


def build_pruned_schema(table_names: Set[str], full_schema: Dict) -> Dict:
    """
    构建裁剪后的 Schema

    Args:
        table_names: 需要保留的表名集合
        full_schema: 完整的 Schema

    Returns:
        裁剪后的 Schema
    """
    pruned = {
        'tables': [],
        'foreign_keys': []
    }

    # 保留相关表
    for table in full_schema['tables']:
        if table['name'] in table_names:
            pruned['tables'].append(table)

    # 保留相关外键
    for fk in full_schema['foreign_keys']:
        if fk['from_table'] in table_names and fk['to_table'] in table_names:
            pruned['foreign_keys'].append(fk)

    return pruned
