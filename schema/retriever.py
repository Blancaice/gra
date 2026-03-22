"""
Schema 检索与路径拓展模块

改进：embedding 检索 + 名词短语直接匹配表名/列名（双通道召回）
"""

import re
from typing import List, Dict, Set
from collections import deque


def retrieve_relevant_schema(
    question: str,
    vectorizer,
    indexer,
    schema_graph: Dict,
    schema: Dict,
    top_k: int = 5,
    query_profile: Dict = None,
) -> Dict:
    """
    检索相关 Schema 并进行关系拓展。

    双通道召回：
    1. Embedding 相似度检索 top-k 列 → 对应表
    2. 名词短语直接匹配表名（零成本，高精度）
    合并后再做外键路径拓展。
    """
    # 1. 向量化问题 → embedding 检索
    query_vector = vectorizer.encode_question(question)
    relevant_columns = indexer.search(query_vector, top_k)

    relevant_tables = set()
    for col in relevant_columns:
        relevant_tables.add(col['table'])

    # 2. 名词短语直接匹配表名（不依赖 embedding，高精度补充）
    if query_profile and query_profile.get("noun_phrases"):
        noun_matched = _match_tables_by_noun_phrases(
            query_profile["noun_phrases"], schema
        )
        if noun_matched:
            relevant_tables |= noun_matched

    # 3. 小型数据库（≤6 表）直接用全部表，避免漏表
    all_table_names = {t["name"] for t in schema.get("tables", [])}
    if len(all_table_names) <= 6:
        relevant_tables = all_table_names

    # 4. 路径拓展：添加通过外键关联的表
    expanded_tables = expand_tables(relevant_tables, schema_graph)

    # 5. 构建裁剪后的 Schema
    pruned_schema = build_pruned_schema(expanded_tables, schema)

    return pruned_schema


def _match_tables_by_noun_phrases(noun_phrases: List[str], schema: Dict) -> Set[str]:
    """
    将问题中的名词短语与 schema 表名/列名做模糊匹配。

    匹配策略：
    - 表名包含名词短语（或反过来）→ 命中
    - 处理复数形式（students → student）
    - 处理下划线分隔（student_course → student, course）
    """
    matched_tables = set()

    # 构建表名查找结构
    table_names = {}  # normalized_name -> original_name
    table_name_parts = {}  # part -> original_name
    for table in schema.get("tables", []):
        orig = table["name"]
        norm = orig.lower().replace("_", " ")
        table_names[norm] = orig
        # 拆分下划线和驼峰
        for part in re.split(r"[_\s]+", orig.lower()):
            if len(part) > 2:  # 跳过太短的部分
                table_name_parts[part] = orig

    for phrase in noun_phrases:
        phrase_lower = phrase.lower().strip()
        if not phrase_lower or len(phrase_lower) <= 2:
            continue

        # 生成变体：原形 + 去尾 s（简单复数处理）
        variants = [phrase_lower]
        if phrase_lower.endswith("s") and len(phrase_lower) > 3:
            variants.append(phrase_lower[:-1])
        if phrase_lower.endswith("ies") and len(phrase_lower) > 4:
            variants.append(phrase_lower[:-3] + "y")

        for variant in variants:
            # 精确匹配表名
            if variant in table_names:
                matched_tables.add(table_names[variant])
                break
            # 表名包含这个词
            for tname_norm, tname_orig in table_names.items():
                if variant in tname_norm or tname_norm in variant:
                    matched_tables.add(tname_orig)
                    break
            else:
                # 匹配表名的某个部分（下划线拆分后）
                if variant in table_name_parts:
                    matched_tables.add(table_name_parts[variant])
                else:
                    # 部分匹配
                    for part, tname_orig in table_name_parts.items():
                        if variant in part or part in variant:
                            matched_tables.add(tname_orig)
                            break

    return matched_tables


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
