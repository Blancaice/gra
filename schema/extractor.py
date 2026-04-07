"""
Schema 提取模块
从数据库中提取表、列、外键等元数据信息
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_table_descriptions(tables_json_path: str, db_id: str) -> Optional[Dict]:
    """
    从 tables.json 加载列的自然语言描述

    Returns:
        {(table_idx, col_name): description} 的映射
    """
    if not Path(tables_json_path).exists():
        return None

    with open(tables_json_path, 'r', encoding='utf-8') as f:
        all_tables = json.load(f)

    for db in all_tables:
        if db['db_id'] == db_id:
            table_names = db.get('table_names_original', db.get('table_names', []))
            col_names = db.get('column_names', [])
            col_names_orig = db.get('column_names_original', [])

            # 构建映射: (table_name, col_name) -> description
            desc_map = {}
            for i, (tbl_idx, desc) in enumerate(col_names):
                if tbl_idx < 0:
                    continue
                tbl_name = table_names[tbl_idx]
                col_name = col_names_orig[i][1] if i < len(col_names_orig) else desc
                desc_map[(tbl_name.lower(), col_name.lower())] = desc
            return desc_map
    return None


def extract_schema(db_path: str, tables_json_path: str = None) -> Dict:
    """
    提取数据库的完整 Schema 信息

    Args:
        db_path: 数据库文件路径
        tables_json_path: tables.json 路径，用于加载列描述

    Returns:
        包含表、列、外键信息的字典
    """
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    cursor = conn.cursor()

    # 尝试加载列描述
    db_id = Path(db_path).stem
    desc_map = None
    if tables_json_path:
        desc_map = load_table_descriptions(tables_json_path, db_id)

    schema = {
        'tables': [],
        'foreign_keys': []
    }

    # 获取所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        table_info = {
            'name': table_name,
            'columns': []
        }

        # 获取表的列信息
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()

        for col in columns:
            col_name = col[1]
            col_data = {
                'name': col_name,
                'type': col[2],
                'primary_key': bool(col[5])
            }
            # 添加列描述
            if desc_map:
                key = (table_name.lower(), col_name.lower())
                if key in desc_map:
                    col_data['description'] = desc_map[key]
            table_info['columns'].append(col_data)

        # 获取外键信息
        cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
        fks = cursor.fetchall()

        for fk in fks:
            schema['foreign_keys'].append({
                'from_table': table_name,
                'from_column': fk[3],
                'to_table': fk[2],
                'to_column': fk[4]
            })

        schema['tables'].append(table_info)

    conn.close()
    return schema


def build_schema_graph(schema: Dict) -> Dict:
    """
    构建 Schema 的图结构，用于路径拓展

    Args:
        schema: extract_schema 返回的 Schema 字典

    Returns:
        邻接表表示的图 {'table_name': ['related_table1', 'related_table2']}
    """
    graph = {}

    # 初始化所有表
    for table in schema['tables']:
        graph[table['name']] = []

    # 添加外键关系
    for fk in schema['foreign_keys']:
        from_table = fk['from_table']
        to_table = fk['to_table']

        # 跳过引用了不存在表的外键（大小写不一致等情况）
        if from_table not in graph or to_table not in graph:
            continue

        if to_table not in graph[from_table]:
            graph[from_table].append(to_table)
        if from_table not in graph[to_table]:
            graph[to_table].append(from_table)

    return graph
