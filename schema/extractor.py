"""
Schema 提取模块
从数据库中提取表、列、外键等元数据信息
"""

import sqlite3
import json
import os
import re
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
        cursor.execute(f"PRAGMA table_info({table_name})")
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
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
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

        if to_table not in graph[from_table]:
            graph[from_table].append(to_table)
        if from_table not in graph[to_table]:
            graph[to_table].append(from_table)

    return graph


def _sample_column_values(db_path: str, table_name: str, col_name: str, limit: int = 5) -> List[str]:
    """采样列的非空去重值"""
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cursor = conn.cursor()
        cursor.execute(
            f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
            f'WHERE "{col_name}" IS NOT NULL LIMIT {limit}'
        )
        rows = cursor.fetchall()
        conn.close()
        return [str(r[0]) for r in rows if r[0] is not None]
    except Exception:
        return []


def generate_column_descriptions(
    db_path: str,
    schema: Dict,
    llm_client,
    cache_dir: str = "data/cache",
) -> Dict[str, Dict[str, str]]:
    """
    为数据库的所有列生成自然语言描述，带磁盘缓存。
    一次 LLM 调用覆盖整个 db 的所有列。

    Returns:
        {table_name: {col_name: description}}
    """
    db_id = Path(db_path).stem
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"col_desc_{db_id}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 构建 schema + 采样值的描述
    lines = []
    for table in schema.get("tables", []):
        tname = table["name"]
        for col in table.get("columns", []):
            cname = col["name"]
            ctype = col.get("type", "")
            samples = _sample_column_values(db_path, tname, cname, limit=5)
            sample_str = ", ".join(samples[:5]) if samples else "(empty)"
            lines.append(f"{tname}.{cname} ({ctype}) samples: [{sample_str}]")

    if not lines:
        return {}

    schema_block = "\n".join(lines)
    prompt = f"""Given the following database columns with their types and sample values, generate a short natural language description for each column. The description should explain what the column stores in plain English.

{schema_block}

Return ONLY a JSON object in this exact format (no markdown, no explanation):
{{"table_name": {{"column_name": "description"}}}}
"""

    try:
        response = llm_client.generate(prompt, max_tokens=2000)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            desc_map = json.loads(json_match.group())
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(desc_map, f, ensure_ascii=False, indent=2)
            return desc_map
    except Exception as exc:
        print(f"  Column description generation failed: {exc}")

    return {}


def enrich_schema_with_descriptions(
    schema: Dict,
    desc_map: Dict[str, Dict[str, str]],
) -> Dict:
    """将生成的列描述注入到 schema 中（仅填充缺失的 description）"""
    for table in schema.get("tables", []):
        tname = table["name"]
        table_descs = desc_map.get(tname, {})
        for col in table.get("columns", []):
            if not col.get("description"):
                desc = table_descs.get(col["name"], "")
                if desc:
                    col["description"] = desc
    return schema
