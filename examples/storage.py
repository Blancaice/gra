"""
示例库存储模块
"""

import json
from typing import List, Dict


def save_examples(examples: List[Dict], filepath: str):
    """
    保存示例到 JSON 文件

    Args:
        examples: 示例列表
        filepath: 保存路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)


def load_examples(filepath: str) -> List[Dict]:
    """
    从 JSON 文件加载示例

    Args:
        filepath: 文件路径

    Returns:
        示例列表
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    # Backward compatibility for old example files.
    for ex in examples:
        if 'source' not in ex:
            ex['source'] = 'unknown'
        if 'token_count' not in ex:
            q = ex.get('question', '')
            s = ex.get('sql', '')
            ex['token_count'] = len(q) // 4 + len(s) // 4 + 20

    return examples


def merge_examples(examples_list: List[List[Dict]]) -> List[Dict]:
    """
    合并多个示例列表

    Args:
        examples_list: 多个示例列表

    Returns:
        合并后的示例列表（去重）
    """
    merged = []
    seen = set()

    for examples in examples_list:
        for example in examples:
            # 使用 SQL 作为唯一标识
            sql = example.get('sql', '')
            if sql and sql not in seen:
                merged.append(example)
                seen.add(sql)

    return merged
