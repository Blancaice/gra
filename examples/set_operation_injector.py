"""
集合操作示例注入器：基于分类器概率 + 关键词检测
"""
import re
from typing import List, Dict


# 手工构造的集合操作示例（不同领域，避免泄露答案）
SET_OPERATION_EXAMPLES = [
    {
        "question": "Find students who are enrolled in both CS101 and MATH202.",
        "sql": "SELECT student_id FROM enrollments WHERE course_id = 'CS101' INTERSECT SELECT student_id FROM enrollments WHERE course_id = 'MATH202'",
        "db_id": "university",
    },
    {
        "question": "List employees who work in either the Sales department or have a salary above 100000.",
        "sql": "SELECT employee_id FROM employees WHERE department = 'Sales' UNION SELECT employee_id FROM employees WHERE salary > 100000",
        "db_id": "company",
    },
    {
        "question": "Find cities that have airports but do not have train stations.",
        "sql": "SELECT city_name FROM cities WHERE has_airport = 1 EXCEPT SELECT city_name FROM cities WHERE has_train_station = 1",
        "db_id": "transportation",
    }
]


def detect_set_operation_keywords(question: str) -> bool:
    """检测集合操作关键词"""
    q = question.lower()
    patterns = [
        r'\b(?:both|and also|as well as)\b',
        r'\b(?:either|or)\b.*\b(?:or)\b',
        r'\b(?:not|without|except|exclude|do not have)\b',
    ]
    return any(re.search(p, q) for p in patterns)


def should_inject_set_examples(
    question: str,
    query_profile: Dict,
    threshold: float = 0.7,
) -> bool:
    """判断是否需要注入集合操作示例"""
    is_complex = query_profile.get("complexity_level") in ["high"]
    set_op_proba = query_profile.get("set_op_proba", 0.0)
    return set_op_proba >= threshold and is_complex


def inject_set_operation_examples(
    question: str,
    existing_examples: List[Dict],
    query_profile: Dict,
    max_inject: int = 2,
    threshold: float = 0.7,
) -> List[Dict]:
    """条件注入集合操作示例"""
    if not should_inject_set_examples(question, query_profile, threshold):
        return existing_examples

    print(f"  [Set Op Injector] Injecting {max_inject} set operation example(s) (proba={query_profile.get('set_op_proba', 0.0):.3f})")
    return SET_OPERATION_EXAMPLES[:max_inject] + existing_examples
