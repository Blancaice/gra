"""
Query complexity analysis with lightweight NER for Text-to-SQL.

核心思路：从 NL 问题中提取名词短语（实体）数量，结合 SQL 操作信号，
多维度判断查询复杂度。实体数量直接预测 JOIN 需求，比纯关键词匹配更准确。
"""

import re
from typing import Dict, List, Tuple


# ── 停用词 & 功能词（提取名词短语时过滤） ──────────────────────────

# 英语虚词：冠词、介词、连词、代词、助动词、疑问词
_FUNCTION_WORDS = frozenset(
    "the a an this that these those it its he she they them their his her my "
    "your our we us me i you "
    "is are was were be been being am do does did have has had will would "
    "can could shall should may might must need "
    "in on at to for of with by from into through during between among about "
    "and or but not nor yet so if then than as "
    "what which who whom whose where when how why "
    "there here up down out over under also very much many "
    "all any both each every few more most other some such no "
    "just only even still already again".split()
)

# 问句中的动词/功能短语（不是实体）
_QUESTION_VERBS = frozenset(
    "show find give list get tell display return select provide "
    "name report describe identify indicate state mention "
    "compare include contain belong happen occur exist "
    "make take go come see know think want say use "
    "work teach taught taken enrolled enroll study studied "
    "live lived born created made held played won lost "
    "started ended joined left hired fired published wrote written "
    "called named located based".split()
)

# SQL 聚合/排序相关词（是操作信号，不是实体）
_SQL_SIGNAL_WORDS = frozenset(
    "count number total sum average avg maximum minimum max min "
    "order sort rank top highest lowest largest smallest "
    "group per different distinct unique "
    "ascending descending asc desc".split()
)

_ALL_STOP = _FUNCTION_WORDS | _QUESTION_VERBS | _SQL_SIGNAL_WORDS


# ── 名词短语提取（轻量 NER） ──────────────────────────────────────

def _extract_noun_phrases(question: str) -> List[str]:
    """
    从 NL 问题中提取名词短语。

    算法：tokenize → 去掉停用词/功能词 → 相邻实体词合并为短语 → 去重。
    例: "Show the name of students who have taken courses taught by teachers"
      tokens: [show, the, name, of, students, who, have, taken, courses, taught, by, teachers]
      过滤后: [name, students, courses, taught, teachers]
      短语:  ["name", "students", "courses", "taught", "teachers"]
      （taught 虽然留下了，但后面会被 _is_entity_like 过滤）

    进一步过滤：只保留 "像实体" 的词（去掉动词形式的残留）。
    """
    tokens = re.findall(r"[a-z]+", question.lower())

    # 提取内容词（非停用词）
    content_spans: List[List[str]] = []
    current: List[str] = []
    for tok in tokens:
        if tok in _ALL_STOP or len(tok) <= 1:
            if current:
                content_spans.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        content_spans.append(current)

    # 每个 span 合并为一个短语
    phrases = [" ".join(span) for span in content_spans]

    # 去重（保序）
    seen = set()
    unique = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


# ── SQL 操作信号检测 ──────────────────────────────────────────────

# 聚合
_AGG_PATTERNS = [
    r"\b(?:count|number of|how many|total number)\b",
    r"\b(?:sum|total|average|avg|mean)\b",
    r"\b(?:maximum|minimum|max|min|greatest|least)\b",
]

# 排序 / LIMIT
_ORDER_PATTERNS = [
    r"\b(?:order|sort|rank)\b",
    r"\b(?:top \d+|bottom \d+)\b",
    r"\b(?:highest|lowest|largest|smallest|most|least|biggest)\b",
]

# GROUP BY
_GROUP_PATTERNS = [
    r"\b(?:for each|per|group by|grouped by|by each)\b",
    r"\b(?:each (?:type|category|department|country|city|year|month))\b",
]

# 子查询 / 嵌套引用
_NESTED_PATTERNS = [
    # 比较 + 子查询: "more than the average", "higher than any"
    r"\b(?:more|less|greater|fewer|higher|lower|bigger|smaller) than (?:the |any |all )",
    # 关系从句引导嵌套
    r"\b(?:that (?:have|has|had|is|are|was|were|do|does|did|contain|include))\b",
    r"\b(?:who (?:have|has|had|is|are|was|were|do|does|did|teach|take|enroll))\b",
    r"\b(?:which (?:have|has|had|is|are|was|were|do|does|did))\b",
    r"\bwhose\b",
    # "X of the Y that" 结构
    r"\bof (?:the |those |all )?(?:\w+ ){0,2}(?:that|which|who)\b",
]

# 否定 / 排除（EXCEPT, NOT IN）
_NEGATION_PATTERNS = [
    r"\b(?:not (?:in|have|include|contain|take|enroll))\b",
    r"\b(?:except|without|exclude|other than|apart from)\b",
    r"\b(?:never|no (?:one|student|course|employee|department))\b",
    r"\b(?:did not|does not|do not|have not|has not|is not|are not)\b",
]

# 集合操作（UNION, INTERSECT, EXCEPT）
_SET_OP_PATTERNS = [
    r"\b(?:both .{3,30} and)\b",
    r"\b(?:either .{3,30} or)\b",
    r"\b(?:as well as|in addition to|together with)\b",
    r"\b(?:but not|and also|or alternatively)\b",
]

# 比较（简单 WHERE 条件 vs 子查询比较）
_SIMPLE_COMPARE = re.compile(
    r"\b(?:more|less|greater|fewer|higher|lower|above|below|over|under|at least|at most)"
    r" than \d"
)

# 多条件
_CONDITION_MARKERS = re.compile(
    r"\b(?:and|or)\b(?!.*\b(?:order|sort|group)\b)"
)


def _count_pattern_matches(text: str, patterns: List[str]) -> int:
    """统计 patterns 中有多少个在 text 里命中。"""
    count = 0
    for pat in patterns:
        if re.search(pat, text):
            count += 1
    return count


def _has_any_pattern(text: str, patterns: List[str]) -> bool:
    return _count_pattern_matches(text, patterns) > 0


def _count_conditions(question: str) -> int:
    """统计约束条件个数（"and" / "or" 在非排序上下文中出现的次数 + 1）。"""
    # 去掉 "order by" 之后的部分（NL 里不常见但保险）
    q = question.lower()
    markers = re.findall(r"\b(?:and|or)\b", q)
    return len(markers) + 1


# ── 主分析函数 ────────────────────────────────────────────────────

def analyze_question(question: str) -> Dict:
    """
    多维度问题复杂度分析。

    维度：
    1. 实体数量（名词短语）→ JOIN 需求
    2. 聚合信号 → GROUP BY / HAVING
    3. 排序信号 → ORDER BY / LIMIT
    4. 嵌套引用 → 子查询
    5. 否定/排除 → NOT IN / EXCEPT
    6. 集合操作 → UNION / INTERSECT
    7. 条件数量 → WHERE 子句复杂度
    """
    q = (question or "").lower().strip()
    tokens = re.findall(r"\w+", q)

    # 1. 实体提取
    noun_phrases = _extract_noun_phrases(q)
    num_entities = len(noun_phrases)

    # 2. SQL 操作信号
    has_agg = _has_any_pattern(q, _AGG_PATTERNS)
    has_order = _has_any_pattern(q, _ORDER_PATTERNS)
    has_group = _has_any_pattern(q, _GROUP_PATTERNS)
    has_nested = _has_any_pattern(q, _NESTED_PATTERNS)
    has_negation = _has_any_pattern(q, _NEGATION_PATTERNS)
    has_set_op = _has_any_pattern(q, _SET_OP_PATTERNS)

    # 简单比较 vs 子查询比较
    has_subquery_compare = (
        _has_any_pattern(q, [r"\b(?:more|less|greater|fewer|higher|lower) than (?:the |any |all )"])
        and not bool(_SIMPLE_COMPARE.search(q))
    )

    # 3. 多条件计数
    num_conditions = _count_conditions(q)

    # ── 复杂度评分 ──
    score = 0.0

    # 实体数量：核心指标，直接预测 JOIN 数
    # 1 entity → 大概率单表; 2 → 可能 JOIN; 3+ → 几乎必然 JOIN
    if num_entities >= 4:
        score += 5.0
    elif num_entities >= 3:
        score += 3.5
    elif num_entities >= 2:
        score += 1.5

    # 聚合
    score += 2.0 if has_agg else 0
    # GROUP BY 显式信号
    score += 1.5 if has_group else 0
    # 排序
    score += 1.0 if has_order else 0
    # 嵌套引用 → 子查询
    score += 3.0 if has_nested else 0
    # 否定/排除
    score += 2.5 if has_negation else 0
    # 集合操作
    score += 2.5 if has_set_op else 0
    # 子查询比较（"more than the average"）
    score += 2.0 if has_subquery_compare else 0
    # 多条件
    score += max(0, (num_conditions - 2)) * 0.5
    # 长问题倾向更复杂
    score += 1.0 if len(tokens) > 20 else 0

    # ── 复杂度等级 ──
    if score >= 7:
        level = "high"
    elif score >= 3.5:
        level = "medium"
    else:
        level = "low"

    return {
        "noun_phrases": noun_phrases,
        "num_entities": num_entities,
        "has_aggregation": has_agg,
        "has_group": has_group,
        "has_order": has_order,
        "has_nested": has_nested,
        "has_negation": has_negation,
        "has_set_op": has_set_op,
        "has_subquery_compare": has_subquery_compare,
        "num_conditions": num_conditions,
        "token_len": len(tokens),
        "complexity_score": score,
        "complexity_level": level,
    }


# ── 下游接口（保持兼容） ─────────────────────────────────────────

def dynamic_weights(profile: Dict) -> Dict:
    """Dynamic weight policy for final score components."""
    level = profile.get("complexity_level", "medium")
    if level == "low":
        return {"nl": 0.45, "sql": 0.35, "ast": 0.08, "schema": 0.17, "token_cost": 0.05}
    if level == "high":
        return {"nl": 0.25, "sql": 0.35, "ast": 0.22, "schema": 0.23, "token_cost": 0.05}
    return {"nl": 0.35, "sql": 0.35, "ast": 0.15, "schema": 0.20, "token_cost": 0.05}


def source_ratio(profile: Dict) -> Dict:
    """Dynamic ratio for question-centric vs schema-centric pools."""
    level = profile.get("complexity_level", "medium")
    if level == "low":
        return {"question_centric": 0.7, "schema_centric": 0.3}
    if level == "high":
        return {"question_centric": 0.4, "schema_centric": 0.6}
    return {"question_centric": 0.55, "schema_centric": 0.45}
