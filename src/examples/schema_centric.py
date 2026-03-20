"""
Schema-centric example generation.

Query-guided flow:
User Question -> Schema Linking -> Anchor Nodes -> Biased Graph Traversal -> Top-M Paths
"""

import json
import random
import re
from typing import Dict, List, Optional, Tuple

from src.examples.example_validator import summarize_rejections, validate_generated_examples


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _name_tokens(name: str) -> List[str]:
    if not name:
        return []
    parts = re.split(r"[^a-zA-Z0-9]+", name)
    out = []
    for p in parts:
        if not p:
            continue
        out.extend(_tokenize(p))
    return out


def _extract_anchor_scores(user_question: str, schema: Dict, schema_hints: Dict = None) -> Dict[str, float]:
    """计算锚点分数。如果有 schema_hints，优先用 LLM 识别的表列作为锚点。"""
    scores: Dict[str, float] = {}

    # 如果有 schema linking 结果，直接用作高分锚点
    if schema_hints and schema_hints.get("tables"):
        linked_tables = {t.lower() for t in schema_hints["tables"]}
        linked_columns = {}
        for tname, cols in schema_hints.get("columns", {}).items():
            linked_columns[tname.lower()] = {c.lower() for c in cols}

        for table in schema.get("tables", []):
            tname = table.get("name", "")
            if tname.lower() in linked_tables:
                scores[tname] = 3.0  # LLM 确认的表，最高分
                # 有具体列匹配的再加分
                if tname.lower() in linked_columns:
                    scores[tname] += 0.5 * len(linked_columns[tname.lower()])
            else:
                scores[tname] = 0.0
        return scores

    # 原始 token 匹配逻辑
    q_tokens = set(_tokenize(user_question))
    scores: Dict[str, float] = {}

    for table in schema.get("tables", []):
        tname = table.get("name", "")
        t_tokens = set(_name_tokens(tname))
        score = 0.0

        # Direct table-name hit.
        if q_tokens.intersection(t_tokens):
            score += 2.0

        # Column-name and column-description hits.
        for col in table.get("columns", []):
            c_tokens = set(_name_tokens(col.get("name", "")))
            d_tokens = set(_tokenize(col.get("description", "")))
            if q_tokens.intersection(c_tokens):
                score += 1.5
            if q_tokens.intersection(d_tokens):
                score += 1.0

        scores[tname] = score

    # Ensure non-zero fallback distribution.
    if not any(v > 0 for v in scores.values()):
        for k in list(scores.keys()):
            scores[k] = 1.0
    return scores


def generate_from_schema(
    schema: Dict,
    schema_graph: Dict,
    llm_client,
    num_paths: int = 3,
    min_nodes: int = 2,
    extra_ratio: float = 0.0,
    min_extra_templates: int = 0,
    db_path: str = None,
    min_valid_ratio: float = 0.7,
    max_retries: int = 1,
    user_question: str = "",
    examples_per_path: int = 2,
    schema_hints: Dict = None,
) -> List[Dict]:
    """
    Build schema-centric examples with query-guided path sampling.
    Defaults: 3 paths x 2 examples per path.
    """
    if not schema.get("tables"):
        return []

    anchor_scores = _extract_anchor_scores(user_question, schema, schema_hints=schema_hints)
    weighted_graph = build_weighted_graph(schema, schema_graph, anchor_scores)
    sampled_paths = sample_paths_query_guided(
        graph=weighted_graph,
        schema=schema,
        anchor_scores=anchor_scores,
        num_paths=max(1, num_paths),
        min_nodes=max(1, min_nodes),
    )

    path_templates: List[str] = []
    for path in sampled_paths:
        path_templates.extend(
            path_to_sql_templates(
                path=path,
                schema=schema,
                user_question=user_question,
                examples_per_path=max(1, examples_per_path),
            )
        )

    all_templates: List[str] = []
    seen = set()
    for sql in path_templates:
        key = (sql or "").strip().lower()
        if key and key not in seen:
            all_templates.append(sql.strip())
            seen.add(key)

    return generate_questions_batch(
        all_templates,
        schema,
        llm_client,
        db_path=db_path,
        min_valid_ratio=min_valid_ratio,
        max_retries=max_retries,
    )


def build_weighted_graph(schema: Dict, schema_graph: Dict, anchor_scores: Dict[str, float]) -> Dict:
    graph = {}
    for table in schema.get("tables", []):
        table_name = table["name"]
        graph[table_name] = []
        for neighbor in schema_graph.get(table_name, []):
            # Edge bias: anchor-connected nodes get higher transition probability.
            weight = 1.0 + 0.4 * anchor_scores.get(neighbor, 0.0)
            graph[table_name].append((neighbor, max(0.1, weight)))
    return graph


def sample_paths_query_guided(
    graph: Dict,
    schema: Dict,
    anchor_scores: Dict[str, float],
    num_paths: int,
    min_nodes: int,
) -> List[List[str]]:
    tables = [t["name"] for t in schema.get("tables", [])]
    if not tables:
        return []

    ranked_anchors = sorted(tables, key=lambda t: anchor_scores.get(t, 0.0), reverse=True)
    paths: List[List[str]] = []
    seen = set()

    # Start from top anchors first.
    for start in ranked_anchors:
        if len(paths) >= num_paths:
            break
        for _ in range(3):
            path = biased_walk(graph, start=start, target_len=min_nodes)
            if len(path) < min_nodes:
                continue
            k = tuple(path)
            if k in seen:
                continue
            seen.add(k)
            paths.append(path)
            if len(paths) >= num_paths:
                break

    # Fallback fill.
    idx = 0
    while len(paths) < num_paths and idx < len(ranked_anchors):
        start = ranked_anchors[idx]
        idx += 1
        path = biased_walk(graph, start=start, target_len=max(1, min_nodes - 1))
        if not path:
            path = [start]
        k = tuple(path)
        if k not in seen:
            seen.add(k)
            paths.append(path)

    return paths[:num_paths]


def biased_walk(graph: Dict, start: str, target_len: int) -> List[str]:
    path = [start]
    visited = {start}
    node = start

    while len(path) < target_len:
        if node not in graph or not graph[node]:
            break
        neighbors = [(n, w) for n, w in graph[node] if n not in visited]
        if not neighbors:
            break
        node = weighted_choice(neighbors)
        path.append(node)
        visited.add(node)

    return path


def weighted_choice(neighbors: List[Tuple[str, float]]) -> str:
    total = sum(w for _, w in neighbors)
    r = random.uniform(0, total)
    cumulative = 0.0
    for node, weight in neighbors:
        cumulative += weight
        if r <= cumulative:
            return node
    return neighbors[-1][0]


def _pick_question_agg(user_question: str) -> str:
    q = (user_question or "").lower()
    if any(k in q for k in ["average", "avg", "mean"]):
        return "AVG"
    if any(k in q for k in ["max", "highest", "largest", "most"]):
        return "MAX"
    if any(k in q for k in ["min", "lowest", "smallest", "least", "youngest"]):
        return "MIN"
    return "COUNT"


def path_to_sql_templates(
    path: List[str],
    schema: Dict,
    user_question: str,
    examples_per_path: int = 2,
) -> List[str]:
    if not path:
        return []

    table_columns = {t["name"]: t.get("columns", []) for t in schema.get("tables", [])}
    join_clause = build_join_clause(path, schema)

    first_table = path[0]
    first_cols = table_columns.get(first_table, [])
    first_col = _pick_first_column(first_cols, default="id")
    num_col = _pick_numeric_column(first_cols, default=first_col)

    templates: List[str] = []

    # Template 1: structure-preserving projection query.
    templates.append(f"SELECT DISTINCT {first_table}.{first_col} FROM {join_clause} LIMIT 20")

    # Template 2: aggregation query guided by user question intent.
    agg = _pick_question_agg(user_question)
    if agg == "COUNT":
        agg_expr = "COUNT(*)"
    else:
        agg_expr = f"{agg}({first_table}.{num_col})"
    templates.append(
        f"SELECT {first_table}.{first_col}, {agg_expr} FROM {join_clause} "
        f"GROUP BY {first_table}.{first_col} ORDER BY {agg_expr} DESC LIMIT 10"
    )

    return templates[: max(1, examples_per_path)]


def build_join_clause(tables: List[str], schema: Dict) -> str:
    if not tables:
        return ""
    if len(tables) == 1:
        return tables[0]

    result = tables[0]
    for i in range(1, len(tables)):
        fk = find_foreign_key(tables[i - 1], tables[i], schema)
        if fk:
            if fk["from_table"] == tables[i]:
                cond = f"{tables[i]}.{fk['from_column']} = {tables[i - 1]}.{fk['to_column']}"
            else:
                cond = f"{tables[i]}.{fk['to_column']} = {tables[i - 1]}.{fk['from_column']}"
            result += f" JOIN {tables[i]} ON {cond}"
        else:
            result += f", {tables[i]}"
    return result


def find_foreign_key(table1: str, table2: str, schema: Dict) -> Optional[Dict]:
    for fk in schema.get("foreign_keys", []):
        if (fk["from_table"] == table1 and fk["to_table"] == table2) or (
            fk["from_table"] == table2 and fk["to_table"] == table1
        ):
            return fk
    return None


def generate_questions_batch(
    sql_templates: List[str],
    schema: Dict,
    llm_client,
    db_path: str = None,
    min_valid_ratio: float = 0.7,
    max_retries: int = 1,
) -> List[Dict]:
    if not sql_templates:
        return []

    schema_str = format_schema(schema)
    sqls_str = "\n".join([f"{i + 1}. {sql}" for i, sql in enumerate(sql_templates)])

    base_prompt = f"""Database Schema:
{schema_str}

SQL Queries:
{sqls_str}

For each SQL, generate:
1) a natural language question
2) short reasoning

Important:
- Keep SQL semantically unchanged.
- Use only table/column names from schema.
- Return only JSON array.

Return format:
[{{"question":"...","sql":"...","reasoning":"..."}}]
"""

    best_valid: List[Dict] = []
    prompt = base_prompt

    for _ in range(max(0, max_retries) + 1):
        response = llm_client.generate(prompt, max_tokens=3000)
        parsed = parse_response(response, sql_templates)
        valid_examples, rejected = validate_generated_examples(
            parsed,
            schema=schema,
            db_path=db_path,
            keep_complex_on_exec_fail=False,
            reject_empty_result=True,
            sample_limit=5,
        )

        if len(valid_examples) > len(best_valid):
            best_valid = valid_examples

        required = max(1, int(len(sql_templates) * min_valid_ratio))
        if len(valid_examples) >= required:
            return valid_examples

        reject_summary = summarize_rejections(rejected)
        prompt = (
            base_prompt
            + "\n\nPrevious output had invalid SQL examples. Regenerate all examples.\n"
            + "Common issues to avoid:\n"
            + (reject_summary or "- invalid syntax/schema/execution")
            + "\nReturn only JSON array."
        )

    return best_valid


def format_schema(schema: Dict) -> str:
    lines = []
    for table in schema.get("tables", []):
        col_parts = []
        for c in table.get("columns", []):
            desc = c.get("description", "")
            if desc and desc.replace(" ", "_").lower() != c["name"].lower():
                col_parts.append(f"{c['name']}: {desc}")
            else:
                col_parts.append(c["name"])
        lines.append(f"{table['name']}: {', '.join(col_parts)}")
    return "\n".join(lines)


def parse_response(response: str, sql_templates: List[str]) -> List[Dict]:
    json_match = re.search(r"\[[\s\S]*\]", response)
    if not json_match:
        return []

    try:
        examples = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    valid = []
    for idx, ex in enumerate(examples):
        if not isinstance(ex, dict) or "question" not in ex:
            continue

        sql = sql_templates[idx] if idx < len(sql_templates) else (ex.get("sql") or "").strip()

        token_count = len(ex.get("question", "")) // 4 + len(sql) // 4 + 20
        valid.append(
            {
                "question": ex["question"],
                "sql": sql,
                "reasoning": ex.get("reasoning", ""),
                "source": "schema_centric",
                "token_count": token_count,
                "complexity_tag": classify_sql_complexity(sql),
            }
        )
    return valid


def classify_sql_complexity(sql: str) -> str:
    s = (sql or "").upper()
    has_set_ops = any(op in s for op in (" UNION ", " INTERSECT ", " EXCEPT "))
    has_sub = s.count("SELECT") > 1
    join_cnt = s.count("JOIN")
    has_group = "GROUP BY" in s
    has_having = "HAVING" in s
    has_order = "ORDER BY" in s

    if has_set_ops or (has_sub and join_cnt > 0) or (has_sub and has_group and has_having):
        return "extra"
    if has_sub or join_cnt >= 2 or (has_group and has_having):
        return "hard"
    if join_cnt >= 1 or has_group or has_order:
        return "medium"
    return "easy"


def _pick_first_column(columns: List[Dict], default: str = "id") -> str:
    if not columns:
        return default
    return columns[0].get("name", default)


def _pick_numeric_column(columns: List[Dict], default: str = "") -> str:
    numeric_types = {"INTEGER", "INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE", "DECIMAL"}
    for col in columns:
        ctype = str(col.get("type", "")).upper()
        if ctype in numeric_types:
            return col.get("name", default)
    return default
