"""
Text-to-SQL main pipeline.
"""

import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from src.examples.retriever import load_train_data, retrieve_question_centric, retrieve_schema_centric
from src.examples.storage import load_examples, merge_examples
from src.generation.generator import (
    generate_sql,
    generate_sql_candidates,
    rank_sql_candidates,
    validate_sql,
    validate_columns_against_schema,
    execute_with_repair,
)
from src.generation.prompt_builder import build_prompt
from src.generation.schema_linker import link_schema, is_complex_question
from src.generation.selector import select_top_k_examples
from src.schema.extractor import build_schema_graph, extract_schema
from src.schema.indexer import SchemaIndexer
from src.schema.retriever import retrieve_relevant_schema
from src.schema.vectorizer import SchemaVectorizer
from src.scoring.final_score import score_all_examples
from src.scoring.query_analyzer import analyze_question
from src.utils.config import Config
from src.utils.llm_client import create_llm_client


class TextToSQLPipeline:
    """End-to-end Text-to-SQL pipeline."""

    def __init__(self, db_path: str, config: Config = None, tables_json_path: str = None, vectorizer=None):
        self.db_path = db_path
        self.config = config or Config()
        self.tables_json_path = tables_json_path

        self.llm_client = create_llm_client(
            provider=self.config.get("llm.provider"),
            model=self.config.get("llm.model"),
        )
        self.vectorizer = vectorizer or SchemaVectorizer(model_name=self.config.get("embedding.model"))

        print("Extracting database schema...")
        self.schema = extract_schema(db_path, tables_json_path)
        self.schema_graph = build_schema_graph(self.schema)

        print("Building schema index...")
        embeddings, metadata = self.vectorizer.vectorize_schema(self.schema)
        self.indexer = SchemaIndexer(dimension=self.vectorizer.dimension)
        self.indexer.build_index(embeddings, metadata)

        self.examples = []

        # 加载训练数据用于检索
        train_paths = self.config.get("examples.train_data_paths",
                                       ["data/spider_data/train_spider.json"])
        if self.config.get("examples.use_train_retrieval", True):
            print("Loading training data for retrieval...")
            self.train_data = load_train_data(train_paths)
        else:
            self.train_data = []

        # 提取 db_id
        self.db_id = os.path.splitext(os.path.basename(db_path))[0]

    def load_examples(self, examples_path: str):
        print(f"Loading examples from {examples_path}...")
        self.examples = load_examples(examples_path)
        print(f"Loaded {len(self.examples)} examples")

    def generate(self, question: str) -> str:
        print(f"\nQuestion: {question}")

        # 分析问题复杂度，用于后续评分
        query_profile = analyze_question(question)

        print("Step 1: Retrieving relevant schema...")
        pruned_schema = retrieve_relevant_schema(
            question,
            self.vectorizer,
            self.indexer,
            self.schema_graph,
            self.schema,
            top_k=self.config.get("schema.top_k", 5),
        )
        print(f"  Selected {len(pruned_schema['tables'])} tables")

        # Step 1.5: 复杂查询触发 schema linking
        schema_hints = None
        if self.config.get("schema_linking.enabled", True) and is_complex_question(question):
            print("  Step 1.5: Complex question detected, triggering schema linking...")
            schema_hints = link_schema(
                question=question,
                pruned_schema=pruned_schema,
                llm_client=self.llm_client,
            )
            if schema_hints:
                print(f"  Linked tables: {schema_hints.get('tables', [])}")
            else:
                print("  Schema linking returned no hints")

        print("Step 2: Retrieving examples...")
        qc_examples = []
        sc_examples = []
        # Question-centric: 训练集 embedding 检索
        if self.config.get("examples.use_train_retrieval", True) and self.train_data:
            qc_examples = retrieve_question_centric(
                question=question,
                vectorizer=self.vectorizer,
                train_data=self.train_data,
                db_id=self.db_id,
                top_n=self.config.get("examples.num_question_centric", 6),
            )

        # Schema-centric: Query-guided Path Sampling + LLM 生成
        if self.config.get("examples.use_schema_centric", True):
            try:
                pruned_graph = build_schema_graph(pruned_schema)
                sc_examples = retrieve_schema_centric(
                    pruned_schema=pruned_schema,
                    schema_graph=pruned_graph,
                    llm_client=self.llm_client,
                    user_question=question,
                    num_paths=self.config.get("examples.schema_centric_num_paths", 3),
                    examples_per_path=self.config.get("examples.schema_centric_examples_per_path", 1),
                    db_path=self.db_path,
                    schema_hints=schema_hints,
                )
            except Exception as exc:
                print(f"  Schema-centric generation failed: {exc}")

        online_examples = merge_examples([qc_examples, sc_examples])
        print(
            f"  Retrieved examples: question_centric={len(qc_examples)}, "
            f"schema_centric={len(sc_examples)}, merged={len(online_examples)}"
        )

        # Prefer online examples by default; optionally merge with loaded library.
        if self.config.get("examples.use_loaded_examples", False) and self.examples:
            candidate_examples = merge_examples([online_examples, self.examples])
            print(f"  Loaded library merged: total candidates={len(candidate_examples)}")
        else:
            candidate_examples = online_examples if online_examples else self.examples
            if self.examples and not online_examples:
                print(f"  Fallback to loaded library examples: {len(candidate_examples)}")

        selected_examples = []
        if candidate_examples:
            print("Step 3: Scoring and selecting examples...")
            relevant_tables = [t["name"] for t in pruned_schema.get("tables", [])]

            scored_examples = score_all_examples(
                user_question=question,
                examples=candidate_examples,
                llm_client=self.llm_client,
                vectorizer=self.vectorizer,
                weights=self.config.get("scoring.weights"),
                relevant_schema=pruned_schema,
                relevant_tables=relevant_tables,
                dynamic_enabled=self.config.get("scoring.dynamic_enabled", True),
            )

            selected_examples = select_top_k_examples(
                scored_examples=scored_examples,
                k=self.config.get("generation.top_k"),
                token_budget=self.config.get("generation.token_budget"),
                user_profile=query_profile,
                mmr_lambda=self.config.get("generation.mmr_lambda", 0.75),
            )
            print(f"  Selected {len(selected_examples)} examples")
        else:
            print("Step 3: No examples available; continue with zero-shot prompt.")

        print("Step 4: Building prompt...")
        prompt = build_prompt(pruned_schema, selected_examples, question, schema_hints=schema_hints)

        print("Step 5: Generating SQL...")
        num_candidates = self.config.get("generation.num_candidates", 1)
        evaluation_mode = self.config.get("generation.evaluation_mode", "general")
        if num_candidates and num_candidates > 1:
            candidates = generate_sql_candidates(
                prompt=prompt,
                llm_client=self.llm_client,
                num_candidates=num_candidates,
                evaluation_mode=evaluation_mode,
            )
            sql, ranked = rank_sql_candidates(
                candidates=candidates,
                user_question=question,
                schema=pruned_schema,
                db_path=self.db_path,
                evaluation_mode=evaluation_mode,
            )
            print(f"  Generated {len(candidates)} candidates, selected best by reranker")
            if ranked:
                print(
                    "  Best candidate score="
                    f"{ranked[0]['score']:.3f}, syntax={ranked[0]['syntax_ok']}, "
                    f"col={ranked[0]['col_ok']}, exec={ranked[0]['exec_ok']}"
                )
        else:
            sql = generate_sql(prompt, self.llm_client, evaluation_mode=evaluation_mode)

        if validate_sql(sql):
            print("SQL syntax validation passed")
        else:
            print("SQL syntax validation failed")

        # 验证列名是否存在于 schema 中
        is_valid, invalid_items = validate_columns_against_schema(sql, pruned_schema)
        if is_valid:
            print("Column validation passed")
        else:
            print(f"Column validation failed: {invalid_items}")

        # Step 6: 语义验证 + 执行验证 + 错误修正
        print("Step 6: Verify and repair...")
        sql, exec_ok = execute_with_repair(
            sql=sql,
            user_question=question,
            schema=pruned_schema,
            db_path=self.db_path,
            llm_client=self.llm_client,
            evaluation_mode=evaluation_mode,
            max_retries=3,
        )
        if exec_ok:
            print("  SQL execution verified OK")
        else:
            print("  SQL execution failed after repair attempts")

        return sql


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Text-to-SQL System")
    parser.add_argument("--db", required=True, help="Database path")
    parser.add_argument("--tables_json", help="tables.json path for column descriptions")
    parser.add_argument("--examples", help="Examples JSON path")
    parser.add_argument("--config", help="Config JSON path")
    parser.add_argument("--question", help="User question")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    config = Config(args.config) if args.config else Config()
    pipeline = TextToSQLPipeline(args.db, config, args.tables_json)

    if args.examples:
        pipeline.load_examples(args.examples)

    if args.interactive:
        print("\n=== Text-to-SQL Interactive Mode ===")
        print("Type 'quit' to exit\n")
        while True:
            question = input("Question: ").strip()
            if question.lower() == "quit":
                break
            if not question:
                continue
            sql = pipeline.generate(question)
            print(f"\nGenerated SQL:\n{sql}\n")
            print("-" * 50)
    elif args.question:
        sql = pipeline.generate(args.question)
        print(f"\nGenerated SQL:\n{sql}")


if __name__ == "__main__":
    main()