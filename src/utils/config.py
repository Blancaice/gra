"""
Configuration management utilities.
"""

import json
from typing import Any, Dict


class Config:
    """Configuration wrapper."""

    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path) if config_path else self.default_config()

    @staticmethod
    def default_config() -> Dict[str, Any]:
        return {
            "llm": {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "max_tokens": 500,
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "schema": {
                "top_k": 5,
                "max_hops": 1,
            },
            "examples": {
                "online_generation": True,
                "use_loaded_examples": False,
                "num_question_centric": 8,
                "num_schema_centric": 6,
                "schema_paths": 3,
                "schema_examples_per_path": 2,
                "schema_min_nodes": 2,
            },
            "scoring": {
                "weights": [0.4, 0.4, 0.2],
                "dynamic_enabled": True,
            },
            "generation": {
                "top_k": 3,
                "num_candidates": 3,
                "token_budget": 2000,
                "use_diversity": True,
                "diversity_threshold": 0.3,
                "mmr_lambda": 0.75,
            },
        }

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_config(self, config_path: str):
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def get(self, key: str, default=None):
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default
