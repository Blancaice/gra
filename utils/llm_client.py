"""
LLM API 客户端封装
"""

from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """LLM API 客户端基类"""

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI API 客户端"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", base_url: Optional[str] = None):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful SQL expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            seed=42
        )
        return response.choices[0].message.content


def create_llm_client(provider: str = "openai", **kwargs) -> LLMClient:
    if provider == "openai":
        return OpenAIClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
