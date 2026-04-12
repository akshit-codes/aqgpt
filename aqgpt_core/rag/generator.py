"""Generation backends for AQGPT RAG answers."""

import json
import importlib
from abc import ABC, abstractmethod

from openai import OpenAI

from aqgpt_core.config import QWEN_API_BASE, QWEN_API_KEY
from aqgpt_core.rag.settings import (
    RAG_GENERATION_PROVIDER,
    RAG_GENERATION_MODEL,
    RAG_OLLAMA_BASE_URL,
)

SYSTEM_PROMPT = """You are an expert assistant for urbanemissions.info, an air pollution knowledge platform focused on India.

Answer the user's question using ONLY the provided context passages.
Rules:
1. Use only provided context.
2. Cite using source indices in square brackets like [1], [2], based only on provided sources.
3. Never cite a source number that is not present in the context block.
4. If context is insufficient, explicitly say so.
5. Be specific and concise.
"""


class BaseRAGGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, context_block: str, chat_history: list[dict] | None = None) -> str:
        pass


class OllamaGenerator(BaseRAGGenerator):
    def __init__(self, model: str = RAG_GENERATION_MODEL, base_url: str = RAG_OLLAMA_BASE_URL):
        try:
            ollama = importlib.import_module("ollama")
        except ImportError as exc:
            raise ImportError("ollama package is required for RAG_GENERATION_PROVIDER=ollama") from exc

        self.model = model
        self.client = ollama.Client(host=base_url)

    def generate(self, question: str, context_block: str, chat_history: list[dict] | None = None) -> str:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context from urbanemissions.info:\n\n{context_block}\n\n"
            f"Question: {question}"
        )
        response = self.client.generate(model=self.model, prompt=prompt, stream=False)
        return response.get("response", "")


class QwenAPIGenerator(BaseRAGGenerator):
    def __init__(self, model: str = RAG_GENERATION_MODEL):
        self.client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_API_BASE)
        self.model = model

    def generate(self, question: str, context_block: str, chat_history: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if chat_history:
            for msg in chat_history[-6:]:
                role = "assistant" if msg.get("role") == "assistant" else "user"
                messages.append({"role": role, "content": msg.get("content", "")})

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Context from urbanemissions.info:\n\n{context_block}\n\n"
                    f"Question: {question}"
                ),
            }
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


class FallbackGenerator(BaseRAGGenerator):
    """Last-resort generation from snippets if model backend is unavailable."""

    def generate(self, question: str, context_block: str, chat_history: list[dict] | None = None) -> str:
        lines = [line for line in context_block.splitlines() if line.strip()]
        if not lines:
            return "I don't have enough information from urbanemissions.info to answer this question fully."
        return (
            "I found relevant urbanemissions.info context but couldn't run the local generator backend. "
            "Here are key extracted lines:\n\n- "
            + "\n- ".join(lines[:8])
        )


def build_generator() -> BaseRAGGenerator:
    provider = (RAG_GENERATION_PROVIDER or "ollama").lower()
    if provider == "ollama":
        return OllamaGenerator()
    if provider == "qwen_api":
        return QwenAPIGenerator()
    return FallbackGenerator()
