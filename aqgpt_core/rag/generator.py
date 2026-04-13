"""Generation backends for AQGPT RAG answers."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod

from aqgpt_core.config import RAG_GENERATION_CONFIG
from aqgpt_core.llm.vllm_provider import VLLMClient
from aqgpt_core.rag.settings import (
    RAG_GENERATION_MODEL,
    RAG_OLLAMA_BASE_URL,
    RAG_VLLM_BASE_URL,
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
            raise ImportError("ollama package is required for the RAG generator") from exc

        self.model = model
        self.client = ollama.Client(host=base_url)

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

        response = self.client.chat(model=self.model, messages=messages)
        return (response.get("message", {}) or {}).get("content", "") or ""


class FallbackGenerator(BaseRAGGenerator):
    def generate(self, question: str, context_block: str, chat_history: list[dict] | None = None) -> str:
        lines = [line for line in context_block.splitlines() if line.strip()]
        if not lines:
            return "I don't have enough information from urbanemissions.info to answer this question fully."
        return (
            "I found relevant urbanemissions.info context but couldn't run the local generator backend. "
            "Here are key extracted lines:\n\n- "
            + "\n- ".join(lines[:8])
        )


class VLLMGenerator(BaseRAGGenerator):
    def __init__(self, model: str = RAG_GENERATION_MODEL, base_url: str = RAG_VLLM_BASE_URL):
        self.model = model
        self.client = VLLMClient(base_url=base_url)

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

        response = self.client.chat(model=self.model, messages=messages, temperature=0.2)
        choices = response.get("choices", [])
        if not choices:
            return ""
        return (choices[0].get("message", {}) or {}).get("content", "") or ""


def build_generator() -> BaseRAGGenerator:
    provider = (RAG_GENERATION_CONFIG.provider or "qwen").lower()
    if provider in {"qwen", "ollama"}:
        return OllamaGenerator(model=RAG_GENERATION_MODEL, base_url=RAG_OLLAMA_BASE_URL)
    if provider == "vllm":
        return VLLMGenerator(model=RAG_GENERATION_MODEL, base_url=RAG_VLLM_BASE_URL)
    return FallbackGenerator()
