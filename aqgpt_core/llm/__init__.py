"""LLM abstraction layer for AQGPT.

Supports pluggable LLM providers: "gemini" and "qwen".
Switch between them via LLM_TEXT_PROVIDER / LLM_FUNCTION_CALLER in .env.
"""

from aqgpt_core.config import LLM_TEXT_PROVIDER, LLM_FUNCTION_CALLER

__all__ = ["get_text_generator", "get_function_caller"]


def get_text_generator():
    """Factory to get the appropriate text generation provider."""
    if LLM_TEXT_PROVIDER == "gemini":
        from aqgpt_core.llm.gemini_provider import GeminiTextGenerator
        return GeminiTextGenerator()
    elif LLM_TEXT_PROVIDER == "qwen":
        from aqgpt_core.llm.qwen_provider import QwenTextGenerator
        return QwenTextGenerator()
    else:
        raise ValueError(f"Unknown LLM_TEXT_PROVIDER: {LLM_TEXT_PROVIDER!r}. Use 'gemini' or 'qwen'.")


def get_function_caller():
    """Factory to get the function calling provider."""
    if LLM_FUNCTION_CALLER == "gemini":
        from aqgpt_core.llm.gemini_provider import GeminiFunctionCaller
        return GeminiFunctionCaller()
    elif LLM_FUNCTION_CALLER == "qwen":
        from aqgpt_core.llm.qwen_provider import QwenFunctionCaller
        return QwenFunctionCaller()
    else:
        raise ValueError(f"Unknown LLM_FUNCTION_CALLER: {LLM_FUNCTION_CALLER!r}. Use 'gemini' or 'qwen'.")
