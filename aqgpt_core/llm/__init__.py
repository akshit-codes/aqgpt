"""LLM abstraction layer for AQGPT."""

from aqgpt_core.config import FUNCTION_MODEL_CONFIG, TEXT_MODEL_CONFIG

__all__ = ["get_text_generator", "get_function_caller"]


def get_text_generator():
    """Factory to get the appropriate text generation provider."""
    if TEXT_MODEL_CONFIG.provider == "gemini":
        from aqgpt_core.llm.gemini_provider import GeminiTextGenerator
        return GeminiTextGenerator(model=TEXT_MODEL_CONFIG.model)
    elif TEXT_MODEL_CONFIG.provider == "qwen":
        from aqgpt_core.llm.qwen_provider import QwenTextGenerator
        return QwenTextGenerator(model=TEXT_MODEL_CONFIG.model, provider=TEXT_MODEL_CONFIG.provider)
    elif TEXT_MODEL_CONFIG.provider == "vllm":
        from aqgpt_core.llm.vllm_provider import VLLMTextGenerator
        return VLLMTextGenerator(model=TEXT_MODEL_CONFIG.model, provider=TEXT_MODEL_CONFIG.provider)
    else:
        raise ValueError(
            f"Unknown text provider: {TEXT_MODEL_CONFIG.provider!r}. Use 'gemini', 'qwen', or 'vllm'."
        )


def get_function_caller():
    """Factory to get the function calling provider."""
    if FUNCTION_MODEL_CONFIG.provider == "gemini":
        from aqgpt_core.llm.gemini_provider import GeminiFunctionCaller
        return GeminiFunctionCaller(model=FUNCTION_MODEL_CONFIG.model)
    elif FUNCTION_MODEL_CONFIG.provider == "qwen":
        from aqgpt_core.llm.qwen_provider import QwenFunctionCaller
        return QwenFunctionCaller(model=FUNCTION_MODEL_CONFIG.model, provider=FUNCTION_MODEL_CONFIG.provider)
    elif FUNCTION_MODEL_CONFIG.provider == "vllm":
        from aqgpt_core.llm.vllm_provider import VLLMFunctionCaller
        return VLLMFunctionCaller(model=FUNCTION_MODEL_CONFIG.model, provider=FUNCTION_MODEL_CONFIG.provider)
    else:
        raise ValueError(
            f"Unknown function-calling provider: {FUNCTION_MODEL_CONFIG.provider!r}. "
            "Use 'gemini', 'qwen', or 'vllm'."
        )
