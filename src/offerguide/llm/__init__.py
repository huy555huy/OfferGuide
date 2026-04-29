"""LLM client — DeepSeek default (OpenAI-compatible). See `client.py` for env-var contract."""

from .client import DEFAULT_DEEPSEEK_BASE, DEFAULT_MODEL, LLMClient, LLMError, LLMResponse

__all__ = ["DEFAULT_DEEPSEEK_BASE", "DEFAULT_MODEL", "LLMClient", "LLMError", "LLMResponse"]
