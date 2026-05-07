"""LLM client — DeepSeek default (OpenAI-compatible). See `client.py` for env-var contract."""

from .budget import (
    DEFAULT_DAILY_CAP_USD,
    BudgetExceeded,
    enforce_daily_budget,
    get_daily_cap_usd,
    get_today_spend_usd,
)
from .client import (
    DEFAULT_DEEPSEEK_BASE,
    DEFAULT_MODEL,
    LLMClient,
    LLMError,
    LLMResponse,
    ToolCall,
)

__all__ = [
    "DEFAULT_DAILY_CAP_USD",
    "DEFAULT_DEEPSEEK_BASE",
    "DEFAULT_MODEL",
    "BudgetExceeded",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "ToolCall",
    "enforce_daily_budget",
    "get_daily_cap_usd",
    "get_today_spend_usd",
]
