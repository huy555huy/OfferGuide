"""Token → USD pricing for cost tracking (W13.7 + W15.15 cache-aware).

Per-token pricing for the models OfferGuide actually uses. Numbers as of
2026-Q2 from Anthropic + DeepSeek public pricing pages.

W15.15 — cache-aware pricing: DeepSeek + Anthropic both serve cached input
tokens at ~10% of full input rate. We track this so the **cost displayed
in /debug + harness_runs.cost_usd reflects reality**, not optimistic
"all input is cache miss" math.

Format change (backwards-compat — old call shape still works):
- Each model entry is now `(input_per_M, input_cached_per_M, output_per_M)`
- ``estimate_cost_usd(... cache_hit_tokens=0)`` defaults to "no hit"
  matching old behavior; pass the tokens count from the LLM response's
  usage object to get accurate cost.

DeepSeek auto prompt caching:
- Server hashes message prefix, hits cache when same prefix seen recently
- Charges cached tokens at ~1/10 normal input rate
- No client-side ``cache_control`` mark needed (works automatically)
- Hit count returned in response usage as ``prompt_cache_hit_tokens``
"""

from __future__ import annotations

# Per-million-token USD rates: (input_per_M, input_cached_per_M, output_per_M)
# Source: vendor pricing pages, 2026-Q2 snapshot.
_PRICING_TABLE: dict[str, tuple[float, float, float]] = {
    # ── Anthropic Claude family — cache_read = 0.1x normal input ──
    "claude-sonnet-4-6":          (3.00, 0.30, 15.00),
    "claude-sonnet-4.5":          (3.00, 0.30, 15.00),
    "claude-sonnet-4-5":          (3.00, 0.30, 15.00),
    "claude-haiku-4-5":           (1.00, 0.10,  5.00),
    "claude-haiku-4-5-20251001":  (1.00, 0.10,  5.00),
    "claude-opus-4-1":            (15.00, 1.50, 75.00),
    "claude-opus-4":              (15.00, 1.50, 75.00),
    # ── DeepSeek family — cache_hit ~ 0.1x normal input ──
    "deepseek-v4-flash":          (0.07, 0.007, 0.28),
    "deepseek-v4-pro":            (0.27, 0.027, 1.10),
    "deepseek-chat":              (0.27, 0.027, 1.10),
    "deepseek-reasoner":          (0.55, 0.055, 2.19),
    # ── OpenAI (some users proxy to) ──
    "gpt-4o":                     (2.50, 1.25, 10.00),
    "gpt-4o-mini":                (0.15, 0.075, 0.60),
    "gpt-4-turbo":                (10.00, 5.00, 30.00),
    # ── Fallback when model name is unknown ──
    "_unknown":                   (3.00, 0.30, 15.00),
}


def estimate_cost_usd(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_hit_tokens: int = 0,
) -> float:
    """Estimate USD cost for a single LLM call (cache-aware).

    Args:
        model: model name string (exact or prefix match against pricing table)
        prompt_tokens: TOTAL input tokens (cache hit + miss). DeepSeek + OpenAI
            return this number directly. Anthropic returns input_tokens +
            cache_read_input_tokens + cache_creation_input_tokens separately;
            sum them to get prompt_tokens equivalent.
        completion_tokens: output tokens
        cache_hit_tokens: subset of ``prompt_tokens`` that hit the cache.
            Charged at the cheaper input_cached rate. Default 0 = treat all
            input as cache miss (conservative).

    Returns:
        USD cost (float, 6 decimal places).
    """
    if not model:
        model = "_unknown"

    rates = _PRICING_TABLE.get(model)
    if rates is None:
        for known, r in _PRICING_TABLE.items():
            if known != "_unknown" and (model.startswith(known) or known.startswith(model[:18])):
                rates = r
                break
    if rates is None:
        rates = _PRICING_TABLE["_unknown"]

    input_rate, input_cached_rate, output_rate = rates
    cache_miss_tokens = max(0, prompt_tokens - cache_hit_tokens)
    cost = (
        cache_miss_tokens * input_rate / 1_000_000.0
        + cache_hit_tokens * input_cached_rate / 1_000_000.0
        + completion_tokens * output_rate / 1_000_000.0
    )
    return round(cost, 6)


def is_known_model(model: str) -> bool:
    """True iff we have explicit pricing for this model (no fallback)."""
    if model in _PRICING_TABLE and model != "_unknown":
        return True
    for known in _PRICING_TABLE:
        if known not in ("_unknown",) and (model.startswith(known) or known.startswith(model[:18])):
            return True
    return False
