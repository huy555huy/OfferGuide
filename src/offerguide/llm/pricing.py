"""Token → USD pricing for cost tracking (W13.7).

Per-token pricing for the models OfferGuide actually uses. Numbers as of
2026-Q1 from Anthropic + DeepSeek public pricing pages.

We deliberately keep this small + lookup-table style:
- Per-million-token rates (the unit Anthropic / DeepSeek publish in)
- Input tokens vs output tokens priced separately (output ≈ 5× input)
- "unknown" model falls back to a conservative estimate so we never
  silently report $0 cost when we should have flagged a known-pricey model.

If the user's proxy (ccvibe, one-api, etc.) routes to a model that maps
to one of these names, we estimate correctly. If the proxy uses a name
we don't recognize, we use the fallback rate.
"""

from __future__ import annotations

# Per-million-token USD rates. Format: (input_per_M, output_per_M)
# Source: vendor pricing pages, 2026-Q1 snapshot.
_PRICING_TABLE: dict[str, tuple[float, float]] = {
    # ── Anthropic Claude family ──
    "claude-sonnet-4-6":      (3.00, 15.00),
    "claude-sonnet-4.5":      (3.00, 15.00),
    "claude-sonnet-4-5":      (3.00, 15.00),
    "claude-haiku-4-5":       (1.00,  5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-opus-4-1":         (15.00, 75.00),
    "claude-opus-4":           (15.00, 75.00),
    # ── DeepSeek family ──
    "deepseek-v4-flash":      (0.07,  0.28),
    "deepseek-v4-pro":        (0.27,  1.10),
    "deepseek-chat":          (0.27,  1.10),
    "deepseek-reasoner":      (0.55,  2.19),
    # ── OpenAI (some users proxy to) ──
    "gpt-4o":                 (2.50, 10.00),
    "gpt-4o-mini":            (0.15,  0.60),
    "gpt-4-turbo":            (10.00, 30.00),
    # ── Fallback when model name is unknown ──
    "_unknown":               (3.00, 15.00),  # mid-range estimate, errs on side of "alert me"
}


def estimate_cost_usd(*, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a single LLM call.

    Returns the dollar cost (as a float) for (prompt_tokens, completion_tokens)
    against the named model. Unknown models use the conservative fallback rate.
    """
    if not model:
        model = "_unknown"

    # Try exact match first; fall back to prefix match (handles model variants
    # like "claude-sonnet-4-6-20251022" matching "claude-sonnet-4-6")
    rates = _PRICING_TABLE.get(model)
    if rates is None:
        for known, r in _PRICING_TABLE.items():
            if known != "_unknown" and (model.startswith(known) or known.startswith(model[:18])):
                rates = r
                break
    if rates is None:
        rates = _PRICING_TABLE["_unknown"]

    input_rate, output_rate = rates
    cost = (
        prompt_tokens * input_rate / 1_000_000.0
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
