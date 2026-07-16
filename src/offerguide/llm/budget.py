"""Optional operator-configured daily LLM cost guardrail.

The agent can run multi-step loops. Without an
hard cap, a runaway worldview update or stuck retry can torch the
budget. This module enforces a per-UTC-day USD cap by summing recent
``skill_runs.cost_usd`` + ``harness_runs.cost_usd``.

Design choice: **enforced at entry points**, not per-LLM-call. The two
entry points that drive 99% of cost are:
The conversation-agent entry point calls ``enforce_daily_budget(store)`` once at start. If over
cap, raises ``BudgetExceeded`` which the caller surfaces as a 429-ish
error. We don't try to be clever with mid-run cancellation — agent
loops are short enough (~$0.10) that "stop the next one" is cheap
enough.

There is no product-level default cap. Set ``OFFERGUIDE_DAILY_BUDGET_USD``
explicitly when an operator wants a hard deployment limit.
"""

from __future__ import annotations

import logging
import os

from ..memory import Store

log = logging.getLogger(__name__)


DEFAULT_DAILY_CAP_USD = 0.0
"""Disabled by default; product quality is not traded for an arbitrary cap."""


class BudgetExceeded(RuntimeError):
    """Today's LLM spend exceeded the daily cap. Caller should surface a
    user-friendly message ('Tomorrow it'll reset / raise the cap')."""

    def __init__(self, today_spent_usd: float, cap_usd: float) -> None:
        self.today_spent_usd = today_spent_usd
        self.cap_usd = cap_usd
        super().__init__(
            f"Daily LLM budget exceeded: ${today_spent_usd:.4f} spent today, "
            f"cap is ${cap_usd:.2f}. Wait until UTC midnight or raise "
            f"OFFERGUIDE_DAILY_BUDGET_USD env."
        )


def get_daily_cap_usd() -> float:
    """Read the daily cap from env, fallback to default."""
    raw = os.environ.get("OFFERGUIDE_DAILY_BUDGET_USD")
    if raw:
        try:
            return float(raw)
        except ValueError:
            log.warning("OFFERGUIDE_DAILY_BUDGET_USD=%r unparseable, using default", raw)
    return DEFAULT_DAILY_CAP_USD


def get_today_spend_usd(store: Store) -> float:
    """Sum cost_usd from skill_runs + harness_runs since UTC midnight.

    Returns 0.0 if tables don't exist yet (graceful for fresh DB).
    """
    total = 0.0
    queries = (
        "SELECT COALESCE(SUM(cost_usd), 0) FROM skill_runs "
        "WHERE created_at >= julianday('now', 'start of day')",
        "SELECT COALESCE(SUM(cost_usd), 0) FROM harness_runs "
        "WHERE started_at >= julianday('now', 'start of day')",
    )
    for q in queries:
        try:
            with store.connect() as conn:
                row = conn.execute(q).fetchone()
            if row and row[0]:
                total += float(row[0])
        except Exception as e:
            # Table not yet migrated (fresh DB) — treat as 0
            log.debug("budget: query failed (likely fresh DB): %s", e)
    return total


def enforce_daily_budget(store: Store, cap_usd: float | None = None) -> None:
    """Raise ``BudgetExceeded`` if today's LLM spend is at/over the cap.

    Call once at the start of an LLM-driven flow. Cheap (1 SQL aggregate query).
    """
    cap = cap_usd if cap_usd is not None else get_daily_cap_usd()
    if cap <= 0:
        # 0/negative cap = disabled (don't enforce)
        return
    spent = get_today_spend_usd(store)
    if spent >= cap:
        raise BudgetExceeded(today_spent_usd=spent, cap_usd=cap)
    if spent > cap * 0.8:
        log.warning(
            "budget: today's LLM spend ${:.4f} is at {:.0%} of cap ${:.2f}",
            spent,
            spent / cap,
            cap,
        )
