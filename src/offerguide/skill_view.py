"""W19 — Reusable SKILL invoke helper for HTML views.

Pre-W19: 4 view functions (apply_pack / post_apply_pack / + 2 others) each
copy-pasted ~70 lines of identical boilerplate:
  1. Validate API key / profile / runtime / spec
  2. Enforce daily budget
  3. Invoke SKILL in a thread
  4. Catch invoke exception
  5. Render error vs success template

Each duplication = 1 more place to forget when CLAUDE.md rule changes.
W19 extracts the pattern into ``invoke_skill_for_view``.

Why a separate file (vs in web.py): web.py is already 4700 lines.
Separation lets both views and tests import without dragging the whole
FastAPI app.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import Store
    from .config import Settings
    from .profile import UserProfile
    from .skills import SkillRuntime, SkillSpec

log = logging.getLogger(__name__)


@dataclass
class SkillViewResult:
    """Either a parsed payload + meta, or an error message.

    Always populated so templates can render uniformly:
    {% if result.error %}<error block>{% else %}<happy path>{% endif %}
    """
    parsed: dict[str, Any] | None = None
    raw_text: str = ""
    skill_run_id: int | None = None
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str | None = None


# Type alias for the inputs builder — takes the SkillSpec + UserProfile,
# returns the input dict to pass to runtime.invoke. Runs ONLY after all
# validation passes, so spec/profile are guaranteed non-None.
InputsBuilder = Callable[["SkillSpec", "UserProfile"], dict[str, Any]]


async def invoke_skill_for_view(
    *,
    skill_name: str,
    inputs_builder: InputsBuilder,
    settings: Settings,
    profile: UserProfile | None,
    runtime: SkillRuntime | None,
    skills: list[SkillSpec],
    store: Store,
) -> SkillViewResult:
    """Validate config + invoke a SKILL for an HTML view. Returns either
    error message or parsed payload + cost/timing meta.

    Validation order matches the user-facing actionability:
      1. LLM key (env config) — most likely cause for new users
      2. Resume — second most likely
      3. Runtime — only fails on init bug
      4. SKILL not registered — only fails after a bad refactor
      5. Daily budget — fails when user has been using a lot

    Each error message is short + actionable in Chinese — so templates can
    render it directly without rewording.
    """
    if not settings.deepseek_api_key:
        return SkillViewResult(error="需要先配 LLM key (.env DEEPSEEK_API_KEY)")
    if profile is None or not profile.raw_resume_text:
        return SkillViewResult(error="未配简历, 去 /profile 上传后回来")
    if runtime is None:
        return SkillViewResult(error="SkillRuntime 未初始化")
    spec = next((s for s in skills if s.name == skill_name), None)
    if spec is None:
        return SkillViewResult(error=f"{skill_name} SKILL 没注册")

    from .llm import BudgetExceeded, enforce_daily_budget
    try:
        enforce_daily_budget(store)
    except BudgetExceeded as e:
        return SkillViewResult(error=str(e))

    inputs = inputs_builder(spec, profile)

    import asyncio as _asyncio
    import time as _time
    t0 = _time.monotonic()
    try:
        sr = await _asyncio.to_thread(runtime.invoke, spec, inputs)
    except Exception as e:
        log.exception("skill_view: invoke %s failed: %s", skill_name, e)
        return SkillViewResult(error=f"调用 SKILL 失败: {e}")

    duration_ms = int((_time.monotonic() - t0) * 1000)
    if sr.parsed is None:
        return SkillViewResult(
            error=f"SKILL 输出非 JSON (skill_run_id={sr.skill_run_id})",
            raw_text=sr.raw_text[:1500],
            skill_run_id=sr.skill_run_id,
            duration_ms=duration_ms,
        )

    return SkillViewResult(
        parsed=sr.parsed,
        skill_run_id=sr.skill_run_id,
        cost_usd=round(sr.cost_usd or 0.0, 5),
        duration_ms=duration_ms,
    )
