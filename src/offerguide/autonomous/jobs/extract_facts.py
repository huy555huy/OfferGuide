"""Daily fact-extraction job — mines user_facts out of recent skill_runs.

Each SKILL run produces structured output (Pydantic-validated). Facts
buried in those outputs ("RemeDi 项目 AUC 0.83", "腾讯一面 GRPO 卡了")
are exactly what we want long-term memory to remember — but no SKILL
caller persists them as facts because they're embedded in ad-hoc shape
per-SKILL.

This job runs an LLM extraction pass over recently completed runs and
appends discovered facts to user_facts (single-pass ADD-only, mem0 v3
style). Every fact carries source_run_id so the user can always
trace "where did this fact come from".

Skip semantics:
- No LLM → skipped (extraction is the LLM's job; no deterministic path)
- Already-extracted runs are deduped via the WHERE clause in
  user_facts.extract_pending_runs

Per-run cap: ``MAX_PER_RUN`` keeps token spend bounded.
"""

from __future__ import annotations

import logging
from typing import Any

from ...user_facts import extract_pending_runs
from ..scheduler import JobContext, JobSpec

log = logging.getLogger(__name__)

MAX_PER_RUN = 30
"""Skill_runs to process per cron tick. With ~8 facts extracted per run
(LLM cap), 30 runs = ~240 fact candidates / day. Single-pass ADD-only
+ UNIQUE on fact_text means duplicates auto-skip."""


def run(ctx: JobContext) -> dict[str, Any]:
    if ctx.llm is None:
        log.info("extract_facts: LLM not configured, skipping")
        return {"skipped": "no_llm"}

    counters = extract_pending_runs(
        ctx.store, llm=ctx.llm, limit=MAX_PER_RUN,
    )

    if ctx.notifier and counters.get("inserted", 0) > 0:
        try:
            ctx.notifier.notify(
                title=f"OfferGuide: 长期记忆 +{counters['inserted']} 条事实",
                body=(
                    f"扫了 {counters['runs_scanned']} 个 SKILL 输出, "
                    f"抽出候选 {counters['candidates']}, "
                    f"新增 {counters['inserted']}, "
                    f"重复/无效 {counters['skipped_dup_or_invalid']}"
                ),
                level="info",
            )
        except Exception:
            log.warning("extract_facts: notify failed", exc_info=True)

    return counters


# Daily 02:00 — runs in the dead-of-night, after the day's SKILL calls
# have settled. Before discover_jobs (06:30) so morning view sees
# fresh memory.
EXTRACT_FACTS_JOB = JobSpec(
    name="extract_facts",
    func=run,
    trigger="cron",
    trigger_kwargs={"hour": 2, "minute": 0},
    misfire_grace_time_s=3600,
)
