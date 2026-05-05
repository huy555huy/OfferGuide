"""Auto-score recently ingested JDs and auto-prepare a 投递包 for the
high-match ones, then inbox them as agent_suggestion.

Why a separate daemon (vs scoring inside the spider/extension paths):

- spider sweeps may add JDs faster than they can score (LLM-bound)
- the chrome extension ingests one JD at a time and shouldn't block on
  apply_assistant generation
- this daemon is the place where ALL un-scored JDs catch up — regardless
  of whether they came from spider, extension, or manual paste

What it actually does each tick:

1. Find jobs with raw_text >= MIN_TEXT and no score_match skill_run
2. score_match each (capped at MAX_PER_RUN to bound cost)
3. For score >= INBOX_THRESHOLD, also pre-generate the apply package
   (apply_assistant SKILL) — this is what makes the inbox suggestion
   "Intent Preview" rather than "TODO" — user sees an actual draft
4. enqueue agent_suggestion to inbox with full attribution (so user
   thumbs flow back to the right SKILL version for evolution)

Idempotent: re-running with same DB does nothing (already-scored jobs
get skipped via the skill_runs lookup).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ... import inbox as inbox_mod
from ...auto_pipeline import (
    INBOX_PROBABILITY_THRESHOLD,
    MIN_TEXT_FOR_AUTO_EVAL,
    evaluate_new_job,
)
from ..scheduler import JobContext

log = logging.getLogger(__name__)

MAX_PER_RUN = 5
"""Per-tick cap on score_match calls. With score_match averaging
~$0.01-0.02 per call (deepseek-v4-flash), 5/tick keeps daily spend
predictable. The agent maintenance tool can override via `limit=`."""


def run(ctx: JobContext, *, limit: int | None = None) -> dict[str, Any]:
    """Find un-scored JDs with enough text, score them, auto-suggest the high-match ones."""
    if ctx.llm is None:
        return {"skipped": "no_llm"}
    if ctx.runtime is None or not ctx.skills or not ctx.user_profile_text:
        # We can score without a profile if we really wanted to, but the
        # match probability is meaningless. Treat profile as a hard prereq.
        return {"skipped": "missing_runtime_or_skills_or_profile"}

    score_spec = next(
        (s for s in ctx.skills if s.name == "score_match"),
        None,
    )
    if score_spec is None:
        return {"skipped": "no_score_match_skill"}
    apply_spec = next(
        (s for s in ctx.skills if s.name == "apply_assistant"),
        None,
    )

    actual_limit = limit if limit is not None else MAX_PER_RUN

    # Find jobs with substantive text but no score_match run yet.
    # Heuristic for "no score_match run": no skill_runs row for this
    # skill_name with input_json containing the job_id.
    with ctx.store.connect() as conn:
        rows = conn.execute(
            "SELECT id FROM jobs "
            "WHERE length(raw_text) >= ? "
            "  AND id NOT IN ("
            "    SELECT CAST(json_extract(input_json, '$.job_id') AS INTEGER) "
            "    FROM skill_runs "
            "    WHERE skill_name = 'score_match' "
            "      AND json_extract(input_json, '$.job_id') IS NOT NULL"
            "  ) "
            "ORDER BY id DESC LIMIT ?",
            (MIN_TEXT_FOR_AUTO_EVAL, actual_limit),
        ).fetchall()
    job_ids = [r[0] for r in rows]

    scored = 0
    auto_suggested = 0
    failures: list[str] = []
    suggestion_ids: list[int] = []

    for job_id in job_ids:
        try:
            res = evaluate_new_job(
                store=ctx.store,
                job_id=job_id,
                score_spec=score_spec,
                runtime=ctx.runtime,
                user_profile_text=ctx.user_profile_text,
            )
        except Exception as e:
            failures.append(f"score job#{job_id}: {type(e).__name__}: {e}")
            continue
        if res is None:
            # raw_text < MIN_TEXT (defensive — query already filters)
            continue
        prob, score_run_id = res
        scored += 1

        if prob < INBOX_PROBABILITY_THRESHOLD:
            continue

        # High-match: pre-generate the apply package + inbox a real
        # Intent Preview suggestion. Pre-generation is what differentiates
        # "agent telling you to do X" from "agent already did X, you
        # just confirm". Failure to pre-generate is non-fatal — we still
        # inbox the suggestion, just without an attached draft.
        apply_run_id: int | None = None
        if apply_spec is not None:
            try:
                with ctx.store.connect() as conn:
                    job_row = conn.execute(
                        "SELECT raw_text, company, title FROM jobs WHERE id = ?",
                        (job_id,),
                    ).fetchone()
                if job_row is not None:
                    pkg_result = ctx.runtime.invoke(
                        apply_spec,
                        {
                            "job_text": job_row[0],
                            "user_profile": ctx.user_profile_text,
                            "company": job_row[1] or "",
                            "title": job_row[2] or "",
                            "job_id": str(job_id),
                        },
                        strict_inputs=False,  # apply_assistant has more inputs than what we pass
                    )
                    apply_run_id = pkg_result.skill_run_id
            except Exception as e:
                # Pre-gen failure is logged but not fatal
                log.warning("pre-gen apply pkg for job#%s failed: %s", job_id, e)

        # enqueue Intent Preview suggestion
        try:
            inbox_id = _enqueue_intent_preview(
                store=ctx.store,
                job_id=job_id,
                probability=prob,
                score_run_id=score_run_id,
                apply_run_id=apply_run_id,
                score_skill_version=score_spec.version,
            )
            if inbox_id is not None:
                auto_suggested += 1
                suggestion_ids.append(inbox_id)
        except Exception as e:
            failures.append(f"inbox enqueue job#{job_id}: {e}")

    summary = {
        "candidates": len(job_ids),
        "scored": scored,
        "auto_suggested": auto_suggested,
        "suggestion_ids": suggestion_ids,
        "failures": failures,
    }
    if ctx.notifier and auto_suggested > 0:
        try:
            ctx.notifier.notify(
                title=f"OfferGuide: 找到 {auto_suggested} 个高匹配岗位",
                body=(
                    f"自动 score 了 {scored} 个新 JD; "
                    f"{auto_suggested} 个 ≥ 阈值已推到 inbox 等你拍板。"
                ),
                level="info",
            )
        except Exception:
            log.warning("auto_score: notify failed", exc_info=True)
    return summary


def _enqueue_intent_preview(
    *,
    store,
    job_id: int,
    probability: float,
    score_run_id: int,
    apply_run_id: int | None,
    score_skill_version: str,
) -> int | None:
    """Enqueue an agent_suggestion inbox item with the Intent Preview format.

    Body is structured 'because X (concrete evidence) → so I built Y' so
    the user can verify the agent's reasoning at a glance. proposed_action
    points at /apply/<job_id> so 1-click approve takes the user straight
    to the (already-generated) draft.
    """
    with store.connect() as conn:
        job_row = conn.execute(
            "SELECT title, company, url, raw_text FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if job_row is None:
        return None
    title = (job_row[0] or "未知岗位")[:80]
    company = (job_row[1] or "未知公司")[:80]
    url = job_row[2] or ""
    jd_excerpt = (job_row[3] or "")[:300]

    # Pull score reasoning if available so the suggestion explains itself
    reasoning_excerpt = ""
    try:
        with store.connect() as conn:
            sr_row = conn.execute(
                "SELECT output_json FROM skill_runs WHERE id = ?", (score_run_id,),
            ).fetchone()
        if sr_row and sr_row[0]:
            sr_data = json.loads(sr_row[0])
            reasoning_excerpt = (sr_data.get("reasoning") or "")[:400]
    except Exception:
        pass

    inbox_title = f"考虑投: {company} · {title} (匹配 {int(probability * 100)}%)"
    body_parts = [
        f"**为什么推荐**: 自动 score_match 算出 {int(probability * 100)}% 匹配。",
    ]
    if reasoning_excerpt:
        body_parts.append(f"\n**判断依据**: {reasoning_excerpt}")
    if apply_run_id is not None:
        body_parts.append(
            f"\n**已为你准备好**: Boss 直聊话术 / Q&A 模板 / 投递时机 / "
            f"pre-submit checklist (apply_assistant 已跑, run #{apply_run_id})。"
            f"批准后跳到 /apply/{job_id} 直接审 + 复制 + 投。",
        )
    else:
        body_parts.append(
            f"\n**接下来**: 批准后跳到 /apply/{job_id} 生成投递包。",
        )
    if jd_excerpt:
        body_parts.append(f"\n\n---\n**JD 节选**: {jd_excerpt}...")
    if url:
        body_parts.append(f"\n\n[原 JD 链接]({url})")

    item = inbox_mod.enqueue_agent_suggestion(
        store,
        title=inbox_title,
        body="".join(body_parts)[:2000],
        source_skill_name="score_match",
        source_skill_version=score_skill_version,
        source_skill_run_id=score_run_id,
        proposed_action={
            "tool": "open_apply",
            "args": {"job_id": job_id, "apply_run_id": apply_run_id},
        },
        payload={
            "job_id": job_id,
            "probability": probability,
            "apply_run_id": apply_run_id,
        },
    )
    return item.id
