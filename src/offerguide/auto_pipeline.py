"""自动化管线：spider → scout.ingest → 可选 score_match → inbox.

The vision in one paragraph: when a spider discovers a JD, the user shouldn't
have to do a thing. The JD goes into the database, gets a first-pass score
if there's enough text to score, and—if the score crosses a threshold—gets
pushed to the inbox so it surfaces in the daily-standup home tomorrow.
The user wakes up to "agent found 3 promising JDs overnight; review these
first" rather than "go run a spider, then score them, then decide".

This module is the glue. Three entry points:

- :func:`run_spider_sweep` — run one spider, ingest all candidates, optionally
  auto-score.  Used by the CLI + the daemon.
- :func:`run_all_spiders` — run every registered spider sequentially.
- :func:`evaluate_new_job` — score a single newly-ingested JD if it has
  enough text. Pulled out so callers (e.g. the Boss browser extension's
  ``/api/extension/ingest`` route) can opt into auto-eval too.

Threshold logic: we don't want to spend tokens on JDs with no real content.
Awesome-jobs entries have raw_text like "公司: 阿里巴巴\\n投递入口: ..." (~50
chars) — those skip auto-eval and become "已扫描" cards. Boss extension
entries have the full JD body (>500 chars usually) — those get scored
automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from . import inbox as inbox_mod
from .llm import LLMClient, LLMError
from .memory import Store
from .platforms import RawJob
from .skills import SkillRuntime, SkillSpec
from .spiders import Spider, SpiderError
from .workers import scout

log = logging.getLogger(__name__)

MIN_TEXT_FOR_AUTO_EVAL = 200
"""Minimum ``raw_text`` length before we spend tokens auto-scoring a JD.
Below this, the JD is metadata-only (company-level entry from a community
list) and not worth a SKILL run."""

INBOX_PROBABILITY_THRESHOLD = 0.55
"""Probability above which a JD auto-pushes to inbox as "考虑投递". Below
this, it's stored but not surfaced — the user can browse via the Pipeline
kanban or the "未跑评估" stat card."""


@dataclass
class SweepResult:
    """Aggregate of one spider-sweep + auto-eval pass."""
    spider_name: str
    candidates_found: int = 0
    """How many ``RawJob`` candidates the spider yielded."""
    new_jobs: list[int] = field(default_factory=list)
    """``jobs.id`` rows newly inserted (excludes dedup hits)."""
    duplicate_count: int = 0
    auto_evaluated: list[int] = field(default_factory=list)
    """``jobs.id`` rows that ran score_match this sweep."""
    pushed_to_inbox: list[int] = field(default_factory=list)
    """``inbox_items.id`` rows enqueued as 'consider_jd' (high-score auto-push)."""
    errors: list[str] = field(default_factory=list)
    spider_errors: list[str] = field(default_factory=list)
    """Soft errors from the spider itself (e.g. one page failed to parse)."""

    def summary(self) -> dict[str, Any]:
        return {
            "spider": self.spider_name,
            "candidates": self.candidates_found,
            "new": len(self.new_jobs),
            "dup": self.duplicate_count,
            "evaluated": len(self.auto_evaluated),
            "inboxed": len(self.pushed_to_inbox),
            "errors": len(self.errors) + len(self.spider_errors),
        }


def run_spider_sweep(
    spider: Spider,
    *,
    store: Store,
    runtime: SkillRuntime | None = None,
    skills: list[SkillSpec] | None = None,
    user_profile_text: str | None = None,
    max_items: int = 30,
    auto_eval: bool = True,
    inbox_threshold: float = INBOX_PROBABILITY_THRESHOLD,
) -> SweepResult:
    """Run one spider, ingest candidates, optionally auto-score.

    ``runtime`` + ``skills`` + ``user_profile_text`` are required for
    auto-eval; without them auto_eval falls back to False (no error,
    just skip the scoring step).

    The function is idempotent at the dedup boundary: re-running the
    same spider over an unchanged DB produces 0 ``new_jobs`` and 0
    ``auto_evaluated`` (we only score newly-inserted jobs).
    """
    result = SweepResult(spider_name=spider.name)

    # Step 1 — discover candidates
    try:
        sp_result = spider.run(max_items=max_items)
    except SpiderError as e:
        result.errors.append(f"spider crashed: {e}")
        return result
    result.candidates_found = len(sp_result.raw_jobs)
    result.spider_errors = list(sp_result.errors)

    # Step 2 — ingest each (dedup handled by scout)
    for rj in sp_result.raw_jobs:
        try:
            was_new, job_id = scout.ingest(store, rj)
        except Exception as e:
            result.errors.append(f"ingest error for {rj.title!r}: {e}")
            continue
        if was_new:
            result.new_jobs.append(job_id)
        else:
            result.duplicate_count += 1

    # Step 3 — auto-evaluate new JDs that have enough text
    can_eval = bool(
        auto_eval
        and runtime is not None
        and skills
        and user_profile_text
    )
    if not can_eval:
        return result

    score_spec = next(
        (s for s in (skills or []) if s.name == "score_match"),
        None,
    )
    if score_spec is None:
        return result

    for job_id in result.new_jobs:
        try:
            scored = evaluate_new_job(
                store=store,
                job_id=job_id,
                score_spec=score_spec,
                runtime=runtime,  # type: ignore[arg-type]
                user_profile_text=user_profile_text,  # type: ignore[arg-type]
            )
        except LLMError as e:
            result.errors.append(f"score_match LLM error on job {job_id}: {e}")
            continue
        except Exception as e:
            result.errors.append(f"score_match crash on job {job_id}: {e}")
            continue
        if scored is None:
            continue
        result.auto_evaluated.append(job_id)

        prob, score_run_id = scored
        if prob >= inbox_threshold:
            inbox_id = _push_to_inbox(
                store, job_id=job_id, prob=prob, score_run_id=score_run_id
            )
            if inbox_id is not None:
                result.pushed_to_inbox.append(inbox_id)

    return result


def run_all_spiders(
    spiders: list[Spider],
    *,
    store: Store,
    runtime: SkillRuntime | None = None,
    skills: list[SkillSpec] | None = None,
    user_profile_text: str | None = None,
    max_items_per_spider: int = 30,
    auto_eval: bool = True,
) -> list[SweepResult]:
    """Run a list of spiders sequentially. Returns one SweepResult per spider.

    A failure in one spider does not stop the others — each is wrapped.
    """
    out: list[SweepResult] = []
    for sp in spiders:
        out.append(
            run_spider_sweep(
                sp,
                store=store,
                runtime=runtime,
                skills=skills,
                user_profile_text=user_profile_text,
                max_items=max_items_per_spider,
                auto_eval=auto_eval,
            )
        )
    return out


def evaluate_new_job(
    *,
    store: Store,
    job_id: int,
    score_spec: SkillSpec,
    runtime: SkillRuntime,
    user_profile_text: str,
) -> tuple[float, int] | None:
    """Run score_match on a single job. Returns (probability, run_id) or None.

    Returns None when the JD's ``raw_text`` is shorter than
    :data:`MIN_TEXT_FOR_AUTO_EVAL` — we skip metadata-only entries
    (e.g. awesome_jobs company-level rows) to avoid wasting tokens.

    The returned probability lets the caller decide whether to push to
    inbox / notify / surface in dashboards.
    """
    with store.connect() as conn:
        row = conn.execute(
            "SELECT title, raw_text FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    raw_text = row[1] or ""
    if len(raw_text) < MIN_TEXT_FOR_AUTO_EVAL:
        return None

    skill_result = runtime.invoke(
        score_spec,
        {
            "job_text": raw_text,
            "user_profile": user_profile_text,
            "job_id": str(job_id),  # so analytics queries can pivot by job_id
        },
    )
    parsed = skill_result.parsed or {}
    prob = float(parsed.get("probability", 0.0))
    return prob, skill_result.skill_run_id


def _push_to_inbox(
    store: Store, *, job_id: int, prob: float, score_run_id: int
) -> int | None:
    """Enqueue a 'consider_jd' inbox item linking to the scored JD."""
    with store.connect() as conn:
        row = conn.execute(
            "SELECT title, company, url FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    if row is None:
        return None
    title, company, url = row[0] or "(无标题)", row[1] or "", row[2] or ""
    inbox_title = (
        f"考虑投递: {company} · {title}" if company else f"考虑投递: {title}"
    )
    body = (
        f"自动评分 {int(prob * 100)}%。"
        + (f"链接: {url}" if url else "")
    )
    item = inbox_mod.enqueue(
        store,
        kind="consider_jd",
        title=inbox_title[:200],
        body=body[:500],
        payload={
            "job_id": job_id,
            "score_run_id": score_run_id,
            "auto_pushed": True,
            "probability": prob,
        },
    )
    return item.id


# ─────────── helpers used by the CLI / daemon entry points ─────────


def build_default_spider_set() -> list[Spider]:
    """Default spider lineup the daemon runs daily.

    Today: just AwesomeJobsSpider. As more spiders graduate from
    experimental → reliable, they get added here.
    """
    from .spiders.awesome_jobs import AwesomeJobsSpider

    return [AwesomeJobsSpider()]


def build_runtime_from_settings(
    *,
    settings_obj: Any,
    store: Store,
) -> SkillRuntime | None:
    """Create a runtime from Settings, or None if no API key configured."""
    if not getattr(settings_obj, "deepseek_api_key", None):
        return None
    llm = LLMClient(
        api_key=settings_obj.deepseek_api_key,
        base_url=settings_obj.deepseek_base_url,
        default_model=settings_obj.default_model,
    )
    return SkillRuntime(llm=llm, store=store)


_ = RawJob  # keep import for downstream extension
