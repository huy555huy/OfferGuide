"""Evaluation tools — the Evaluation sub-agent's hands.

Wraps 6 high-value SKILLs as registry tools so the sub-agent can call
them by name:
- score_job (score_match SKILL)
- generate_apply_pack (apply_assistant SKILL)
- generate_interview_prep (prepare_interview SKILL)
- tailor_resume (tailor_resume SKILL)
- find_resume_gaps (analyze_gaps SKILL)
- compare_jobs (compare_jobs SKILL)

Plus 2 read tools for context:
- read_job (load JD + extras)
- read_my_recommendations (currently scored/unscored jobs)

The 6 SKILL prompts stay 100% in `src/offerguide/skills/<name>/SKILL.md`.
Tool handlers just unpack args → load context → invoke SkillRuntime →
return result.parsed as JSON. No SKILL prompt change. No new LLM call
pattern.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .registry import registry, tool_error, tool_result

log = logging.getLogger(__name__)


def _load_job(store, job_id: int) -> dict[str, Any] | None:
    """Load 1 job row → flat dict."""
    with store.connect() as conn:
        row = conn.execute(
            "SELECT id, source, source_id, url, title, company, location, "
            "       raw_text, extras_json "
            "FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": int(row[0]), "source": row[1] or "",
        "source_id": row[2] or "", "url": row[3] or "",
        "title": row[4] or "", "company": row[5] or "",
        "location": row[6] or "", "raw_text": row[7] or "",
        "extras_json": row[8] or "{}",
    }


def _format_jd(job: dict[str, Any]) -> str:
    """Render JD as compact text for SKILL input."""
    parts = [
        f"# {job['title']}",
        f"公司: {job['company']}" if job["company"] else "",
        f"地点: {job['location']}" if job["location"] else "",
        f"链接: {job['url']}" if job["url"] else "",
        "",
        job["raw_text"][:6000],
    ]
    return "\n".join(p for p in parts if p)


def _find_skill(skills, name):
    for s in skills:
        if s.name == name:
            return s
    return None


def _invoke_skill(*, name, inputs, store, runtime, skills):
    """Run a SKILL via runtime + write follow_through-eligible harness_event.

    Wraps the existing SkillRuntime to:
    1. Return parsed JSON as tool_result
    2. Write {kind: 'skill_run_for_job', job_id, skill_run_id} harness_event
       so app_outcome later attributes to this SKILL
    """
    if runtime is None:
        return tool_error("internal: runtime not provided (LLM key missing?)")
    spec = _find_skill(skills, name)
    if spec is None:
        return tool_error(f"SKILL '{name}' not registered")
    try:
        result = runtime.invoke(spec, inputs)
    except ValueError as e:
        return tool_error(f"{name}: invalid inputs: {e}")
    except Exception as e:
        return tool_error(f"{name}: invoke raised: {type(e).__name__}: {e}")
    if result.parsed is None:
        return tool_error(
            f"{name}: SKILL returned non-JSON",
            raw=(result.raw_text or "")[:300],
            skill_run_id=result.skill_run_id,
        )
    return tool_result({
        **result.parsed,
        "_meta": {
            "skill_run_id": result.skill_run_id,
            "cost_usd": result.cost_usd,
            "skill_version": result.skill_version,
        },
    })


def _link_skill_to_job(store, *, job_id, skill_run_id, skill_name, skill_version):
    """Write harness_event linking this SKILL run to a job_id. Mirrors
    the W20.5 logic that lets W20.5 multi-SKILL app_outcome attribution
    find which SKILLs touched a given job."""
    try:
        from ..agent_runtime import _schema as _hs
        _hs.init_agent_runtime_schema(store)
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO harness_events(kind, job_id, note, source) "
                "VALUES (?, ?, ?, ?)",
                (
                    f"{skill_name}_generated", job_id,
                    json.dumps({
                        "skill_run_id": skill_run_id,
                        "skill_name": skill_name,
                        "skill_version": skill_version,
                    }, ensure_ascii=False),
                    "evaluation_tool",
                ),
            )
    except Exception:
        log.debug("link_skill_to_job failed", exc_info=True)


# ── score_job ─────────────────────────────────────────────────────


def _score_job_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    runtime = rt.get("runtime")
    skills = rt.get("skills") or []
    user_profile_text = rt.get("user_profile_text") or ""
    if not store:
        return tool_error("internal: store missing")
    job_id = args.get("job_id")
    if not isinstance(job_id, int):
        return tool_error("score_job needs integer job_id")
    job = _load_job(store, job_id)
    if job is None:
        return tool_error(f"job#{job_id} not found")
    if not user_profile_text:
        return tool_error("user_profile_text missing — agent needs profile loaded")

    out = _invoke_skill(
        name="score_match",
        inputs={
            "job_text": _format_jd(job)[:4000],
            "user_profile": user_profile_text[:4000],
        },
        store=store, runtime=runtime, skills=skills,
    )
    # Parse to extract skill_run_id for the harness_event link
    try:
        parsed = json.loads(out)
        if "_meta" in parsed and parsed["_meta"].get("skill_run_id"):
            _link_skill_to_job(
                store, job_id=job_id,
                skill_run_id=parsed["_meta"]["skill_run_id"],
                skill_name="score_match",
                skill_version=parsed["_meta"]["skill_version"],
            )
            # Also write the W15.22-style 'scored' harness_event so
            # /recommended view can rank by this score.
            try:
                with store.connect() as conn:
                    conn.execute(
                        "INSERT INTO harness_events(kind, job_id, note, source) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            "scored", job_id,
                            json.dumps({
                                "probability": parsed.get("probability"),
                                "skill_run_id": parsed["_meta"]["skill_run_id"],
                                "deal_breakers": parsed.get("deal_breakers") or [],
                            }, ensure_ascii=False),
                            "score_job_tool",
                        ),
                    )
            except Exception:
                pass
    except json.JSONDecodeError:
        pass
    return out


registry.register(
    name="score_job",
    group="evaluation",
    schema={
        "name": "score_job",
        "description": (
            "对 1 个 job 跑 score_match SKILL, 返回 calibrated probability "
            "(0..1) + reasoning + dimensions + deal_breakers. 写到 "
            "harness_events 'scored' 让 /recommended 用. "
            "**1 个 LLM call ~10s ~$0.0003**."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer"},
            },
            "required": ["job_id"],
        },
    },
    handler=_score_job_handler,
    deprecated_aliases=("score_match",),
)


# ── generate_apply_pack ───────────────────────────────────────────


def _generate_apply_pack_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    runtime = rt.get("runtime")
    skills = rt.get("skills") or []
    user_profile_text = rt.get("user_profile_text") or ""
    if not store:
        return tool_error("internal: store missing")
    job_id = args.get("job_id")
    if not isinstance(job_id, int):
        return tool_error("needs integer job_id")
    job = _load_job(store, job_id)
    if job is None:
        return tool_error(f"job#{job_id} not found")
    if not user_profile_text:
        return tool_error("user_profile_text missing")

    out = _invoke_skill(
        name="apply_assistant",
        inputs={
            "company": job["company"],
            "role_focus": job["title"],
            "job_text": _format_jd(job)[:5000],
            "user_profile": user_profile_text[:5000],
        },
        store=store, runtime=runtime, skills=skills,
    )
    # Link for downstream app_outcome
    try:
        parsed = json.loads(out)
        if "_meta" in parsed and parsed["_meta"].get("skill_run_id"):
            _link_skill_to_job(
                store, job_id=job_id,
                skill_run_id=parsed["_meta"]["skill_run_id"],
                skill_name="apply_assistant",
                skill_version=parsed["_meta"]["skill_version"],
            )
    except json.JSONDecodeError:
        pass
    return out


registry.register(
    name="generate_apply_pack",
    group="evaluation",
    schema={
        "name": "generate_apply_pack",
        "description": (
            "对 1 个 job 跑 apply_assistant SKILL → 自我介绍话术 + 网申 QA "
            "模板 + 投递策略 + pre-submit checklist + skip_reasons. "
            "用户用这套自己 copy 去投. **~$0.0003 / call**."
        ),
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
    },
    handler=_generate_apply_pack_handler,
    deprecated_aliases=("apply_assistant",),
)


# ── generate_interview_prep ───────────────────────────────────────


def _generate_interview_prep_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    runtime = rt.get("runtime")
    skills = rt.get("skills") or []
    user_profile_text = rt.get("user_profile_text") or ""
    if not store:
        return tool_error("internal: store missing")
    job_id = args.get("job_id")
    if not isinstance(job_id, int):
        return tool_error("needs integer job_id")
    job = _load_job(store, job_id)
    if job is None:
        return tool_error(f"job#{job_id} not found")
    if not user_profile_text:
        return tool_error("user_profile_text missing")
    try:
        from .. import project_vault as _pv
        user_profile_text = _pv.append_to_profile_text(
            store, user_profile_text, max_project_chars=3500,
        )
    except Exception:
        pass

    # Pull past 面经 if any
    past_experiences_text = ""
    try:
        from .. import interview_corpus
        experiences = interview_corpus.fetch_for_company(
            store, job["company"], limit=8,
        )
        if experiences:
            past_experiences_text = interview_corpus.render_snippets(
                experiences, max_chars=4000,
            )
    except Exception:
        pass

    out = _invoke_skill(
        name="prepare_interview",
        inputs={
            "company": job["company"],
            "job_text": _format_jd(job)[:5000],
            "user_profile": user_profile_text[:8500],
            "past_experiences": past_experiences_text or "(无过往面经)",
        },
        store=store, runtime=runtime, skills=skills,
    )
    try:
        parsed = json.loads(out)
        if "_meta" in parsed and parsed["_meta"].get("skill_run_id"):
            _link_skill_to_job(
                store, job_id=job_id,
                skill_run_id=parsed["_meta"]["skill_run_id"],
                skill_name="prepare_interview",
                skill_version=parsed["_meta"]["skill_version"],
            )
    except json.JSONDecodeError:
        pass
    return out


registry.register(
    name="generate_interview_prep",
    group="evaluation",
    schema={
        "name": "generate_interview_prep",
        "description": (
            "对 1 个 job 跑 prepare_interview SKILL → 公司画像 + 高频题 "
            "(校准过的 likelihood) + 备战重点 + 弱点. 自动拉 interview "
            "corpus 同公司面经. **~$0.0003 / call**."
        ),
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
    },
    handler=_generate_interview_prep_handler,
    deprecated_aliases=("prepare_interview",),
)


# ── tailor_resume ────────────────────────────────────────────────


def _tailor_resume_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    runtime = rt.get("runtime")
    skills = rt.get("skills") or []
    user_profile_text = rt.get("user_profile_text") or ""
    if not store:
        return tool_error("internal: store missing")
    job_id = args.get("job_id")
    if not isinstance(job_id, int):
        return tool_error("needs integer job_id")
    job = _load_job(store, job_id)
    if job is None:
        return tool_error(f"job#{job_id} not found")
    if not user_profile_text:
        return tool_error("user_profile_text missing")
    try:
        from .. import project_vault as _pv
        user_profile_text = _pv.append_to_profile_text(
            store, user_profile_text, max_project_chars=3500,
        )
    except Exception:
        pass

    out = _invoke_skill(
        name="tailor_resume",
        inputs={
            "master_resume": user_profile_text[:9000],
            "job_text": _format_jd(job)[:4000],
            "company": job["company"],
            "successful_profile_json": "{}",
        },
        store=store, runtime=runtime, skills=skills,
    )
    try:
        parsed = json.loads(out)
        if "_meta" in parsed and parsed["_meta"].get("skill_run_id"):
            _link_skill_to_job(
                store, job_id=job_id,
                skill_run_id=parsed["_meta"]["skill_run_id"],
                skill_name="tailor_resume",
                skill_version=parsed["_meta"]["skill_version"],
            )
    except json.JSONDecodeError:
        pass
    return out


registry.register(
    name="tailor_resume",
    group="evaluation",
    schema={
        "name": "tailor_resume",
        "description": (
            "对 1 个 job 跑 tailor_resume SKILL → 针对该 JD 微调过的简历 "
            "内容 (重新组织项目顺序 / 突出相关关键词 / 调整语气). "
            "不动 master_resume 原文."
        ),
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
    },
    handler=_tailor_resume_handler,
)


# ── find_resume_gaps ─────────────────────────────────────────────


def _find_resume_gaps_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    runtime = rt.get("runtime")
    skills = rt.get("skills") or []
    user_profile_text = rt.get("user_profile_text") or ""
    if not store:
        return tool_error("internal: store missing")
    job_id = args.get("job_id")
    if not isinstance(job_id, int):
        return tool_error("needs integer job_id")
    job = _load_job(store, job_id)
    if job is None:
        return tool_error(f"job#{job_id} not found")
    if not user_profile_text:
        return tool_error("user_profile_text missing")

    return _invoke_skill(
        name="analyze_gaps",
        inputs={
            "job_text": _format_jd(job)[:4000],
            "user_profile": user_profile_text[:4000],
        },
        store=store, runtime=runtime, skills=skills,
    )


registry.register(
    name="find_resume_gaps",
    group="evaluation",
    schema={
        "name": "find_resume_gaps",
        "description": (
            "对 1 个 job 跑 analyze_gaps SKILL → 简历相对 JD 的弱点清单 "
            "+ 短期可补 vs 长期需积累的区分. 给用户决定是否投."
        ),
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
    },
    handler=_find_resume_gaps_handler,
)


# ── compare_jobs ─────────────────────────────────────────────────


def _compare_jobs_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    runtime = rt.get("runtime")
    skills = rt.get("skills") or []
    user_profile_text = rt.get("user_profile_text") or ""
    if not store:
        return tool_error("internal: store missing")
    job_ids = args.get("job_ids") or []
    if not isinstance(job_ids, list) or len(job_ids) < 2:
        return tool_error("compare_jobs needs ≥2 job_ids in a list")

    jobs_text: list[str] = []
    for jid in job_ids[:5]:  # cap 5 jobs (token budget)
        job = _load_job(store, int(jid))
        if job is None:
            continue
        jobs_text.append(
            f"## job#{jid}: {job['company']} · {job['title']}\n"
            + _format_jd(job)[:1500]
        )

    if len(jobs_text) < 2:
        return tool_error("need ≥2 valid jobs to compare")

    return _invoke_skill(
        name="compare_jobs",
        inputs={
            "jobs_text": "\n\n---\n\n".join(jobs_text),
            "user_profile": user_profile_text[:4000],
        },
        store=store, runtime=runtime, skills=skills,
    )


registry.register(
    name="compare_jobs",
    group="evaluation",
    schema={
        "name": "compare_jobs",
        "description": (
            "对 2-5 个 job 跑 compare_jobs SKILL → 多岗对比表 "
            "(优劣 / 投递性价比 / 建议先后顺序). 用户挑岗辅助决策."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 5,
                },
            },
            "required": ["job_ids"],
        },
    },
    handler=_compare_jobs_handler,
)


# ── read_job (shared, useful for any agent that needs JD details) ──


def _read_job_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    if not store:
        return tool_error("internal: store missing")
    job_id = args.get("job_id")
    if not isinstance(job_id, int):
        return tool_error("needs integer job_id")
    job = _load_job(store, job_id)
    if job is None:
        return tool_error(f"job#{job_id} not found")
    # Don't pass internal columns
    return tool_result(job)


registry.register(
    name="read_job",
    group="shared",
    schema={
        "name": "read_job",
        "description": (
            "Load 1 job row by id → {id, source, url, title, company, "
            "location, raw_text, extras_json}. 任何 agent 看 JD 详情用."
        ),
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
    },
    handler=_read_job_handler,
)
