"""W13.1 evolution.evolve — generate variant prompts via LLM.

This is the "creative" step of the evolution loop: given a SKILL whose
fitness has dropped below threshold, ask the LLM to write N improved
versions of its prompt. Each variant lands in ``skill_variants`` as
``status='shadow'``, ready for the gray-release pipeline (Step 8) to
promote one to canary, then to live (or fail it).

Why this is invoked as a tool by the agent (not a cron):
- The agent already has full system context when it decides to evolve
- The agent can write a richer "what's wrong with the current prompt"
  message to the variant generator than a fixed cron job could
- The agent surfaces evolution decisions in its trajectory, so the user
  can audit them in the /agent UI

Pipeline:

    evolve_skill(skill_name, num_variants=3) [tool_call]
        ↓
    1. Resolve current body (live variant > seed disk SKILL.md)
    2. Pull recent low-scoring skill_runs for context
    3. LLM call: "here's the prompt + 3 examples where it underperformed.
                  give me 3 better versions"
    4. Insert each generated variant into skill_variants (status='shadow')
    5. Return summary with version numbers + brief diff
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..llm import LLMClient
from ..memory import Store
from ..skills import SkillSpec, discover_skills
from . import registry
from .fitness import compute_fitness

log = logging.getLogger(__name__)


SKILLS_ROOT = Path(__file__).parent.parent / "skills"


_VARIANT_GEN_SYSTEM = """你是 OfferGuide 的 SKILL 进化引擎。你看一个表现欠佳的 SKILL prompt + 用户的真实反馈信号 + 几条最近的 skill_runs (输入 + 当时的输出)，生成 N 个改进版本的 prompt 候选。

# 你能做的（鼓励）
- 让 prompt 更明确、约束更紧（"严禁编造"、"必须 JSON"、"长度 ≤ X 字"）
- 调整 output_schema 的字段优先级（把最重要的字段放前面）
- 加入 chain-of-thought 提示（"先分 3 步思考再输出"）
- 删冗余 / 简化语言（让 prompt 更短能降低 token 成本同时提高聚焦）
- 加入 few-shot 例子（如果原 prompt 没有）

# 你严禁做的
- 改 SKILL 的核心目的 / 输出 schema 的字段名（会破坏下游消费者）
- 引入 master_resume / job_text 之外的虚构上下文
- 长度爆炸（新 prompt ≤ 原 prompt 的 1.5 倍长度）

# 输出（严格 JSON, 不要 markdown 代码块）

{
  "variants": [
    {
      "version_label": <str, 一句话描述这个变种的核心改动, ≤ 30 字>,
      "body": <str, 完整的新 prompt body 文本>,
      "rationale": <str, 一段话: 我改了什么 + 为什么 + 解决哪个观察到的问题>
    },
    ...
  ]
}

每个 variant 必须真的不同（不能换几个标点就交差）。
"""


@dataclass
class EvolveResult:
    """Summary of one evolve_skill invocation — returned to the agent."""
    skill_name: str
    parent_version: str
    candidates_generated: int
    candidates_persisted: int
    variant_versions: list[str]
    notes: str


def evolve_skill(
    *,
    store: Store,
    llm: LLMClient,
    skill_name: str,
    num_variants: int = 3,
    num_eval_examples: int = 5,
) -> EvolveResult:
    """Generate ``num_variants`` candidate prompt variants for ``skill_name``.

    Each variant is persisted as a ``shadow`` row in ``skill_variants``,
    waiting for the gray-release pipeline to promote one. This function
    does NOT run the variants on real inputs (synthetic eval is left to
    the canary-rollout pipeline in Step 8 — it'd double the LLM cost to
    eval here AND in canary).

    Returns an EvolveResult summary; the agent surfaces this in its
    final answer + trajectory.
    """
    # 1. Resolve current body
    seed_spec = _load_seed_spec(skill_name)
    if seed_spec is None:
        return EvolveResult(
            skill_name=skill_name, parent_version="?",
            candidates_generated=0, candidates_persisted=0,
            variant_versions=[],
            notes=f"ERROR: SKILL '{skill_name}' not found in skills/ directory",
        )

    live = registry.get_live_variant(store, skill_name)
    if live:
        parent_body = live.body_md
        parent_version = live.version
    else:
        parent_body = seed_spec.body
        parent_version = seed_spec.version

    # 2. Pull eval context (recent low-scoring signals + their associated runs)
    eval_context = _build_eval_context(
        store, skill_name=skill_name,
        skill_version=parent_version,
        n_examples=num_eval_examples,
    )

    # 3. LLM call to generate variants
    user_msg = (
        f"## SKILL 名: {skill_name}\n"
        f"## 当前版本: {parent_version}\n"
        f"## 当前 fitness: {eval_context['fitness']}\n\n"
        f"## 当前 prompt body\n```\n{parent_body[:4000]}\n```\n\n"
        f"## 最近表现欠佳的 N 条调用（输入摘要 + 评分）\n"
        f"{eval_context['examples_text']}\n\n"
        f"## 任务\n"
        f"生成 {num_variants} 个改进版本。每个版本必须实质不同。"
    )
    try:
        from ..llm.client import _parse_tool_arguments
        resp = llm.chat(
            messages=[
                {"role": "system", "content": _VARIANT_GEN_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,  # higher temp = more variety in generated variants
            json_mode=True,
        )
        data = _parse_tool_arguments(resp.content)
    except Exception as e:
        log.exception("evolve_skill LLM call failed")
        return EvolveResult(
            skill_name=skill_name, parent_version=parent_version,
            candidates_generated=0, candidates_persisted=0,
            variant_versions=[],
            notes=f"ERROR: variant generation LLM failed: {e}",
        )

    raw_variants = (data or {}).get("variants", []) if isinstance(data, dict) else []
    if not raw_variants or not isinstance(raw_variants, list):
        return EvolveResult(
            skill_name=skill_name, parent_version=parent_version,
            candidates_generated=0, candidates_persisted=0,
            variant_versions=[],
            notes=f"ERROR: LLM did not return variants list (got: {str(data)[:200]})",
        )

    # 4. Persist each as shadow variant
    persisted: list[str] = []
    for i, v in enumerate(raw_variants[:num_variants]):
        if not isinstance(v, dict):
            continue
        body = (v.get("body") or "").strip()
        if not body:
            continue
        # Don't persist no-op variants (essentially equal to parent)
        if _is_essentially_same(body, parent_body):
            continue
        new_version = registry.bump_version(parent_version, suffix=f"shadow-{i}")
        notes_md = (
            f"label: {v.get('version_label', '?')}\n"
            f"rationale: {v.get('rationale', '?')[:500]}"
        )
        sid = registry.insert_shadow_variant(
            store,
            skill_name=skill_name, version=new_version,
            parent_version=parent_version,
            body_md=body, notes=notes_md,
        )
        if sid:
            persisted.append(new_version)

    return EvolveResult(
        skill_name=skill_name,
        parent_version=parent_version,
        candidates_generated=len(raw_variants),
        candidates_persisted=len(persisted),
        variant_versions=persisted,
        notes=(
            f"生成 {len(raw_variants)} 个候选, 持久化 {len(persisted)} 个 shadow 变种。"
            f"可用 promote_to_canary 将其灰度放量。"
            if persisted
            else "所有候选都跟父版本基本相同, 没有持久化任何变种。"
        ),
    )


# ─────────────────────── helpers ───────────────────────


def _load_seed_spec(skill_name: str) -> SkillSpec | None:
    """Find the seed (on-disk) SkillSpec by name."""
    for spec in discover_skills(SKILLS_ROOT):
        if spec.name == skill_name:
            return spec
    return None


def _build_eval_context(
    store: Store, *, skill_name: str, skill_version: str, n_examples: int,
) -> dict:
    """Pull fitness + a few recent input/output examples for the LLM context.

    We deliberately surface the WORST examples (lowest signal_value) so the
    variant generator sees the failure mode it should fix.
    """
    fit = compute_fitness(store, skill_name=skill_name, skill_version=skill_version)
    fit_str = f"{fit.fitness:.2f}" if fit.fitness is not None else "(no signal yet)"

    # Pull worst-N skill_runs by joining with evolution_signals
    examples_lines: list[str] = []
    try:
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT s.skill_run_id, s.signal_value, s.notes, "
                "       r.input_json, r.output_json "
                "FROM evolution_signals s "
                "LEFT JOIN skill_runs r ON r.id = s.skill_run_id "
                "WHERE s.skill_name = ? AND s.skill_version = ? "
                "  AND s.signal_kind IN ('critic', 'user_thumbs') "
                "  AND s.skill_run_id IS NOT NULL "
                "ORDER BY s.signal_value ASC LIMIT ?",
                (skill_name, skill_version, n_examples),
            ).fetchall()
    except Exception as e:
        log.warning("eval_context query failed: %s", e)
        rows = []

    if not rows:
        examples_lines.append("(无历史 skill_runs 关联到 signals)")
    for run_id, value, _notes, input_json, output_json in rows:
        try:
            inputs = json.loads(input_json or "{}")
        except json.JSONDecodeError:
            inputs = {}
        # Clip aggressively for prompt budget
        input_summary = ", ".join(
            f"{k}={str(v)[:80]}…" if len(str(v)) > 80 else f"{k}={v}"
            for k, v in inputs.items()
        )[:400]
        output_preview = (output_json or "")[:400].replace("\n", " ")
        examples_lines.append(
            f"- run#{run_id} (signal_value={value:.2f}): "
            f"input={input_summary} | output={output_preview}…"
        )

    return {
        "fitness": fit_str,
        "fitness_report": fit,
        "examples_text": "\n".join(examples_lines),
    }


def _is_essentially_same(a: str, b: str) -> bool:
    """Cheap diff check — variants whose body matches parent's are dropped."""
    a_norm = "".join(a.split())[:2000]
    b_norm = "".join(b.split())[:2000]
    if a_norm == b_norm:
        return True
    # Length proxy: if difference is < 5%, skip
    if a_norm and b_norm and abs(len(a_norm) - len(b_norm)) / max(len(a_norm), len(b_norm)) < 0.05:
        # Could still be reordered — do a quick char-set diff
        diff = sum(1 for x, y in zip(a_norm, b_norm, strict=False) if x != y)
        if diff / max(1, len(a_norm)) < 0.05:
            return True
    return False
