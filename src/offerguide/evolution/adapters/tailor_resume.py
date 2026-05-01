"""``tailor_resume`` SKILL adapter — anti-fabrication metric.

This SKILL's value is *not* in the tailored markdown looking pretty —
it's in **never editorial-overstepping into fabrication**. The metric
weights heavily toward catching fabrication, change_log specificity,
and ATS keyword grounding (each claimed keyword must substring-match
the tailored markdown).

Metric axes:

- ``schema``                  — output validates as TailorResumeResult
- ``no_fabrication``          — no master_resume-absent claims appear in tailored
                                (verified by string-overlap heuristic)
- ``change_log_specific``     — every change_log entry has rationale referencing
                                JD line / profile field / ATS keyword
- ``ats_keywords_grounded``   — every ats_keywords_used substring appears in tailored
- ``honest_missing_keywords`` — ats_keywords_missing non-empty when JD has obvious
                                requirements not in master_resume (anti-overpromise)
- ``self_audit_active``       — cannot_fake_warnings non-empty on borderline cases
                                (signals the LLM is reasoning about temptation)

Pre-dogfood the metric is structural — once we have user-acceptance feedback on
each change_log entry, we can replace ``change_log_specific`` weight with
``user_acceptance_rate``.
"""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from ...skills.tailor_resume.helpers import TailorResumeResult
from ._base import MetricBreakdown, parse_json_output

name: str = "tailor_resume"
INPUT_NAMES: list[str] = [
    "master_resume", "job_text", "company", "successful_profile_json",
]
METRIC_AXES: list[str] = [
    "total",
    "schema",
    "no_fabrication",
    "change_log_specific",
    "ats_keywords_grounded",
    "honest_missing_keywords",
    "self_audit_active",
]

_W_SCHEMA = 0.15
_W_NO_FAB = 0.30  # heaviest — fabrication is the existential risk
_W_CHANGELOG = 0.15
_W_ATS_GROUNDED = 0.15
_W_HONEST_MISSING = 0.10
_W_SELF_AUDIT = 0.15


@dataclass(frozen=True)
class TailorResumeExample:
    name: str
    master_resume: str
    job_text: str
    company: str
    successful_profile_json: str
    band: str = "real"
    notes: str = ""


# ── examples ───────────────────────────────────────────────────────


_REAL_MASTER_RESUME = """\
# 胡阳

上海财经大学 应用统计 专硕 (2027 届毕业)

## 实习经历
- **法至科技** (2025/3 至今, NLP 工程师)
  - 用 LangChain + LangGraph 搭建多 agent 评测系统, 引入 GAIA benchmark
  - RAG retrieval 失败 OOD 用例做 query rewriting, hit@5 从 0.61 提升到 0.74

## 项目经历
- **RemeDi** — 基于 BERT 的医疗文本双流推荐, AUC 提升 0.04
- **Deep Research Agent** — 基于 LangGraph + DSPy 的开源研究助手

## 技能
Python, PyTorch, LangGraph, DSPy, BGE, Pydantic
"""


_BYTEDANCE_AGENT_JD = """\
字节跳动 Seed - AI Agent 后端实习

【职责】
1. 参与下一代 LLM Agent 系统设计与实现
2. 用 LangGraph / DSPy 等框架搭建复杂 agent workflow
3. 设计高 QPS agent inference pipeline
4. 评估 + 优化 agent 在 GAIA / SWE-bench 等 benchmark 上的表现

【要求】
- Python + PyTorch / TensorFlow 熟练
- 了解 Transformer / Attention / RAG / Agent 框架原理
- 熟悉 LangGraph / LangChain / DSPy 任一
- 算法基础（leetcode hot100 级别）
- 加分: 有开源 agent 项目经验 / GRPO / Speculative Decoding

【地点】上海 / 北京
"""


_BYTEDANCE_PROFILE = _json.dumps({
    "company": "字节跳动",
    "role_focus": "AI Agent 后端实习",
    "evidence_count": 3,
    "evidence_kinds": ["offer_post", "interview", "project_share"],
    "background_pattern": {
        "education_level": "硕士占多数", "school_tier": "985 头部",
        "majors": ["计算机"], "internships": ["美团 / 头部 AI 实验室"],
        "competitions": [], "publications": [],
    },
    "skill_pattern": {
        "must_have": ["LangGraph", "RAG", "Pydantic", "leetcode hot100"],
        "highly_valued": ["DSPy", "GAIA benchmark"],
        "differentiators": ["开源 agent 评测项目"],
    },
    "project_pattern": {
        "typical_project_themes": ["LLM agent 评测"],
        "common_tech_stacks": ["LangGraph", "BGE"],
        "scale_signals": ["200+ 测试用例"], "outcome_signals": [],
    },
    "interview_pattern": {
        "common_questions": [
            {"question": "LangGraph state 设计", "category": "technical",
             "evidence_count": 2}
        ],
        "behavioral_themes": [],
        "decision_factors": ["项目深度", "系统设计"],
    },
    "why_they_passed": ["项目深度被表扬 (来自 2 条 offer_post)"],
    "evidence_sources": [],
    "uncertainty_notes": [],
}, ensure_ascii=False)


EXAMPLES: tuple[TailorResumeExample, ...] = (
    TailorResumeExample(
        name="bytedance_agent_strong_fit",
        band="real",
        master_resume=_REAL_MASTER_RESUME,
        job_text=_BYTEDANCE_AGENT_JD,
        company="字节跳动",
        successful_profile_json=_BYTEDANCE_PROFILE,
        notes="高匹配 — tailor 应该 emphasize LangGraph/RAG/DSPy + reorder 实习到最前; "
              "ats_keywords_missing 应该列 GRPO / Speculative / leetcode hot100",
    ),
)


# ── metric ─────────────────────────────────────────────────────────


def metric(
    example: TailorResumeExample,
    raw_output: str | dict,
) -> MetricBreakdown:
    parsed = parse_json_output(raw_output)
    if not parsed:
        return _zero(example, "OUTPUT_PARSE_FAILURE: invalid JSON")

    result: TailorResumeResult | None = None
    schema_score = 0.0
    schema_note = ""
    try:
        result = TailorResumeResult.model_validate(parsed)
        schema_score = 1.0
        schema_note = "schema 通过 ✓"
    except ValidationError as e:
        err = e.errors()[0]
        loc = ".".join(str(x) for x in err.get("loc", []))
        schema_note = f"schema 失败: {loc}: {err.get('msg', '...')}"

    if result is None:
        return MetricBreakdown(
            total=0.0,
            breakdown={ax: 0.0 for ax in METRIC_AXES if ax != "total"},
            feedback=f"案例: {example.name}\n{schema_note}",
        )

    # no_fabrication: heuristic — does tailored claim companies/projects
    # not in master_resume? Look for proper-noun-style entities in
    # tailored that are absent from master.
    fab_violations = _check_fabrication(
        master=example.master_resume, tailored=result.tailored_markdown
    )
    if not fab_violations:
        no_fab_score = 1.0
        no_fab_note = "无编造 ✓"
    else:
        no_fab_score = max(0.0, 1.0 - 0.4 * len(fab_violations))
        no_fab_note = f"⚠ 检测到 {len(fab_violations)} 处可疑编造: {fab_violations[:3]}"

    # change_log_specific: every entry must reference JD/profile/ATS
    if result.change_log:
        ok = sum(
            1 for c in result.change_log
            if any(k in c.rationale for k in (
                "JD", "jd", "画像", "profile", "ATS", "ats", "关键词",
                "must_have", "第", "条",
            ))
        )
        changelog_score = ok / len(result.change_log)
        changelog_note = (
            f"change_log 含具体依据 {ok}/{len(result.change_log)}"
        )
    else:
        changelog_score = 0.3  # empty change_log on a non-trivial JD is weak
        changelog_note = "change_log 为空"

    # ats_keywords_grounded: every claimed keyword must substring-match tailored
    if result.ats_keywords_used:
        grounded = sum(
            1 for k in result.ats_keywords_used
            if k in result.tailored_markdown
        )
        ats_score = grounded / len(result.ats_keywords_used)
        ats_note = f"ats_keywords_used 全部入文 {grounded}/{len(result.ats_keywords_used)}"
    else:
        ats_score = 0.5
        ats_note = "ats_keywords_used 为空 (中性)"

    # honest_missing_keywords: should be non-empty if JD has obvious gaps
    # (heuristic — if any of these keywords appear in JD but not in master_resume,
    # the model should have surfaced them)
    obvious_jd_skills = ["GRPO", "Speculative", "TensorFlow", "leetcode hot100"]
    expected_missing = [
        k for k in obvious_jd_skills
        if k in example.job_text and k not in example.master_resume
    ]
    if expected_missing:
        if result.ats_keywords_missing:
            honest_score = 1.0
            honest_note = f"诚实列了 {len(result.ats_keywords_missing)} 个缺失 ✓"
        else:
            honest_score = 0.0
            honest_note = (
                f"⚠ JD 明显缺失 {expected_missing} 但 missing 列空"
            )
    else:
        honest_score = 1.0
        honest_note = "无明显 JD-vs-master gap (skip)"

    # self_audit_active: cannot_fake_warnings present on borderline cases
    # Borderline = JD asks for must_have not in master_resume
    if expected_missing:
        if result.cannot_fake_warnings:
            audit_score = 1.0
            audit_note = f"自审主动 {len(result.cannot_fake_warnings)} 条 ✓"
        else:
            audit_score = 0.5
            audit_note = "borderline 案例但无 self-audit 警告"
    else:
        audit_score = 1.0
        audit_note = "非 borderline (skip)"

    total = (
        _W_SCHEMA * schema_score
        + _W_NO_FAB * no_fab_score
        + _W_CHANGELOG * changelog_score
        + _W_ATS_GROUNDED * ats_score
        + _W_HONEST_MISSING * honest_score
        + _W_SELF_AUDIT * audit_score
    )

    feedback = "\n".join([
        f"案例: {example.name} ({example.band})",
        schema_note, no_fab_note, changelog_note, ats_note,
        honest_note, audit_note,
        f"score: schema={schema_score:.2f} no_fab={no_fab_score:.2f}"
        f" changelog={changelog_score:.2f} ats={ats_score:.2f}"
        f" honest={honest_score:.2f} audit={audit_score:.2f}"
        f" → total={total:.2f}",
    ])

    return MetricBreakdown(
        total=total,
        breakdown={
            "schema": schema_score,
            "no_fabrication": no_fab_score,
            "change_log_specific": changelog_score,
            "ats_keywords_grounded": ats_score,
            "honest_missing_keywords": honest_score,
            "self_audit_active": audit_score,
        },
        feedback=feedback,
    )


# ── fabrication detector (heuristic, not perfect) ────────────────


def _check_fabrication(*, master: str, tailored: str) -> list[str]:
    """Heuristic: extract company-like and project-like tokens from
    tailored, check they all appear in master_resume.

    False-positives are tolerable (the metric just slightly penalizes
    a perfectly-honest tailor); false-negatives are more dangerous so
    the patterns are aggressive.
    """
    violations: list[str] = []
    # Companies: 2-6 CJK chars + 公司|科技|集团 | famous bare names
    cjk_company_pat = re.compile(
        r"([一-鿿]{2,8})(公司|科技|集团|实验室|股份)"
    )
    for m in cjk_company_pat.finditer(tailored):
        full = m.group(0)
        if full not in master and m.group(1) not in master:
            violations.append(full)

    # Famous bare-name companies — must be in master to claim
    famous = {"字节", "字节跳动", "阿里", "阿里巴巴", "腾讯", "美团",
              "百度", "京东", "拼多多", "小红书", "网易", "华为", "小米"}
    for f in famous:
        if f in tailored and f not in master:
            violations.append(f"提到 {f} 但 master 无")

    # Famous awards / certifications absent from master
    awards = ["ACM 金牌", "ACM 银牌", "ACM 区域赛", "Kaggle"]
    for a in awards:
        if a in tailored and a not in master:
            violations.append(f"提到 {a} 但 master 无")

    return list(set(violations))


def _zero(example: TailorResumeExample, reason: str) -> MetricBreakdown:
    return MetricBreakdown(
        total=0.0,
        breakdown={ax: 0.0 for ax in METRIC_AXES if ax != "total"},
        feedback=f"案例: {example.name} ({example.band})\n{reason}",
    )


def split_train_val(
    *,
    val_fraction: float = 0.5,
    seed: int = 0,
) -> tuple[list[TailorResumeExample], list[TailorResumeExample]]:
    import random
    rng = random.Random(seed)
    by_band: dict[str, list[TailorResumeExample]] = {}
    for ex in sorted(EXAMPLES, key=lambda e: e.name):
        by_band.setdefault(ex.band, []).append(ex)
    train: list[TailorResumeExample] = []
    val: list[TailorResumeExample] = []
    for _band, exs in sorted(by_band.items()):
        n_val = max(1, round(len(exs) * val_fraction))
        shuffled = exs[:]
        rng.shuffle(shuffled)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val
