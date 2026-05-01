"""Corpus quality classifier — 真实用户内容 vs 卖课/引流/低质量内容.

A successful-candidate profile is only as good as its training corpus. If we
synthesize a profile from posts that turn out to be marketer-written, the
profile recommends courses, not skills. That's a non-starter for a job-hunt
agent that's supposed to give the user **actionable** preparation guidance.

This module is the trust layer underneath ``successful_profile_synth``. It
runs each corpus item through an LLM-driven classifier that returns:

- ``content_kind`` — interview/offer_post/project_share/reflection/other
- ``quality_score`` — 0..1, "should we trust this for profile synthesis"
- ``quality_signals`` — structured evidence so the user can audit decisions

Why an LLM and not regex/ML: the marketer-vs-real signal is high-context.
"加微信" alone could be a real user offering to share, or a marketer hook;
"私信我" before "我手撕了 6 道题" is a marketer, after that is a friendly
sharer.  The LLM reads the whole post and weights the signal mix.

We cache the verdict on the row (``quality_classified_at`` timestamp) so
re-runs of the classifier are cheap — only newly-ingested items touch the
LLM.

A *separate* deterministic pre-filter (``_obvious_marketer``) catches the
most blatant cases (post is 90% 加 V / 私信 hook) without an LLM call. This
saves tokens on the trivial cases while the LLM handles the gray zone.

This module ALSO classifies the ``content_kind`` so the rest of the system
can ask "show me only high-quality offer-posts about 字节跳动 AI Agent
position" in SQL — that's the precursor to ``successful_profile_synth``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .llm import LLMClient, LLMError
from .memory import Store

log = logging.getLogger(__name__)

ContentKind = Literal[
    "interview",        # 面经 — 面试题、流程
    "offer_post",       # 拿到 offer 后的复盘 / 经验贴
    "reflection",       # 失败 / 沉默 / 学习反思
    "project_share",    # 项目经历分享（高质量项目案例）
    "other",            # 其他（公告、问答、求助等不归类的）
    "marketer",         # 卖课 / 引流 / 培训 / 包过广告
]
"""Content categories. ``marketer`` is the negative class — these get
quality_score forced to 0 and are excluded from profile synthesis."""

VALID_KINDS: frozenset[str] = frozenset(ContentKind.__args__)

# Threshold for inclusion in successful-profile synthesis. Tunable but the
# system biases conservatively — we'd rather drop a borderline real post
# than poison the profile with a marketer post.
HIGH_QUALITY_THRESHOLD = 0.6
"""Items with quality_score >= this are considered "trustworthy" for profile
synthesis. The synthesizer reaches up the score ranks until it has enough
items, so a high threshold means smaller but cleaner corpus."""

LOW_QUALITY_THRESHOLD = 0.3
"""Items below this are dropped entirely — never shown in the UI, never used
for synthesis. The agent treats them as noise."""


@dataclass(frozen=True)
class QualitySignals:
    """Structured evidence for the quality verdict.

    These are the granular flags the LLM extracts; ``quality_score`` is
    derived from them. Storing them lets the user audit the verdict
    ("why did you mark this as low quality?") and lets us debug
    classifier behavior over time.
    """

    # Real-user signals (positive)
    has_specific_timeline: bool = False
    """Mentions concrete dates / weeks / 投递→面试间隔."""

    has_round_detail: bool = False
    """Distinguishes 一面/二面/三面/HR 面 with content per round."""

    has_specific_questions: bool = False
    """Quotes actual interview questions, not generic categories."""

    has_project_detail: bool = False
    """Discusses specific projects with names / tech stacks / numbers."""

    has_failure_or_struggle: bool = False
    """Mentions things that didn't work, places they got stuck — high
    signal of authenticity (marketers paint everything as success)."""

    has_concrete_outcome: bool = False
    """Says whether they got the offer, what package, what level — vs
    vague 'success' claims."""

    # Marketer signals (negative)
    has_contact_hook: bool = False
    """加微信 / 私信我 / 加 V / DM me when content is contact-soliciting,
    not just contact-offering."""

    has_course_pitch: bool = False
    """资料包 / 训练营 / 包过 / 课程 / 网课 / 简历修改服务 / 1v1 mentioned
    as primary CTA."""

    has_repetitive_promo_language: bool = False
    """Marketer copywriting patterns: same phrases used in dozens of posts,
    formulaic intros, overuse of emoji."""

    has_external_funnel: bool = False
    """Pushes users to 公众号 / 知识星球 / 小程序 / 群."""

    # Verdict
    kind: ContentKind = "other"
    quality_score: float = 0.5
    """Aggregate trust score. ``score = positives - 1.5 * negatives`` rough
    formula, clamped to 0..1."""

    rationale: str = ""
    """One-sentence Chinese explanation for the user / audit log."""


@dataclass
class ClassificationVerdict:
    """What ``classify_one`` returns — signals + persistence-ready dict."""
    signals: QualitySignals
    raw_llm_output: dict[str, Any] = field(default_factory=dict)
    """Whatever JSON the LLM emitted, before our parsing. Useful for
    debugging when our extraction misses a field."""

    skipped_llm: bool = False
    """True when the deterministic pre-filter handled this item without
    calling the LLM (e.g. obvious marketer post)."""


# ───────────── deterministic pre-filter ─────────────────────────────


_BLATANT_MARKETER_RE = re.compile(
    r"(加(微信|V|wx|w[Xx])|私信我|关注公众号|训练营|包过|"
    r"内推码.{0,15}(微信|\+v|加我)|资料包.{0,5}(限免|私信|DM)|"
    r"1v1.{0,5}(辅导|带练)|网课.{0,5}(代报|代购))",
    re.IGNORECASE,
)
"""Heuristic regex for "obviously a marketer" content. If the post matches
≥ 2 distinct patterns AND has no real-content signals, we skip the LLM
and mark it as marketer. False-positive rate is intentionally tolerated —
real users rarely paste "加微信训练营包过" into a 面经."""

_REAL_CONTENT_RE = re.compile(
    r"(一面|二面|三面|终面|HR 面|笔试|手撕|代码题|"
    r"问.{0,3}(项目|经历|场景)|leetcode|面试官|被问|"
    r"挂了|过了|拿到 offer|意向书|薪资|base|n\+\d|n＋\d)",
)
"""Signals of genuine 面经/offer content. If a post has any of these AND
also marketer signals, it's gray zone — needs the LLM."""


def _obvious_marketer(text: str) -> bool:
    """Cheap pre-filter: post is 90%+ marketing copy with no real content."""
    if len(text) < 30:
        return False  # too short to confidently classify
    marketer_hits = len(set(_BLATANT_MARKETER_RE.findall(text)))
    real_hits = len(set(_REAL_CONTENT_RE.findall(text)))
    return marketer_hits >= 2 and real_hits == 0


# ───────────── LLM-based classifier ─────────────────────────────


_CLASSIFY_PROMPT = """\
你是 OfferGuide 的语料质量审核员，要把一篇贴子分类、并判断它是不是真实用户写的。

返回 JSON, 字段（全部必填，类型严格）：
{
  "kind": "interview" | "offer_post" | "project_share" | "reflection" | "other" | "marketer",
  "has_specific_timeline": true|false,        // 提到具体日期/周/面试间隔
  "has_round_detail": true|false,             // 区分一面/二面/三面/HR面 + 每轮内容
  "has_specific_questions": true|false,       // 引用了具体面试题，不是泛泛的"问 SQL"
  "has_project_detail": true|false,           // 提到具体项目名/技术栈/数字
  "has_failure_or_struggle": true|false,      // 提到没答出/卡住/挂了等真实细节
  "has_concrete_outcome": true|false,         // 说了是否拿 offer / 等级 / 薪资
  "has_contact_hook": true|false,             // 加微信/私信/加V 是引流而非分享
  "has_course_pitch": true|false,             // 训练营/课程/资料包/包过 作为主要 CTA
  "has_repetitive_promo_language": true|false,// 模板化文案 / 营销腔
  "has_external_funnel": true|false,          // 推 公众号/星球/小程序
  "rationale": "一句中文说明判断依据，<60 字"
}

判断要点：
- "marketer" 类的特征：私聊/加 V 是 CTA、训练营/课程是主推、内容空洞
- "offer_post" 通常含具体 base / 等级 / 时间线 / 心得
- "project_share" 要有具体技术栈 + 真实问题 / 数据 / 评估指标
- 真实用户经常带"挂了/沉默/卡住/没答上来"等不光鲜的细节，是高可信信号
- 如果一个贴同时有真实细节和加 V 引流, 仍然倾向 "marketer" — 引流目的污染信任

只返回 JSON，不要解释。
"""


def classify_one(
    *,
    text: str,
    company: str | None = None,
    role_hint: str | None = None,
    llm: LLMClient | None = None,
) -> ClassificationVerdict:
    """Classify one corpus item. Returns ``ClassificationVerdict``.

    If ``llm`` is None, we run only the deterministic pre-filter. That's
    enough to catch obvious marketer content; gray-zone items get a
    middle-of-the-road default verdict (kind='other', score=0.5) until
    the LLM is available.
    """
    # Deterministic fast path
    if _obvious_marketer(text):
        return ClassificationVerdict(
            signals=QualitySignals(
                has_contact_hook=True,
                has_course_pitch=True,
                kind="marketer",
                quality_score=0.0,
                rationale="纯卖课/引流文案，无真实面试内容",
            ),
            skipped_llm=True,
        )

    if llm is None:
        # Best we can do without LLM: assume 'other' middle quality
        return ClassificationVerdict(
            signals=QualitySignals(
                kind="other",
                quality_score=0.5,
                rationale="未调用 LLM，默认中性",
            ),
            skipped_llm=True,
        )

    user_msg = (
        f"【公司】{company or '未知'}\n"
        f"【岗位线索】{role_hint or '无'}\n"
        f"【正文】\n{text[:6000]}"
    )

    try:
        resp = llm.chat(
            messages=[
                {"role": "system", "content": _CLASSIFY_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            json_mode=True,
        )
    except LLMError as e:
        log.warning("corpus_quality LLM failed: %s", e)
        return ClassificationVerdict(
            signals=QualitySignals(
                kind="other", quality_score=0.5,
                rationale=f"LLM 调用失败：{e}",
            ),
            skipped_llm=True,
        )

    try:
        raw = json.loads(resp.content)
    except json.JSONDecodeError:
        return ClassificationVerdict(
            signals=QualitySignals(
                kind="other", quality_score=0.5,
                rationale="LLM 返回非 JSON",
            ),
            skipped_llm=False,
        )

    signals = _signals_from_llm(raw)
    return ClassificationVerdict(signals=signals, raw_llm_output=raw)


def _signals_from_llm(raw: dict) -> QualitySignals:
    """Map LLM JSON fields to our QualitySignals dataclass + score."""
    kind_raw = raw.get("kind", "other")
    kind: ContentKind = kind_raw if kind_raw in VALID_KINDS else "other"  # type: ignore[assignment]

    pos_count = sum(1 for k in (
        "has_specific_timeline", "has_round_detail", "has_specific_questions",
        "has_project_detail", "has_failure_or_struggle", "has_concrete_outcome",
    ) if raw.get(k))
    neg_count = sum(1 for k in (
        "has_contact_hook", "has_course_pitch",
        "has_repetitive_promo_language", "has_external_funnel",
    ) if raw.get(k))

    # Marketer kind always scores low regardless of mixed signals
    if kind == "marketer":
        score = 0.0
    else:
        # Linear blend: each positive worth +0.15, each negative -0.20.
        # Base 0.5, clamped to [0, 1].
        raw_score = 0.5 + 0.15 * pos_count - 0.20 * neg_count
        score = max(0.0, min(1.0, raw_score))

    return QualitySignals(
        has_specific_timeline=bool(raw.get("has_specific_timeline")),
        has_round_detail=bool(raw.get("has_round_detail")),
        has_specific_questions=bool(raw.get("has_specific_questions")),
        has_project_detail=bool(raw.get("has_project_detail")),
        has_failure_or_struggle=bool(raw.get("has_failure_or_struggle")),
        has_concrete_outcome=bool(raw.get("has_concrete_outcome")),
        has_contact_hook=bool(raw.get("has_contact_hook")),
        has_course_pitch=bool(raw.get("has_course_pitch")),
        has_repetitive_promo_language=bool(raw.get("has_repetitive_promo_language")),
        has_external_funnel=bool(raw.get("has_external_funnel")),
        kind=kind,
        quality_score=round(score, 3),
        rationale=str(raw.get("rationale", "")).strip()[:200],
    )


# ───────────── persistence layer ─────────────────────────────


def persist_verdict(
    store: Store, *, item_id: int, verdict: ClassificationVerdict
) -> None:
    """Write the verdict back to interview_experiences row.

    Updates: content_kind, quality_score, quality_signals_json,
    quality_classified_at. Does NOT touch raw_text or other fields.
    """
    signals_dict = {
        # positive signals
        "has_specific_timeline":         verdict.signals.has_specific_timeline,
        "has_round_detail":              verdict.signals.has_round_detail,
        "has_specific_questions":        verdict.signals.has_specific_questions,
        "has_project_detail":            verdict.signals.has_project_detail,
        "has_failure_or_struggle":       verdict.signals.has_failure_or_struggle,
        "has_concrete_outcome":          verdict.signals.has_concrete_outcome,
        # negative signals
        "has_contact_hook":              verdict.signals.has_contact_hook,
        "has_course_pitch":              verdict.signals.has_course_pitch,
        "has_repetitive_promo_language": verdict.signals.has_repetitive_promo_language,
        "has_external_funnel":           verdict.signals.has_external_funnel,
        # meta
        "rationale":                     verdict.signals.rationale,
        "skipped_llm":                   verdict.skipped_llm,
    }
    with store.connect() as conn:
        conn.execute(
            "UPDATE interview_experiences "
            "SET content_kind = ?, quality_score = ?, "
            "    quality_signals_json = ?, quality_classified_at = julianday('now') "
            "WHERE id = ?",
            (
                verdict.signals.kind,
                verdict.signals.quality_score,
                json.dumps(signals_dict, ensure_ascii=False),
                item_id,
            ),
        )


def classify_pending(
    store: Store,
    *,
    llm: LLMClient | None = None,
    limit: int = 50,
) -> dict[str, int]:
    """Classify all unscored corpus items, persist verdicts.

    "Unscored" = ``quality_classified_at IS NULL``. Capped to ``limit``
    per call so a daemon job can process incrementally without
    hammering the LLM.

    Returns counters: ``{processed, marketer, high_quality, low_quality}``.
    """
    counters = {"processed": 0, "marketer": 0, "high_quality": 0, "low_quality": 0}
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, company, role_hint, raw_text "
            "FROM interview_experiences "
            "WHERE quality_classified_at IS NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    for r in rows:
        item_id, company, role_hint, raw_text = r
        verdict = classify_one(
            text=raw_text or "",
            company=company,
            role_hint=role_hint,
            llm=llm,
        )
        persist_verdict(store, item_id=item_id, verdict=verdict)
        counters["processed"] += 1
        if verdict.signals.kind == "marketer":
            counters["marketer"] += 1
        elif verdict.signals.quality_score >= HIGH_QUALITY_THRESHOLD:
            counters["high_quality"] += 1
        elif verdict.signals.quality_score < LOW_QUALITY_THRESHOLD:
            counters["low_quality"] += 1

    return counters


def fetch_high_quality(
    store: Store,
    *,
    company: str,
    role_hint: str | None = None,
    kinds: tuple[str, ...] = ("interview", "offer_post", "project_share", "reflection"),
    min_score: float = HIGH_QUALITY_THRESHOLD,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Pull the top-N high-quality corpus items for profile synthesis.

    Filters: company match (LIKE) + role_hint LIKE if provided +
    content_kind in (kinds) + quality_score >= min_score.

    Sorted by quality_score DESC, then created_at DESC. ``role_hint``
    is a soft filter — if no rows match strictly, falls back to no
    role filter.
    """
    placeholders = ",".join("?" * len(kinds))
    base_sql = (
        f"SELECT id, company, role_hint, raw_text, source, source_url, "
        f"  content_kind, quality_score, quality_signals_json, created_at "
        f"FROM interview_experiences "
        f"WHERE company LIKE ? "
        f"  AND content_kind IN ({placeholders}) "
        f"  AND quality_score >= ? "
    )
    role_clause = " AND role_hint LIKE ? " if role_hint else ""
    order_sql = " ORDER BY quality_score DESC, created_at DESC LIMIT ? "

    params: list[Any] = [f"%{company}%", *kinds, min_score]
    if role_hint:
        params.append(f"%{role_hint}%")
    params.append(limit)

    with store.connect() as conn:
        rows = conn.execute(
            base_sql + role_clause + order_sql,
            tuple(params),
        ).fetchall()

    if not rows and role_hint:
        # Fall back to no role filter if strict match found nothing
        return fetch_high_quality(
            store,
            company=company,
            role_hint=None,
            kinds=kinds,
            min_score=min_score,
            limit=limit,
        )

    return [
        {
            "id": r[0],
            "company": r[1],
            "role_hint": r[2],
            "raw_text": r[3],
            "source": r[4],
            "source_url": r[5],
            "content_kind": r[6],
            "quality_score": r[7],
            "quality_signals": json.loads(r[8]) if r[8] else {},
            "created_at": r[9],
        }
        for r in rows
    ]
