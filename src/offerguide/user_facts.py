"""Long-term memory layer — user_facts table + retrieval, mem0 v3-inspired.

Borrowed from `mem0 <https://github.com/mem0ai/mem0>`_ (54k ⭐, 2026-04 v3 algorithm)
two key insights:

1. **Single-pass ADD-only**: new facts append, never UPDATE/DELETE. Mem0 v3
   benchmarks (LoCoMo 91.6, LongMemEval 93.4) showed accumulation outperforms
   clobber-update — older "stale-looking" facts often turn out to still be
   relevant context. Confidence is per-fact, not per-row-version.
2. **Multi-signal retrieval**: BM25 keyword + entity match scored in parallel
   and fused. We skip semantic embeddings for now (sqlite-vec is available but
   adds embedding-call cost on every retrieve); BM25 + entity LIKE is
   surprisingly competitive on Chinese 校招 text.

Why we need this:
Each SKILL invocation in OfferGuide is currently *stateless* — score_match
re-extracts the user's project list from raw resume text every call. A user
who's already told us "RemeDi project, AUC 0.83, BERT 双塔" should not have
that re-derived 100 times.

Long-term memory closes the loop:
- Each SKILL run gets a follow-up extraction pass that mines facts out of the
  output (via :func:`extract_facts_from_run`).
- Every subsequent SKILL run preloads the most-relevant facts into the prompt
  (via :func:`retrieve_for_prompt`).
- The user can always audit / delete facts via a UI page.

Fact kinds (controlled vocabulary):
    profile         — fixed user attributes (school, major, year)
    preference      — stated preferences (location / company size / role)
    experience      — work / project experience details
    feedback        — past 反馈 from interviews / HR / rejections
    project         — specific project facts (tech stack, results)
    company_signal  — observed company-specific insights

Anti-pattern guarded against:
- Don't extract opinions as facts. The LLM is prompted to output only
  *factual* claims with measurable / verifiable specifics.
- Don't extract derived inferences. Only what the source text directly says.
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

FactKind = Literal[
    "profile",
    "preference",
    "experience",
    "feedback",
    "project",
    "company_signal",
]

VALID_KINDS: frozenset[str] = frozenset(FactKind.__args__)

DEFAULT_RETRIEVE_TOP_K = 8
"""Top-K facts to inject into a prompt. 8 is a heuristic — enough context to
remember key projects + preferences, small enough to not blow the prompt."""

MIN_FACT_LEN = 8
MAX_FACT_LEN = 240
"""Fact text length bounds. Below MIN it's not informative; above MAX it's a
narrative not a fact (and will hurt retrieval precision)."""


@dataclass(frozen=True)
class Fact:
    """One row of user_facts."""
    id: int
    fact_text: str
    kind: FactKind
    source_skill: str | None
    source_run_id: int | None
    confidence: float
    entities: list[str]
    used_count: int
    created_at: float
    last_used_at: float | None


@dataclass
class FactCandidate:
    """A fact pending insertion. Returned by extractors before persistence."""
    fact_text: str
    kind: FactKind
    confidence: float = 0.7
    entities: list[str] = field(default_factory=list)


# ─────────── persistence ─────────────────────────────


def add_fact(
    store: Store,
    *,
    fact_text: str,
    kind: FactKind,
    source_skill: str | None = None,
    source_run_id: int | None = None,
    confidence: float = 0.7,
    entities: list[str] | None = None,
) -> int | None:
    """Insert one fact. Returns row id, or ``None`` if duplicate / invalid.

    ADD-only: same fact_text is a UNIQUE conflict → silently skipped (returns
    None). The caller can then optionally bump confidence on the existing row
    via :func:`bump_confidence` if they want to reinforce.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"unknown kind {kind!r}; must be one of {sorted(VALID_KINDS)}")

    text = (fact_text or "").strip()
    if not (MIN_FACT_LEN <= len(text) <= MAX_FACT_LEN):
        return None

    confidence = max(0.0, min(1.0, confidence))
    entities_json = json.dumps(entities or [], ensure_ascii=False)

    with store.connect() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO user_facts"
            "(fact_text, kind, source_skill, source_run_id, confidence, entities_json) "
            "VALUES (?,?,?,?,?,?) RETURNING id",
            (text, kind, source_skill, source_run_id, confidence, entities_json),
        )
        row = cur.fetchone()
    return int(row[0]) if row else None


def bump_confidence(store: Store, *, fact_id: int, delta: float = 0.05) -> None:
    """Increase confidence on an existing fact (clamped to 1.0)."""
    with store.connect() as conn:
        conn.execute(
            "UPDATE user_facts "
            "SET confidence = MIN(1.0, confidence + ?) WHERE id = ?",
            (delta, fact_id),
        )


def list_facts(
    store: Store,
    *,
    kind: FactKind | None = None,
    limit: int = 200,
) -> list[Fact]:
    """List facts, optionally filtered by kind. Newest first."""
    with store.connect() as conn:
        if kind is not None:
            rows = conn.execute(
                "SELECT id, fact_text, kind, source_skill, source_run_id, "
                "       confidence, entities_json, used_count, created_at, last_used_at "
                "FROM user_facts WHERE kind = ? ORDER BY created_at DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, fact_text, kind, source_skill, source_run_id, "
                "       confidence, entities_json, used_count, created_at, last_used_at "
                "FROM user_facts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_row_to_fact(r) for r in rows]


def delete_fact(store: Store, *, fact_id: int) -> bool:
    """Remove one fact. The user can purge wrong / outdated facts via UI."""
    with store.connect() as conn:
        cur = conn.execute("DELETE FROM user_facts WHERE id = ?", (fact_id,))
    return (cur.rowcount or 0) > 0


# ─────────── retrieval ─────────────────────────────


def retrieve(
    store: Store,
    *,
    query: str,
    kinds: tuple[FactKind, ...] | None = None,
    top_k: int = DEFAULT_RETRIEVE_TOP_K,
    min_confidence: float = 0.3,
) -> list[Fact]:
    """Retrieve top-K facts by combined BM25-style + entity-match score.

    Score = 0.6 * keyword_overlap + 0.3 * entity_overlap + 0.1 * confidence.

    Keyword overlap is a simple Jaccard between query tokens and fact tokens
    (Chinese: split by character + 2-grams). Entity overlap is exact match
    between query-extracted entities and fact's entities_json.

    On match, increments ``used_count`` and ``last_used_at`` so unused facts
    can be GC'd later if they accumulate.
    """
    if not query.strip():
        return []
    q_tokens = _tokenize(query)
    q_entities = _extract_entities_from_text(query)

    with store.connect() as conn:
        if kinds:
            placeholders = ",".join("?" * len(kinds))
            rows = conn.execute(
                f"SELECT id, fact_text, kind, source_skill, source_run_id, "
                f"       confidence, entities_json, used_count, created_at, last_used_at "
                f"FROM user_facts "
                f"WHERE kind IN ({placeholders}) AND confidence >= ? "
                f"ORDER BY used_count DESC, created_at DESC",
                (*kinds, min_confidence),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, fact_text, kind, source_skill, source_run_id, "
                "       confidence, entities_json, used_count, created_at, last_used_at "
                "FROM user_facts WHERE confidence >= ? "
                "ORDER BY used_count DESC, created_at DESC",
                (min_confidence,),
            ).fetchall()

    if not rows:
        return []

    # Score every candidate
    scored: list[tuple[float, Fact]] = []
    for r in rows:
        fact = _row_to_fact(r)
        f_tokens = _tokenize(fact.fact_text)
        keyword_score = _jaccard(q_tokens, f_tokens)
        entity_score = (
            len(set(q_entities) & set(fact.entities)) / max(1, len(q_entities))
            if q_entities
            else 0.0
        )
        combined = (
            0.6 * keyword_score
            + 0.3 * entity_score
            + 0.1 * fact.confidence
        )
        if combined > 0.05:  # filter out near-zero matches
            scored.append((combined, fact))

    scored.sort(key=lambda t: -t[0])
    top = [f for _s, f in scored[:top_k]]

    # Mark retrieved facts as used (single SQL trip)
    if top:
        ids = [f.id for f in top]
        placeholders = ",".join("?" * len(ids))
        with store.connect() as conn:
            conn.execute(
                f"UPDATE user_facts "
                f"SET used_count = used_count + 1, "
                f"    last_used_at = julianday('now') "
                f"WHERE id IN ({placeholders})",
                tuple(ids),
            )
    return top


def retrieve_for_prompt(
    store: Store,
    *,
    query: str,
    kinds: tuple[FactKind, ...] | None = None,
    top_k: int = DEFAULT_RETRIEVE_TOP_K,
    header: str = "## 已知用户长期事实（来自历史 SKILL 调用）",
) -> str:
    """Render top-K facts as a prompt-ready section.

    Returns "" if no relevant facts — caller can conditionally include.
    Each fact line: "- [kind] fact_text (置信度 0.XX)".
    """
    facts = retrieve(store, query=query, kinds=kinds, top_k=top_k)
    if not facts:
        return ""
    lines = [header]
    for f in facts:
        lines.append(
            f"- [{f.kind}] {f.fact_text} (置信度 {f.confidence:.2f})"
        )
    return "\n".join(lines)


# ─────────── LLM-driven extraction ─────────────────────────────


_EXTRACT_PROMPT = """你是 OfferGuide 的事实抽取员。给定一段 SKILL 输出文本，
抽取其中**对未来 SKILL 调用有用的、可验证的、单句的事实**。

返回 JSON 数组，每条 fact 是：
{{
  "fact_text":  <中文一句话, 8-240 字, 必须是单一事实, 含具体名词/数字/时间>,
  "kind":       "profile" | "preference" | "experience" | "feedback" | "project" | "company_signal",
  "confidence": <0..1, LLM 自评事实可信度>,
  "entities":   <list[str], 出现的公司/项目/技能名, 0-5 个>
}}

判断要点：
- ✅ 抽: "用户 RemeDi 项目用 BERT 双塔模型, AUC 提升 0.04"  (实验性/具体)
- ✅ 抽: "字节 AI Agent 实习要求 LangGraph 经验, 来自 2 条 offer 复盘"  (聚合证据)
- ✅ 抽: "用户上次面试 GRPO 题答得卡, 需要补 RL 基础"  (反馈)
- ❌ 不抽: "用户综合素质优秀"  (空泛)
- ❌ 不抽: "建议加强算法基础"  (建议不是事实)
- ❌ 不抽: "可能字节比较卷"  (主观推断)

如果没有合格事实, 返回 []。最多返回 8 条。
**只返回 JSON 数组, 不要 markdown 代码块, 不要解释。**

文本来源 SKILL: {skill_name}
"""


def extract_facts_from_text(
    *,
    text: str,
    source_skill: str,
    llm: LLMClient | None,
) -> list[FactCandidate]:
    """Run LLM extraction over arbitrary text. Returns FactCandidate list.

    Falls back to empty list if no LLM (deterministic path doesn't help here —
    extraction is the LLM's job).
    """
    if llm is None or not text.strip():
        return []

    try:
        resp = llm.chat(
            messages=[
                {"role": "system", "content": _EXTRACT_PROMPT.format(
                    skill_name=source_skill
                )},
                {"role": "user", "content": text[:8000]},
            ],
            temperature=0.0,
            json_mode=True,
        )
    except LLMError as e:
        log.warning("user_facts extract LLM failed: %s", e)
        return []

    try:
        raw = json.loads(resp.content)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []

    out: list[FactCandidate] = []
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        fact_text = str(item.get("fact_text", "")).strip()
        kind = item.get("kind", "experience")
        if kind not in VALID_KINDS:
            kind = "experience"
        try:
            confidence = float(item.get("confidence", 0.7))
        except (ValueError, TypeError):
            confidence = 0.7
        ents_raw = item.get("entities", [])
        entities = (
            [str(e).strip() for e in ents_raw if e][:5]
            if isinstance(ents_raw, list)
            else []
        )
        if MIN_FACT_LEN <= len(fact_text) <= MAX_FACT_LEN:
            out.append(FactCandidate(
                fact_text=fact_text, kind=kind,  # type: ignore[arg-type]
                confidence=confidence, entities=entities,
            ))
    return out


def extract_and_persist(
    store: Store,
    *,
    text: str,
    source_skill: str,
    source_run_id: int | None = None,
    llm: LLMClient | None,
) -> dict[str, int]:
    """Extract facts from text + persist them. Returns counters."""
    candidates = extract_facts_from_text(
        text=text, source_skill=source_skill, llm=llm,
    )
    inserted = 0
    skipped = 0
    for c in candidates:
        new_id = add_fact(
            store,
            fact_text=c.fact_text,
            kind=c.kind,
            source_skill=source_skill,
            source_run_id=source_run_id,
            confidence=c.confidence,
            entities=c.entities,
        )
        if new_id is None:
            skipped += 1
        else:
            inserted += 1
    return {
        "candidates": len(candidates),
        "inserted": inserted,
        "skipped_dup_or_invalid": skipped,
    }


# ─────────── batch extract from skill_runs (for daemon) ─────────


def extract_pending_runs(
    store: Store,
    *,
    llm: LLMClient | None,
    skill_names: tuple[str, ...] = (
        "score_match", "analyze_gaps", "prepare_interview",
        "deep_project_prep", "post_interview_reflection",
        "successful_profile", "profile_resume_gap",
    ),
    limit: int = 30,
) -> dict[str, Any]:
    """Sweep recent skill_runs that haven't been extracted yet.

    "Haven't been extracted" = no row in user_facts with this run_id.
    Capped to ``limit`` per call (daemon safety).

    Returns aggregated counters for logging.
    """
    with store.connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, skill_name, output_json
            FROM skill_runs
            WHERE skill_name IN ({",".join("?" * len(skill_names))})
              AND id NOT IN (
                  SELECT DISTINCT source_run_id FROM user_facts
                  WHERE source_run_id IS NOT NULL
              )
            ORDER BY created_at DESC LIMIT ?
            """,
            (*skill_names, limit),
        ).fetchall()

    total = {
        "runs_scanned": 0,
        "candidates": 0,
        "inserted": 0,
        "skipped_dup_or_invalid": 0,
    }
    for r in rows:
        run_id, skill_name, output_text = r
        result = extract_and_persist(
            store,
            text=output_text or "",
            source_skill=skill_name,
            source_run_id=int(run_id),
            llm=llm,
        )
        total["runs_scanned"] += 1
        total["candidates"] += result["candidates"]
        total["inserted"] += result["inserted"]
        total["skipped_dup_or_invalid"] += result["skipped_dup_or_invalid"]
    return total


# ─────────── helpers ─────────────────────────────


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    """Simple tokenizer that works on mixed Chinese/English.

    English: regex \\w+ words. Chinese: each character + 2-grams (sliding).
    Returns set for Jaccard scoring.
    """
    if not text:
        return set()
    eng = set(m.group().lower() for m in _TOKEN_RE.finditer(text) if not _is_cjk(m.group()))
    # Chinese: char + 2-gram bag
    chars: list[str] = []
    for ch in text:
        if "一" <= ch <= "鿿":
            chars.append(ch)
    cjk_tokens: set[str] = set(chars)
    cjk_tokens.update(
        text[i:i + 2] for i in range(len(text) - 1)
        if "一" <= text[i] <= "鿿" and "一" <= text[i + 1] <= "鿿"
    )
    return eng | cjk_tokens


def _is_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _extract_entities_from_text(text: str) -> list[str]:
    """Heuristic entity extractor — pulls Chinese 公司/项目-style proper nouns.

    Looks for runs of 2-6 CJK chars followed by 公司/科技/股份/集团/项目 etc.,
    plus capitalized-English company names. Cheap; LLM-free.
    """
    out: set[str] = set()
    # English caps companies (>= 2 chars)
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9]{1,15})\b", text):
        out.add(m.group(1))
    # Chinese: 字节跳动/腾讯/阿里巴巴/美团/RemeDi
    # known company suffixes hint
    for m in re.finditer(
        r"([一-鿿]{2,6})(公司|科技|集团|股份|项目|实验室|大学)",
        text,
    ):
        out.add(m.group(1))
    # Famous bare names (no suffix) — small whitelist
    famous = {
        "字节跳动", "字节", "阿里巴巴", "阿里", "腾讯", "美团", "百度",
        "京东", "拼多多", "小红书", "网易", "华为", "小米", "蔚来", "理想",
        "小鹏", "OpenAI", "Anthropic", "DeepSeek",
    }
    for f in famous:
        if f in text:
            out.add(f)
    return list(out)[:10]


def _row_to_fact(row: tuple) -> Fact:
    return Fact(
        id=int(row[0]),
        fact_text=row[1],
        kind=row[2],
        source_skill=row[3],
        source_run_id=int(row[4]) if row[4] is not None else None,
        confidence=float(row[5]),
        entities=json.loads(row[6]) if row[6] else [],
        used_count=int(row[7]),
        created_at=float(row[8]),
        last_used_at=float(row[9]) if row[9] is not None else None,
    )
