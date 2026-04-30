"""STAR + Reflection story bank.

Borrowed pattern from `Career-Ops <https://github.com/santifer/career-ops>`_
(MIT, santifer): accumulate 5-10 master behavioral narratives across
interview evaluations rather than regenerating from scratch each time.

Why this matters:
- Behavioral interview answers benefit from rehearsal repetition
- The same "讲一个跨团队协作的例子" story can be reused across companies
- Adding ``reflection`` to STAR (= STAR+R) makes answers stickier and
  more memorable, per the Career-Ops methodology

Stories are tagged by theme — ``prepare_interview`` and
``deep_project_prep`` SKILLs query by tag at retrieval time so the
generated questions can reference EXACTLY the user's pre-prepped
materials. Each retrieval bumps ``used_count`` so the dashboard
surfaces stale stories that haven't been used in a while.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .memory import Store

# Common theme tags — not enforced at the schema level (the column is a
# JSON list of arbitrary strings) but used as a recommended vocabulary
# in the UI dropdown.
RECOMMENDED_TAGS: tuple[str, ...] = (
    "collaboration",      # cross-team / cross-functional
    "conflict",           # disagreement resolution
    "failure",            # what went wrong + recovery
    "learning",           # picking up new tech fast
    "leadership",         # owning a decision
    "ambiguity",          # navigating unclear requirements
    "tradeoff",           # design / engineering choices
    "deadline",           # working under pressure
    "feedback",           # giving / receiving criticism
    "ownership",          # initiative beyond assigned scope
)


@dataclass(frozen=True)
class Story:
    """One behavioral story (STAR + Reflection)."""

    id: int
    title: str
    situation: str
    task: str
    action: str
    result: str
    reflection: str | None
    tags: list[str]
    used_count: int
    confidence: float
    created_at: float


# ── CRUD ───────────────────────────────────────────────────────────


def insert(
    store: Store,
    *,
    title: str,
    situation: str,
    task: str,
    action: str,
    result: str,
    reflection: str | None = None,
    tags: list[str] | None = None,
    confidence: float = 0.5,
) -> Story:
    """Add a story to the bank. Validates required STAR fields are non-empty."""
    for name, val in (
        ("title", title), ("situation", situation),
        ("task", task), ("action", action), ("result", result),
    ):
        if not val or not val.strip():
            raise ValueError(f"{name} is required")
    tags_clean = [t.strip() for t in (tags or []) if t and t.strip()]
    tags_json = json.dumps(tags_clean, ensure_ascii=False)

    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO behavioral_stories"
            "(title, situation, task, action, result, reflection, tags_json, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                title.strip(), situation.strip(), task.strip(),
                action.strip(), result.strip(),
                (reflection or "").strip() or None,
                tags_json,
                max(0.0, min(1.0, confidence)),
            ),
        )
        story_id = int(cur.lastrowid or 0)
        row = conn.execute(
            "SELECT id, title, situation, task, action, result, reflection, "
            "tags_json, used_count, confidence, created_at "
            "FROM behavioral_stories WHERE id = ?",
            (story_id,),
        ).fetchone()
    return _row_to_story(row)


def get(store: Store, story_id: int) -> Story | None:
    with store.connect() as conn:
        row = conn.execute(
            "SELECT id, title, situation, task, action, result, reflection, "
            "tags_json, used_count, confidence, created_at "
            "FROM behavioral_stories WHERE id = ?",
            (story_id,),
        ).fetchone()
    return _row_to_story(row) if row else None


def list_all(store: Store, *, limit: int = 100) -> list[Story]:
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, title, situation, task, action, result, reflection, "
            "tags_json, used_count, confidence, created_at "
            "FROM behavioral_stories ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_story(r) for r in rows]


def search_by_tag(store: Store, tag: str, *, limit: int = 5) -> list[Story]:
    """Return stories whose tags include ``tag`` — for SKILL retrieval.

    Bumps ``used_count`` on each returned story so the dashboard can
    show stale/under-used ones.
    """
    if not tag.strip():
        return []
    pattern = f'%"{tag.strip()}"%'  # crude JSON-substring match
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, title, situation, task, action, result, reflection, "
            "tags_json, used_count, confidence, created_at "
            "FROM behavioral_stories WHERE tags_json LIKE ? "
            "ORDER BY used_count ASC, confidence DESC LIMIT ?",
            (pattern, limit),
        ).fetchall()
        if rows:
            ids = [int(r[0]) for r in rows]
            conn.execute(
                f"UPDATE behavioral_stories SET used_count = used_count + 1 "
                f"WHERE id IN ({','.join('?' * len(ids))})",
                tuple(ids),
            )
    return [_row_to_story(r) for r in rows]


def update_confidence(store: Store, story_id: int, confidence: float) -> Story | None:
    confidence = max(0.0, min(1.0, confidence))
    with store.connect() as conn:
        conn.execute(
            "UPDATE behavioral_stories SET confidence = ? WHERE id = ?",
            (confidence, story_id),
        )
    return get(store, story_id)


def delete(store: Store, story_id: int) -> bool:
    with store.connect() as conn:
        cur = conn.execute(
            "DELETE FROM behavioral_stories WHERE id = ?", (story_id,)
        )
    return (cur.rowcount or 0) > 0


# ── rendering helper for SKILL prompts ─────────────────────────────


def render_for_skill(stories: list[Story], *, max_chars: int = 3000) -> str:
    """Render stories into a single labelled blob suitable for SKILL prompts.

    Used by ``prepare_interview`` / ``deep_project_prep`` when building
    behavioral_questions_tailored — the SKILL sees the user's actual
    rehearsed material and can write per-question ``answer_outline`` that
    reference these stories instead of generic advice.
    """
    if not stories:
        return ""
    out: list[str] = []
    used = 0
    for s in stories:
        block = (
            f"### {s.title}  [tags: {', '.join(s.tags)}]\n"
            f"S: {s.situation}\n"
            f"T: {s.task}\n"
            f"A: {s.action}\n"
            f"R: {s.result}\n"
        )
        if s.reflection:
            block += f"+R: {s.reflection}\n"
        if used + len(block) > max_chars:
            out.append(f"…（还有 {len(stories) - len(out)} 条 stories 因长度省略）")
            break
        out.append(block)
        used += len(block)
    return "\n".join(out).strip()


# ── internal ───────────────────────────────────────────────────────


def _row_to_story(row: tuple) -> Story:
    try:
        tags = json.loads(row[7]) if row[7] else []
    except (json.JSONDecodeError, TypeError):
        tags = []
    return Story(
        id=int(row[0]),
        title=row[1],
        situation=row[2],
        task=row[3],
        action=row[4],
        result=row[5],
        reflection=row[6],
        tags=list(tags),
        used_count=int(row[8]),
        confidence=float(row[9]),
        created_at=float(row[10]),
    )
