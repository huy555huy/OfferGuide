"""Lightweight HR-email classifier — paste-in version.

This is the **paste-in** alternative to a full IMAP integration:
the user copies one or more HR emails into a textarea, the backend
runs cheap regex pattern matching to classify each into an
``application_events.kind`` (or ``"unrelated"``) and tries to find
which application the email is talking about (by company-name match).

When patterns are ambiguous and a high-stakes kind is at play
(interview / offer / rejected), the caller may optionally fall back
to an LLM for a second opinion. That hand-off is *not* implemented
here — this module is pure-Python deterministic.

Real HR-email patterns (from sourced templates and scraped sample
subjects across 字节跳动 / 阿里巴巴 / 腾讯 / 美团 / 京东 校招):

- 笔试 / OA: '笔试通知', '在线测评', '笔试链接', 'OA test', '在线笔试'
- 面试: '面试邀请', '面试通知', '邀请您参加', 'interview invitation',
        '面试时间', '面试地点'
- 拒信: '很遗憾', '暂时不适合', '感谢您的关注', '留下了深刻印象',
        'thank you for your application', '未能进入下一轮'
- Offer: '录用通知', 'offer', '非常荣幸', '录用意向'
- 收到回复: '已收到您的简历', '收到您的投递', '您的申请已收到'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

EmailKind = Literal[
    "submitted",     # 投递回执
    "viewed",        # 简历已查看 (rare in email — usually platform-only)
    "replied",       # 收到您的简历回执
    "assessment",    # 笔试 / OA
    "interview",     # 面试邀约
    "rejected",      # 拒信
    "offer",         # offer / 录用通知
    "unrelated",     # 与求职无关——不该入 application_events
]


# Patterns ordered roughly by specificity (most specific first wins).
# Each entry: (kind, regex, weight)
# Higher weight = stronger signal. Multiple matches accumulate weight per kind.
_PATTERNS: list[tuple[EmailKind, str, int]] = [
    # ── offer (highest stakes, must catch first) ──
    ("offer",      r"录用通知|录用意向|offer 函|offer letter", 5),
    ("offer",      r"\boffer\b(?!.*拒).*(发出|签订|收到)", 4),
    ("offer",      r"非常荣幸.*(录用|加入|成为)", 3),
    ("offer",      r"(欢迎.*加入|期待.*入职)", 2),

    # ── interview ──
    ("interview",  r"面试邀请|面试通知|interview invitation", 5),
    ("interview",  r"邀请您?参加.*(面试|interview)", 4),
    ("interview",  r"面试时间|面试地点|面试形式|面试链接", 3),
    ("interview",  r"(一面|二面|三面|HR\s*面|终面|复试|technical interview)", 3),

    # ── assessment / OA / 笔试 ──
    ("assessment", r"笔试通知|笔试邀请|笔试链接|笔试时间", 5),
    ("assessment", r"在线测评|在线笔试|online assessment|\bOA test\b", 4),
    ("assessment", r"测试链接|测评地址|笔试系统", 3),

    # ── rejected (must check AFTER above; some rejects mention 'interview') ──
    ("rejected",   r"很遗憾.*(未能|不能|无法).*(进入|继续|匹配)", 5),
    ("rejected",   r"暂时不适合|无法.*入选|未通过.*筛选", 4),
    ("rejected",   r"感谢.*(关注|投递|参与).*(留下|印象)", 3),
    ("rejected",   r"thank you for.*application", 2),

    # ── replied (received your resume — neutral acknowledgment) ──
    ("replied",    r"已收到.*简历|收到您的投递|申请已收到|application received", 4),
    ("replied",    r"我们将.*尽快.*(回复|联系)", 2),

    # ── submitted (回执 from the platform itself) ──
    ("submitted",  r"投递成功|投递回执|您的简历已经成功投递", 4),
]


@dataclass(frozen=True)
class EmailClassification:
    """Result of running pattern matching over one email."""

    kind: EmailKind
    confidence: float
    """0-1, ratio of best match weight to total matched weight.
    >= 0.7 = high confidence; lower = ambiguous (caller may want LLM
    second opinion)."""

    matched_company: str | None
    """Best-guess company name in the email body, matched against the
    ``jobs.company`` set provided by the caller. ``None`` if no
    confident match."""

    matched_application_id: int | None
    """If we could uniquely identify an application this email refers
    to (matched company has exactly one active application), this is
    its id. Otherwise None — the UI lets the user disambiguate."""

    evidence: list[str]
    """Substrings from the email that triggered the classification —
    for transparency in the UI."""


def classify(
    text: str,
    *,
    known_companies: list[str] | None = None,
    known_apps_by_company: dict[str, list[int]] | None = None,
) -> EmailClassification:
    """Classify a single email's text into an EmailKind.

    ``known_companies`` lets the classifier match company names from
    the user's actual ``jobs.company`` set (so we don't blindly trust
    the email sender domain). ``known_apps_by_company`` maps company →
    list of active application ids; if exactly one app exists for the
    matched company, we attach it to the result.
    """
    if not text or not text.strip():
        return EmailClassification(
            kind="unrelated",
            confidence=0.0,
            matched_company=None,
            matched_application_id=None,
            evidence=["empty input"],
        )

    # Score each kind by summed weight of matched patterns
    scores: dict[EmailKind, int] = {}
    evidence: list[str] = []
    for kind, pattern, weight in _PATTERNS:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            scores[kind] = scores.get(kind, 0) + weight
            snippet = m.group(0)[:60]
            evidence.append(f"{kind}: {snippet!r}")

    if not scores:
        return EmailClassification(
            kind="unrelated",
            confidence=0.0,
            matched_company=_match_company(text, known_companies or []),
            matched_application_id=None,
            evidence=["no patterns matched"],
        )

    best_kind, best_weight = max(scores.items(), key=lambda kv: kv[1])
    total_weight = sum(scores.values())
    confidence = best_weight / total_weight if total_weight > 0 else 0.0

    company = _match_company(text, known_companies or [])
    app_id: int | None = None
    if company and known_apps_by_company:
        ids = known_apps_by_company.get(company, [])
        if len(ids) == 1:
            app_id = ids[0]

    return EmailClassification(
        kind=best_kind,
        confidence=confidence,
        matched_company=company,
        matched_application_id=app_id,
        evidence=evidence,
    )


def _match_company(text: str, known_companies: list[str]) -> str | None:
    """Find the best-matching company name in the text.

    Substring match, prefer longest match (so '阿里云' wins over '阿里').
    """
    if not known_companies:
        return None
    matches = [c for c in known_companies if c and c in text]
    if not matches:
        return None
    return max(matches, key=len)


def classify_batch(
    emails: list[str],
    *,
    known_companies: list[str] | None = None,
    known_apps_by_company: dict[str, list[int]] | None = None,
) -> list[EmailClassification]:
    """Classify a list of emails. Just maps over ``classify``."""
    return [
        classify(
            e,
            known_companies=known_companies,
            known_apps_by_company=known_apps_by_company,
        )
        for e in emails
    ]


def split_email_dump(blob: str) -> list[str]:
    """Best-effort split of a pasted email *dump* into individual emails.

    Heuristic: blocks separated by 2+ blank lines, OR lines starting with
    'From:' (mbox-style). Each chunk goes to ``classify`` as its own
    email. If the blob has neither separator, returns ``[blob]``.
    """
    if not blob.strip():
        return []

    # Try mbox-style 'From: ' separator first
    if re.search(r"^From:\s", blob, flags=re.MULTILINE):
        chunks = re.split(r"(?=^From:\s)", blob, flags=re.MULTILINE)
        return [c.strip() for c in chunks if c.strip()]

    # Else split on 2+ blank lines
    chunks = re.split(r"\n\s*\n\s*\n+", blob)
    return [c.strip() for c in chunks if c.strip()]
