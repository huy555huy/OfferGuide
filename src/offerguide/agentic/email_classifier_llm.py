"""LLM-driven email classifier — replaces the regex one.

Sends an email's text to DeepSeek and asks for a structured JSON:
``{kind, confidence, matched_company, extracted, evidence}``.

This is what the user meant by "should be agentic, not regex":

- The LLM reads context, not just keywords. A rejection email that
  mentions "面试" in passing won't be misclassified.
- The LLM extracts structured info (interview time, contact name,
  reference job title) that regex literally can't.
- New email patterns from new HR templates work out-of-the-box; no
  regex updates needed.

Cost: each classification is one DeepSeek-V4-flash call (~200 tokens
output). At V4-flash pricing, ~$0.0002 per email.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from ..llm import LLMClient, LLMError

EmailKind = Literal[
    "submitted",
    "viewed",
    "replied",
    "assessment",
    "interview",
    "rejected",
    "offer",
    "unrelated",
]


@dataclass(frozen=True)
class LLMEmailClassification:
    """LLM output, normalized to the same shape as the regex classifier
    so the UI doesn't care which path was taken."""

    kind: EmailKind
    confidence: float  # 0-1, model's self-reported certainty
    matched_company: str | None
    matched_application_id: int | None
    extracted: dict[str, Any]
    """Structured info — interview_time / contact_name / referenced_role
    / etc. — that regex literally cannot pull out."""

    evidence: list[str]
    """Short quoted snippets from the email that justified the
    classification. Surfaced in the UI for transparency."""

    raw_response: str
    """The LLM's exact JSON for debugging / audit."""


_SYSTEM_PROMPT = """你是 HR 邮件分类器。给定一封邮件全文，输出严格 JSON：

{
  "kind": "submitted" | "viewed" | "replied" | "assessment" | "interview"
        | "rejected" | "offer" | "unrelated",
  "confidence": <float 0-1>,
  "matched_company": <str | null, 邮件提到的招聘方公司名>,
  "extracted": {
    "interview_time": <str | null, 形如 '2026-05-20 14:00' 或 null>,
    "contact_name": <str | null, HR 姓名>,
    "referenced_role": <str | null, 邮件提到的岗位名>,
    "interview_round": <"一面"|"二面"|"三面"|"终面"|"HR 面"|null>,
    "assessment_link": <str | null>,
    "deadline": <str | null>
  },
  "evidence": [<str, 1-3 条邮件原文截取作为证据>]
}

类型定义：
- submitted: 投递回执（"您的简历已成功投递"）
- viewed: HR 已查看（极少出现在邮件，主要是平台站内信）
- replied: 收到您的简历（中性回执，未表达态度）
- assessment: 笔试 / OA / 在线测评通知
- interview: 面试邀约（含一面/二面/终面/HR 面）
- rejected: 拒信（"很遗憾"、"暂时不适合"、"未能进入下一轮"）
- offer: 录用通知 / Offer / "非常荣幸通知您"
- unrelated: newsletter / 营销邮件 / 系统通知 / 其它非求职相关

重要规则：
1. **仔细看上下文**——拒信里出现"面试"是描述以前面试，不是新邀约
2. **confidence 要诚实**——能 100% 确定才给 0.95+；模糊的给 0.5-0.7
3. **extracted 字段没找到必须填 null，不要编造**
4. **evidence 必须是邮件原文片段**，不是你的解读
5. 输出严格 JSON，**不要 markdown 代码块**"""


def classify_email_llm(
    text: str,
    *,
    llm: LLMClient,
    known_companies: list[str] | None = None,
    known_apps_by_company: dict[str, list[int]] | None = None,
    model: str | None = None,
) -> LLMEmailClassification:
    """Classify one email using the LLM.

    ``known_companies`` is passed in the user message so the LLM can
    pick the canonical name from the user's actual company set
    (instead of inventing close variants like "字节" vs "字节跳动").
    """
    if not text or not text.strip():
        return LLMEmailClassification(
            kind="unrelated",
            confidence=0.0,
            matched_company=None,
            matched_application_id=None,
            extracted={},
            evidence=["empty input"],
            raw_response="",
        )

    company_hint = ""
    if known_companies:
        company_hint = (
            f"\n\n【已知公司列表（matched_company 必须从这里选或填 null）】\n"
            f"{', '.join(known_companies)}"
        )

    user_msg = f"【邮件全文】\n{text.strip()}{company_hint}"

    try:
        resp = llm.chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            temperature=0.0,  # classification → deterministic
            json_mode=True,
        )
    except LLMError as e:
        # Fail soft → return unclassified with evidence about the error
        return LLMEmailClassification(
            kind="unrelated",
            confidence=0.0,
            matched_company=None,
            matched_application_id=None,
            extracted={},
            evidence=[f"LLM call failed: {e}"],
            raw_response="",
        )

    try:
        parsed = json.loads(resp.content)
    except json.JSONDecodeError:
        return LLMEmailClassification(
            kind="unrelated",
            confidence=0.0,
            matched_company=None,
            matched_application_id=None,
            extracted={},
            evidence=[f"LLM returned non-JSON: {resp.content[:200]}"],
            raw_response=resp.content,
        )

    kind = parsed.get("kind", "unrelated")
    if kind not in EmailKind.__args__:  # type: ignore[attr-defined]
        kind = "unrelated"

    matched_company = parsed.get("matched_company") or None
    app_id: int | None = None
    if matched_company and known_apps_by_company:
        ids = known_apps_by_company.get(matched_company, [])
        if len(ids) == 1:
            app_id = ids[0]

    return LLMEmailClassification(
        kind=kind,
        confidence=float(parsed.get("confidence", 0.0)),
        matched_company=matched_company,
        matched_application_id=app_id,
        extracted=parsed.get("extracted", {}) or {},
        evidence=list(parsed.get("evidence", [])),
        raw_response=resp.content,
    )


def classify_email_batch_llm(
    texts: list[str],
    *,
    llm: LLMClient,
    known_companies: list[str] | None = None,
    known_apps_by_company: dict[str, list[int]] | None = None,
    model: str | None = None,
) -> list[LLMEmailClassification]:
    """Classify a list of emails. Each is a separate LLM call.

    No batching into one call — keeps each classification independent
    and lets the LLM focus. ~$0.0002 per email at V4-flash pricing.
    """
    return [
        classify_email_llm(
            t,
            llm=llm,
            known_companies=known_companies,
            known_apps_by_company=known_apps_by_company,
            model=model,
        )
        for t in texts
    ]
