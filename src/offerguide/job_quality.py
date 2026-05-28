"""Deterministic quality gates for discovered jobs.

The matcher can decide whether a decent JD is worth applying to. This module
keeps obviously unusable rows out of that expensive/user-facing path: dead
links, aggregator articles, and roles that are clearly outside a tech/AI
graduate search.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from .platforms._spec import RawJob


TENCENT_CAMPUS_DEAD_DETAIL_RE = re.compile(
    r"^https://join\.qq\.com/jobdesc\.html\?postId=(\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class JobQualityVerdict:
    usable: bool
    reason: str = ""
    detail: str = ""


def normalize_known_job_url(
    *, source: str | None, url: str | None, source_id: str | None = None,
) -> str | None:
    """Return a current public URL for sources with known stale URL shapes."""
    if not url:
        return url
    source_l = (source or "").lower()
    if source_l == "tencent_campus":
        post_id = _extract_tencent_campus_post_id(url) or (source_id or "").strip()
        if post_id:
            return "https://join.qq.com/post_detail.html?" + urlencode({"postid": post_id})
    return url


def quality_verdict_for_raw_job(rj: RawJob) -> JobQualityVerdict:
    return quality_verdict_for_job(
        source=rj.source,
        title=rj.title,
        company=rj.company,
        url=rj.url,
        raw_text=rj.raw_text,
        extras=rj.extras,
    )


def quality_verdict_for_job(
    *,
    source: str | None,
    title: str | None,
    company: str | None = None,
    url: str | None = None,
    raw_text: str | None = None,
    extras_json: str | None = None,
    extras: dict[str, Any] | None = None,
) -> JobQualityVerdict:
    """Classify whether a row is fit for recommendation/scoring.

    This is intentionally conservative. It blocks only rows with strong signals
    of being unusable for the current product promise.
    """
    source_l = (source or "").lower()
    extras_obj = extras if isinstance(extras, dict) else _load_extras(extras_json)
    title_s = title or ""
    url_s = url or ""

    if extras_obj.get("dead"):
        return JobQualityVerdict(False, "dead_url", "user/platform marked URL dead")

    if source_l == "zerovoice_repo" and extras_obj.get("link_type") != "official_ats":
        return JobQualityVerdict(False, "not_direct_apply_link", "0voice row is not an official ATS URL")

    if _is_known_dead_url(source_l, url_s):
        return JobQualityVerdict(False, "known_dead_url", url_s)

    blocked = _blocked_role_token(title_s)
    if blocked:
        return JobQualityVerdict(False, "off_target_role", blocked)

    if source_l == "nowcoder" and _has_nowcoder_platform_signal(url_s, extras_obj):
        target = _nowcoder_target_signal(title_s, raw_text or "", extras_obj)
        if not target:
            return JobQualityVerdict(False, "low_relevance_nowcoder", "no tech/AI/intern/campus signal")

    return JobQualityVerdict(True)


def is_job_usable_for_recommendation(
    *,
    source: str | None,
    title: str | None,
    company: str | None = None,
    url: str | None = None,
    raw_text: str | None = None,
    extras_json: str | None = None,
    extras: dict[str, Any] | None = None,
) -> bool:
    return quality_verdict_for_job(
        source=source,
        title=title,
        company=company,
        url=url,
        raw_text=raw_text,
        extras_json=extras_json,
        extras=extras,
    ).usable


def _is_known_dead_url(source: str, url: str) -> bool:
    if source == "tencent_campus" and TENCENT_CAMPUS_DEAD_DETAIL_RE.match(url):
        return True
    return url.lower().endswith("/404.html")


def _extract_tencent_campus_post_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "join.qq.com":
        return None
    qs = parse_qs(parsed.query)
    for key in ("postid", "postId"):
        values = qs.get(key)
        if values and values[0].strip():
            return values[0].strip()
    return None


_HARD_BLOCKED_ROLE_TOKENS: tuple[str, ...] = (
    # HR / finance / legal / admin / education roles observed in nowcoder dogfood.
    "人力", "hr", "招聘", "薪酬", "行政", "法务", "合同",
    "会计", "财务", "结算", "出纳", "审计", "税务",
    "教师", "老师", "家教", "辅导", "考研", "保研", "课程顾问",
    # Factory / service / sales noise.
    "坐岗", "长白班", "普工", "厂工", "操作工", "装配工", "流水线",
    "包吃住", "倒班", "夜班", "焊工", "钳工", "电工", "包装工",
    "拣货", "分拣员", "理货员", "保洁", "保安", "收银员", "服务员",
    "厨师", "外卖", "骑手", "配送", "司机", "美容", "美甲", "按摩",
    "电销", "房产中介", "保险代理", "直播带货",
)

_SOFT_BLOCKED_ROLE_TOKENS: tuple[str, ...] = (
    "合规", "销售", "客服",
)

_TECH_TITLE_TOKENS: tuple[str, ...] = (
    "算法", "研发", "开发", "工程师", "数据", "ai", "aigc", "agent",
    "llm", "rag", "nlp", "机器学习", "深度学习", "大模型", "智能体",
)


def _blocked_role_token(title: str) -> str:
    lower = title.lower()
    for token in _HARD_BLOCKED_ROLE_TOKENS:
        if token.lower() in lower:
            return token
    if any(token.lower() in lower for token in _TECH_TITLE_TOKENS):
        return ""
    for token in _SOFT_BLOCKED_ROLE_TOKENS:
        if token.lower() in lower:
            return token
    return ""


_TARGET_TITLE_TOKENS: tuple[str, ...] = (
    "ai", "aigc", "agent", "llm", "rag", "nlp", "cv", "机器学习",
    "深度学习", "大模型", "智能体", "算法", "推荐", "搜索", "搜广推",
    "数据", "后端", "前端", "客户端", "服务端", "研发", "开发", "工程师",
    "java", "python", "golang", "go", "c++", "web", "vue", "react",
    "实习", "校招", "应届",
)

_TARGET_CAREER_TOKENS: tuple[str, ...] = (
    "算法", "数据", "开发", "研发", "工程", "后端", "前端", "客户端",
    "服务端", "java", "python", "golang", "c++", "测试", "运维",
    "产品经理",
)


def _nowcoder_target_signal(title: str, raw_text: str, extras: dict[str, Any]) -> bool:
    hay_title = title.lower()
    if any(token.lower() in hay_title for token in _TARGET_TITLE_TOKENS):
        return True

    discovered_keyword = str(extras.get("discovered_keyword") or "")
    if discovered_keyword and "sitemap walk" not in discovered_keyword.lower():
        if any(
            token.lower() in discovered_keyword.lower()
            for token in _TARGET_TITLE_TOKENS + ("diffusion", "rlhf", "pytorch")
        ):
            return True

    for key in ("careerJobName", "industryName", "jobKeys"):
        value = extras.get(key)
        if isinstance(value, list):
            text = " ".join(str(v) for v in value)
        else:
            text = str(value or "")
        if any(token.lower() in text.lower() for token in _TARGET_CAREER_TOKENS):
            return True

    # Body-only signal is weaker, but useful for titles like "研究实习生".
    body = raw_text.lower()
    return any(token.lower() in body for token in ("pytorch", "机器学习", "大模型", "算法", "python"))


def _has_nowcoder_platform_signal(url: str, extras: dict[str, Any]) -> bool:
    if "nowcoder.com/jobs/detail/" in (url or "").lower():
        return True
    return any(
        key in extras
        for key in ("careerJobName", "industryName", "graduationYear", "jobKeys")
    )


def _load_extras(extras_json: str | None) -> dict[str, Any]:
    if not extras_json:
        return {}
    try:
        parsed = json.loads(extras_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
