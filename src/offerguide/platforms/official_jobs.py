"""Verified official recruitment sources.

This module is intentionally narrower than the generic web-search agent. It
only exposes sources we have actually probed and can parse into real JDs. When
an official site is a login wall or a JavaScript shell whose public API is not
verified, we report that as a source status instead of pretending it works.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

import httpx

from ._spec import RawJob


DEFAULT_TIMEOUT_S = 15.0
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 Safari/605.1.15"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
}

TENCENT_CAMPUS_SEARCH_URL = "https://join.qq.com/api/v1/position/searchPosition"
TENCENT_CAMPUS_DETAIL_URL = "https://join.qq.com/api/v1/jobDetails/getJobDetailsByPostId"
TENCENT_SOCIAL_SEARCH_URL = "https://careers.tencent.com/tencentcareer/api/post/Query"
TENCENT_SOCIAL_DETAIL_URL = "https://careers.tencent.com/tencentcareer/api/post/ByPostId"
BAIDU_CAMPUS_LIST_URL = "https://talent.baidu.com/jobs/list"


@dataclass(frozen=True)
class SourceStatus:
    """Observed capability status for a company/source."""

    company_key: str
    label: str
    status: str
    evidence_url: str
    note: str


@dataclass
class SourceSearchResult:
    """One source probe result."""

    source: str
    status: str
    evidence_url: str
    jobs: list[RawJob] = field(default_factory=list)
    note: str = ""


SOURCE_LANDSCAPE: dict[str, SourceStatus] = {
    "tencent": SourceStatus(
        company_key="tencent",
        label="腾讯",
        status="verified_public",
        evidence_url=TENCENT_CAMPUS_SEARCH_URL,
        note="校招 join.qq.com 和社招 careers.tencent.com 均有可读 JSON 接口。",
    ),
    "baidu": SourceStatus(
        company_key="baidu",
        label="百度",
        status="verified_public",
        evidence_url=BAIDU_CAMPUS_LIST_URL,
        note="校招列表页服务端渲染 window.__INITIAL_DATA__，包含真实岗位全文字段。",
    ),
    "bytedance": SourceStatus(
        company_key="bytedance",
        label="字节跳动",
        status="unverified_js_shell",
        evidence_url="https://jobs.bytedance.com/campus/",
        note="robots 允许访问 campus，但静态 HTML 是前端壳；本轮未验证出稳定公开 JD API。",
    ),
    "alibaba": SourceStatus(
        company_key="alibaba",
        label="阿里巴巴",
        status="unverified_js_shell",
        evidence_url="https://talent.alibaba.com/",
        note="官网为前端壳；已看到招聘前端资产，但未验证出稳定公开 JD API。",
    ),
    "meituan": SourceStatus(
        company_key="meituan",
        label="美团",
        status="login_required",
        evidence_url="https://zhaopin.meituan.com/web/campus",
        note="岗位 API 探测返回未登录；自动化应走用户浏览器会话或官方登录后人工确认。",
    ),
    "xiaohongshu": SourceStatus(
        company_key="xiaohongshu",
        label="小红书",
        status="login_required",
        evidence_url="https://job.xiaohongshu.com/",
        note="招聘 API 探测返回用户未登录；不能声明后台可无登录抓取。",
    ),
    "boss": SourceStatus(
        company_key="boss",
        label="BOSS 直聘",
        status="browser_session_required",
        evidence_url="https://www.zhipin.com/",
        note="BOSS 强登录/风控，真实路径是用户浏览器扩展采集和点击发送。",
    ),
}


def search_verified_official_jobs(
    *,
    company: str | None,
    keyword: str,
    limit: int = 5,
    client: httpx.Client | None = None,
) -> list[SourceSearchResult]:
    """Search only verified official sources and report unsupported ones honestly."""

    keyword = keyword.strip()
    if not keyword:
        return [
            SourceSearchResult(
                source="official_sources",
                status="error",
                evidence_url="",
                note="keyword is required",
            )
        ]
    limit = max(1, min(int(limit or 5), 20))
    key = company_key(company)
    all_supported = key is None

    own_client = client or httpx.Client(
        timeout=DEFAULT_TIMEOUT_S,
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
    )
    results: list[SourceSearchResult] = []
    try:
        if all_supported or key == "tencent":
            results.extend(search_tencent_jobs(keyword=keyword, limit=limit, client=own_client))
        if all_supported or key == "baidu":
            results.append(search_baidu_campus_jobs(keyword=keyword, limit=limit, client=own_client))
        if key in SOURCE_LANDSCAPE and SOURCE_LANDSCAPE[key].status != "verified_public":
            st = SOURCE_LANDSCAPE[key]
            results.append(
                SourceSearchResult(
                    source=st.company_key,
                    status=st.status,
                    evidence_url=st.evidence_url,
                    note=st.note,
                )
            )
        if not results:
            results.append(
                SourceSearchResult(
                    source="official_sources",
                    status="unsupported_company",
                    evidence_url="",
                    note=(
                        f"暂未验证 {company or '该公司'} 的官方招聘接口；"
                        "请让 agent 先 web_search/fetch_url 探路，不要直接假设可抓。"
                    ),
                )
            )
    finally:
        if client is None:
            own_client.close()
    return results


def company_key(company: str | None) -> str | None:
    """Normalize a user/company string to one of our observed source keys."""

    if not company:
        return None
    s = company.strip().lower()
    if not s:
        return None
    if any(token in s for token in ("巨头", "大厂", "bat", "互联网")):
        return None
    mappings = [
        ("tencent", ("腾讯", "tencent", "qq.com")),
        ("baidu", ("百度", "baidu")),
        ("bytedance", ("字节", "bytedance", "douyin", "抖音", "tiktok")),
        ("alibaba", ("阿里", "alibaba", "taobao", "淘宝", "aliyun")),
        ("meituan", ("美团", "meituan")),
        ("xiaohongshu", ("小红书", "xiaohongshu", "rednote")),
        ("boss", ("boss", "zhipin", "直聘")),
    ]
    for key, aliases in mappings:
        if any(alias in s for alias in aliases):
            return key
    return "unknown"


def search_tencent_jobs(
    *,
    keyword: str,
    limit: int,
    client: httpx.Client,
) -> list[SourceSearchResult]:
    """Search verified Tencent campus + social JSON APIs."""

    campus_limit = max(1, limit)
    social_limit = max(1, limit)
    return [
        search_tencent_campus_jobs(keyword=keyword, limit=campus_limit, client=client),
        search_tencent_social_jobs(keyword=keyword, limit=social_limit, client=client),
    ]


def search_tencent_campus_jobs(
    *,
    keyword: str,
    limit: int,
    client: httpx.Client,
) -> SourceSearchResult:
    headers = {
        "Referer": "https://join.qq.com/post.html",
        "Origin": "https://join.qq.com",
        "Accept": "application/json, text/plain, */*",
    }
    payload = {"keyword": keyword, "pageIndex": 1, "pageSize": limit}
    try:
        resp = client.post(TENCENT_CAMPUS_SEARCH_URL, json=payload, headers=headers)
        data = resp.json()
    except Exception as e:
        return SourceSearchResult(
            source="tencent_campus",
            status="error",
            evidence_url=TENCENT_CAMPUS_SEARCH_URL,
            note=f"search failed: {type(e).__name__}: {e}",
        )
    if resp.status_code != 200 or data.get("status") != 0:
        return SourceSearchResult(
            source="tencent_campus",
            status="error",
            evidence_url=TENCENT_CAMPUS_SEARCH_URL,
            note=f"search returned HTTP {resp.status_code}: {str(data)[:300]}",
        )

    jobs: list[RawJob] = []
    errors: list[str] = []
    for item in (data.get("data") or {}).get("positionList") or []:
        post_id = str(item.get("postId") or "").strip()
        if not post_id:
            continue
        try:
            detail_resp = client.get(
                TENCENT_CAMPUS_DETAIL_URL,
                params={"postId": post_id},
                headers=headers,
            )
            detail_data = detail_resp.json()
            if detail_resp.status_code != 200 or detail_data.get("status") != 0:
                errors.append(f"{post_id}: detail status {detail_resp.status_code}/{detail_data}")
                continue
            detail = detail_data.get("data") or {}
            jobs.append(raw_job_from_tencent_campus(item, detail))
        except Exception as e:
            errors.append(f"{post_id}: {type(e).__name__}: {e}")
    note = f"public JSON API; count={(data.get('data') or {}).get('count')}"
    if errors:
        note += "; detail_errors=" + " | ".join(errors[:3])
    return SourceSearchResult(
        source="tencent_campus",
        status="ok",
        evidence_url=TENCENT_CAMPUS_SEARCH_URL,
        jobs=jobs,
        note=note,
    )


def raw_job_from_tencent_campus(item: dict[str, Any], detail: dict[str, Any]) -> RawJob:
    post_id = str(detail.get("postId") or item.get("postId") or "").strip()
    title = str(detail.get("title") or item.get("positionTitle") or "腾讯校招岗位").strip()
    location = _join_list(detail.get("workCityList")) or str(item.get("workCities") or "").strip()
    url = f"https://join.qq.com/jobdesc.html?postId={post_id}" if post_id else "https://join.qq.com/post.html"
    raw_text = _join_lines(
        [
            f"职位名: {title}",
            "公司: 腾讯",
            f"项目: {item.get('projectName') or detail.get('projectType') or ''}",
            f"岗位族: {detail.get('tidName') or ''}",
            f"BG: {item.get('bgs') or ''}",
            f"工作城市: {location}",
            f"面试城市: {_join_list(detail.get('recruitCityList'))}",
            "",
            "## 岗位介绍",
            str(detail.get("desc") or "").strip(),
            "",
            "## 任职要求 / 岗位职责",
            str(detail.get("request") or "").strip(),
        ]
    )
    return RawJob(
        source="tencent_campus",
        source_id=post_id or None,
        url=url,
        title=title,
        company="腾讯",
        location=location or None,
        raw_text=raw_text,
        extras={
            "source_verified": True,
            "source_kind": "official_json_api",
            "evidence_url": TENCENT_CAMPUS_SEARCH_URL,
            "detail_evidence_url": TENCENT_CAMPUS_DETAIL_URL,
            "platform": "join.qq.com",
            "project_name": item.get("projectName"),
            "recruit_label": item.get("recruitLabelName"),
            "raw_item": item,
            "raw_detail": detail,
        },
    )


def search_tencent_social_jobs(
    *,
    keyword: str,
    limit: int,
    client: httpx.Client,
) -> SourceSearchResult:
    params = {
        "timestamp": "1",
        "countryId": "",
        "cityId": "",
        "bgIds": "",
        "productId": "",
        "categoryId": "",
        "parentCategoryId": "",
        "attrId": "",
        "keyword": keyword,
        "pageIndex": 1,
        "pageSize": limit,
        "language": "zh-cn",
        "area": "cn",
    }
    headers = {"Referer": "https://careers.tencent.com/"}
    try:
        resp = client.get(TENCENT_SOCIAL_SEARCH_URL, params=params, headers=headers)
        data = resp.json()
    except Exception as e:
        return SourceSearchResult(
            source="tencent_social",
            status="error",
            evidence_url=TENCENT_SOCIAL_SEARCH_URL,
            note=f"search failed: {type(e).__name__}: {e}",
        )
    if resp.status_code != 200 or data.get("Code") != 200:
        return SourceSearchResult(
            source="tencent_social",
            status="error",
            evidence_url=TENCENT_SOCIAL_SEARCH_URL,
            note=f"search returned HTTP {resp.status_code}: {str(data)[:300]}",
        )

    jobs: list[RawJob] = []
    errors: list[str] = []
    for item in (data.get("Data") or {}).get("Posts") or []:
        post_id = str(item.get("PostId") or "").strip()
        if not post_id:
            continue
        try:
            detail_resp = client.get(
                TENCENT_SOCIAL_DETAIL_URL,
                params={"timestamp": "1", "postId": post_id, "language": "zh-cn"},
                headers={"Referer": f"https://careers.tencent.com/jobdesc.html?postId={post_id}"},
            )
            detail_data = detail_resp.json()
            if detail_resp.status_code != 200 or detail_data.get("Code") != 200:
                errors.append(f"{post_id}: detail status {detail_resp.status_code}/{detail_data}")
                continue
            jobs.append(raw_job_from_tencent_social(detail_data.get("Data") or item))
        except Exception as e:
            errors.append(f"{post_id}: {type(e).__name__}: {e}")
    note = f"public JSON API; count={(data.get('Data') or {}).get('Count')}"
    if errors:
        note += "; detail_errors=" + " | ".join(errors[:3])
    return SourceSearchResult(
        source="tencent_social",
        status="ok",
        evidence_url=TENCENT_SOCIAL_SEARCH_URL,
        jobs=jobs,
        note=note,
    )


def raw_job_from_tencent_social(detail: dict[str, Any]) -> RawJob:
    post_id = str(detail.get("PostId") or "").strip()
    title = str(detail.get("RecruitPostName") or "腾讯社招岗位").strip()
    location = str(detail.get("LocationName") or "").strip()
    url = str(detail.get("PostURL") or "").strip()
    if post_id and (not url or url.startswith("http://")):
        url = f"https://careers.tencent.com/jobdesc.html?postId={post_id}"
    raw_text = _join_lines(
        [
            f"职位名: {title}",
            "公司: 腾讯",
            f"BG: {detail.get('BGName') or ''}",
            f"产品/平台: {detail.get('ProductName') or ''}",
            f"类别: {detail.get('CategoryName') or ''}",
            f"地点: {location}",
            f"经验要求: {detail.get('RequireWorkYearsName') or ''}",
            f"更新时间: {detail.get('LastUpdateTime') or ''}",
            "",
            "## 岗位职责",
            str(detail.get("Responsibility") or "").strip(),
            "",
            "## 任职要求",
            str(detail.get("Requirement") or "").strip(),
            "",
            "## 岗位介绍",
            str(detail.get("Introduction") or "").strip(),
        ]
    )
    return RawJob(
        source="tencent_social",
        source_id=post_id or None,
        url=url or None,
        title=title,
        company="腾讯",
        location=location or None,
        raw_text=raw_text,
        extras={
            "source_verified": True,
            "source_kind": "official_json_api",
            "evidence_url": TENCENT_SOCIAL_SEARCH_URL,
            "detail_evidence_url": TENCENT_SOCIAL_DETAIL_URL,
            "platform": "careers.tencent.com",
            "raw_detail": detail,
        },
    )


def search_baidu_campus_jobs(
    *,
    keyword: str,
    limit: int,
    client: httpx.Client,
) -> SourceSearchResult:
    url = f"{BAIDU_CAMPUS_LIST_URL}?search={quote_plus(keyword)}"
    try:
        resp = client.get(url, headers={"Referer": BAIDU_CAMPUS_LIST_URL})
        html = resp.text
        if resp.status_code != 200:
            raise ValueError(f"HTTP {resp.status_code}")
        jobs = parse_baidu_campus_jobs(html, evidence_url=url)[:limit]
    except Exception as e:
        return SourceSearchResult(
            source="baidu_campus",
            status="error",
            evidence_url=url,
            note=f"SSR parse failed: {type(e).__name__}: {e}",
        )
    return SourceSearchResult(
        source="baidu_campus",
        status="ok",
        evidence_url=url,
        jobs=jobs,
        note="server-rendered window.__INITIAL_DATA__.listData.listDetailData",
    )


_BAIDU_INITIAL_RE = re.compile(r"window\.__INITIAL_DATA__\s*=\s*", re.MULTILINE)
_JOB_CODE_RE = re.compile(r"\((J\d+)\)")


def parse_baidu_campus_jobs(html: str, *, evidence_url: str | None = None) -> list[RawJob]:
    data = parse_baidu_initial_data(html)
    list_data = data.get("listData") or {}
    recruit_type = str(list_data.get("recruitType") or "GRADUATE")
    rows = list_data.get("listDetailData") or []
    if not isinstance(rows, list):
        return []
    return [
        raw_job_from_baidu_campus(row, recruit_type=recruit_type, evidence_url=evidence_url)
        for row in rows
        if isinstance(row, dict)
    ]


def parse_baidu_initial_data(html: str) -> dict[str, Any]:
    m = _BAIDU_INITIAL_RE.search(html)
    if not m:
        raise ValueError("window.__INITIAL_DATA__ not found")
    payload = html[m.end() :]
    payload = re.sub(r":\s*undefined(?=[,}])", ": null", payload)
    data, _consumed = json.JSONDecoder().raw_decode(payload)
    if not isinstance(data, dict):
        raise ValueError("window.__INITIAL_DATA__ is not an object")
    return data


def raw_job_from_baidu_campus(
    row: dict[str, Any],
    *,
    recruit_type: str,
    evidence_url: str | None = None,
) -> RawJob:
    title = str(row.get("name") or "百度校招岗位").strip()
    post_id = str(row.get("postId") or "").strip()
    job_code = _extract_job_code(title)
    location = str(row.get("workPlace") or "").strip()
    url = (
        f"https://talent.baidu.com/jobs/detail/{recruit_type}/{post_id}"
        if post_id
        else evidence_url
    )
    raw_text = _join_lines(
        [
            f"职位名: {title}",
            "公司: 百度",
            f"项目: {row.get('projectType') or ''}",
            f"类别: {row.get('postType') or ''}",
            f"地点: {location}",
            f"发布时间: {row.get('publishDate') or ''}",
            f"更新时间: {row.get('updateDate') or ''}",
            f"笔试: {row.get('writeExaminationDate') or ''}",
            f"面试: {row.get('interviewDate') or ''}",
            "",
            "## 工作内容",
            str(row.get("workContent") or "").strip(),
            "",
            "## 任职要求",
            str(row.get("serviceCondition") or "").strip(),
        ]
    )
    return RawJob(
        source="baidu_campus",
        source_id=post_id or job_code,
        url=url,
        title=title,
        company="百度",
        location=location or None,
        raw_text=raw_text,
        extras={
            "source_verified": True,
            "source_kind": "official_ssr",
            "evidence_url": evidence_url or BAIDU_CAMPUS_LIST_URL,
            "platform": "talent.baidu.com",
            "job_code": job_code,
            "job_id": row.get("jobId"),
            "post_id": post_id,
            "recruit_type": recruit_type,
            "raw_row": row,
        },
    )


def _extract_job_code(title: str) -> str | None:
    m = _JOB_CODE_RE.search(title)
    return m.group(1) if m else None


def _join_list(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value or "").strip()


def _join_lines(parts: list[str]) -> str:
    return "\n".join(p for p in parts if p is not None and str(p).strip() != "").strip()
