"""W17 — recruit type classifier (deterministic, no LLM).

Classifies a job into one of:
- ``summer_intern``    — 暑期实习 (5-9 月入职, 通常带转正 path)
- ``daily_intern``     — 日常实习 (任何时间, 不一定有转正)
- ``campus_fulltime``  — 校招正式岗 (毕业入职)
- ``social``           — 社招 (要工作经验, 应届生不该投)
- ``unknown``          — 没明确信号

Why deterministic (not LLM): the platforms expose explicit enum fields
(腾讯 projectName='应届实习', 百度 projectType='暑期实习项目'/'日常实习项目')
and we shouldn't burn LLM cost on what's already structured.

Empirically observed enum values (probed 2026-05-10):

腾讯校招 (`join.qq.com` /api/v1/position/searchPosition) projectName 当前
全部为 ``'应届实习'`` (5 月节奏 — 应届实习就是暑期可转正实习). 秋招会有
`'正式校招'` 等其他值.

百度 (`talent.baidu.com/jobs/list`):
  - ``recruitType=GRADUATE``: ``projectType ∈ {'校招', 'AIDU项目', '管培生项目'}``
  - ``recruitType=INTERN``:   ``projectType ∈ {'暑期实习项目', '日常实习项目'}``

腾讯社招 (`careers.tencent.com`): 全部归 social (不是用户投的方向).

nowcoder: 没显式 enum, 看 title + extras.graduationYear:
  - title 含 "暑期实习" → summer_intern
  - title 含 "日常实习" → daily_intern
  - title 含 "实习" 没标暑期/日常 → daily_intern (大厂暑期都会标)
  - title 含 "校招" → campus_fulltime
  - 其余 → unknown
"""

from __future__ import annotations

import json
from typing import Any

# Stable canonical labels — UI templates + tests reference these.
SUMMER_INTERN = "summer_intern"
DAILY_INTERN = "daily_intern"
CAMPUS_FULLTIME = "campus_fulltime"
SOCIAL = "social"
UNKNOWN = "unknown"

ALL_TYPES = (SUMMER_INTERN, DAILY_INTERN, CAMPUS_FULLTIME, SOCIAL, UNKNOWN)

# Display labels for the UI (Chinese, matches what users see on platforms).
LABEL_ZH: dict[str, str] = {
    SUMMER_INTERN:   "暑期实习",
    DAILY_INTERN:    "日常实习",
    CAMPUS_FULLTIME: "校招正式",
    SOCIAL:          "社招",
    UNKNOWN:         "未分类",
}

# Color hint for tile/pill (matches existing /recommended palette).
COLOR_FOR: dict[str, str] = {
    SUMMER_INTERN:   "green",   # 优先级最高 — 5 月节奏的核心
    DAILY_INTERN:    "yellow",  # 次要, 不一定有转正
    CAMPUS_FULLTIME: "primary", # 9-10 月秋招主体
    SOCIAL:          "gray",    # 应届生不该投
    UNKNOWN:         "gray",
}


def classify_recruit_type(job: dict[str, Any]) -> str:
    """Classify a jobs row dict.

    Accepts a dict-like with keys: ``source``, ``title``, ``extras_json``
    (JSON str) OR ``extras`` (already-parsed dict). Other keys ignored.
    """
    source = (job.get("source") or "").lower()
    title = job.get("title") or ""
    extras = _load_extras(job)

    # Tencent social: all social, even if title says "实习生"
    if source == "tencent_social":
        return SOCIAL

    # W19+ — Bytedance jobs.bytedance.com API: empirically all recruit_type='正式'
    # (verified 2026-05-11, 1334 results all 正式). 应届实习走 0voice 间接.
    if source == "bytedance_jobs":
        return SOCIAL

    # Baidu (校招 GRADUATE 或 实习 INTERN): explicit projectType enum
    # (verified 2026-05-10 — INTERN endpoint returns 暑期实习项目 / 日常实习项目)
    if source in ("baidu_campus", "baidu_intern"):
        pt = _baidu_project_type(extras)
        if "暑期实习" in pt:
            return SUMMER_INTERN
        if "日常实习" in pt:
            return DAILY_INTERN
        # 校招 / AIDU项目 / 管培生项目 → all 校招正式
        if pt in {"校招", "AIDU项目", "管培生项目"}:
            return CAMPUS_FULLTIME
        # Unrecognized projectType, fall through to title heuristic
        return _classify_by_title(title)

    # Tencent campus: projectName / recruitLabelName enum
    if source == "tencent_campus":
        pn = (extras.get("project_name") or extras.get("recruit_label") or "")
        if "应届实习" in pn or "暑期实习" in pn:
            # 腾讯叫"应届实习" = 暑期可转正
            return SUMMER_INTERN
        if "日常实习" in pn:
            return DAILY_INTERN
        if "正式校招" in pn or "校招" in pn:
            return CAMPUS_FULLTIME
        return _classify_by_title(title)

    # W20 — shixiseng /interns 端点全部是实习 (校招走 resume.shixiseng.com/xiaozhao
    # 不在这个 adapter 里). title 含 "暑期"/"summer" → SUMMER, 否则 DAILY.
    # 不会 fall 到 UNKNOWN 因为 source 已经保证是实习.
    if source == "shixiseng":
        t_lower = (title or "").lower()
        if "暑期" in title or "summer" in t_lower:
            return SUMMER_INTERN
        return DAILY_INTERN

    # nowcoder + boss_extension + user_paste_* → title heuristic
    return _classify_by_title(title)


def _classify_by_title(title: str) -> str:
    """Last-resort title-based classification."""
    t = title or ""
    if "暑期实习" in t:
        return SUMMER_INTERN
    if "日常实习" in t:
        return DAILY_INTERN
    if "实习" in t:
        # Convention observed: 大厂暑期实习 always labels '暑期'.
        # "实习" alone implies 日常 by default.
        return DAILY_INTERN
    if "校招" in t or "管培生" in t or "应届" in t:
        return CAMPUS_FULLTIME
    return UNKNOWN


def _baidu_project_type(extras: dict[str, Any]) -> str:
    """Pull baidu's projectType from extras, handling both flat + nested cases."""
    # Flat: extras['project_type'] (set in W17 by raw_job_from_baidu_*)
    flat = extras.get("project_type")
    if flat:
        return str(flat)
    # Nested: extras['raw_row']['projectType'] (codex's W16 shape)
    raw_row = extras.get("raw_row")
    if isinstance(raw_row, dict):
        return str(raw_row.get("projectType") or "")
    return ""


def _load_extras(job: dict[str, Any]) -> dict[str, Any]:
    """Robust extras loader — accepts dict or JSON str."""
    raw = job.get("extras")
    if isinstance(raw, dict):
        return raw
    raw = job.get("extras_json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def is_intern(recruit_type: str) -> bool:
    """Convenience: True if summer_intern OR daily_intern."""
    return recruit_type in (SUMMER_INTERN, DAILY_INTERN)


# Default filter for an应届校招 user's /recommended view. The user
# (上财应统专硕 2027 届) cares about 暑期实习 (now) + 日常实习 (filler);
# 校招正式 is fall season; social is wrong audience.
DEFAULT_FILTER_TYPES = (SUMMER_INTERN, DAILY_INTERN)
