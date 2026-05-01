"""GitHub Awesome-style 校招清单 spider — 抄社区维护好的 markdown 列表。

社区驱动的 GitHub 仓库（如 ``namewyf/Campus2026``，393 ⭐）把 2026 届校招 +
实习信息整理成 markdown 表格，每周由志愿者更新。这种数据源比直接爬招聘网
站可靠得多——markdown 是结构化的、Git 有版本历史、维护者会及时下架失效
JD。我们的策略是定期 ``git pull`` 等价物（从 raw.githubusercontent.com
拉 README），解析其中的 markdown 表格，每行作为一个候选 JD。

为什么这是合法且最佳的做法：

- 仓库 LICENSE 通常允许复用（namewyf/Campus2026 是 MIT）
- README.md 是公开 raw 内容，无 cookie / API key
- 维护者本身就期望被消费——这就是 awesome-style 列表的价值
- 没有反爬压力——拉 GitHub raw 是无限免费的

预设源（W10）：

- ``namewyf/Campus2026`` —— 互联网/AI/外企/游戏/车企/IC 等多分类
  - 393 ⭐ · 公开 ATTRIBUTION 已加
  - 表头格式: ``| 公司 | 招聘状态&&投递链接 | 更新日期 | 地点 | 备注 |``

要新增源，传 ``sources=`` 参数。每个 source 是 ``(repo_slug, file_path)``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final
from urllib.parse import urljoin

from ..platforms import RawJob
from ._base import (
    SpiderError,
    SpiderResult,
    rate_limited_get,
)

DEFAULT_SOURCES: Final[list[tuple[str, str]]] = [
    ("namewyf/Campus2026", "README.md"),
]
"""``(repo, path)`` pairs to fetch via raw.githubusercontent. Pre-configured
with the most active 2026 list as of 2026-04. Users can extend or replace."""

GITHUB_RAW: Final[str] = "https://raw.githubusercontent.com"


@dataclass(frozen=True)
class _TableRow:
    """One markdown-table row from an awesome-jobs README."""
    company: str
    apply_url: str
    apply_label: str
    """The visible link text — usually `[校招正式批]` or `[内推]`. Useful as
    a hint about whether this is full校招 or 提前batch / 内推."""
    update_date: str
    """Raw value as seen — could be ``2025/7/20`` or ``2025-07-20`` or empty."""
    location: str
    note: str
    section: str
    """The H2/H3 section path for display — e.g. ``校招正式批 · 互联网 && AI``."""
    leaf_section: str = ""
    """The deepest heading only (e.g. ``互联网 && AI``). Used by the
    spider's section filter so ancestor headings don't pollute matches —
    a 银行&&保险 row whose H2 ancestor is 校招正式批 should be dropped
    when filtering on ``("互联网","AI","校招","实习")``."""


class AwesomeJobsSpider:
    """从 GitHub awesome-style markdown 清单拉 JD 候选。

    可指定多个源；run() 顺序拉取并合并去重（基于公司名+URL）。
    """

    name: str = "awesome_jobs"

    def __init__(
        self,
        *,
        sources: list[tuple[str, str]] | None = None,
        branch: str = "main",
        sections_keep_substrings: tuple[str, ...] = (
            "互联网", "AI", "校招", "实习",
        ),
    ) -> None:
        self.sources = list(sources or DEFAULT_SOURCES)
        self.branch = branch
        self.sections_keep_substrings = sections_keep_substrings
        """Only emit rows whose enclosing section name contains any of these
        substrings. Drops irrelevant categories like 银行 / 国企 by default;
        pass an empty tuple to keep everything."""

    def run(self, *, max_items: int = 30) -> SpiderResult:
        out = SpiderResult(spider_name=self.name)
        seen: set[tuple[str, str]] = set()  # (company, apply_url) — light dedup

        for repo, path in self.sources:
            url = f"{GITHUB_RAW}/{repo}/{self.branch}/{path}"
            try:
                resp = rate_limited_get(url)
            except SpiderError as e:
                out.errors.append(f"{repo}@{path}: {e}")
                continue
            out.pages_fetched += 1

            try:
                rows = parse_markdown_tables(resp.text)
            except Exception as e:
                out.errors.append(f"{repo}@{path}: parse failed: {e}")
                continue

            for row in rows:
                if not self._section_ok(row):
                    continue
                key = (row.company, row.apply_url)
                if key in seen:
                    continue
                seen.add(key)
                rj = self._row_to_raw_job(row, repo=repo)
                out.raw_jobs.append(rj)
                if len(out.raw_jobs) >= max_items:
                    return out
        return out

    def _section_ok(self, row: _TableRow) -> bool:
        """Match against the row's deepest heading, not the H2 ancestor.

        Falls back to the full section path when the row predates the
        leaf-tracking field (defensive)."""
        if not self.sections_keep_substrings:
            return True
        target = row.leaf_section or row.section
        return any(s in target for s in self.sections_keep_substrings)

    def _row_to_raw_job(self, row: _TableRow, *, repo: str) -> RawJob:
        # raw_text is intentionally compact — score_match needs a couple of
        # signals (company, role hint, location) and any extra stuff.
        raw_text = "\n".join(filter(None, [
            f"公司: {row.company}",
            f"招聘类型: {row.apply_label}",
            f"地点: {row.location}" if row.location else "",
            f"更新: {row.update_date}" if row.update_date else "",
            f"备注: {row.note}" if row.note else "",
            f"分类: {row.section}",
            f"投递入口: {row.apply_url}",
            f"来源: github:{repo}",
        ]))
        return RawJob(
            source="awesome_jobs",
            source_id=_stable_id(row.company, row.apply_url),
            url=row.apply_url,
            title=f"{row.company} · {row.apply_label}",
            company=row.company,
            location=row.location or None,
            raw_text=raw_text,
            extras={
                "section": row.section,
                "apply_label": row.apply_label,
                "update_date": row.update_date,
                "note": row.note,
                "github_repo": repo,
            },
        )


# ─────────── markdown table parser ───────────────────


_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_ROW_RE: Final[re.Pattern[str]] = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP_RE: Final[re.Pattern[str]] = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]+\|?\s*$")
_LINK_RE: Final[re.Pattern[str]] = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def parse_markdown_tables(md_text: str) -> list[_TableRow]:
    """Parse all rows from all markdown tables in the document.

    The parser is intentionally simple — it works on namewyf/Campus2026's
    structure and similar 5-column awesome-jobs READMEs. Header detection
    is heuristic: any row directly followed by a separator (``| --- |``)
    is treated as a header; subsequent ``|...|`` rows are data until a
    blank line or non-table content interrupts.

    Section context tracking: most recent ``##`` / ``###`` heading is
    associated with each row.
    """
    rows: list[_TableRow] = []
    lines = md_text.splitlines()
    cur_section: list[str] = []  # stack of (level, text) so we can show H2 / H3 path
    in_table = False
    headers: list[str] = []
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        # Heading update — keep last 2 levels for context
        m = _HEADING_RE.match(line)
        if m:
            level_hashes, text = m.groups()
            level = len(level_hashes)
            # Trim section stack to deeper levels
            cur_section = [s for s in cur_section if int(s.split(":", 1)[0]) < level]
            cur_section.append(f"{level}:{text.strip()}")
            in_table = False
            i += 1
            continue

        # Table detection: header row + separator line
        if _TABLE_ROW_RE.match(line) and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            headers = [c.strip() for c in line.strip().strip("|").split("|")]
            in_table = True
            i += 2
            continue

        # Table data row
        if in_table and _TABLE_ROW_RE.match(line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            row = _row_from_cells(cells, headers, cur_section)
            if row is not None:
                rows.append(row)
            i += 1
            continue

        # Blank line breaks tables
        if not line.strip():
            in_table = False
        i += 1
    return rows


def _row_from_cells(
    cells: list[str], headers: list[str], section_stack: list[str]
) -> _TableRow | None:
    """Map cells to fields, using header names as hints.

    Headers we recognize (case + chinese-spacing variants):
      - 公司 / company
      - 招聘状态&&投递链接 / 投递链接 / link / 入口
      - 更新日期 / 更新 / date
      - 地点 / location
      - 备注 / note
    """
    if len(cells) < 2:
        return None

    def find(*names: str) -> str:
        for n in names:
            for j, h in enumerate(headers):
                if n in h.replace(" ", "").replace("&&", "").lower() if h else False:
                    return cells[j] if j < len(cells) else ""
        return ""

    # Direct positional mapping for namewyf/Campus2026's exact 5-col layout
    by_name = {
        "company": find("公司", "company"),
        "link":    find("投递链接", "投递", "链接", "link", "入口", "招聘状态"),
        "date":    find("更新日期", "更新", "date"),
        "location": find("地点", "location"),
        "note":    find("备注", "note", "remark"),
    }

    # Fallback: position-based for 5-col tables (公司 | 链接 | 日期 | 地点 | 备注)
    if not by_name["company"] and len(cells) >= 5:
        by_name = {
            "company": cells[0],
            "link":    cells[1],
            "date":    cells[2],
            "location": cells[3],
            "note":    cells[4],
        }

    company = _strip_md_inline(by_name["company"])
    if not company:
        return None
    if company.lower() in {"公司", "company"}:
        return None  # second header row inside a table — skip

    link_match = _LINK_RE.search(by_name["link"])
    if link_match:
        apply_label = _strip_md_inline(link_match.group(1))
        apply_url = link_match.group(2)
    else:
        apply_label = _strip_md_inline(by_name["link"])
        apply_url = ""

    if not apply_url or not apply_url.startswith(("http://", "https://")):
        return None  # require a real outbound link

    # Section uses the *deepest* heading only — that's the one that
    # carries the meaningful filter signal (e.g. ``银行 && 保险`` vs
    # ``互联网 && AI``). The H2 ancestor (e.g. ``校招正式批``) is in
    # ``parent_section`` for display.
    if section_stack:
        leaf = section_stack[-1].split(":", 1)[1]
        parent = (
            section_stack[-2].split(":", 1)[1] if len(section_stack) >= 2 else ""
        )
        section_display = (
            f"{parent} · {leaf}" if parent and parent != leaf else leaf
        )
    else:
        leaf = ""
        section_display = ""

    return _TableRow(
        company=company,
        apply_url=apply_url,
        apply_label=apply_label,
        update_date=_strip_md_inline(by_name["date"]),
        location=_strip_md_inline(by_name["location"]),
        note=_strip_md_inline(by_name["note"]),
        section=section_display,
        leaf_section=leaf,
    )


_INLINE_FORMAT_RE: Final[re.Pattern[str]] = re.compile(r"\*\*|__|`|~~")


def _strip_md_inline(s: str) -> str:
    return _INLINE_FORMAT_RE.sub("", s).strip()


def _stable_id(company: str, url: str) -> str:
    """Deterministic short id from company + URL — used as RawJob.source_id."""
    import hashlib
    h = hashlib.md5(f"{company}|{url}".encode()).hexdigest()
    return h[:12]


_ = urljoin  # keep import for downstream
