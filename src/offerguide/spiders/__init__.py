"""Job-discovery spiders — turn the agent from "粘贴 JD" into "我们爬好了给你"。

Each spider implements :class:`Spider` and yields :class:`offerguide.platforms.RawJob`
objects. Existing ``workers.scout.ingest`` handles dedup + storage, so a spider
just needs to produce ``RawJob`` and let scout do the rest.

Why a separate module from ``platforms/``: ``platforms/`` parses **one**
JD (e.g. given a Boss URL, produce a RawJob from its HTML). Spiders **discover**
JDs proactively — they crawl listing pages, RSS feeds, Awesome-style markdown
catalogs, etc. The two concerns compose: a spider yields URLs, then for each
URL a platform parser produces the RawJob.

Live spiders (W10):

- ``yingjiesheng`` — 应届生网 BBS RSS / category page (校招主战场，反爬温和)
- ``shixiseng``    — 实习僧岗位列表 (实习专门)
- ``nowcoder_list`` — 牛客校招日历 (公司 + 截止日期)
- ``awesome_jobs`` — GitHub Awesome-style markdown JD catalogs

Each is rate-limited + cache-aware. The autonomous daemon's
``corpus_refresh`` job will iterate spiders and feed scout.

NOT scrape-friendly (use the browser extension instead):

- Boss 直聘 — strong anti-scraping, sliders, cookie-based auth
- 拉勾   — IP-frequency-banned, login wall on most pages
- 看准网 — Boss-owned, same restrictions
- 智联招聘 — IP whitelist heuristics, login wall
- 微信公众号 — closed garden, no programmatic access
"""

from __future__ import annotations

from ._base import (
    Spider,
    SpiderError,
    SpiderResult,
    rate_limited_get,
)

__all__ = [
    "Spider",
    "SpiderError",
    "SpiderResult",
    "rate_limited_get",
]
