"""W20 audit — real network end-to-end test of 实习僧 adapter through
the ambient daemon's _crawl_shixiseng_per_keyword helper.

Goal: prove the wiring is right (adapter → ambient → DB → /recommended)
not just that the adapter alone works.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["OFFERGUIDE_NO_AMBIENT"] = "1"

import offerguide
from offerguide.application_plan import build_application_plan
from offerguide.agent_runtime import _schema as harness_schema
from offerguide.recruit_type import classify_recruit_type
from offerguide.workers.ambient import _crawl_shixiseng_per_keyword

# ---- 1. Setup ---------------------------------------------------------

tmp = Path(tempfile.mkdtemp(prefix="ogfd_w20_ssseng_"))
db = tmp / "store.db"
store = offerguide.Store(db)
store.init_schema()
harness_schema.init_agent_runtime_schema(store)
print(f"DB: {db}")

# ---- 2. Real crawl 2 keywords ----------------------------------------

KEYWORDS = ["AI Agent", "大模型"]
print(f"\nCrawling shixiseng with keywords: {KEYWORDS}, limit_per_kw=8")
t0 = time.monotonic()
summary = _crawl_shixiseng_per_keyword(
    store=store, keywords=KEYWORDS, limit_per_kw=8,
)
elapsed = time.monotonic() - t0
print(f"Elapsed: {elapsed:.1f}s")
print(f"Summary: {json.dumps(summary, ensure_ascii=False, indent=2)}")

# ---- 3. Read back what was ingested --------------------------------

with store.connect() as conn:
    rows = conn.execute(
        "SELECT id, source, title, company, location, url, extras_json "
        "FROM jobs WHERE source = 'shixiseng' ORDER BY id"
    ).fetchall()

print(f"\n{len(rows)} shixiseng jobs in DB:")
print("-" * 110)
print(f"{'ID':>4} {'COMPANY':<20} {'TITLE':<30} {'CITY':<10} {'KW':<15} {'TYPE':<14}")
print("-" * 110)

by_kw: dict[str, int] = {}
by_company: dict[str, int] = {}
by_type: dict[str, int] = {}
for r in rows:
    job = {
        "id": r[0], "source": r[1], "title": r[2] or "", "company": r[3] or "",
        "location": r[4] or "", "url": r[5] or "", "extras_json": r[6] or "{}",
    }
    extras = json.loads(job["extras_json"])
    kw = extras.get("discovered_keyword", "?")
    rt = classify_recruit_type(job)
    print(f"{r[0]:>4} {(job['company'][:18]):<20} {(job['title'][:28]):<30} "
          f"{(job['location'][:8]):<10} {kw[:13]:<15} {rt:<14}")
    by_kw[kw] = by_kw.get(kw, 0) + 1
    by_company[job["company"]] = by_company.get(job["company"], 0) + 1
    by_type[rt] = by_type.get(rt, 0) + 1

print("-" * 110)
print(f"\nDistribution by keyword: {by_kw}")
print(f"Distribution by recruit_type: {by_type}")
print(f"Company diversity: {len(by_company)} unique companies "
      f"(avg {len(rows) / max(len(by_company), 1):.1f} jobs/company)")

# Check niche AI startups vs 大厂
big_techs = {"百度", "阿里巴巴", "腾讯", "字节跳动", "美团", "美图", "小米",
             "华为", "京东", "网易", "拼多多", "快手", "京东", "淘宝", "蚂蚁",
             "蚂蚁集团", "支付宝"}
big = sum(1 for c in by_company if any(b in c for b in big_techs))
niche = len(by_company) - big
print(f"大厂: {big} 家 / 中小厂 (niche): {niche} 家")

# ---- 4. Sample one job → application_plan ---------------------------

if rows:
    sample_job = {
        "id": rows[0][0], "source": rows[0][1], "title": rows[0][2],
        "company": rows[0][3], "location": rows[0][4], "url": rows[0][5],
        "extras_json": rows[0][6],
    }
    plan = build_application_plan(sample_job)
    print(f"\n--- application_plan for sample (job#{sample_job['id']}: {sample_job['title']}) ---")
    print(f"  platform: {plan.platform}")
    print(f"  platform_label: {plan.platform_label}")
    print(f"  channel_note: {plan.channel_note[:80]}...")
    print(f"  steps: {len(plan.steps)} 步")
    print(f"  fields: {len(plan.fields)} 字段")
    print(f"  material_checklist: {len(plan.material_checklist)} 条")
    print(f"  post_apply_actions: {len(plan.post_apply_actions)} 条")

# ---- 5. Result ---------------------------------------------------------

print("\n" + "=" * 78)
ok = (
    summary["inserted_total"] > 0
    and len(by_company) >= 4  # at least some company diversity
    and not summary["errors"]
    and rows[0] is not None
)
print(f"PASS: {ok}")
if not ok:
    print("FAIL reasons:")
    if summary["inserted_total"] == 0:
        print("  - 0 inserted")
    if len(by_company) < 4:
        print(f"  - low diversity: {len(by_company)} companies")
    if summary["errors"]:
        print(f"  - errors: {summary['errors']}")
print("=" * 78)
sys.exit(0 if ok else 1)
