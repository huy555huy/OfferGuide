"""W19+ audit — real TestClient probe of /jobs/N/apply-pack across 8 sources.

Goal (audit doc item #3): prove the 8 host-based application_plan branches
really render in HTML, not just "the unit test passes the build_application_plan
function call". Renders page, greps HTML for plan-specific 中文 needles per
source.

Run:
    PYTHONPATH=src python scripts/audit_w19_apply_pack_real_render.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Force ambient daemon off — we're testing apply-pack render, not crawler
os.environ["OFFERGUIDE_NO_AMBIENT"] = "1"

from fastapi.testclient import TestClient

import offerguide
from offerguide.config import Settings
from offerguide.agent_runtime import _schema as harness_schema
from offerguide.profile.schema import UserProfile
from offerguide.ui.web import create_app

# ---- 1. Setup tempdir + store + profile + app ----------------------------

tmp = Path(tempfile.mkdtemp(prefix="ogfd_w19_render_"))
db = tmp / "store.db"
store = offerguide.Store(db)
store.init_schema()
harness_schema.init_agent_runtime_schema(store)

profile = UserProfile(
    raw_resume_text="测试用简历: AI Agent / LLM 应用方向, 上海财经大学应用统计专硕 2027 届",
)

# Settings — empty key on purpose so the LLM call returns error (we only care
# about application_plan rendering, which is independent of LLM).
settings = Settings(deepseek_api_key=None, db_path=db, disable_ambient_crawl=True)

app = create_app(
    settings=settings, store=store, profile=profile,
    skills=[], runtime=None, notifier=None,
)

# ---- 2. Insert one job per source (8 host-based + nowcoder + boss + unknown) ----

CASES: list[dict] = [
    # (id_hint, source, url, title, company, extras_json, expected needles in HTML)
    {
        "source": "tencent_campus",
        "url": "https://join.qq.com/post.html?pid=12345",
        "title": "AI Agent 工程师 - 应届实习",
        "company": "腾讯",
        "extras": {"source_verified": True, "evidence_url": "https://join.qq.com/api/v1/position/searchPosition"},
        "expect_needles": [
            "腾讯校招 · join.qq.com",
            "微信扫码登录 join.qq.com",
            "腾讯 ATS 不限简历字数",
        ],
    },
    {
        "source": "tencent_social",
        "url": "https://careers.tencent.com/jobdesc.html?postId=999",
        "title": "高级 AI 工程师",
        "company": "腾讯",
        "extras": {"source_verified": True},
        "expect_needles": [
            "腾讯社招 · ⚠ 应届生慎投",
            "QQ / 微信扫码登录 careers.tencent.com",
        ],
    },
    {
        "source": "baidu_intern",
        "url": "https://talent.baidu.com/jobs/list?recruitType=INTERN&jobId=222",
        "title": "AI 大模型 - 暑期实习",
        "company": "百度",
        "extras": {"source_verified": True, "project_type": "暑期实习项目"},
        "expect_needles": [
            "百度 · 暑期/日常实习",
            "百度账号登录 talent.baidu.com",
            "暑期项目一般要求 3-4 月以上",
        ],
    },
    {
        "source": "baidu_campus",
        "url": "https://talent.baidu.com/jobs/list?recruitType=GRADUATE&jobId=333",
        "title": "AI 大模型 - 校招正式",
        "company": "百度",
        "extras": {"source_verified": True, "project_type": "校招"},
        "expect_needles": [
            "百度 · 校招正式",
            "校招岗位选「毕业入职」时间",
        ],
    },
    {
        "source": "bytedance_jobs",
        "url": "https://jobs.bytedance.com/experienced/position/7545051719739115783/detail",
        "title": "AI Agent 高级工程师 - TikTok",
        "company": "字节跳动",
        "extras": {"source_verified": True},
        "expect_needles": [
            "字节跳动 · jobs.bytedance.com",
            "飞书扫码登录 jobs.bytedance.com",
            "字节走自研飞书 People ATS",
        ],
    },
    {
        "source": "zerovoice_repo",  # 阿里 via 0voice — host detection routes to alibaba_plan
        "url": "https://campus-talent.alibaba.com/positionDetail?positionId=88888",
        "title": "通义实验室 - 大模型算法实习",
        "company": "阿里巴巴",
        "extras": {"source_verified": True, "evidence_url": "https://github.com/0voice/2026-Computer-Spring-Recruitment-Job-Compilation/blob/main/README.md", "link_type": "official_ats"},
        "expect_needles": [
            "阿里巴巴 · talent.alibaba.com",
            "淘宝 / 支付宝账号登录 talent.alibaba.com",
        ],
    },
    {
        "source": "zerovoice_repo",  # 北森 SaaS via 0voice
        "url": "https://app.mokahr.com/campus_apply/sensetime/12345",
        "title": "多模态大模型算法工程师",
        "company": "商汤科技",
        "extras": {"source_verified": True, "link_type": "official_ats"},
        "expect_needles": [
            "北森 SaaS",
            "微信扫码或手机号注册 app.mokahr.com",
            "ATS 解析较严格",
        ],
    },
    {
        "source": "agent_search",
        "url": "https://www.zhipuai.cn/jobs/llm-agent",
        "title": "LLM Agent 应用研发实习",
        "company": "智谱AI",
        "extras": {"discovered_via": "agent_search"},
        "expect_needles": [
            "agent 搜到的外部岗 · ⚠ 先核验",
            "先打开链接核验",
            "不是 verified API",
        ],
    },
    # Sanity: the 3 non-host-based plans we already had
    {
        "source": "nowcoder",
        "url": "https://www.nowcoder.com/jobs/detail/12345",
        "title": "AI Agent 暑期实习",
        "company": "某中厂",
        "extras": {"discovered_via": "nowcoder_sitemap"},
        "expect_needles": ["牛客", "牛客打招呼"],
    },
    {
        "source": "boss",
        "url": "https://www.zhipin.com/job_detail/abcdef.html",
        "title": "LLM 应用工程师",
        "company": "某创业公司",
        "extras": {},
        "expect_needles": ["BOSS 直聘", "立即沟通"],
    },
    {
        "source": "manual",  # unknown source, just URL
        "url": "https://example.com/careers/job/1",
        "title": "数据科学家",
        "company": "某公司",
        "extras": {},
        "expect_needles": ["官网 / ATS 网申"],
    },
]

# Insert into DB
job_ids: list[int] = []
with store.connect() as conn:
    for c in CASES:
        cur = conn.execute(
            "INSERT INTO jobs (source, title, company, location, url, raw_text, "
            "content_hash, extras_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                c["source"], c["title"], c["company"], "上海",
                c["url"],
                f"测试 raw_text for {c['title']}" * 20,
                f"hash_{c['source']}_{len(job_ids)}",
                json.dumps(c["extras"]),
            ),
        )
        job_ids.append(cur.lastrowid or 0)
    conn.commit()

print(f"Inserted {len(job_ids)} test jobs into {db}")

# ---- 3. Real TestClient GET /jobs/{id}/apply-pack each ---------------------

results: list[dict] = []
with TestClient(app) as client:
    for c, jid in zip(CASES, job_ids, strict=True):
        r = client.get(f"/jobs/{jid}/apply-pack")
        body = r.text
        found = []
        missing = []
        for n in c["expect_needles"]:
            if n in body:
                found.append(n)
            else:
                missing.append(n)
        # Also confirm the LLM-error block renders (since key is empty)
        llm_err_present = "需要先配 LLM key" in body
        results.append({
            "case": f"{c['source']} → {c['company']} · {c['title'][:25]}",
            "status": r.status_code,
            "html_len": len(body),
            "found": found,
            "missing": missing,
            "llm_err_block": llm_err_present,
        })

# ---- 4. Print real results ------------------------------------------------

print("\n" + "=" * 78)
print(f"{'CASE':<55} {'STATUS':>6} {'HTML KB':>7} {'NEEDLES':>10}")
print("-" * 78)
ok = 0
fail = 0
for r in results:
    nlen = len(r["found"]) + len(r["missing"])
    nfnd = len(r["found"])
    flag = "✓" if not r["missing"] and r["status"] == 200 else "✗"
    print(f"{flag} {r['case']:<53} {r['status']:>6} {r['html_len'] // 1024:>7} {nfnd}/{nlen:>2}")
    if r["missing"]:
        print(f"    ❌ MISSING: {r['missing']}")
        fail += 1
    else:
        ok += 1

print("-" * 78)
print(f"PASS: {ok} / FAIL: {fail} / TOTAL: {len(results)}")
print("=" * 78)

if fail:
    sys.exit(1)
