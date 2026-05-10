"""W19 — application_plan host-based 大厂细化 tests.

Verifies that ingest sources (baidu_intern, baidu_campus, tencent_campus,
tencent_social, agent_search) route to per-company plans rather than the
generic 'official_site' fallback. Plus host-detection works for
user-pasted URLs (e.g. talent.alibaba.com → alibaba_plan).
"""
from __future__ import annotations

from offerguide.application_plan import build_application_plan

# ─────────────────── Source-based routing ───────────────────


class TestSourceRouting:
    def test_baidu_intern_routes_to_baidu_plan(self):
        plan = build_application_plan({
            "source": "baidu_intern",
            "url": "https://talent.baidu.com/jobs/detail/INTERN/abc",
            "title": "AI Agent 实习生", "company": "百度",
            "extras_json": '{"source_verified": true}',
        })
        assert plan.platform == "baidu_intern"
        assert "百度" in plan.platform_label
        assert "暑期/日常实习" in plan.platform_label
        assert "百度账号登录" in plan.channel_note
        # Verified source pill
        assert "已核验" in plan.platform_label
        # Step set should mention 暑期/日常/AIDU project distinction
        assert any("暑期项目" in s for s in plan.steps)

    def test_baidu_campus_routes_to_baidu_plan_no_intern_label(self):
        plan = build_application_plan({
            "source": "baidu_campus",
            "url": "https://talent.baidu.com/jobs/detail/GRADUATE/xyz",
            "title": "AIDU-数据平台",
            "company": "百度",
            "extras_json": '{"source_verified": true}',
        })
        assert plan.platform == "baidu_campus"
        assert "校招正式" in plan.platform_label
        assert any("毕业入职" in s for s in plan.steps)

    def test_tencent_campus_routes_to_tencent_plan(self):
        plan = build_application_plan({
            "source": "tencent_campus",
            "url": "https://join.qq.com/jobdesc.html?postId=1234",
            "title": "应届实习", "company": "腾讯",
            "extras_json": '{"source_verified": true}',
        })
        assert plan.platform == "tencent_campus"
        assert "腾讯校招" in plan.platform_label
        assert "微信扫码登录" in plan.steps[0]
        # 内推码 强烈建议
        assert any("内推码" in s for s in plan.steps)
        # 内推码 字段
        assert any("内推码" in f.label for f in plan.fields)

    def test_tencent_social_warns_应届生(self):
        plan = build_application_plan({
            "source": "tencent_social",
            "url": "https://careers.tencent.com/jobdesc.html?postId=5678",
            "title": "AI Agent 应用架构师", "company": "腾讯",
            "extras_json": "{}",
        })
        assert plan.platform == "tencent_social"
        assert "应届生慎投" in plan.platform_label or "社招" in plan.platform_label
        # Warning content
        assert "工作经验" in plan.channel_note or "1-3 年" in plan.channel_note

    def test_agent_search_routes_to_verify_first_plan(self):
        """agent_search source must show 'verify first' framing — these JDs
        come from LLM web search, not verified API."""
        plan = build_application_plan({
            "source": "agent_search",
            "url": "https://www.zhipuai.cn/careers/abc",
            "title": "Agent 工程师", "company": "智谱AI",
            "extras_json": "{}",
        })
        assert plan.platform == "agent_search_external"
        assert "先核验" in plan.platform_label
        # Should say 'not verified' explicitly
        assert "not verified" in plan.channel_note.lower() or "verified API" in plan.channel_note
        # First step should be 'open + verify it still loads'
        assert "核验" in plan.steps[0] or "404" in " ".join(plan.steps)
        # verified_source must be False (no false promise)
        assert plan.verified_source is False


# ─────────────────── Host-based routing (user-pasted URLs) ───────────────────


class TestHostRouting:
    def test_pasted_bytedance_url_routes_to_bytedance_plan(self):
        plan = build_application_plan({
            "source": "user_paste_url",
            "url": "https://jobs.bytedance.com/campus/position/123",
            "title": "AI 实习", "company": "字节跳动",
            "extras_json": "{}",
        })
        assert plan.platform == "bytedance"
        assert "字节跳动" in plan.platform_label
        # Bytedance plan should strongly recommend 内推码
        assert "内推码" in plan.steps[0]
        # 飞书 People ATS mention
        assert "飞书" in plan.channel_note

    def test_pasted_alibaba_url_routes_to_alibaba_plan(self):
        plan = build_application_plan({
            "source": "user_paste_url",
            "url": "https://talent.alibaba.com/off-campus/position/12345",
            "title": "AI 实习", "company": "阿里巴巴",
            "extras_json": "{}",
        })
        assert plan.platform == "alibaba"
        assert "阿里巴巴" in plan.platform_label
        assert "北森" in plan.channel_note  # 阿里 uses 北森 ATS
        # 业务线 awareness
        assert any("通义" in s or "蚂蚁" in s or "达摩院" in s
                   for s in plan.material_checklist)

    def test_pasted_meituan_url_routes_to_meituan_plan(self):
        plan = build_application_plan({
            "source": "user_paste_url",
            "url": "https://zhaopin.meituan.com/web/campus/12345",
            "title": "AI 实习", "company": "美团",
            "extras_json": "{}",
        })
        assert plan.platform == "meituan"
        assert "美团" in plan.platform_label
        assert "登录" in plan.platform_label  # warns about login wall
        assert any("登录" in s for s in plan.steps)

    def test_agent_search_with_bytedance_host_uses_bytedance_plan(self):
        """Source-vs-host precedence: when agent_search ingests a
        jobs.bytedance.com URL, host wins (bytedance plan, not generic
        agent_search verify-first)."""
        plan = build_application_plan({
            "source": "agent_search",
            "url": "https://jobs.bytedance.com/campus/x",
            "title": "AI 实习", "company": "字节跳动",
            "extras_json": "{}",
        })
        assert plan.platform == "bytedance"
        # NOT the generic agent_search_external plan
        assert plan.platform != "agent_search_external"


# ─────────────────── Backward compat (codex's W16 sources still work) ───────────────────


class TestBackwardCompat:
    def test_boss_extension_still_routes_to_boss_plan(self):
        plan = build_application_plan({
            "source": "boss_extension",
            "url": "https://www.zhipin.com/job_detail/abc.html",
            "title": "AI 实习", "company": "字节",
        })
        assert plan.platform == "boss_zhipin"

    def test_nowcoder_still_routes_to_nowcoder_plan(self):
        plan = build_application_plan({
            "source": "nowcoder",
            "url": "https://www.nowcoder.com/jobs/detail/123",
            "title": "AI 实习", "company": "公司",
        })
        assert plan.platform == "nowcoder"

    def test_unknown_source_no_url_falls_back_to_unknown(self):
        plan = build_application_plan({
            "source": "manual", "url": None,
            "title": "AI 实习", "company": "未知公司",
        })
        assert plan.platform == "unknown"

    def test_unknown_source_with_unrecognized_host_falls_back_to_official_site(self):
        """Generic ATS path still applies for hosts we haven't carved out."""
        plan = build_application_plan({
            "source": "user_paste_url",
            "url": "https://careers.unknownco.com/job/123",
            "title": "x", "company": "未知公司",
        })
        assert plan.platform == "official_site"
