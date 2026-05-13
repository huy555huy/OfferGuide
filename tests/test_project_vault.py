"""Project vault: truthful project records for resume/interview grounding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide.config import Settings
from offerguide.llm import LLMResponse
from offerguide.profile import UserProfile
from offerguide.project_vault import (
    append_to_profile_text,
    draft_market_context,
    insert,
    list_all,
    render_for_skill,
)
from offerguide.skills import SkillRuntime, discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


class _StubLLM:
    def __init__(self, content: str = "{}") -> None:
        self._content = content
        self.calls: list[dict] = []

    def chat(self, messages, **kw):
        self.calls.append({"messages": list(messages), **kw})
        return LLMResponse(content=self._content, model="stub")


class _StubSearch:
    name = "stub"

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, *, max_results=10):
        self.calls.append((query, max_results))
        return self.hits[:max_results]


class _Hit:
    def __init__(self, title, url, snippet):
        self.title = title
        self.url = url
        self.snippet = snippet


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "project_vault.db")
    s.init_schema()
    return s


@pytest.fixture
def app_client(store):
    skills = discover_skills(SKILLS_ROOT)
    runtime = SkillRuntime(_StubLLM(), store)  # type: ignore[arg-type]
    profile = UserProfile(raw_resume_text="简历正文", source_pdf="/tmp/x.pdf")
    app = create_app(
        settings=Settings(deepseek_api_key="x", default_model="stub"),
        store=store,
        profile=profile,
        skills=skills,
        runtime=runtime,
        notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


def test_insert_project_record_requires_truth_grounding_fields(store):
    with pytest.raises(ValueError, match="my_work"):
        insert(
            store,
            title="课程项目",
            mainstream_direction="数据分析",
            project_task="完成一个分析报告",
            my_work="",
        )


def test_project_record_renders_guardrails_not_fake_metrics(store):
    record = insert(
        store,
        title="OfferGuide",
        mainstream_direction="AI Agent",
        typical_problem="把求职流程中的 JD 评估、简历调整、面试准备串起来",
        project_task="构建本地优先的求职 copilot",
        my_work="实现 agent loop、SKILL 调用和本地 SQLite 状态管理",
        contribution_type="integration",
        contribution_detail="整合成熟工具形成闭环，不声称提出新算法",
        market_context="同类 AI 研究助手通常先解释它如何帮助用户完成资料收集、阅读和报告生成。",
        reference_sources="- OpenAI Deep Research: https://example.com/openai",
        project_outputs="可运行 Web UI、测试、文档",
        do_not_claim="不要写支撑上万用户；没有线上用户数据",
        tags=["resume", "interview"],
        confidence=0.8,
    )

    rendered = render_for_skill([record])
    assert "项目事实档案" in rendered
    assert "AI Agent" in rendered
    assert "组合实现" in rendered
    assert "外部表达参考" in rendered
    assert "不能把参考来源里的指标" in rendered
    assert "不要写支撑上万用户" in rendered
    assert "不要自动生成百分比提升" in rendered


def test_draft_market_context_summarizes_public_expression_without_claiming_results():
    llm = _StubLLM(json.dumps({
        "market_context": [
            "同类工具通常先说明用户输入复杂问题后，系统会搜索、阅读并整理证据。",
            "表达重点放在减少人工资料整理负担，而不是宣称替代研究判断。",
        ],
        "reference_sources": [
            "OpenAI Deep Research: https://example.com/deep-research"
        ],
        "warnings": [
            "不要借用公开来源的准确率、用户规模或发布时间。"
        ],
    }, ensure_ascii=False))
    search = _StubSearch([
        _Hit(
            "OpenAI Deep Research",
            "https://example.com/deep-research",
            "A tool that searches, reads, and writes sourced reports.",
        )
    ])

    draft = draft_market_context(
        title="Deep Research Workspace",
        mainstream_direction="AI Agent",
        project_task="生成带来源的研究报告",
        my_work="实现搜索、抓取和报告流程",
        search=search,
        llm=llm,
    )

    assert search.calls
    assert "用户输入复杂问题" in draft.market_context
    assert "OpenAI Deep Research" in draft.reference_sources
    assert "不要借用公开来源" in draft.warnings[0]


def test_draft_market_context_falls_back_to_sources_when_llm_missing():
    search = _StubSearch([
        _Hit("Project README", "https://example.com/readme", "Architecture overview")
    ])
    draft = draft_market_context(
        mainstream_direction="后端系统",
        project_task="完成商品发布和检索",
        search=search,
        llm=None,
    )
    assert "可参考这些公开资料" in draft.market_context
    assert "Project README" in draft.reference_sources
    assert "LLM 总结不可用" in draft.warnings[0]


def test_append_to_profile_text_is_noop_when_vault_empty(store):
    assert append_to_profile_text(store, "base resume") == "base resume"


def test_append_to_profile_text_includes_project_vault(store):
    insert(
        store,
        title="RemeDi",
        mainstream_direction="NLP",
        project_task="复现论文中的核心流程",
        my_work="整理数据处理脚本并跑通训练流程",
        contribution_type="reproduction",
    )
    text = append_to_profile_text(store, "base resume")
    assert text.startswith("base resume")
    assert "项目事实档案" in text
    assert "复现理解" in text


def test_project_vault_page_and_insert(app_client):
    client, store = app_client
    resp = client.get("/project-vault")
    assert resp.status_code == 200
    assert "项目事实档案" in resp.text
    assert "结果与数据" not in resp.text

    resp = client.post(
        "/api/project-vault/insert",
        data={
            "title": "校园二手交易系统",
            "mainstream_direction": "后端系统",
            "project_task": "完成商品发布、检索和订单流转",
            "my_work": "负责数据库表设计和商品检索接口",
            "contribution_type": "engineering_improvement",
            "market_context": "同类后端项目通常先讲清楚业务流程，再解释数据模型和接口设计。",
            "reference_sources": "某技术博客: https://example.com/backend",
            "do_not_claim": "不要写高并发；没有压测",
            "confidence": "0.7",
        },
    )
    assert resp.status_code == 200
    assert "校园二手交易系统" in resp.text
    assert "同类后端项目通常先讲清楚业务流程" in resp.text
    assert "不要写高并发" in resp.text
    assert len(list_all(store)) == 1


def test_project_vault_market_context_endpoint_uses_search_stub(app_client, monkeypatch):
    client, _store = app_client

    class _SearchWithClose(_StubSearch):
        def close(self):
            pass

    def fake_build_default_search():
        return _SearchWithClose([
            _Hit("Project README", "https://example.com/readme", "Architecture overview")
        ])

    import offerguide.agentic.search as search_mod
    monkeypatch.setattr(search_mod, "build_default_search", fake_build_default_search)

    resp = client.post(
        "/api/project-vault/draft-market-context",
        data={
            "title": "校园二手交易系统",
            "mainstream_direction": "后端系统",
            "project_task": "完成商品发布和检索",
            "my_work": "负责接口和数据库设计",
        },
    )

    assert resp.status_code == 200
    assert "外部表达参考" in resp.text
    assert "Project README" in resp.text
    assert "LLM 总结不可用" in resp.text


def test_tailor_resume_tool_input_includes_project_vault(store):
    from offerguide.tools.evaluation import _tailor_resume_handler

    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
            "VALUES ('manual', 'AI 实习', '字节', ?, 'h1')",
            ("JD " * 100,),
        )
    insert(
        store,
        title="OfferGuide",
        mainstream_direction="AI Agent",
        project_task="构建本地求职 copilot",
        my_work="实现 agent loop 和 SKILL 调用",
        do_not_claim="不要写模型准确率提升",
    )

    class _Runtime:
        def __init__(self) -> None:
            self.inputs = None

        def invoke(self, spec, inputs):
            self.inputs = inputs
            from offerguide.skills._runtime import SkillResult
            payload = {
                "company": "字节",
                "role_focus": "AI 实习",
                "tailored_markdown": "",
                "change_log": [],
                "inserted_claims": [],
                "ats_keywords_used": [],
                "ats_keywords_missing": [],
                "cannot_fake_warnings": [],
                "fit_estimate": {"before": 0.3, "after": 0.3, "rationale": "x"},
                "suggested_filename": "x.pdf",
            }
            return SkillResult(
                raw_text=json.dumps(payload, ensure_ascii=False),
                parsed=payload,
                skill_name=spec.name,
                skill_version=spec.version,
                skill_run_id=1,
                input_hash="h",
                cost_usd=0.0,
                latency_ms=1,
            )

    spec = type("Spec", (), {"name": "tailor_resume", "version": "0.2.0"})()
    runtime = _Runtime()
    out = _tailor_resume_handler(
        {"job_id": 1},
        store=store,
        runtime=runtime,
        skills=[spec],
        user_profile_text="原始简历",
    )
    assert "tailored_markdown" in out
    assert runtime.inputs is not None
    assert "项目事实档案" in runtime.inputs["master_resume"]
    assert "不要写模型准确率提升" in runtime.inputs["master_resume"]


def test_prepare_interview_tool_input_includes_project_vault(store):
    from offerguide.tools.evaluation import _generate_interview_prep_handler

    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
            "VALUES ('manual', '算法实习', '腾讯', ?, 'h1')",
            ("JD " * 100,),
        )
    insert(
        store,
        title="论文复现项目",
        mainstream_direction="科研复现",
        project_task="复现论文核心实验流程",
        my_work="整理数据、跑通 baseline、记录复现偏差",
        contribution_type="reproduction",
    )

    class _Runtime:
        def __init__(self) -> None:
            self.inputs = None

        def invoke(self, spec, inputs):
            self.inputs = inputs
            from offerguide.skills._runtime import SkillResult
            payload = {
                "company_snapshot": "暂无面经数据，下方推断基于 JD。",
                "expected_questions": [],
                "prep_focus_areas": [],
                "weak_spots": [],
            }
            return SkillResult(
                raw_text=json.dumps(payload, ensure_ascii=False),
                parsed=payload,
                skill_name=spec.name,
                skill_version=spec.version,
                skill_run_id=2,
                input_hash="h",
                cost_usd=0.0,
                latency_ms=1,
            )

    spec = type("Spec", (), {"name": "prepare_interview", "version": "0.1.0"})()
    runtime = _Runtime()
    _generate_interview_prep_handler(
        {"job_id": 1},
        store=store,
        runtime=runtime,
        skills=[spec],
        user_profile_text="原始简历",
    )
    assert runtime.inputs is not None
    assert "项目事实档案" in runtime.inputs["user_profile"]
    assert "复现理解" in runtime.inputs["user_profile"]
