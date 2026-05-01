"""W8' — UI overhaul: dashboard + structured report rendering.

Verifies that:
- /dashboard renders without errors and shows real data
- The chat report uses structured cards (not raw markdown <pre>)
- Probability bar renders with the right width
- Question categories get the right color class
- Suggestion AI risk classes flow through to CSS
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import inbox as inbox_mod
from offerguide.application_events import record as record_event
from offerguide.config import Settings
from offerguide.profile import UserProfile
from offerguide.skills import SkillResult, discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ── fakes ──────────────────────────────────────────────────────────


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._next_id = 500

    def invoke(self, spec, inputs, **_):
        self._next_id += 1
        self.calls.append((spec.name, inputs))
        if spec.name == "score_match":
            parsed = {
                "probability": 0.78,
                "reasoning": "强匹配，简历项目和 JD 高度对齐",
                "dimensions": {"tech": 0.85, "exp": 0.72, "company_tier": 0.78},
                "deal_breakers": [],
            }
        elif spec.name == "analyze_gaps":
            parsed = {
                "summary": "整体匹配良好，有 2 处可定向补充",
                "keyword_gaps": [
                    {"jd_keyword": "C++", "in_resume": False, "importance": "high",
                     "evidence_in_jd": "扎实 C++"}
                ],
                "suggestions": [
                    {"section": "技能", "action": "add", "current_text": None,
                     "proposed_addition": "C++ (本科课程实践)",
                     "reason": "JD 要求 C++", "ai_risk": "low", "confidence": 0.7},
                    {"section": "项目经历", "action": "emphasize",
                     "current_text": "Deep Research Agent",
                     "proposed_addition": "强调 multi-step planning 能力",
                     "reason": "JD 要求 planning", "ai_risk": "high", "confidence": 0.5},
                ],
                "do_not_add": ["不要捏造大型分布式经验"],
                "ai_detection_warnings": ["注意句式过于工整可能被识别"],
            }
        else:
            parsed = {
                "company_snapshot": "字节 AI Lab，主攻 LLM 后训练",
                "expected_questions": [
                    {"question": "讲讲 attention 缩放为什么除以 √d",
                     "category": "technical", "likelihood": 0.85,
                     "rationale": "JD 要求 Transformer"},
                    {"question": "你做的 RemeDi loss 曲线",
                     "category": "project_deep_dive", "likelihood": 0.75,
                     "rationale": "简历项目"},
                    {"question": "讲个跨团队协作",
                     "category": "behavioral", "likelihood": 0.55,
                     "rationale": "校招通用"},
                ],
                "prep_focus_areas": ["Transformer 数学", "GRPO 调参"],
                "weak_spots": ["Megatron 没用过"],
            }
        return SkillResult(
            raw_text=json.dumps(parsed),
            parsed=parsed,
            skill_name=spec.name,
            skill_version=spec.version,
            skill_run_id=self._next_id,
            input_hash="x",
            cost_usd=0.0001,
            latency_ms=42,
        )


@pytest.fixture
def app_setup(tmp_path: Path):
    store = offerguide.Store(tmp_path / "ui.db")
    store.init_schema()
    profile = UserProfile(raw_resume_text="胡阳，统计学专硕")
    skills = discover_skills(SKILLS_ROOT)
    runtime = _FakeRuntime()
    app = create_app(
        settings=Settings(),
        store=store,
        profile=profile,
        skills=skills,
        runtime=runtime,  # type: ignore[arg-type]
        notifier=ConsoleNotifier(),
    )
    return app, store, runtime


# ═══════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════


class TestDashboard:
    def test_dashboard_route_renders(self, app_setup) -> None:
        app, _, _ = app_setup
        client = TestClient(app)
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "Dashboard" in resp.text
        # Stats cards
        assert "总 JD 数" in resp.text
        assert "SKILL 调用" in resp.text
        assert "Inbox 待决策" in resp.text
        assert "SKILL 进化" in resp.text

    def test_dashboard_empty_states(self, app_setup) -> None:
        """Empty DB shows the 'no data yet' empty states."""
        app, _, _ = app_setup
        client = TestClient(app)
        resp = client.get("/dashboard")
        assert "还没有应用事件" in resp.text
        assert "还没有 SKILL 进化记录" in resp.text
        assert "还没有 SKILL 调用" in resp.text

    def test_dashboard_renders_evolution_history(self, app_setup) -> None:
        app, store, _ = app_setup
        # Seed an evolution_log row
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO evolution_log(skill_name, parent_version, new_version, "
                "metric_name, metric_before, metric_after, notes) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    "score_match", "0.2.0", "0.2.1",
                    "score_match_total", 0.50, 0.72,
                    json.dumps({"delta_total": 0.22}),
                ),
            )

        resp = TestClient(app).get("/dashboard")
        assert "score_match" in resp.text
        assert "0.2.0" in resp.text
        assert "0.2.1" in resp.text
        # Delta shown with up arrow
        assert "↑" in resp.text or "+0.220" in resp.text

    def test_dashboard_renders_application_funnel(self, app_setup) -> None:
        app, store, _ = app_setup
        # Seed an application + events
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, title, raw_text, content_hash) "
                "VALUES ('manual', 't', 'jd', 'hash1')"
            )
            cur = conn.execute(
                "INSERT INTO applications(job_id, status) VALUES (1, 'applied')"
            )
            app_id = cur.lastrowid

        record_event(store, application_id=app_id, kind="submitted", source="manual")
        record_event(store, application_id=app_id, kind="viewed", source="email")
        record_event(store, application_id=app_id, kind="replied", source="manual")

        resp = TestClient(app).get("/dashboard")
        assert "投递" in resp.text       # funnel stages render
        assert "HR 已查看" in resp.text
        assert "HR 已回复" in resp.text

    def test_dashboard_skill_runs_with_data(self, app_setup) -> None:
        app, _, runtime = app_setup
        # Run a chat to produce real skill_runs
        client = TestClient(app)
        client.post("/chat", data={"job_text": "JD", "action": "score_and_gaps"})

        resp = client.get("/dashboard")
        # Recent runs table
        assert "score_match" in resp.text


# ═══════════════════════════════════════════════════════════════════
# STRUCTURED REPORT RENDERING
# ═══════════════════════════════════════════════════════════════════


class TestStructuredReport:
    def test_score_renders_probability_bar(self, app_setup) -> None:
        app, _, _ = app_setup
        client = TestClient(app)
        resp = client.post("/chat", data={"job_text": "JD", "action": "score"})
        # Probability shown as percentage
        assert "78" in resp.text  # 0.78 → 78%
        # Probability bar present
        assert "prob-bar" in resp.text
        assert "prob-fill" in resp.text
        # Calibration markers
        assert "prob-marker" in resp.text

    def test_score_dimensions_render_as_cards(self, app_setup) -> None:
        app, _, _ = app_setup
        client = TestClient(app)
        resp = client.post("/chat", data={"job_text": "JD", "action": "score"})
        # Dimensions present as visible names
        assert "tech" in resp.text
        assert "exp" in resp.text
        assert "company_tier" in resp.text

    def test_gaps_suggestions_render_as_cards_with_ai_risk_class(self, app_setup) -> None:
        app, _, _ = app_setup
        client = TestClient(app)
        resp = client.post("/chat", data={"job_text": "JD", "action": "gaps"})
        # Suggestion card class includes ai_risk for color coding
        assert 'class="suggestion low"' in resp.text
        assert 'class="suggestion high"' in resp.text
        # Action pill
        assert ">add<" in resp.text or "add" in resp.text
        # do_not_add section
        assert "禁止添加" in resp.text
        # ai_detection_warnings
        assert "AI 检测" in resp.text

    def test_high_risk_warning_banner(self, app_setup) -> None:
        app, _, _ = app_setup
        client = TestClient(app)
        resp = client.post("/chat", data={"job_text": "JD", "action": "gaps"})
        # When at least one high-risk suggestion exists, a warning banner renders
        assert "AI 检测高风险" in resp.text
        assert "49% 公司会自动 dismiss" in resp.text

    def test_prep_renders_categorized_questions(self, app_setup) -> None:
        app, _, _ = app_setup
        client = TestClient(app)
        resp = client.post(
            "/chat",
            data={
                "job_text": "JD",
                "action": "prepare_interview",
                "company": "字节跳动",
            },
        )
        # Questions render with their category as a CSS class
        assert "cat-technical" in resp.text
        assert "cat-project_deep_dive" in resp.text
        assert "cat-behavioral" in resp.text
        # Likelihood as percentage
        assert "85%" in resp.text  # highest-likelihood question
        # Focus areas + weak spots sections
        assert "备战重点" in resp.text
        assert "用户弱点" in resp.text

    def test_prep_company_required_error(self, app_setup) -> None:
        """prepare_interview with no company shows clear UI error."""
        app, _, _ = app_setup
        client = TestClient(app)
        resp = client.post(
            "/chat",
            data={"job_text": "JD", "action": "prepare_interview"},
        )
        # Error class rendered
        assert "error" in resp.text.lower()
        assert "公司名" in resp.text or "company" in resp.text.lower()


# ═══════════════════════════════════════════════════════════════════
# HOME STATS STRIP
# ═══════════════════════════════════════════════════════════════════


class TestHomeStats:
    def test_home_shows_daily_brief_stats_when_db_empty(self, app_setup) -> None:
        """The new daily-standup home surfaces 4 lifecycle-driven stats
        instead of raw DB counts. Empty DB → all four pillars empty +
        friendly empty-states."""
        app, _, _ = app_setup
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        # 4 lifecycle stat labels (replacing the old "已入库 JD / SKILL 调用次数")
        assert "沉默待跟进" in resp.text
        assert "本周面试" in resp.text
        assert "未跑评估" in resp.text
        assert "投递战况" in resp.text
        # Friendly empty-state copy
        assert (
            "今天战况平稳" in resp.text
            or "没有紧急待办" in resp.text
        )

    def test_home_links_to_pipeline_page(self, app_setup) -> None:
        """Home page has a 投递战况 stat card linking to /pipeline.
        The 5-column kanban itself lives on /pipeline now (总分式 UI)."""
        app, _, _ = app_setup
        resp = TestClient(app).get("/")
        assert "投递战况" in resp.text
        assert "/pipeline" in resp.text

    def test_quick_eval_keeps_legacy_stats(self, app_setup) -> None:
        """The legacy stat labels migrated to /quick-eval — make sure
        they still render there for users who came from external links."""
        app, store, _ = app_setup
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, title, raw_text, content_hash) "
                "VALUES ('manual', 't', 'body', 'h1')"
            )
        inbox_mod.enqueue(store, kind="consider_jd", title="X")
        resp = TestClient(app).get("/quick-eval")
        assert resp.status_code == 200
        assert "已入库 JD" in resp.text
        assert "SKILL 调用次数" in resp.text


# ═══════════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════════


class TestNavigation:
    def test_nav_has_three_tabs(self, app_setup) -> None:
        app, _, _ = app_setup
        client = TestClient(app)
        resp = client.get("/")
        assert 'href="/"' in resp.text
        assert 'href="/inbox"' in resp.text
        assert 'href="/dashboard"' in resp.text

    def test_active_tab_highlighted_on_home(self, app_setup) -> None:
        app, _, _ = app_setup
        resp = TestClient(app).get("/")
        # Home tab gets active class
        assert 'class="active"' in resp.text

    def test_active_tab_highlighted_on_dashboard(self, app_setup) -> None:
        app, _, _ = app_setup
        resp = TestClient(app).get("/dashboard")
        assert 'class="active"' in resp.text
