"""W8 +3 — post_interview_reflection SKILL + /reflect endpoint
+ auto-feedback loop + cover letter PDF print page.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import offerguide
from offerguide import briefs, story_bank
from offerguide.briefs import CompanyBrief
from offerguide.config import Settings
from offerguide.evolution.adapters import get_adapter
from offerguide.evolution.adapters import (
    post_interview_reflection as reflect_adapter,
)
from offerguide.profile import UserProfile
from offerguide.skills import SkillResult, discover_skills, load_skill
from offerguide.skills.post_interview_reflection.helpers import (
    PostInterviewReflection,
    QuestionMatch,
)
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ── fixtures ───────────────────────────────────────────────────────


def _make_store(tmp_path: Path) -> offerguide.Store:
    store = offerguide.Store(tmp_path / "reflect.db")
    store.init_schema()
    return store


def _good_reflection() -> dict:
    return {
        "company": "字节跳动",
        "hit_rate": 0.6,
        "matched_predictions": [
            {"predicted_question": "Transformer attention 缩放推导",
             "predicted_likelihood": 0.85,
             "match_kind": "exact",
             "actual_question": "为什么除以 √d",
             "user_self_rating": 0.85},
            {"predicted_question": "GRPO vs PPO",
             "predicted_likelihood": 0.7,
             "match_kind": "exact",
             "actual_question": "GRPO 和 PPO 区别在哪",
             "user_self_rating": 0.9},
            {"predicted_question": "company_specific 业务对比",
             "predicted_likelihood": 0.4,
             "match_kind": "miss",
             "actual_question": None,
             "user_self_rating": None},
        ],
        "surprises": [
            {"question": "设计 100w QPS agent inference pipeline",
             "category": "system_design",
             "why_we_missed": "system_design 没在我们的 deep_project_prep 里高优先级，"
                              "字节实习一面其实常出"},
        ],
        "user_performance_summary": "前 3 题答得不错，system design 没准备没答好",
        "suggested_stories": [
            {"title": "RemeDi 训练流水线",
             "suggested_situation": "面试官追问 RemeDi loss 曲线",
             "suggested_task": "解释训练崩溃 + 恢复",
             "suggested_action": "二分回退 + 对比 wandb logs",
             "suggested_result": "定位 LR scheduler bug 修复",
             "suggested_reflection": "训练前先冻结实验跑短跑",
             "suggested_tags": ["failure", "learning"],
             "triggered_by": "面试官追问 RemeDi loss 曲线"},
        ],
        "brief_delta": {
            "interview_style_addition": "字节 Seed 实习一面会出 system design",
            "new_recent_signals": ["100w QPS pipeline 是 surprise"],
            "confidence_adjustment": -0.05,
        },
        "weak_spots_to_practice": [
            "Speculative decoding 数学",
            "Agent inference pipeline 工程设计",
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# SCHEMA
# ═══════════════════════════════════════════════════════════════════


class TestSchema:
    def test_validates_well_formed(self) -> None:
        r = PostInterviewReflection.model_validate(_good_reflection())
        assert r.hit_rate == 0.6
        assert len(r.matched_predictions) == 3
        assert len(r.surprises) == 1
        assert len(r.suggested_stories) == 1

    def test_rejects_extra_keys(self) -> None:
        bad = _good_reflection()
        bad["bonus_field"] = "x"
        with pytest.raises(ValidationError):
            PostInterviewReflection.model_validate(bad)

    def test_rejects_invalid_match_kind(self) -> None:
        bad = _good_reflection()
        bad["matched_predictions"][0]["match_kind"] = "rocket_launch"
        with pytest.raises(ValidationError):
            PostInterviewReflection.model_validate(bad)

    def test_rejects_hit_rate_oor(self) -> None:
        bad = _good_reflection()
        bad["hit_rate"] = 1.5
        with pytest.raises(ValidationError):
            PostInterviewReflection.model_validate(bad)

    def test_calibration_score_lower_when_well_calibrated(self) -> None:
        # Likelihood 0.85 + match → small error
        # Likelihood 0.40 + miss → small error
        good = _good_reflection()
        r = PostInterviewReflection.model_validate(good)
        # 0.85 vs 1.0 = 0.15; 0.7 vs 1.0 = 0.3; 0.4 vs 0.0 = 0.4 → mean ≈ 0.283
        assert r.calibration_score() < 0.4

    def test_calibration_score_higher_when_miscalibrated(self) -> None:
        # Likelihood 0.95 + miss → big error
        bad = _good_reflection()
        bad["matched_predictions"] = [
            {"predicted_question": "Q", "predicted_likelihood": 0.95,
             "match_kind": "miss", "actual_question": None, "user_self_rating": None},
        ]
        r = PostInterviewReflection.model_validate(bad)
        assert r.calibration_score() > 0.5

    def test_question_match_direct(self) -> None:
        m = QuestionMatch(
            predicted_question="Q", predicted_likelihood=0.5,
            match_kind="exact", actual_question="Q",
            user_self_rating=0.8,
        )
        assert m.match_kind == "exact"


# ═══════════════════════════════════════════════════════════════════
# ADAPTER
# ═══════════════════════════════════════════════════════════════════


def _example():
    return next(e for e in reflect_adapter.EXAMPLES if e.name == "bytedance_high_hit")


class TestAdapter:
    def test_skill_loads(self) -> None:
        spec = load_skill(SKILLS_ROOT / "post_interview_reflection")
        assert spec.name == "post_interview_reflection"
        assert set(spec.inputs) == {"company", "prep_questions_json", "actual_transcript"}

    def test_registered(self) -> None:
        assert get_adapter("post_interview_reflection") is reflect_adapter

    def test_invalid_json_zero(self) -> None:
        ex = _example()
        result = reflect_adapter.metric(ex, "not json")
        assert result.total == 0.0

    def test_good_output_scores_high(self) -> None:
        ex = _example()
        result = reflect_adapter.metric(
            ex, json.dumps(_good_reflection(), ensure_ascii=False)
        )
        assert result.total > 0.5
        assert result.breakdown["schema"] == 1.0

    def test_hit_rate_out_of_band_penalized(self) -> None:
        # bytedance_high_hit expects 0.5-0.85; output 0.1 is way below
        ex = _example()
        bad = _good_reflection()
        bad["hit_rate"] = 0.1
        result = reflect_adapter.metric(ex, json.dumps(bad, ensure_ascii=False))
        assert result.breakdown["hit_rate_in_band"] < 1.0

    def test_generic_miss_explanation_penalized(self) -> None:
        ex = _example()
        bad = _good_reflection()
        bad["surprises"] = [
            {"question": "X", "category": "technical", "why_we_missed": "意外问题"}
        ]
        result = reflect_adapter.metric(ex, json.dumps(bad, ensure_ascii=False))
        assert result.breakdown["surprises_explained"] < 1.0

    def test_empty_transcript_case(self) -> None:
        """The empty-transcript example expects hit_rate ≈ 0."""
        ex = next(e for e in reflect_adapter.EXAMPLES if e.name == "empty_transcript")
        out = _good_reflection()
        out["hit_rate"] = 0.0
        out["matched_predictions"] = []
        out["surprises"] = []
        out["suggested_stories"] = []
        out["brief_delta"]["interview_style_addition"] = None
        result = reflect_adapter.metric(ex, json.dumps(out, ensure_ascii=False))
        assert result.breakdown["hit_rate_in_band"] == 1.0


# ═══════════════════════════════════════════════════════════════════
# /reflect endpoint + auto-feedback loop
# ═══════════════════════════════════════════════════════════════════


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._next_id = 800

    def invoke(self, spec, inputs, **_):
        self._next_id += 1
        self.calls.append((spec.name, dict(inputs)))
        parsed = (
            _good_reflection() if spec.name == "post_interview_reflection" else {}
        )
        return SkillResult(
            raw_text=json.dumps(parsed, ensure_ascii=False),
            parsed=parsed, skill_name=spec.name, skill_version=spec.version,
            skill_run_id=self._next_id, input_hash="x",
            cost_usd=0.0001, latency_ms=42,
        )


@pytest.fixture
def app_setup(tmp_path: Path):
    store = _make_store(tmp_path)
    profile = UserProfile(raw_resume_text="x")
    skills = discover_skills(SKILLS_ROOT)
    runtime = _FakeRuntime()
    app = create_app(
        settings=Settings(), store=store, profile=profile,
        skills=skills, runtime=runtime,  # type: ignore[arg-type]
        notifier=ConsoleNotifier(),
    )
    return app, store, runtime


class TestReflectEndpoint:
    def test_get_reflect_page(self, app_setup) -> None:
        app, _, _ = app_setup
        resp = TestClient(app).get("/reflect")
        assert resp.status_code == 200
        assert "面试复盘" in resp.text
        assert "actual_transcript" in resp.text

    def test_post_runs_skill(self, app_setup) -> None:
        app, _, runtime = app_setup
        resp = TestClient(app).post(
            "/api/reflect/run",
            data={
                "company": "字节跳动",
                "actual_transcript": "面完了 attention 缩放问到了 GRPO 答得好 ...",
            },
        )
        assert resp.status_code == 200
        assert any(c[0] == "post_interview_reflection" for c in runtime.calls)
        assert "60%" in resp.text  # hit_rate

    def test_post_required_fields(self, app_setup) -> None:
        app, _, _ = app_setup
        # Missing fields → FastAPI 422 (Form(...) Pydantic validation)
        resp = TestClient(app).post(
            "/api/reflect/run", data={},
        )
        assert resp.status_code == 422

        # Empty strings (fields present but blank) → 200 with error fragment
        resp2 = TestClient(app).post(
            "/api/reflect/run",
            data={"company": "  ", "actual_transcript": "  "},
        )
        assert resp2.status_code == 200
        assert "都必填" in resp2.text

    def test_no_profile_returns_error(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        app = create_app(
            settings=Settings(), store=store, profile=None,
            skills=discover_skills(SKILLS_ROOT), runtime=None,
            notifier=ConsoleNotifier(),
        )
        resp = TestClient(app).post(
            "/api/reflect/run",
            data={"company": "字节", "actual_transcript": "x"},
        )
        assert "未加载简历" in resp.text


class TestAutoFeedbackLoop:
    def test_auto_apply_stories_inserts_into_story_bank(self, app_setup) -> None:
        app, store, _ = app_setup
        # Initially empty
        assert len(story_bank.list_all(store)) == 0

        TestClient(app).post(
            "/api/reflect/run",
            data={
                "company": "字节跳动",
                "actual_transcript": "...",
                "auto_apply_stories": "1",
            },
        )
        # The fake reflection has 1 suggested story → should be inserted
        rows = story_bank.list_all(store)
        assert len(rows) == 1
        assert rows[0].title == "RemeDi 训练流水线"
        assert "failure" in rows[0].tags

    def test_no_auto_apply_no_insert(self, app_setup) -> None:
        app, store, _ = app_setup
        TestClient(app).post(
            "/api/reflect/run",
            data={
                "company": "字节跳动",
                "actual_transcript": "...",
                # no auto_apply_stories
            },
        )
        assert len(story_bank.list_all(store)) == 0

    def test_auto_apply_brief_appends_interview_style(self, app_setup) -> None:
        app, store, _ = app_setup
        # Seed an existing brief
        from offerguide.briefs import _upsert
        _upsert(store, "字节跳动", CompanyBrief(
            summary="x",
            current_app_limit=2,
            interview_style="原有: 项目深挖",
            recent_signals=["旧信号"],
            hiring_trend="expanding",
            confidence=0.6,
        ))

        TestClient(app).post(
            "/api/reflect/run",
            data={
                "company": "字节跳动",
                "actual_transcript": "...",
                "auto_apply_brief": "1",
            },
        )
        updated = briefs.get_brief(store, "字节跳动")
        assert updated is not None
        # Style addition merged in
        assert "system design" in updated.brief.interview_style
        # New signal appended
        assert any("100w QPS" in s for s in updated.brief.recent_signals)
        # Confidence dropped by 0.05
        assert updated.brief.confidence == pytest.approx(0.55, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
# Cover letter print page
# ═══════════════════════════════════════════════════════════════════


def _seed_cover_letter_run(store, run_id: int = 999) -> None:
    """Seed a fake write_cover_letter skill_runs row."""
    output = {
        "opening_hook": "Looking at the GRPO + DeepSpeed track at Seed",
        "narrative_body": ["Para 1 with PyTorch context", "Para 2 with RemeDi"],
        "closing_call_to_action": "Available May 2026.",
        "customization_signals": ["Mentioned Seed's recent paper"],
        "ats_keywords_used": ["PyTorch", "GRPO"],
        "ai_risk_warnings": [],
        "suggested_tone": "warm_concise",
        "personalization_score": 0.72,
        "overall_word_count": 220,
    }
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO skill_runs(id, skill_name, skill_version, input_hash, "
            "input_json, output_json, latency_ms) "
            "VALUES (?, 'write_cover_letter', '0.1.0', 'h', '{}', ?, 42)",
            (run_id, json.dumps(output, ensure_ascii=False)),
        )


class TestCoverLetterPrint:
    def test_print_page_renders(self, app_setup) -> None:
        app, store, _ = app_setup
        _seed_cover_letter_run(store)

        resp = TestClient(app).get("/cover-letter/999.html")
        assert resp.status_code == 200
        # Standalone HTML doc (no extends base.html)
        assert "<!DOCTYPE html>" in resp.text
        assert "@page" in resp.text  # print CSS
        assert "Looking at the GRPO" in resp.text
        # Print button only visible on screen
        assert "no-print" in resp.text

    def test_404_for_unknown_run(self, app_setup) -> None:
        app, _, _ = app_setup
        resp = TestClient(app).get("/cover-letter/9999.html")
        assert resp.status_code == 404

    def test_404_for_wrong_skill_kind(self, app_setup) -> None:
        app, store, _ = app_setup
        # Seed a non-cover-letter run with id 50
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_runs(id, skill_name, skill_version, input_hash, "
                "input_json, output_json, latency_ms) "
                "VALUES (50, 'score_match', '0.2.0', 'h', '{}', '{}', 1)"
            )
        resp = TestClient(app).get("/cover-letter/50.html")
        assert resp.status_code == 404


def test_topbar_has_reflect_link(app_setup) -> None:
    app, _, _ = app_setup
    resp = TestClient(app).get("/")
    assert 'href="/reflect"' in resp.text
