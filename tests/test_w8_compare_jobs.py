"""W8'' — compare_jobs SKILL: schema + adapter + /compare endpoint.

Tests the 5th evolvable SKILL across:
- COMPANY_APPLICATION_LIMITS lookup
- Pydantic schema (extra='forbid', rank ≥ 1, action enum, list bounds)
- Adapter metric scoring on good and pathological outputs
- /compare GET (lists company groups) + POST /compare/run (invokes SKILL)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import offerguide
from offerguide.config import Settings
from offerguide.evolution.adapters import compare_jobs as adapter
from offerguide.evolution.adapters import get_adapter
from offerguide.profile import UserProfile
from offerguide.skills import SkillResult, discover_skills, load_skill
from offerguide.skills.compare_jobs.helpers import (
    COMPANY_APPLICATION_LIMITS,
    CompareJobsResult,
    JobComparison,
    ProfileAlignment,
    lookup_application_limit,
)
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ── Frontmatter / loading ──────────────────────────────────────────


def test_skill_loads() -> None:
    spec = load_skill(SKILLS_ROOT / "compare_jobs")
    assert spec.name == "compare_jobs"
    assert spec.version == "0.1.0"
    assert set(spec.inputs) == {"company", "user_profile", "jobs_json"}


def test_skill_appears_in_discover() -> None:
    skills = discover_skills(SKILLS_ROOT)
    assert "compare_jobs" in {s.name for s in skills}


# ── lookup_application_limit ───────────────────────────────────────


class TestLookup:
    def test_exact_match(self) -> None:
        assert lookup_application_limit("字节跳动") == 2
        assert lookup_application_limit("淘天") == 3
        assert lookup_application_limit("阿里巴巴") == 3

    def test_substring_match_falls_through(self) -> None:
        # 阿里云 should match 阿里 (substring)
        assert lookup_application_limit("阿里云") == 3
        # 字节跳动-Doubao should match 字节跳动 (and so 字节)
        assert lookup_application_limit("字节跳动-Doubao") == 2

    def test_default_when_unknown(self) -> None:
        assert lookup_application_limit("某不知名公司") == 3
        assert lookup_application_limit("某不知名公司", default=5) == 5

    def test_empty_string_returns_default(self) -> None:
        assert lookup_application_limit("") == 3

    def test_known_table_completeness(self) -> None:
        """Sanity check on the hardcoded data — must include real
        sources from the web research."""
        assert "字节跳动" in COMPANY_APPLICATION_LIMITS
        assert "阿里巴巴" in COMPANY_APPLICATION_LIMITS
        assert "腾讯" in COMPANY_APPLICATION_LIMITS


# ── Pydantic schema ────────────────────────────────────────────────


def _good_output(*, n_jobs: int = 5) -> dict:
    return {
        "company": "字节跳动",
        "application_limit_estimate": 2,
        "application_limit_source": "known",
        "rankings": [
            {
                "job_id": 100 + i,
                "title": f"Position {i}",
                "rank": i + 1,
                "action": "apply_first" if i < 2 else (
                    "apply_backup" if i < 4 else "skip"
                ),
                "match_probability": 0.8 - 0.1 * i,
                "competitiveness_estimate": 0.6,
                "profile_alignment": {"tech": 0.8, "exp": 0.7, "culture": 0.5},
                "distinguishing_factors": [f"unique{i}_a", f"unique{i}_b"],
                "risk_factors": [f"risk{i}"],
                "reasoning": f"position {i} reasoning sentence here.",
            }
            for i in range(n_jobs)
        ],
        "recommended_apply_count": 2,
        "strategic_summary": "投 #100 + #101，跳 #103 #104 因为不沾。",
    }


class TestSchema:
    def test_valid_passes(self) -> None:
        result = CompareJobsResult.model_validate(_good_output())
        assert len(result.rankings) == 5
        assert result.recommended_apply_count == 2

    def test_extra_field_rejected(self) -> None:
        bad = _good_output()
        bad["bonus"] = "x"
        with pytest.raises(ValidationError):
            CompareJobsResult.model_validate(bad)

    def test_rank_must_be_positive(self) -> None:
        bad = _good_output()
        bad["rankings"][0]["rank"] = 0
        with pytest.raises(ValidationError):
            CompareJobsResult.model_validate(bad)

    def test_match_probability_oor(self) -> None:
        bad = _good_output()
        bad["rankings"][0]["match_probability"] = 1.5
        with pytest.raises(ValidationError):
            CompareJobsResult.model_validate(bad)

    def test_invalid_action_enum(self) -> None:
        bad = _good_output()
        bad["rankings"][0]["action"] = "definitely_apply_omg"
        with pytest.raises(ValidationError):
            CompareJobsResult.model_validate(bad)

    def test_zero_jobs_rejected(self) -> None:
        bad = _good_output()
        bad["rankings"] = []
        with pytest.raises(ValidationError):
            CompareJobsResult.model_validate(bad)

    def test_alignment_oor(self) -> None:
        bad = _good_output()
        bad["rankings"][0]["profile_alignment"]["tech"] = 1.2
        with pytest.raises(ValidationError):
            CompareJobsResult.model_validate(bad)

    def test_helper_methods(self) -> None:
        result = CompareJobsResult.model_validate(_good_output())
        first = result.by_action("apply_first")
        assert len(first) == 2
        top = result.top_picks()
        assert top[0].rank == 1
        ids = result.all_ids()
        assert ids == {100, 101, 102, 103, 104}

    def test_direct_instantiation(self) -> None:
        align = ProfileAlignment(tech=0.5, exp=0.5, culture=0.5)
        jc = JobComparison(
            job_id=1, title="X", rank=1, action="apply_first",
            match_probability=0.5, competitiveness_estimate=0.5,
            profile_alignment=align,
            distinguishing_factors=[], risk_factors=[],
            reasoning="ok",
        )
        assert jc.rank == 1


# ── Adapter ────────────────────────────────────────────────────────


def _example():
    return next(e for e in adapter.EXAMPLES if e.name == "bytedance_5_options")


class TestAdapter:
    def test_registered(self) -> None:
        assert get_adapter("compare_jobs") is adapter

    def test_examples_non_empty(self) -> None:
        assert len(adapter.EXAMPLES) >= 3

    def test_invalid_json_zero(self) -> None:
        ex = _example()
        result = adapter.metric(ex, "not valid")
        assert result.total == 0.0

    def test_schema_valid_scores_positive(self) -> None:
        ex = _example()
        # The good output has 5 jobs but the example has job_ids (101, 102, 103, 104, 105).
        # Need the rankings to use those exact ids.
        out = _good_output(n_jobs=5)
        for i, r in enumerate(out["rankings"]):
            r["job_id"] = 101 + i  # match example.job_ids
        result = adapter.metric(ex, json.dumps(out, ensure_ascii=False))
        assert result.total > 0.5
        assert result.breakdown["schema"] == 1.0

    def test_rank_validity_fails_when_id_missing(self) -> None:
        ex = _example()
        out = _good_output(n_jobs=4)  # only 4 rankings, but example has 5 ids
        for i, r in enumerate(out["rankings"]):
            r["job_id"] = 101 + i
        result = adapter.metric(ex, json.dumps(out, ensure_ascii=False))
        assert result.breakdown["rank_validity"] == 0.0

    def test_limit_consistency_penalty(self) -> None:
        ex = _example()
        out = _good_output(n_jobs=5)
        for i, r in enumerate(out["rankings"]):
            r["job_id"] = 101 + i
        out["application_limit_estimate"] = 8  # 实际 2，偏离 6
        result = adapter.metric(ex, json.dumps(out, ensure_ascii=False))
        assert result.breakdown["limit_consistency"] < 1.0

    def test_distinguishing_quality_low_when_repeated(self) -> None:
        ex = _example()
        out = _good_output(n_jobs=5)
        for i, r in enumerate(out["rankings"]):
            r["job_id"] = 101 + i
            r["distinguishing_factors"] = ["要求 Python", "需要项目经验"]  # all same
        result = adapter.metric(ex, json.dumps(out, ensure_ascii=False))
        assert result.breakdown["distinguishing_quality"] < 0.5

    def test_all_misfit_case_zero_apply_first_required(self) -> None:
        ex = next(e for e in adapter.EXAMPLES if e.name == "all_misfit_quant")
        out = {
            "company": "某私募",
            "application_limit_estimate": 3,
            "application_limit_source": "inferred",
            "rankings": [
                {
                    "job_id": 301, "title": "x", "rank": 1,
                    "action": "apply_first",  # WRONG — should be skip
                    "match_probability": 0.15,
                    "competitiveness_estimate": 0.5,
                    "profile_alignment": {"tech": 0.2, "exp": 0.1, "culture": 0.5},
                    "distinguishing_factors": [], "risk_factors": [],
                    "reasoning": "x",
                },
                {
                    "job_id": 302, "title": "y", "rank": 2,
                    "action": "skip",
                    "match_probability": 0.10,
                    "competitiveness_estimate": 0.5,
                    "profile_alignment": {"tech": 0.1, "exp": 0.1, "culture": 0.5},
                    "distinguishing_factors": ["x"], "risk_factors": ["y"],
                    "reasoning": "y",
                },
            ],
            "recommended_apply_count": 1,
            "strategic_summary": "投 #301。",
        }
        result = adapter.metric(ex, json.dumps(out, ensure_ascii=False))
        # all_misfit case demands apply_first_count == 0 → action_coherence drops hard
        assert result.breakdown["action_coherence"] == 0.0

    def test_split_train_val(self) -> None:
        train, val = adapter.split_train_val(seed=0)
        assert len(train) + len(val) == len(adapter.EXAMPLES)


# ── /compare endpoint ──────────────────────────────────────────────


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list = []

    def invoke(self, spec, inputs, **_):
        self.calls.append((spec.name, inputs))
        # Return a minimal valid CompareJobsResult shape
        jobs = json.loads(inputs["jobs_json"])
        rankings = [
            {
                "job_id": j["job_id"],
                "title": j["title"],
                "rank": i + 1,
                "action": "apply_first" if i < 2 else "skip",
                "match_probability": 0.7 - 0.1 * i,
                "competitiveness_estimate": 0.6,
                "profile_alignment": {"tech": 0.7, "exp": 0.6, "culture": 0.5},
                "distinguishing_factors": [f"factor_{i}_a", f"factor_{i}_b"],
                "risk_factors": [f"risk_{i}"],
                "reasoning": f"reasoning {i}",
            }
            for i, j in enumerate(jobs)
        ]
        parsed = {
            "company": inputs["company"],
            "application_limit_estimate": 2,
            "application_limit_source": "known",
            "rankings": rankings,
            "recommended_apply_count": 2,
            "strategic_summary": f"投 top 2 of {inputs['company']}.",
        }
        return SkillResult(
            raw_text=json.dumps(parsed, ensure_ascii=False),
            parsed=parsed,
            skill_name=spec.name, skill_version=spec.version,
            skill_run_id=42, input_hash="x", cost_usd=0.0001, latency_ms=10,
        )


@pytest.fixture
def app_setup(tmp_path: Path):
    store = offerguide.Store(tmp_path / "cmp.db")
    store.init_schema()
    profile = UserProfile(raw_resume_text="胡阳的简历")
    skills = discover_skills(SKILLS_ROOT)
    runtime = _FakeRuntime()
    app = create_app(
        settings=Settings(),
        store=store, profile=profile, skills=skills,
        runtime=runtime,  # type: ignore[arg-type]
        notifier=ConsoleNotifier(),
    )
    return app, store, runtime


def _seed_jobs_for_company(store, company: str, n: int = 3) -> list[int]:
    ids: list[int] = []
    for i in range(n):
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
                "VALUES ('manual', ?, ?, ?, ?)",
                (f"职位 {i}", company, f"jd content {i}", f"hash_{company}_{i}"),
            )
            ids.append(int(cur.lastrowid or 0))
    return ids


class TestCompareEndpoint:
    def test_compare_get_lists_company_groups(self, app_setup) -> None:
        app, store, _ = app_setup
        _seed_jobs_for_company(store, "字节跳动", n=3)
        _seed_jobs_for_company(store, "阿里巴巴", n=2)
        _seed_jobs_for_company(store, "唯一一个职位的公司", n=1)

        resp = TestClient(app).get("/compare")
        assert resp.status_code == 200
        assert "字节跳动" in resp.text
        assert "阿里巴巴" in resp.text
        # Singleton company should NOT appear
        assert "唯一一个职位的公司" not in resp.text

    def test_compare_get_with_company_param(self, app_setup) -> None:
        app, store, _ = app_setup
        _seed_jobs_for_company(store, "字节跳动", n=3)
        resp = TestClient(app).get("/compare?company=字节跳动")
        assert "字节跳动" in resp.text
        assert "已知投递限额" in resp.text
        # Form to select jobs
        assert 'name="job_ids"' in resp.text

    def test_compare_run_invokes_skill(self, app_setup) -> None:
        app, store, runtime = app_setup
        ids = _seed_jobs_for_company(store, "字节跳动", n=3)

        client = TestClient(app)
        resp = client.post(
            "/compare/run",
            data={
                "company": "字节跳动",
                "application_limit": "2",
                "job_ids": [str(i) for i in ids],
            },
        )
        assert resp.status_code == 200
        # SKILL was invoked
        assert any(c[0] == "compare_jobs" for c in runtime.calls)
        # Rendered with some apply_first entries
        assert "apply_first" in resp.text

    def test_compare_run_too_few_jobs(self, app_setup) -> None:
        app, store, _ = app_setup
        ids = _seed_jobs_for_company(store, "字节跳动", n=2)
        resp = TestClient(app).post(
            "/compare/run",
            data={
                "company": "字节跳动",
                "application_limit": "2",
                "job_ids": [str(ids[0])],  # only 1
            },
        )
        assert "至少要选 2 个" in resp.text

    def test_compare_run_no_profile(self, tmp_path: Path) -> None:
        store = offerguide.Store(tmp_path / "x.db")
        store.init_schema()
        ids = _seed_jobs_for_company(store, "字节", n=2)
        app = create_app(
            settings=Settings(), store=store, profile=None,
            skills=discover_skills(SKILLS_ROOT), runtime=_FakeRuntime(),  # type: ignore[arg-type]
            notifier=ConsoleNotifier(),
        )
        resp = TestClient(app).post(
            "/compare/run",
            data={"company": "字节", "job_ids": [str(i) for i in ids]},
        )
        assert "未加载简历" in resp.text


def test_nav_has_compare_link(app_setup) -> None:
    app, _, _ = app_setup
    resp = TestClient(app).get("/")
    assert 'href="/compare"' in resp.text
