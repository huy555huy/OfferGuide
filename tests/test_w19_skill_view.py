"""W19 — SKILL invoke helper tests + nowcoder discovered_via attribution."""
from __future__ import annotations

import json as _json
from pathlib import Path

import pytest

import offerguide
from offerguide.config import Settings
from offerguide.agent_runtime import _schema as harness_schema
from offerguide.profile import UserProfile
from offerguide.skill_view import SkillViewResult, invoke_skill_for_view
from offerguide.skills import discover_skills

# ─────────────────── invoke_skill_for_view validation paths ───────────────────


@pytest.fixture
def real_skills():
    return discover_skills(Path(__file__).parent.parent / "src/offerguide/skills")


@pytest.fixture
def real_store(tmp_path):
    s = offerguide.Store(tmp_path / "w19.db")
    s.init_schema()
    harness_schema.init_agent_runtime_schema(s)
    return s


def _profile_with_resume(text: str = "x" * 200) -> UserProfile:
    return UserProfile(raw_resume_text=text)


@pytest.mark.asyncio
async def test_returns_error_when_no_api_key(real_store, real_skills):
    settings = Settings(deepseek_api_key="", default_model="stub")
    result = await invoke_skill_for_view(
        skill_name="apply_assistant",
        inputs_builder=lambda _spec, _p: {"x": "y"},
        settings=settings, profile=_profile_with_resume(),
        runtime=None, skills=real_skills, store=real_store,
    )
    assert result.error is not None
    assert "LLM key" in result.error
    assert result.parsed is None


@pytest.mark.asyncio
async def test_returns_error_when_no_profile(real_store, real_skills):
    settings = Settings(deepseek_api_key="x", default_model="stub")
    result = await invoke_skill_for_view(
        skill_name="apply_assistant",
        inputs_builder=lambda _spec, _p: {"x": "y"},
        settings=settings, profile=None,
        runtime=None, skills=real_skills, store=real_store,
    )
    assert result.error is not None
    assert "简历" in result.error


@pytest.mark.asyncio
async def test_returns_error_when_no_runtime(real_store, real_skills):
    settings = Settings(deepseek_api_key="x", default_model="stub")
    result = await invoke_skill_for_view(
        skill_name="apply_assistant",
        inputs_builder=lambda _spec, _p: {"x": "y"},
        settings=settings, profile=_profile_with_resume(),
        runtime=None, skills=real_skills, store=real_store,
    )
    assert result.error is not None
    assert "SkillRuntime" in result.error


@pytest.mark.asyncio
async def test_returns_error_when_skill_missing(real_store, real_skills):
    settings = Settings(deepseek_api_key="x", default_model="stub")

    class _StubRuntime:
        def invoke(self, *a, **kw):
            return None

    result = await invoke_skill_for_view(
        skill_name="not_a_real_skill",
        inputs_builder=lambda _spec, _p: {"x": "y"},
        settings=settings, profile=_profile_with_resume(),
        runtime=_StubRuntime(),  # type: ignore[arg-type]
        skills=real_skills, store=real_store,
    )
    assert result.error is not None
    assert "not_a_real_skill" in result.error
    assert "没注册" in result.error


@pytest.mark.asyncio
async def test_invokes_skill_and_returns_parsed(real_store, real_skills):
    """Happy path with stub runtime returning valid JSON."""
    settings = Settings(deepseek_api_key="x", default_model="stub")

    captured_inputs: dict = {}

    class _StubRuntime:
        def invoke(self, spec, inputs, **kw):
            captured_inputs.update(inputs)
            from offerguide.skills._runtime import SkillResult
            return SkillResult(
                raw_text='{"company":"X","self_intro_snippet":{"text":"hi"}}',
                parsed={"company": "X", "self_intro_snippet": {"text": "hi"}},
                skill_name=spec.name, skill_version=spec.version,
                skill_run_id=42, input_hash="h",
                cost_usd=0.001, latency_ms=120,
            )

    result = await invoke_skill_for_view(
        skill_name="apply_assistant",
        inputs_builder=lambda _spec, p: {
            "company": "字节", "role_focus": "AI 实习",
            "job_text": "x" * 100, "user_profile": p.raw_resume_text[:200],
        },
        settings=settings, profile=_profile_with_resume("test resume content"),
        runtime=_StubRuntime(),  # type: ignore[arg-type]
        skills=real_skills, store=real_store,
    )
    assert result.error is None
    assert result.parsed == {"company": "X", "self_intro_snippet": {"text": "hi"}}
    assert result.skill_run_id == 42
    assert result.cost_usd == 0.001
    assert result.duration_ms >= 0
    # Builder was called with correct args
    assert captured_inputs["company"] == "字节"
    assert captured_inputs["user_profile"] == "test resume content"


@pytest.mark.asyncio
async def test_returns_error_when_skill_outputs_invalid_json(real_store, real_skills):
    """SKILL JSON parse failure → error with raw_text preserved for debug."""
    settings = Settings(deepseek_api_key="x", default_model="stub")

    class _StubRuntime:
        def invoke(self, spec, inputs, **kw):
            from offerguide.skills._runtime import SkillResult
            return SkillResult(
                raw_text="<garbage that's not JSON>",
                parsed=None,
                skill_name=spec.name, skill_version=spec.version,
                skill_run_id=99, input_hash="h",
                cost_usd=0.0, latency_ms=80,
            )

    result = await invoke_skill_for_view(
        skill_name="apply_assistant",
        inputs_builder=lambda _spec, _p: {"x": "y"},
        settings=settings, profile=_profile_with_resume(),
        runtime=_StubRuntime(),  # type: ignore[arg-type]
        skills=real_skills, store=real_store,
    )
    assert result.error is not None
    assert "JSON" in result.error
    assert result.raw_text == "<garbage that's not JSON>"
    assert result.skill_run_id == 99


@pytest.mark.asyncio
async def test_returns_error_when_invoke_raises(real_store, real_skills):
    """Runtime crashing → error message, no leak of stack to UI."""
    settings = Settings(deepseek_api_key="x", default_model="stub")

    class _CrashingRuntime:
        def invoke(self, *a, **kw):
            raise RuntimeError("kaboom")

    result = await invoke_skill_for_view(
        skill_name="apply_assistant",
        inputs_builder=lambda _spec, _p: {"x": "y"},
        settings=settings, profile=_profile_with_resume(),
        runtime=_CrashingRuntime(),  # type: ignore[arg-type]
        skills=real_skills, store=real_store,
    )
    assert result.error is not None
    assert "kaboom" in result.error


# ─────────────────── nowcoder discovered_via attribution ───────────────────


def test_nowcoder_crawl_tags_discovered_via(monkeypatch, real_store):
    """W18 → W19 — every nowcoder ingest should land with extras_json having
    discovered_via='nowcoder_sitemap' so /recommended cards show source."""
    from offerguide.platforms import RawJob
    from offerguide.workers import scout

    fake_jobs = [
        RawJob(
            source="nowcoder", source_id="111",
            url="https://www.nowcoder.com/jobs/detail/111",
            title="Test 暑期实习", company="测试公司", location="北京",
            raw_text="x" * 300, extras={},
        ),
    ]

    class _FakeClient:
        def close(self):
            pass

    monkeypatch.setattr(
        "offerguide.workers.scout.nowcoder.NowcoderClient",
        lambda **kw: _FakeClient(),
    )
    monkeypatch.setattr(
        "offerguide.workers.scout.nowcoder.iter_jd_urls",
        lambda client: iter(["https://www.nowcoder.com/jobs/detail/111"]),
    )
    monkeypatch.setattr(
        "offerguide.workers.scout.nowcoder.fetch_and_parse",
        lambda client, url: fake_jobs[0],
    )

    counters = scout.crawl_nowcoder(real_store, limit=1)
    assert counters["ingested_new"] == 1

    with real_store.connect() as conn:
        row = conn.execute(
            "SELECT extras_json FROM jobs WHERE title = 'Test 暑期实习'"
        ).fetchone()
    assert row is not None
    extras = _json.loads(row[0])
    assert extras.get("discovered_via") == "nowcoder_sitemap"
    assert extras.get("discovered_keyword") == "(sitemap walk, no keyword)"


def test_nowcoder_crawl_does_not_clobber_existing_attribution(
    monkeypatch, real_store,
):
    """If a downstream parser already set discovered_via, don't overwrite."""
    from offerguide.platforms import RawJob
    from offerguide.workers import scout

    rj = RawJob(
        source="nowcoder", source_id="222",
        url="https://www.nowcoder.com/jobs/detail/222",
        title="某 niche 岗位", company="某公司", location="上海",
        raw_text="y" * 300,
        extras={"discovered_via": "custom_tag", "discovered_keyword": "Diffusion"},
    )

    class _FakeClient:
        def close(self):
            pass

    monkeypatch.setattr(
        "offerguide.workers.scout.nowcoder.NowcoderClient",
        lambda **kw: _FakeClient(),
    )
    monkeypatch.setattr(
        "offerguide.workers.scout.nowcoder.iter_jd_urls",
        lambda client: iter(["https://www.nowcoder.com/jobs/detail/222"]),
    )
    monkeypatch.setattr(
        "offerguide.workers.scout.nowcoder.fetch_and_parse",
        lambda client, url: rj,
    )

    scout.crawl_nowcoder(real_store, limit=1)
    with real_store.connect() as conn:
        row = conn.execute(
            "SELECT extras_json FROM jobs WHERE title = '某 niche 岗位'"
        ).fetchone()
    extras = _json.loads(row[0])
    # Existing attribution preserved (setdefault, not overwrite)
    assert extras.get("discovered_via") == "custom_tag"
    assert extras.get("discovered_keyword") == "Diffusion"


# ─────────────────── SkillViewResult dataclass shape ───────────────────


def test_skill_view_result_default_fields():
    """All fields default-populated so templates can do .field with no errors."""
    r = SkillViewResult()
    assert r.parsed is None
    assert r.raw_text == ""
    assert r.skill_run_id is None
    assert r.cost_usd == 0.0
    assert r.duration_ms == 0
    assert r.error is None
