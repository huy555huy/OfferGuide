"""End-to-end W1 tests: every module wires up correctly and the example would run."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import offerguide
from offerguide.memory.vec import vec_version
from offerguide.skills import SkillParseError, discover_skills, load_skill

REPO_ROOT = Path(__file__).parent.parent
SKILLS_ROOT = REPO_ROOT / "src/offerguide/skills"

# The plan-mode authors' real resume — used for end-to-end PDF parsing if present.
SAMPLE_RESUME = Path(
    "/Users/huy/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
    "xwechat_files/wxid_5fq3lbda9swi22_c4de/msg/file/2026-04/"
    "胡阳-上海财经大学-应用统计专硕(4).pdf"
)


# ---- skills ---------------------------------------------------------------


def test_score_match_skill_loads() -> None:
    """Locks down on-disk shape only; version-specific assertions live in the per-week tests."""
    spec = load_skill(SKILLS_ROOT / "score_match")
    assert spec.name == "score_match"
    assert spec.version  # any non-empty version
    assert "calibrated" in spec.description.lower()
    assert spec.inputs == ("job_text", "user_profile")
    assert "评估这个岗位" in spec.triggers
    assert "matching" in spec.tags
    assert spec.body.strip()
    assert spec.helper_scripts and spec.helper_scripts[0].name == "helpers.py"


def test_discover_skills_finds_all() -> None:
    skills = discover_skills(SKILLS_ROOT)
    names = {s.name for s in skills}
    assert "score_match" in names


def test_skill_loader_rejects_missing_frontmatter(tmp_path: Path) -> None:
    bad = tmp_path / "bad_skill"
    bad.mkdir()
    (bad / "SKILL.md").write_text("no frontmatter here")
    with pytest.raises(SkillParseError):
        load_skill(bad)


def test_skill_loader_rejects_missing_required_fields(tmp_path: Path) -> None:
    bad = tmp_path / "missing_fields"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: x\n---\n\nbody")
    with pytest.raises(SkillParseError):
        load_skill(bad)


def test_skill_loader_handles_evolved_at(tmp_path: Path) -> None:
    sd = tmp_path / "evo"
    sd.mkdir()
    (sd / "SKILL.md").write_text(
        "---\n"
        "name: evo\n"
        "description: test\n"
        "version: 0.2.0\n"
        "evolved_at: '2026-04-28T10:00:00+08:00'\n"
        "parent_version: '0.1.0'\n"
        "---\n\nbody"
    )
    spec = load_skill(sd)
    assert spec.evolved_at is not None
    assert spec.parent_version == "0.1.0"


# ---- memory ---------------------------------------------------------------


def test_store_init_schema_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "store.db"
    store = offerguide.Store(db)
    store.init_schema()
    store.init_schema()  # second call must not crash

    assert db.exists()
    counts = store.health_check()
    assert counts == {
        "profile": 0,
        "jobs": 0,
        "applications": 0,
        "application_events": 0,
        "skill_runs": 0,
        "feedback": 0,
        "interviews": 0,
        "evolution_log": 0,
        "inbox_items": 0,
        "interview_experiences": 0,
    }


def test_store_pragmas_applied(tmp_path: Path) -> None:
    store = offerguide.Store(tmp_path / "store.db")
    store.init_schema()
    with store.connect() as conn:
        (fk,) = conn.execute("PRAGMA foreign_keys").fetchone()
        (jm,) = conn.execute("PRAGMA journal_mode").fetchone()
    assert fk == 1
    assert jm.lower() == "wal"


def test_sqlite_vec_extension_loads(tmp_path: Path) -> None:
    store = offerguide.Store(tmp_path / "store.db")
    store.init_schema()
    with store.connect(with_vec=True) as conn:
        v = vec_version(conn)
    assert v.startswith("v")


def test_init_vec_schema_creates_virtual_table(tmp_path: Path) -> None:
    from offerguide.memory.vec import init_vec_schema

    store = offerguide.Store(tmp_path / "store.db")
    store.init_schema()
    with store.connect(with_vec=True) as conn:
        init_vec_schema(conn, embedding_dim=8)
        # confirm table is queryable
        (cnt,) = conn.execute("SELECT COUNT(*) FROM vec_embeddings").fetchone()
        assert cnt == 0
        # confirm the metadata sidecar table exists too
        (cnt2,) = conn.execute("SELECT COUNT(*) FROM vec_metadata").fetchone()
        assert cnt2 == 0


# ---- profile --------------------------------------------------------------


def test_load_resume_pdf_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pdf"
    with pytest.raises(FileNotFoundError):
        offerguide.load_resume_pdf(missing)


@pytest.mark.skipif(not SAMPLE_RESUME.exists(), reason="sample resume PDF not on this machine")
def test_load_real_resume_extracts_text() -> None:
    profile = offerguide.load_resume_pdf(SAMPLE_RESUME)
    assert len(profile.raw_resume_text) > 500
    assert profile.source_pdf is not None
    # The resume is in Chinese — sanity-check that some Chinese characters survived
    assert any("一" <= ch <= "鿿" for ch in profile.raw_resume_text)


# ---- agent ----------------------------------------------------------------


def test_graph_builds_and_routes_to_summarize_when_no_action() -> None:
    """W4: with no `requested_action` and no runtime, graph still composes a summary."""
    skills = discover_skills(SKILLS_ROOT)
    graph = offerguide.build_graph(skills=skills, runtime=None)
    # No `requested_action` → defaults to score_and_gaps but runtime=None → error path
    # → summarize still runs and produces a final_response
    result = graph.invoke({"requested_action": None})
    assert result.get("final_response") is not None


def test_graph_runs_with_empty_skills() -> None:
    """build_graph must accept an empty skills iterable for topology-only tests."""
    graph = offerguide.build_graph(skills=[], runtime=None)
    # With empty skills, score_node would raise — but with no action it routes straight to summarize
    result = graph.invoke({"requested_action": None})
    assert "final_response" in result


# ---- public API -----------------------------------------------------------


def test_public_api_surface() -> None:
    """Lock down what offerguide exports at the top level — refactors must update __all__."""
    expected = {
        "AgentState",
        "SkillSpec",
        "Store",
        "UserProfile",
        "__version__",
        "build_graph",
        "discover_skills",
        "load_resume_pdf",
        "load_skill",
    }
    assert set(offerguide.__all__) == expected
    for name in expected:
        assert hasattr(offerguide, name), f"offerguide.{name} missing"


# ---- raw sqlite sanity ----------------------------------------------------


def test_can_insert_and_query_jobs(tmp_path: Path) -> None:
    """Smoke test: the schema is shaped correctly for the W2 Scout to write into."""
    store = offerguide.Store(tmp_path / "store.db")
    store.init_schema()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, source_id, url, title, company, location, raw_text, content_hash) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("nowcoder", "abc123", "https://x", "Backend SE Intern", "ByteDance", "Shanghai", "JD text", "hash1"),
        )
    with store.connect() as conn:
        row = conn.execute("SELECT title, company FROM jobs").fetchone()
    assert row == ("Backend SE Intern", "ByteDance")


def test_jobs_dedup_by_content_hash(tmp_path: Path) -> None:
    store = offerguide.Store(tmp_path / "store.db")
    store.init_schema()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, raw_text, content_hash) VALUES (?,?,?)",
            ("nowcoder", "JD text", "hashA"),
        )
    with store.connect() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO jobs(source, raw_text, content_hash) VALUES (?,?,?)",
            ("nowcoder", "JD text", "hashA"),
        )
