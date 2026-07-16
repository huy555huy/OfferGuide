"""W13.1 evolution.release — gray-release state machine tests."""

from __future__ import annotations

import pytest

import offerguide
from offerguide.evolution.registry import (
    get_canary_variants,
    get_live_variant,
    get_variant_by_version,
    insert_shadow_variant,
    promote_to_canary,
)
from offerguide.evolution.release import (
    DEFAULT_CANARY_TRAFFIC_PCT,
    run_release_cycle,
    run_release_cycle_for_skill,
)
from offerguide.evolution.signals import record_critic_signal


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "release.db")
    s.init_schema()
    return s


# ═══════════════════════════════════════════════════════════════════
# Single-skill cycles
# ═══════════════════════════════════════════════════════════════════


class TestPerSkillCycle:
    def test_no_variants_is_noop(self, store):
        action = run_release_cycle_for_skill(store, "x")
        assert action.action == "noop"

    def test_first_shadow_promotes_directly_to_live(self, store):
        """When no live exists, the first shadow goes straight to live."""
        insert_shadow_variant(
            store, skill_name="x", version="0.1.1-shadow", parent_version="0.1.0", body_md="b"
        )
        action = run_release_cycle_for_skill(store, "x")
        assert action.action == "promote_to_live"
        live = get_live_variant(store, "x")
        assert live is not None
        assert live.version == "0.1.1-shadow"

    def test_shadow_with_existing_live_goes_to_canary(self, store):
        # Set up: live exists
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now'))",
                ("x", "0.1.0", "live body", "live"),
            )
        # Add a shadow
        insert_shadow_variant(
            store,
            skill_name="x",
            version="0.2.0-shadow",
            parent_version="0.1.0",
            body_md="new body",
        )
        action = run_release_cycle_for_skill(store, "x")
        assert action.action == "promote_to_canary"
        canaries = get_canary_variants(store, "x")
        assert len(canaries) == 1
        assert canaries[0].canary_traffic_pct == pytest.approx(DEFAULT_CANARY_TRAFFIC_PCT)

    def test_canary_with_too_few_signals_is_noop(self, store):
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now'))",
                ("x", "0.1.0", "live", "live"),
            )
        insert_shadow_variant(
            store, skill_name="x", version="0.2.0", parent_version="0.1.0", body_md="canary"
        )
        promote_to_canary(store, skill_name="x", version="0.2.0", traffic_pct=0.2)
        # Too few signals on canary
        for _ in range(2):
            record_critic_signal(
                store, skill_name="x", skill_version="0.2.0", skill_run_id=None, score=0.7
            )
        action = run_release_cycle_for_skill(store, "x", min_canary_signals=8)
        assert action.action == "noop"
        # Canary still canary
        v = get_variant_by_version(store, "x", "0.2.0")
        assert v is not None
        assert v.status == "canary"

    def test_canary_clearly_better_promotes_to_live(self, store):
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now'))",
                ("x", "0.1.0", "live", "live"),
            )
        insert_shadow_variant(
            store, skill_name="x", version="0.2.0", parent_version="0.1.0", body_md="canary"
        )
        promote_to_canary(store, skill_name="x", version="0.2.0", traffic_pct=0.2)
        # Live signals at 0.4
        for _ in range(8):
            record_critic_signal(
                store, skill_name="x", skill_version="0.1.0", skill_run_id=None, score=0.4
            )
        # Canary signals at 0.85
        for _ in range(8):
            record_critic_signal(
                store, skill_name="x", skill_version="0.2.0", skill_run_id=None, score=0.85
            )
        action = run_release_cycle_for_skill(store, "x", min_canary_signals=8)
        assert action.action == "promote_to_live"
        # Canary is now live, old live is retired
        new_live = get_live_variant(store, "x")
        assert new_live is not None
        assert new_live.version == "0.2.0"
        old = get_variant_by_version(store, "x", "0.1.0")
        assert old is not None
        assert old.status == "retired"

    def test_canary_clearly_worse_is_failed(self, store):
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now'))",
                ("x", "0.1.0", "live", "live"),
            )
        insert_shadow_variant(
            store, skill_name="x", version="0.2.0", parent_version="0.1.0", body_md="bad canary"
        )
        promote_to_canary(store, skill_name="x", version="0.2.0", traffic_pct=0.2)
        for _ in range(8):
            record_critic_signal(
                store, skill_name="x", skill_version="0.1.0", skill_run_id=None, score=0.85
            )
        for _ in range(8):
            record_critic_signal(
                store, skill_name="x", skill_version="0.2.0", skill_run_id=None, score=0.3
            )
        action = run_release_cycle_for_skill(store, "x", min_canary_signals=8)
        assert action.action == "fail_canary"
        c = get_variant_by_version(store, "x", "0.2.0")
        assert c is not None
        assert c.status == "failed"
        # Live unchanged
        live = get_live_variant(store, "x")
        assert live is not None
        assert live.version == "0.1.0"

    def test_dry_run_doesnt_mutate(self, store):
        insert_shadow_variant(
            store, skill_name="x", version="0.1.1-shadow", parent_version="0.1.0", body_md="b"
        )
        action = run_release_cycle_for_skill(store, "x", dry_run=True)
        assert action.action == "promote_to_live"
        assert "would" in action.reason
        # Not actually promoted
        v = get_variant_by_version(store, "x", "0.1.1-shadow")
        assert v is not None
        assert v.status == "shadow"


# ═══════════════════════════════════════════════════════════════════
# All-skill cycle
# ═══════════════════════════════════════════════════════════════════


class TestRunReleaseCycle:
    def test_empty_db(self, store):
        result = run_release_cycle(store)
        assert result.actions == []
        assert result.skipped == []

    def test_processes_each_skill_once(self, store):
        for skill in ["a", "b", "c"]:
            insert_shadow_variant(
                store,
                skill_name=skill,
                version="0.1.1-shadow",
                parent_version="0.1.0",
                body_md=f"body {skill}",
            )
        result = run_release_cycle(store)
        assert len(result.actions) == 3
        action_skills = {a.skill_name for a in result.actions}
        assert action_skills == {"a", "b", "c"}
        # All promoted (no live existed → direct-to-live)
        assert all(a.action == "promote_to_live" for a in result.actions)

    def test_render_summary_human_readable(self, store):
        insert_shadow_variant(
            store,
            skill_name="x",
            version="0.1.1-shadow",
            parent_version="0.1.0",
            body_md="b",
        )
        result = run_release_cycle(store)
        text = result.render_summary()
        assert "x" in text
        assert "promote" in text


# ═══════════════════════════════════════════════════════════════════
# /evolution UI route
# ═══════════════════════════════════════════════════════════════════


class TestEvolutionUIRoute:
    def test_evolution_page_renders_empty(self, tmp_path):
        from pathlib import Path

        from fastapi.testclient import TestClient

        from offerguide.config import Settings
        from offerguide.skills import discover_skills
        from offerguide.ui.notify import ConsoleNotifier
        from offerguide.ui.web import create_app

        store = offerguide.Store(tmp_path / "evo_ui.db")
        store.init_schema()
        skills = discover_skills(Path(__file__).parent.parent / "src/offerguide/skills")
        s = Settings(deepseek_api_key="", deepseek_base_url="x", default_model="m")
        app = create_app(
            settings=s,
            store=store,
            master_source=None,
            skills=skills,
            runtime=None,
            notifier=ConsoleNotifier(),
        )
        client = TestClient(app)
        resp = client.get("/evolution")
        assert resp.status_code == 200
        assert "SKILL 进化" in resp.text or "进化" in resp.text
        # Only current SKILLs should appear as cards.
        assert "apply_assistant" in resp.text

    def test_evolution_release_cycle_endpoint(self, tmp_path):
        from pathlib import Path

        from fastapi.testclient import TestClient

        from offerguide.config import Settings
        from offerguide.evolution.registry import insert_shadow_variant
        from offerguide.skills import discover_skills
        from offerguide.ui.notify import ConsoleNotifier
        from offerguide.ui.web import create_app

        store = offerguide.Store(tmp_path / "evo_ui2.db")
        store.init_schema()
        # Add a shadow that should promote to live
        insert_shadow_variant(
            store,
            skill_name="apply_assistant",
            version="0.1.1-shadow",
            parent_version="0.1.0",
            body_md="evolved",
        )

        skills = discover_skills(Path(__file__).parent.parent / "src/offerguide/skills")
        s = Settings(deepseek_api_key="", deepseek_base_url="x", default_model="m")
        app = create_app(
            settings=s,
            store=store,
            master_source=None,
            skills=skills,
            runtime=None,
            notifier=ConsoleNotifier(),
        )
        client = TestClient(app)

        # dry_run first
        resp = client.post("/api/evolution/release_cycle?dry_run=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["actions"]
        assert "would" in data["actions"][0]["reason"]

        # real cycle
        resp = client.post("/api/evolution/release_cycle")
        assert resp.status_code == 200
        # Variant should be live now
        with store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM skill_variants WHERE skill_name=? AND version=?",
                ("apply_assistant", "0.1.1-shadow"),
            ).fetchone()
        assert row[0] == "live"
