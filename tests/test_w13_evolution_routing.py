"""W13.1 evolution.registry + SkillRuntime variant routing.

These tests prove the W13.1 invariant: evolved variants in skill_variants
override the on-disk SKILL.md body, but the seed is preserved as fallback.
"""

from __future__ import annotations

import random

import pytest

import offerguide
from offerguide.evolution.registry import (
    bump_version,
    fail_variant,
    get_canary_variants,
    get_live_variant,
    get_shadow_variants,
    get_variant_by_version,
    insert_shadow_variant,
    list_all_variants,
    promote_to_canary,
    promote_to_live,
    select_variant_for_invoke,
    update_fitness_score,
)
from offerguide.llm import LLMResponse
from offerguide.skills import SkillRuntime, SkillSpec


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "evo_route.db")
    s.init_schema()
    return s


@pytest.fixture
def seed_spec():
    return SkillSpec(
        name="test_skill", description="d", version="0.1.0",
        body="SEED PROMPT BODY", inputs=("text",),
    )


# ═══════════════════════════════════════════════════════════════════
# bump_version
# ═══════════════════════════════════════════════════════════════════


class TestBumpVersion:
    def test_basic_bump(self):
        assert bump_version("0.1.0") == "0.1.1"
    def test_with_suffix(self):
        assert bump_version("0.1.0", "shadow") == "0.1.1-shadow"
    def test_strips_parent_suffix(self):
        assert bump_version("0.1.5-shadow") == "0.1.6"
    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            bump_version("not.a.version")
    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            bump_version("0.1.a")


# ═══════════════════════════════════════════════════════════════════
# CRUD on skill_variants
# ═══════════════════════════════════════════════════════════════════


class TestRegistryCRUD:
    def test_insert_and_read_shadow(self, store):
        sid = insert_shadow_variant(
            store, skill_name="x", version="0.2.0-shadow",
            parent_version="0.1.0", body_md="EVOLVED BODY",
            notes="trial 1",
        )
        assert sid is not None and sid > 0
        v = get_variant_by_version(store, "x", "0.2.0-shadow")
        assert v is not None
        assert v.status == "shadow"
        assert v.body_md == "EVOLVED BODY"
        assert v.parent_version == "0.1.0"
        assert v.canary_traffic_pct == 0.0

    def test_get_shadow_variants(self, store):
        for ver in ["0.2.0-a", "0.2.0-b", "0.2.0-c"]:
            insert_shadow_variant(store, skill_name="x", version=ver,
                                   parent_version="0.1.0", body_md=f"body {ver}")
        shadows = get_shadow_variants(store, "x")
        assert len(shadows) == 3
        assert all(s.status == "shadow" for s in shadows)

    def test_promote_to_canary_changes_status(self, store):
        insert_shadow_variant(store, skill_name="x", version="0.2.0",
                               parent_version="0.1.0", body_md="b")
        ok = promote_to_canary(store, skill_name="x", version="0.2.0",
                                traffic_pct=0.3)
        assert ok is True
        v = get_variant_by_version(store, "x", "0.2.0")
        assert v.status == "canary"
        assert v.canary_traffic_pct == pytest.approx(0.3)
        assert v.promoted_at is not None

    def test_promote_to_canary_rejects_invalid_pct(self, store):
        insert_shadow_variant(store, skill_name="x", version="0.2.0",
                               parent_version="0.1.0", body_md="b")
        with pytest.raises(ValueError):
            promote_to_canary(store, skill_name="x", version="0.2.0",
                               traffic_pct=1.5)
        with pytest.raises(ValueError):
            promote_to_canary(store, skill_name="x", version="0.2.0",
                               traffic_pct=0.0)

    def test_promote_to_live_demotes_existing(self, store):
        # Set up: there's a 0.1.0 marked live (the seed promoted earlier)
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now') - 5)",
                ("x", "0.1.0", "old body", "live"),
            )
        # And a 0.2.0 canary
        insert_shadow_variant(store, skill_name="x", version="0.2.0",
                               parent_version="0.1.0", body_md="new body")
        promote_to_canary(store, skill_name="x", version="0.2.0", traffic_pct=0.3)

        # Promote to live
        ok = promote_to_live(store, skill_name="x", version="0.2.0")
        assert ok is True

        # 0.1.0 is now retired, 0.2.0 is live
        v_old = get_variant_by_version(store, "x", "0.1.0")
        v_new = get_variant_by_version(store, "x", "0.2.0")
        assert v_old.status == "retired"
        assert v_new.status == "live"
        assert v_new.canary_traffic_pct == 0.0

    def test_fail_variant_marks_failed(self, store):
        insert_shadow_variant(store, skill_name="x", version="0.2.0",
                               parent_version="0.1.0", body_md="bad")
        ok = fail_variant(store, skill_name="x", version="0.2.0",
                           reason="critic = 0.2")
        assert ok is True
        v = get_variant_by_version(store, "x", "0.2.0")
        assert v.status == "failed"
        assert "critic = 0.2" in (v.notes or "")

    def test_update_fitness_score(self, store):
        insert_shadow_variant(store, skill_name="x", version="0.2.0",
                               parent_version="0.1.0", body_md="b")
        update_fitness_score(store, skill_name="x", version="0.2.0", fitness=0.78)
        v = get_variant_by_version(store, "x", "0.2.0")
        assert v.fitness_score == pytest.approx(0.78)


# ═══════════════════════════════════════════════════════════════════
# select_variant_for_invoke (the hot path)
# ═══════════════════════════════════════════════════════════════════


class TestSelectVariantForInvoke:
    def test_no_variants_falls_back_to_seed(self, store):
        sel = select_variant_for_invoke(store, skill_name="empty")
        assert sel.use_disk_seed is True
        assert sel.selected_variant is None

    def test_live_variant_overrides_seed(self, store):
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now'))",
                ("x", "0.2.0", "evolved body", "live"),
            )
        sel = select_variant_for_invoke(store, skill_name="x")
        assert sel.use_disk_seed is False
        assert sel.selected_variant.version == "0.2.0"
        assert "live" in sel.selection_reason

    def test_canary_traffic_split_uses_canary(self, store):
        # Live + canary at 100% traffic split
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now') - 2)",
                ("x", "0.1.0", "live body", "live"),
            )
        insert_shadow_variant(store, skill_name="x", version="0.2.0",
                               parent_version="0.1.0", body_md="canary body")
        promote_to_canary(store, skill_name="x", version="0.2.0", traffic_pct=1.0)

        # With 100% canary traffic, every roll should land on canary
        rng = random.Random(42)
        for _ in range(5):
            sel = select_variant_for_invoke(store, skill_name="x", rng=rng)
            assert sel.selected_variant.version == "0.2.0"
            assert "canary" in sel.selection_reason

    def test_canary_low_pct_mostly_routes_to_live(self, store):
        """A 1% canary should route to live ~99% of the time."""
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now') - 2)",
                ("x", "0.1.0", "live", "live"),
            )
        insert_shadow_variant(store, skill_name="x", version="0.2.0",
                               parent_version="0.1.0", body_md="canary")
        promote_to_canary(store, skill_name="x", version="0.2.0", traffic_pct=0.01)
        rng = random.Random(42)
        live_hits = canary_hits = 0
        for _ in range(200):
            sel = select_variant_for_invoke(store, skill_name="x", rng=rng)
            if sel.selected_variant.version == "0.1.0":
                live_hits += 1
            else:
                canary_hits += 1
        # 1% canary out of 200 rolls — expect live to dominate (>= 195 hits)
        assert live_hits >= 195
        assert canary_hits <= 5

    def test_shadow_never_selected_for_traffic(self, store):
        """Shadow variants must never get production traffic."""
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now'))",
                ("x", "0.1.0", "live", "live"),
            )
        # Insert several shadows (status=shadow) — none should ever be selected
        for ver in ["0.2.0-a", "0.2.0-b"]:
            insert_shadow_variant(store, skill_name="x", version=ver,
                                   parent_version="0.1.0", body_md="shadow")
        # 100 rolls: should always pick the live variant
        for _ in range(100):
            sel = select_variant_for_invoke(store, skill_name="x")
            assert sel.selected_variant.version == "0.1.0"


# ═══════════════════════════════════════════════════════════════════
# SkillRuntime variant routing integration
# ═══════════════════════════════════════════════════════════════════


class _CapturingLLM:
    """Records every system-prompt sent to chat() so we can assert which body the runtime used."""
    def __init__(self):
        self.calls: list[str] = []
    def chat(self, messages, **kw):
        sysmsg = next((m["content"] for m in messages if m.get("role") == "system"), "")
        self.calls.append(sysmsg)
        return LLMResponse(content="{}", model="stub")


class TestSkillRuntimeVariantRouting:
    def test_uses_seed_when_no_variants(self, store, seed_spec):
        llm = _CapturingLLM()
        rt = SkillRuntime(llm=llm, store=store)
        rt.invoke(seed_spec, {"text": "hi"})
        assert any("SEED PROMPT BODY" in s for s in llm.calls)

    def test_live_variant_overrides_seed_body(self, store, seed_spec):
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now'))",
                (seed_spec.name, "0.2.0", "EVOLVED BODY V2", "live"),
            )
        llm = _CapturingLLM()
        rt = SkillRuntime(llm=llm, store=store)
        result = rt.invoke(seed_spec, {"text": "hi"})

        # The system prompt must be the EVOLVED body, not SEED
        assert any("EVOLVED BODY V2" in s for s in llm.calls)
        assert not any("SEED PROMPT BODY" in s for s in llm.calls)
        # SkillResult reports the version that ran (the live variant)
        assert result.skill_version == "0.2.0"

    def test_skill_runs_records_effective_version_not_seed(self, store, seed_spec):
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now'))",
                (seed_spec.name, "0.3.0", "live body 3", "live"),
            )
        llm = _CapturingLLM()
        rt = SkillRuntime(llm=llm, store=store)
        rt.invoke(seed_spec, {"text": "hello"})
        with store.connect() as conn:
            row = conn.execute(
                "SELECT skill_name, skill_version FROM skill_runs WHERE skill_name = ?",
                (seed_spec.name,),
            ).fetchone()
        assert row[1] == "0.3.0"  # the variant version, not seed 0.1.0

    def test_consult_variant_registry_false_uses_seed(self, store, seed_spec):
        """Bypass flag for meta_evolve_skill testing candidates."""
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now'))",
                (seed_spec.name, "0.2.0", "should-not-be-used", "live"),
            )
        llm = _CapturingLLM()
        rt = SkillRuntime(llm=llm, store=store)
        rt.invoke(seed_spec, {"text": "hi"}, consult_variant_registry=False)
        # We bypassed the registry, so seed body wins
        assert any("SEED PROMPT BODY" in s for s in llm.calls)
        assert not any("should-not-be-used" in s for s in llm.calls)
