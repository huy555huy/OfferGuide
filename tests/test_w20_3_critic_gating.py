"""W20.3 — critic gating tests.

User concern (verbatim): "critic 这个得想清楚了, 不能因为他而反耽误了我的
模型的使用".

Gating rules tested:
1. user_button / manual / chat / sse → SKIP critic (interactive UX)
2. cron_wake / scheduler → KEEP critic (background, time available)
3. trivial run (≤1 iter + 0 SKILL) → SKIP regardless of trigger
4. OFFERGUIDE_CRITIC_DISABLED=1 → SKIP all
5. OFFERGUIDE_CRITIC_SAMPLE_RATE=0 → SKIP all (probabilistic)
6. critic_enabled=False at constructor → SKIP all
7. timeout: critic LLM hang > N seconds → return None, no block
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

import offerguide
from offerguide.agent.loop import AgentLoop


@pytest.fixture
def loop_factory(tmp_path):
    """Build a minimal AgentLoop for gating tests — no real LLM, no real run."""
    def _factory(**overrides):
        store = offerguide.Store(tmp_path / "loop.db")
        store.init_schema()
        kwargs = {
            "llm": MagicMock(),
            "runtime": MagicMock(),
            "store": store,
            "skills": [],
            "master_resume_text": "resume",
            # W20.4 — critic OFF by default for the agent itself; gating
            # tests below assume critic is ON so we can test the gates.
            "critic_enabled": True,
        }
        kwargs.update(overrides)
        return AgentLoop(**kwargs)
    return _factory


# ── _should_run_critic gating ──────────────────────────────────────────


class TestCriticGating:
    """W20.3 — verify _should_run_critic decision matrix."""

    def test_user_button_trigger_skips_critic(self, loop_factory):
        loop = loop_factory()
        run, reason = loop._should_run_critic(
            trigger_kind="user_button",
            iterations=3,
            skill_invocations={"x": {"skill_name": "score_match"}},
        )
        assert run is False
        assert "interactive_trigger" in reason

    def test_manual_trigger_skips_critic(self, loop_factory):
        loop = loop_factory()
        run, reason = loop._should_run_critic(
            trigger_kind="manual", iterations=3, skill_invocations={"x": {}},
        )
        assert run is False
        assert "interactive" in reason

    def test_chat_trigger_skips_critic(self, loop_factory):
        loop = loop_factory()
        run, _ = loop._should_run_critic(
            trigger_kind="chat", iterations=2, skill_invocations={"x": {}},
        )
        assert run is False

    def test_sse_trigger_skips_critic(self, loop_factory):
        """SSE = user-triggered streaming; UX latency matters."""
        loop = loop_factory()
        run, _ = loop._should_run_critic(
            trigger_kind="sse", iterations=2, skill_invocations={"x": {}},
        )
        assert run is False

    def test_cron_wake_trigger_keeps_critic(self, loop_factory):
        """Cron has time, needs signal for evolve."""
        loop = loop_factory()
        run, reason = loop._should_run_critic(
            trigger_kind="cron_wake",
            iterations=3,
            skill_invocations={"x": {"skill_name": "score_match"}},
        )
        assert run is True
        assert reason is None

    def test_scheduler_trigger_keeps_critic(self, loop_factory):
        loop = loop_factory()
        run, _ = loop._should_run_critic(
            trigger_kind="scheduler", iterations=2,
            skill_invocations={"x": {}},
        )
        assert run is True

    def test_autonomous_trigger_keeps_critic(self, loop_factory):
        loop = loop_factory()
        run, _ = loop._should_run_critic(
            trigger_kind="autonomous", iterations=2,
            skill_invocations={"x": {}},
        )
        assert run is True

    def test_unknown_trigger_keeps_critic(self, loop_factory):
        """Unknown trigger_kind defaults to keeping critic (safer for evolve)."""
        loop = loop_factory()
        run, _ = loop._should_run_critic(
            trigger_kind="unknown_new_trigger", iterations=2,
            skill_invocations={"x": {}},
        )
        assert run is True

    def test_trivial_run_skips_critic_even_for_cron(self, loop_factory):
        """≤1 iteration AND 0 SKILL = nothing to critique, skip even cron."""
        loop = loop_factory()
        run, reason = loop._should_run_critic(
            trigger_kind="cron_wake", iterations=1, skill_invocations={},
        )
        assert run is False
        assert reason == "trivial_run"

    def test_trivial_check_requires_both_conditions(self, loop_factory):
        """≤1 iter BUT skill ran → not trivial."""
        loop = loop_factory()
        run, _ = loop._should_run_critic(
            trigger_kind="cron_wake", iterations=1,
            skill_invocations={"x": {"skill_name": "score_match"}},
        )
        assert run is True

        run2, _ = loop._should_run_critic(
            trigger_kind="cron_wake", iterations=5, skill_invocations={},
        )
        assert run2 is True  # iterations > 1, even with no SKILL

    def test_constructor_disable_skips_critic(self, loop_factory):
        loop = loop_factory(critic_enabled=False)
        run, reason = loop._should_run_critic(
            trigger_kind="cron_wake", iterations=3,
            skill_invocations={"x": {}},
        )
        assert run is False
        assert reason == "constructor_disabled"

    def test_env_disable_skips_critic(self, loop_factory, monkeypatch):
        monkeypatch.setenv("OFFERGUIDE_CRITIC_DISABLED", "1")
        loop = loop_factory()  # constructor enabled
        run, reason = loop._should_run_critic(
            trigger_kind="cron_wake", iterations=3,
            skill_invocations={"x": {}},
        )
        assert run is False
        assert reason == "env_disabled"

    def test_env_sample_rate_zero_always_skips(self, loop_factory, monkeypatch):
        monkeypatch.setenv("OFFERGUIDE_CRITIC_SAMPLE_RATE", "0")
        loop = loop_factory()
        run, reason = loop._should_run_critic(
            trigger_kind="cron_wake", iterations=3,
            skill_invocations={"x": {}},
        )
        assert run is False
        assert "sampled_out" in reason

    def test_env_sample_rate_one_always_runs(self, loop_factory, monkeypatch):
        monkeypatch.setenv("OFFERGUIDE_CRITIC_SAMPLE_RATE", "1.0")
        loop = loop_factory()
        run, _ = loop._should_run_critic(
            trigger_kind="cron_wake", iterations=3,
            skill_invocations={"x": {}},
        )
        assert run is True

    def test_env_sample_rate_invalid_falls_back_to_one(self, loop_factory, monkeypatch):
        monkeypatch.setenv("OFFERGUIDE_CRITIC_SAMPLE_RATE", "not-a-number")
        loop = loop_factory()
        run, _ = loop._should_run_critic(
            trigger_kind="cron_wake", iterations=3,
            skill_invocations={"x": {}},
        )
        assert run is True


# ── Timeout: critic doesn't block run() longer than N seconds ────────


class TestCriticTimeout:
    def test_critic_timeout_returns_quickly(self, loop_factory):
        """If critic LLM call hangs, _self_critique_with_timeout returns
        within self._critic_timeout_s + small overhead."""
        loop = loop_factory(critic_timeout_s=0.5)

        # Patch _self_critique to hang forever
        def _hang(*args, **kwargs):
            time.sleep(10)
            return 1.0, "should never see this"

        loop._self_critique = _hang  # type: ignore[assignment]

        t0 = time.monotonic()
        score, notes = loop._self_critique_with_timeout(
            goal="x", final_answer="y", events=[],
        )
        elapsed = time.monotonic() - t0

        assert score is None
        assert "timeout" in notes
        assert elapsed < 1.5, (
            f"timeout fence broken — elapsed {elapsed:.2f}s "
            f"with timeout=0.5s"
        )

    def test_critic_normal_call_completes(self, loop_factory):
        """Sanity: when critic doesn't hang, normal flow returns score."""
        loop = loop_factory(critic_timeout_s=5.0)

        def _ok(*args, **kwargs):
            return 0.75, "looked fine"

        loop._self_critique = _ok  # type: ignore[assignment]

        score, notes = loop._self_critique_with_timeout(
            goal="x", final_answer="y", events=[],
        )
        assert score == 0.75
        assert notes == "looked fine"

    def test_critic_default_OFF_for_user_concerns(self, tmp_path):
        """W20.4 — critic should default to OFF.

        User original objection: 'critic 这个得想清楚了, 不能因为他而反耽误
        我的模型的使用. 不能使用一个纯系统代码的 critic 来掩饰, 因为他凭
        什么能评判你?'

        Resolution: critic LLM (same model judging same model) is epistemically
        weak. Default OFF. Real evolve signal comes from user_thumbs /
        app_outcome / follow_through. Opt-in critic via env if user wants it.
        """
        store = offerguide.Store(tmp_path / "loop_default.db")
        store.init_schema()
        # Bare AgentLoop without critic_enabled override — should default OFF
        loop = AgentLoop(
            llm=MagicMock(), runtime=MagicMock(),
            store=store, skills=[],
        )
        assert loop._critic_enabled is False

    def test_env_force_on_overrides_default_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OFFERGUIDE_CRITIC_ENABLED", "1")
        store = offerguide.Store(tmp_path / "loop_envon.db")
        store.init_schema()
        loop = AgentLoop(
            llm=MagicMock(), runtime=MagicMock(),
            store=store, skills=[],
        )
        assert loop._critic_enabled is True

    def test_critic_exception_propagates(self, loop_factory):
        """If critic raises (not hangs), exception propagates so the existing
        try/except at call site can catch + log it."""
        loop = loop_factory(critic_timeout_s=2.0)

        def _crash(*args, **kwargs):
            raise RuntimeError("critic LLM 500")

        loop._self_critique = _crash  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="critic LLM 500"):
            loop._self_critique_with_timeout(
                goal="x", final_answer="y", events=[],
            )
