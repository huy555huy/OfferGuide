"""W21 — sub-agent prompt structure tests.

Key invariant: sub-agent system prompts are MINIMAL + GENERIC. They
don't contain developer-authored opinions about the user, the market,
or job-hunting strategy. Task framing comes from the main agent at
delegation time (via the goal/instructions passed into run()).
"""
from __future__ import annotations

from offerguide.agents.discovery import DISCOVERY_SYSTEM_PROMPT
from offerguide.agents.evaluation import EVALUATION_SYSTEM_PROMPT


class TestPromptsAreGenericNotOpinionated:
    """Regression guard: my last commit had 100 lines of "你这个用户
    上财应统专硕..." baked into discovery.py. Wrong. User authors their
    own situation; sub-agent prompt is generic glue."""

    def test_discovery_prompt_is_short_generic(self):
        # Should be short — no opinionated walls of text
        assert len(DISCOVERY_SYSTEM_PROMPT) < 600

    def test_discovery_prompt_has_no_user_specifics(self):
        # These were in the bad version. Should NOT be present.
        forbidden = [
            "上海财经大学", "应用统计", "2027 届",
            "985", "AI Agent 暑期实习",
            "不脱颖", "HR 默认 deprioritize",
            "智谱", "月之暗面", "MiniMax",
            "用户自己上 BOSS",
        ]
        for term in forbidden:
            assert term not in DISCOVERY_SYSTEM_PROMPT, (
                f"DISCOVERY_SYSTEM_PROMPT must not contain '{term}' — "
                "that's developer-imposed opinion. Move to runtime "
                "user message authored by main agent."
            )

    def test_discovery_prompt_explains_sub_agent_role(self):
        assert "sub-agent" in DISCOVERY_SYSTEM_PROMPT
        assert "done" in DISCOVERY_SYSTEM_PROMPT
        # Says "user message has the actual task"
        assert "user message" in DISCOVERY_SYSTEM_PROMPT

    def test_evaluation_prompt_is_short_generic(self):
        assert len(EVALUATION_SYSTEM_PROMPT) < 600

    def test_evaluation_prompt_has_no_judgment_philosophy(self):
        forbidden = [
            "probability 不是数字, 是一个故事",
            "用户的现实",
            "你的评估哲学",
            "盲目链 SKILL",  # was in the bad version
            "上财", "应统",
        ]
        for term in forbidden:
            assert term not in EVALUATION_SYSTEM_PROMPT
