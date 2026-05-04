"""W13.7 held-out eval set — protect against prompt regressions.

The W13.1 evolution_signals tells us how a SKILL is doing on lived
experience (critic + thumbs + outcomes). But signals are slow to
accumulate AND drift over time. Without an held-out eval set we can't:

- Catch when a GEPA-evolved variant looks better in synthetic eval
  but is actually worse on objective ground-truth
- Run nightly regression to detect prompt drift
- Compare two model providers (claude vs deepseek) on identical inputs

This module is a lightweight take on the DeepEval pattern: each SKILL
has a small (3-10 examples) hand-curated test set with `expected_keys`
or `expected_phrases` the output should contain, plus optional Pydantic
validation. Run via ``python -m offerguide.eval``.
"""

from .runner import EvalCase, EvalReport, EvalResult, run_eval, run_eval_for_skill

__all__ = ["EvalCase", "EvalReport", "EvalResult", "run_eval", "run_eval_for_skill"]
