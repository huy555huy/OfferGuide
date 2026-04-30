"""Backward-compat shim — delegates to ``adapters/score_match.py``.

The score_match-specific examples live in
``offerguide.evolution.adapters.score_match`` since W8' (when other
SKILLs joined the evolvable set). This module remains as a re-export so
existing imports keep working::

    from offerguide.evolution import GoldenExample, GOLDEN_EXAMPLES

still resolves.
"""

from __future__ import annotations

from .adapters.score_match import EXAMPLES as GOLDEN_EXAMPLES
from .adapters.score_match import ScoreMatchExample as GoldenExample
from .adapters.score_match import split_train_val

__all__ = ["GOLDEN_EXAMPLES", "GoldenExample", "split_train_val"]
