"""Backward-compat shim — delegates to ``adapters/_base.py`` and ``adapters/score_match.py``.

The generic ``MetricBreakdown`` and ``aggregate`` moved to
``adapters/_base.py``; ``score_match_metric`` moved to
``adapters/score_match.py``. This module keeps existing imports working::

    from offerguide.evolution.metrics import score_match_metric, aggregate
"""

from __future__ import annotations

from .adapters._base import MetricBreakdown, aggregate, parse_json_output
from .adapters.score_match import metric as score_match_metric

# Older alias used in tests
parse_score_match_output = parse_json_output

__all__ = [
    "MetricBreakdown",
    "aggregate",
    "parse_json_output",
    "parse_score_match_output",
    "score_match_metric",
]
