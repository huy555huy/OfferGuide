"""Shared types for SKILL-specific evolution adapters.

The adapter pattern lets each evolvable SKILL plug its own
``(GoldenExample type, metric function, axis names)`` into the generic
GEPA runner without making the runner know about that SKILL's domain.

To add a new evolvable SKILL:
1. Drop a module ``adapters/<skill_name>.py`` next to score_match.py
2. The module must export exactly what the SkillAdapter Protocol below
   requires (EXAMPLES, INPUT_NAMES, METRIC_AXES, metric, aggregate,
   split_train_val)
3. Register it in ``adapters/__init__.py``'s registry dict

The runner calls only methods on the adapter — it never imports concrete
example or metric types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class MetricBreakdown:
    """Per-axis breakdown of one metric evaluation.

    Generic across all SKILLs: ``breakdown`` carries arbitrary axis names
    (e.g. score_match uses ``prob/recall/anti``; analyze_gaps uses
    ``schema/keyword_recall/ai_risk``).  ``total`` is the weighted sum the
    adapter computed; GEPA ranks candidates on this scalar.
    """

    total: float
    breakdown: dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    """Human-readable explanation for the GEPA reflection LM. Write it the
    way you'd write to a teammate: 'too high; reasoning didn't mention X'.
    The reflection LM consumes this verbatim when proposing prompt
    mutations."""

    # ─ Backward-compat shims so existing score_match tests stay green ─

    @property
    def prob_score(self) -> float:
        return self.breakdown.get("prob", 0.0)

    @property
    def recall_score(self) -> float:
        return self.breakdown.get("recall", 0.0)

    @property
    def anti_score(self) -> float:
        return self.breakdown.get("anti", 0.0)


def aggregate(metrics: list[MetricBreakdown]) -> dict[str, float]:
    """Average ``total`` and each per-axis score across a list of breakdowns.

    Generic — the returned dict has an ``n`` field, a ``total`` field, and
    one entry per axis name that appeared in *any* breakdown. Adapters
    using fixed axis names get stable output keys.
    """
    if not metrics:
        return {"total": 0.0, "n": 0}
    n = len(metrics)
    out: dict[str, float] = {
        "total": sum(m.total for m in metrics) / n,
        "n": n,
    }
    axis_names: set[str] = set()
    for m in metrics:
        axis_names |= set(m.breakdown.keys())
    for ax in axis_names:
        out[ax] = sum(m.breakdown.get(ax, 0.0) for m in metrics) / n
    return out


def parse_json_output(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Coerce a SKILL output into a dict, accepting pre-parsed JSON or string.

    Returns ``{}`` for unparseable input — the caller's metric handles that
    by giving a low score, which is the right outcome (a parse failure on
    the output IS a failure of the prompt).
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


@runtime_checkable
class SkillAdapter(Protocol):
    """Structural type that ``adapters/__init__.py::get_adapter`` returns.

    Every evolvable SKILL ships a module satisfying this protocol.  The
    Protocol is structural (no inheritance required) so the adapter can be
    a plain module that exposes module-level names.
    """

    name: str
    """Skill identifier, matches the directory under ``src/offerguide/skills/``."""

    INPUT_NAMES: list[str]
    """The SKILL's ordered input field names. Used by ``build_trainset``
    to pull values off each GoldenExample-like instance."""

    METRIC_AXES: list[str]
    """Axis names the adapter's metric function fills into
    ``MetricBreakdown.breakdown``. Used by the CLI to print a fixed-order
    metric table."""

    def metric(
        self,
        example: Any,
        raw_output: str | dict[str, Any],
    ) -> MetricBreakdown:
        """Score one (example, model output) pair."""

    def split_train_val(
        self, *, val_fraction: float = 0.4, seed: int = 0
    ) -> tuple[list[Any], list[Any]]:
        """Deterministic train/val split — adapters can stratify by class."""
