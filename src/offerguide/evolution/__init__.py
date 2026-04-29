"""GEPA-driven SKILL evolution.

Public surface (kept small — most callers want either the metric or the runner):

- ``GoldenExample``, ``GOLDEN_EXAMPLES``  — handcrafted score_match cases
- ``score_match_metric``, ``aggregate``    — metric primitives
- ``evolve_skill``, ``EvolutionResult``    — end-to-end runner
- ``bump_version``, ``write_evolved_skill``, ``log_evolution`` — partial pipeline pieces
"""

from .golden_trainset import GOLDEN_EXAMPLES, GoldenExample, split_train_val
from .metrics import MetricBreakdown, aggregate, score_match_metric
from .runner import (
    EvolutionResult,
    build_trainset,
    bump_version,
    evolve_skill,
    log_evolution,
    make_score_match_metric,
    run_gepa,
    write_evolved_skill,
)

__all__ = [
    "GOLDEN_EXAMPLES",
    "EvolutionResult",
    "GoldenExample",
    "MetricBreakdown",
    "aggregate",
    "build_trainset",
    "bump_version",
    "evolve_skill",
    "log_evolution",
    "make_score_match_metric",
    "run_gepa",
    "score_match_metric",
    "split_train_val",
    "write_evolved_skill",
]
