"""GEPA-driven SKILL evolution.

Public surface (kept small — most callers want either the metric or the runner):

- ``GoldenExample``, ``GOLDEN_EXAMPLES``  — handcrafted score_match cases
- ``score_match_metric``, ``aggregate``    — metric primitives
- ``evolve_skill``, ``EvolutionResult``    — end-to-end runner
- ``bump_version``, ``write_evolved_skill``, ``log_evolution`` — partial pipeline pieces
"""

from .adapters import (
    MetricBreakdown,
    aggregate,
    get_adapter,
    list_evolvable_skills,
)
from .diff import (
    DiffReport,
    EvolutionRecord,
    build_diff_report,
    latest_evolution,
    list_evolutions,
    render_markdown,
)
from .golden_trainset import GOLDEN_EXAMPLES, GoldenExample, split_train_val
from .metrics import score_match_metric
from .runner import (
    EvolutionResult,
    build_trainset,
    bump_version,
    evolve_skill,
    log_evolution,
    make_metric_for_adapter,
    make_score_match_metric,
    run_gepa,
    write_evolved_skill,
)

__all__ = [
    "GOLDEN_EXAMPLES",
    "DiffReport",
    "EvolutionRecord",
    "EvolutionResult",
    "GoldenExample",
    "MetricBreakdown",
    "aggregate",
    "build_diff_report",
    "build_trainset",
    "bump_version",
    "evolve_skill",
    "get_adapter",
    "latest_evolution",
    "list_evolutions",
    "list_evolvable_skills",
    "log_evolution",
    "make_metric_for_adapter",
    "make_score_match_metric",
    "render_markdown",
    "run_gepa",
    "score_match_metric",
    "split_train_val",
    "write_evolved_skill",
]
