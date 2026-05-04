"""W13.1 evolution — closed-loop SKILL prompt evolution.

This package replaces the W6 DSPy/GEPA framework. The old design had a
sound metric primitive but no real signal pipeline — the trainset was
hand-stitched from skill_runs without quality labels, so evolution
was theoretical not driven by lived experience.

W13.1 architecture:

    skill_run executed
        │
        ├─ AgentLoop critic LLM judges trajectory  ──┐
        ├─ user 👍/👎 in /agent inbox                ├─→ evolution_signals
        ├─ application reaches outcome (offer/rej)   │
        └─ user did/didn't follow through            ┘
                                                      ↓
                                              fitness.compute_fitness
                                                      ↓
                                              fitness.detect_evolution_candidates
                                                      ↓
                                              meta_evolve_skill (a SKILL the agent calls)
                                                      ↓
                                              generates N variants
                                                      ↓
                                       runs each on recent real inputs
                                                      ↓
                                              registry.insert_shadow_variant
                                                      ↓
                                       gray-release: promote_to_canary
                                                      ↓
                                  SkillRuntime routes traffic per canary_traffic_pct
                                                      ↓
                                       fitness.compare_versions decides
                                                      ↓
                                       promote_to_live  /  fail_variant

Public API:

- ``signals.record_*`` — write feedback signals from any source
- ``fitness.compute_fitness`` — read signals, return aggregate score
- ``fitness.detect_evolution_candidates`` — which SKILLs need evolution now
- ``fitness.compare_versions`` — A/B comparison for promote/rollback
- ``registry.select_variant_for_invoke`` — used by SkillRuntime hot path
- ``registry.insert_shadow_variant`` / ``promote_to_canary`` /
  ``promote_to_live`` / ``fail_variant`` — variant lifecycle

Promote/rollback orchestration lives in ``release.py`` (W13.1 Step 8).
The agent-callable SKILL that ties it all together is
``skills/meta_evolve_skill/`` (W13.1 Step 7).
"""

from .fitness import (
    EVOLUTION_COOLDOWN_DAYS,
    EVOLUTION_THRESHOLD,
    MIN_SIGNALS_FOR_TRIGGER,
    EvolutionTrigger,
    FitnessReport,
    compare_versions,
    compute_fitness,
    detect_evolution_candidates,
    normalize_signal_value,
)
from .registry import (
    Variant,
    VariantSelection,
    VariantStatus,
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
from .signals import (
    DEFAULT_WEIGHTS,
    SignalKind,
    SignalRecord,
    fetch_signals,
    record_app_outcome,
    record_critic_signal,
    record_follow_through,
    record_synthetic_eval,
    record_user_thumbs,
)

__all__ = [
    # signals
    "DEFAULT_WEIGHTS",
    "SignalKind",
    "SignalRecord",
    "fetch_signals",
    "record_app_outcome",
    "record_critic_signal",
    "record_follow_through",
    "record_synthetic_eval",
    "record_user_thumbs",
    # fitness
    "EVOLUTION_COOLDOWN_DAYS",
    "EVOLUTION_THRESHOLD",
    "MIN_SIGNALS_FOR_TRIGGER",
    "EvolutionTrigger",
    "FitnessReport",
    "compare_versions",
    "compute_fitness",
    "detect_evolution_candidates",
    "normalize_signal_value",
    # registry
    "Variant",
    "VariantSelection",
    "VariantStatus",
    "bump_version",
    "fail_variant",
    "get_canary_variants",
    "get_live_variant",
    "get_shadow_variants",
    "get_variant_by_version",
    "insert_shadow_variant",
    "list_all_variants",
    "promote_to_canary",
    "promote_to_live",
    "select_variant_for_invoke",
    "update_fitness_score",
]
