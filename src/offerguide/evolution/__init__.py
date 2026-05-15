"""W13.1 evolution — closed-loop SKILL prompt evolution.

This package replaces the W6 DSPy/GEPA framework. The old design had a
sound metric primitive but no real signal pipeline — the trainset was
hand-stitched from skill_runs without quality labels, so evolution
was theoretical not driven by lived experience.

Closed-loop architecture (post-W21 — harness drives, no LLM self-critique):

    skill_run executed
        │
        ├─ user 👍/👎 in /agent inbox                ┐
        ├─ application reaches outcome (offer/rej)   ├─→ evolution_signals
        ├─ user did/didn't follow through            │   (3 real-feedback channels;
        │                                            │   no LLM-self-critique — that
        │                                            │   pattern was retired with W13
        │                                            │   AgentLoop. A 'critic' slot
        │                                            │   exists for future external
        │                                            │   annotators, but isn't wired.)
        ↓
    fitness.compute_fitness  (aggregate signals into one score per skill_version)
        ↓
    The agent runtime calls detect_evolution_candidates as a tool when it
    suspects a SKILL is underperforming. If results say "ripe":
        ↓
    agent calls evolve_skill(name)  →  evolution.evolve.evolve_skill()
        ↓
    generates N variants, persisted as 'shadow' rows in skill_variants
        ↓
    agent (or cron, or user UI) calls run_release_cycle:
      - shadow → canary  (small traffic split via SkillRuntime variant routing)
      - canary signals ripen → fitness.compare_versions
      - canary → live (winner) or fail_variant (loser)
        ↓
    SkillRuntime.invoke consults skill_variants per call → routes to live
    (or canary slice). New SKILL invocations write fresh signals. Loop.

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
    # fitness
    "EVOLUTION_COOLDOWN_DAYS",
    "EVOLUTION_THRESHOLD",
    "MIN_SIGNALS_FOR_TRIGGER",
    "EvolutionTrigger",
    "FitnessReport",
    "SignalKind",
    "SignalRecord",
    # registry
    "Variant",
    "VariantSelection",
    "VariantStatus",
    "bump_version",
    "compare_versions",
    "compute_fitness",
    "detect_evolution_candidates",
    "fail_variant",
    "fetch_signals",
    "get_canary_variants",
    "get_live_variant",
    "get_shadow_variants",
    "get_variant_by_version",
    "insert_shadow_variant",
    "list_all_variants",
    "normalize_signal_value",
    "promote_to_canary",
    "promote_to_live",
    "record_app_outcome",
    "record_critic_signal",
    "record_follow_through",
    "record_synthetic_eval",
    "record_user_thumbs",
    "select_variant_for_invoke",
    "update_fitness_score",
]
