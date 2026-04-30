"""GEPA evolution orchestrator — compose trainset + metric + LM, run, write back.

This module is the only one that imports DSPy. Tests outside this file run
without a DSPy install. The runner is split into pieces so each can be
tested in isolation:

1. ``build_trainset(...)`` — pure: GoldenExample → list[dspy.Example]
2. ``make_metric(...)``    — pure-ish: returns a closure that GEPA calls
3. ``run_gepa(...)``       — the only path that hits real LLMs
4. ``write_evolved_skill(...)`` + ``log_evolution(...)`` — pure file IO + DB IO

W8': The runner is SKILL-agnostic — it dispatches to a per-SKILL adapter
(``evolution.adapters.<skill_name>``) for the trainset and metric.
``score_match`` is one such adapter; ``analyze_gaps`` and
``prepare_interview`` plug in the same way.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import dspy

from ..memory import Store
from ..skills import load_skill
from ..skills._spec import SkillSpec
from .adapters import get_adapter
from .adapters._base import MetricBreakdown, aggregate
from .adapters.score_match import ScoreMatchExample as GoldenExample
from .adapters.score_match import metric as score_match_metric
from .dspy_module import build_student, read_instructions


@dataclass(frozen=True)
class EvolutionResult:
    skill_name: str
    parent_version: str
    new_version: str
    new_skill_path: Path
    """Path to the SKILL.md that now holds the evolved instructions."""

    metric_before: dict[str, float]
    """Aggregated baseline metric on the val set, with the original prompt."""

    metric_after: dict[str, float]
    """Same aggregation post-evolution. ``metric_after['total'] - metric_before['total']``
    is the headline number for the README before/after diff."""

    evolution_log_id: int


# ---------- 1. Trainset construction -----------------------------------


def build_trainset(
    examples: list[GoldenExample],
    *,
    input_names: list[str],
) -> list[dspy.Example]:
    """Convert GoldenExamples into dspy.Examples with the golden attached.

    Each input field declared in the spec gets pulled from the matching
    GoldenExample attribute (``job_text``, ``user_profile``, ...). The full
    GoldenExample is also attached as ``_golden`` so the metric can recover it
    after DSPy round-trips.
    """
    out: list[dspy.Example] = []
    for ex in examples:
        kwargs = {name: getattr(ex, name) for name in input_names}
        kwargs["_golden_name"] = ex.name
        out.append(dspy.Example(**kwargs).with_inputs(*input_names))
    return out


def _golden_by_name(examples: list) -> dict[str, Any]:
    return {ex.name: ex for ex in examples}


# ---------- 2. Metric closure ------------------------------------------


def make_metric_for_adapter(
    adapter: ModuleType,
    examples: list,
) -> Callable[..., dspy.Prediction]:
    """Return a GEPA-compatible metric callable for a given SKILL adapter.

    The callable looks up the example by name (round-tripped through
    ``_golden_name`` on dspy.Example) and calls ``adapter.metric(...)``,
    returning a ``dspy.Prediction(score, feedback)`` for GEPA.
    """
    by_name = _golden_by_name(examples)

    def metric(
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any = None,
        *_args: Any,
        **_kwargs: Any,
    ) -> dspy.Prediction:
        golden = by_name.get(getattr(example, "_golden_name", ""))
        if golden is None:
            return dspy.Prediction(
                score=0.0,
                feedback="ERROR: example missing _golden_name; metric cannot lookup target.",
            )
        raw = getattr(prediction, "response_json", None) or ""
        breakdown: MetricBreakdown = adapter.metric(golden, raw)
        return dspy.Prediction(score=breakdown.total, feedback=breakdown.feedback)

    return metric


def make_score_match_metric(
    examples: list[GoldenExample],
) -> Callable[..., dspy.Prediction]:
    """Backward-compat wrapper — same as ``make_metric_for_adapter(score_match_adapter, ...)``."""
    from .adapters import score_match as score_match_adapter

    return make_metric_for_adapter(score_match_adapter, examples)


# ---------- 3. The actual GEPA call ------------------------------------


def run_gepa(
    *,
    student: dspy.Module,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    metric: Callable,
    main_lm: dspy.LM,
    reflection_lm: dspy.LM,
    auto: str = "light",
    log_dir: str | None = None,
) -> dspy.Module:
    """Run dspy.GEPA. Returns the optimized student.

    `auto="light"` is the cheapest preset (~$2 / ~20 min on small trainsets;
    bumps to "medium"/"heavy" trade more API spend for more rollouts). With our
    11-case golden set, "light" is the right default.
    """
    dspy.configure(lm=main_lm)
    optimizer = dspy.GEPA(
        metric=metric,
        auto=auto,
        reflection_lm=reflection_lm,
        skip_perfect_score=True,
        track_stats=True,
        log_dir=log_dir,
    )
    return optimizer.compile(student=student, trainset=trainset, valset=valset)


# ---------- 4. SKILL writeback + evolution_log -------------------------


_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def bump_version(version: str) -> str:
    """Bump the patch component if x.y.z, else suffix '.evolved' once.

    GEPA's evolution is a content change at semver patch level (the prompt is
    different but the contract — inputs and outputs — is unchanged). If the
    version doesn't match x.y.z we append `.evolved` so we never lose info.
    """
    m = _VERSION_RE.match(version)
    if m is None:
        return f"{version}.evolved"
    major, minor, patch = m.groups()
    return f"{major}.{minor}.{int(patch) + 1}"


def write_evolved_skill(
    skill_dir: Path,
    *,
    parent_spec: SkillSpec,
    new_instructions: str,
    new_version: str | None = None,
    backup: bool = True,
) -> Path:
    """Overwrite SKILL.md with evolved body + bumped version + parent pointer.

    A timestamped backup of the parent file is written to ``SKILL.md.<vN>.bak``
    when ``backup=True`` (default). Git history is the official record, but the
    on-disk backup is convenient when running the CLI outside a git checkout.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(skill_md)
    raw = skill_md.read_text(encoding="utf-8")

    if backup:
        backup_path = skill_dir / f"SKILL.md.v{parent_spec.version}.bak"
        shutil.copy2(skill_md, backup_path)

    new_version = new_version or bump_version(parent_spec.version)
    new_frontmatter = _patched_frontmatter(
        raw, new_version=new_version, parent_version=parent_spec.version
    )
    new_text = f"{new_frontmatter}\n{new_instructions.rstrip()}\n"
    skill_md.write_text(new_text, encoding="utf-8")
    return skill_md


def _patched_frontmatter(
    raw_md: str, *, new_version: str, parent_version: str
) -> str:
    """Return the frontmatter block (between two `---`) with version+evolution fields updated.

    Doesn't try to be a full YAML round-tripper — the SKILL frontmatter is
    well-controlled (we own all writes). We use line-level surgery for the
    three keys we touch and append any that were absent.
    """
    lines = raw_md.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md missing opening frontmatter delimiter")
    end = next(
        (i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if end is None:
        raise ValueError("SKILL.md missing closing frontmatter delimiter")
    fm_lines = lines[1:end]

    iso_now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    updates = {
        "version": new_version,
        "parent_version": parent_version,
        "evolved_at": iso_now,
    }
    seen: set[str] = set()
    out: list[str] = []
    for line in fm_lines:
        key_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*", line)
        if key_match and key_match.group(1) in updates:
            key = key_match.group(1)
            seen.add(key)
            out.append(f"{key}: {_yaml_scalar(updates[key])}\n")
        else:
            out.append(line)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}: {_yaml_scalar(v)}\n")

    return "---\n" + "".join(out) + "---"


def _yaml_scalar(value: str) -> str:
    """Emit a YAML scalar safely. Quoted unless it's already a simple identifier."""
    if value is None:
        return "null"
    if re.match(r"^[A-Za-z0-9._\-+:]+$", str(value)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def log_evolution(
    store: Store,
    *,
    skill_name: str,
    parent_version: str,
    new_version: str,
    metric_before: dict[str, float],
    metric_after: dict[str, float],
    notes: str = "",
) -> int:
    """Append a row to evolution_log; returns the new row's id."""
    notes_full = json.dumps(
        {
            "metric_before": metric_before,
            "metric_after": metric_after,
            "delta_total": metric_after.get("total", 0.0)
            - metric_before.get("total", 0.0),
            "extra": notes,
        },
        ensure_ascii=False,
    )
    metric_name = f"{skill_name}_total"
    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO evolution_log(skill_name, parent_version, new_version, "
            "metric_name, metric_before, metric_after, notes) VALUES (?,?,?,?,?,?,?)",
            (
                skill_name,
                parent_version,
                new_version,
                metric_name,
                metric_before.get("total", 0.0),
                metric_after.get("total", 0.0),
                notes_full,
            ),
        )
        return int(cur.lastrowid or 0)


# ---------- 5. Baseline / post-eval helpers ----------------------------


def evaluate_module(
    *,
    student: dspy.Module,
    examples: list,
    input_names: list[str],
    metric_fn: Callable[[Any, str], MetricBreakdown] = score_match_metric,
) -> dict[str, float]:
    """Run a student on all examples and aggregate per-axis metric scores.

    Used for both the pre-evolution baseline and the post-evolution comparison.
    A ``forward()`` exception is treated as a hard parse failure (score 0)
    rather than a soft signal — the model didn't even produce output, so
    partial credit on individual axes would be misleading.

    ``metric_fn`` defaults to score_match's metric for backward compat. For
    other SKILLs, pass the adapter's ``metric`` function (or use
    ``evolve_skill`` which dispatches automatically).
    """
    breakdowns: list[MetricBreakdown] = []
    for ex in examples:
        kwargs = {name: getattr(ex, name) for name in input_names}
        try:
            pred = student(**kwargs)
            raw: str = getattr(pred, "response_json", None) or ""
        except Exception:
            # Empty string forces metric into OUTPUT_PARSE_FAILURE branch
            # which returns a hard 0 across all axes.
            raw = ""
        breakdowns.append(metric_fn(ex, raw))
    return aggregate(breakdowns)


# ---------- 6. End-to-end orchestration --------------------------------


def evolve_skill(
    skill_dir: Path,
    *,
    store: Store,
    main_lm: dspy.LM,
    reflection_lm: dspy.LM,
    auto: str = "light",
    examples: list | None = None,
    val_fraction: float = 0.4,
    backup: bool = True,
    log_dir: str | None = None,
    adapter: ModuleType | None = None,
) -> EvolutionResult:
    """Full evolution pipeline: load → baseline eval → GEPA → write back → log.

    Dispatches to ``evolution.adapters.<skill_name>`` for the trainset
    and metric. ``adapter`` can be passed explicitly to override; otherwise
    it's looked up from the SKILL name.

    Returns an EvolutionResult with the metric delta — the headline number
    for the W6 deliverable.
    """
    parent_spec = load_skill(skill_dir)
    if not parent_spec.inputs:
        raise ValueError(f"{parent_spec.name} has no declared inputs; cannot evolve.")

    # Resolve adapter for this SKILL
    if adapter is None:
        try:
            adapter = get_adapter(parent_spec.name)
        except KeyError as e:
            raise ValueError(
                f"No evolution adapter registered for SKILL {parent_spec.name!r}. "
                f"Add one in offerguide.evolution.adapters and register it in "
                f"adapters/__init__.py::REGISTRY."
            ) from e

    if examples is None:
        examples = list(adapter.EXAMPLES)
    train_examples, val_examples = adapter.split_train_val(val_fraction=val_fraction)

    # The adapter declares which input fields it produces. The SKILL declares
    # which inputs it consumes. Sanity-check they match.
    input_names = list(parent_spec.inputs)
    adapter_inputs = set(adapter.INPUT_NAMES)
    spec_inputs = set(input_names)
    if adapter_inputs != spec_inputs:
        raise ValueError(
            f"Adapter input mismatch for {parent_spec.name}: "
            f"adapter declares {sorted(adapter_inputs)}, SKILL declares {sorted(spec_inputs)}."
        )

    metric_fn = adapter.metric

    # Baseline: how well does the original prompt do on val?
    dspy.configure(lm=main_lm)
    baseline_student = build_student(parent_spec)
    metric_before = evaluate_module(
        student=baseline_student,
        examples=val_examples,
        input_names=input_names,
        metric_fn=metric_fn,
    )

    # GEPA evolution
    student = build_student(parent_spec)
    trainset = build_trainset(train_examples, input_names=input_names)
    valset = build_trainset(val_examples, input_names=input_names)
    metric = make_metric_for_adapter(adapter, examples)
    optimized = run_gepa(
        student=student,
        trainset=trainset,
        valset=valset,
        metric=metric,
        main_lm=main_lm,
        reflection_lm=reflection_lm,
        auto=auto,
        log_dir=log_dir,
    )

    new_instructions = read_instructions(optimized)
    new_version = bump_version(parent_spec.version)

    # Post: how well does the evolved prompt do on val?
    metric_after = evaluate_module(
        student=optimized,
        examples=val_examples,
        input_names=input_names,
        metric_fn=metric_fn,
    )

    new_path = write_evolved_skill(
        skill_dir,
        parent_spec=parent_spec,
        new_instructions=new_instructions,
        new_version=new_version,
        backup=backup,
    )
    log_id = log_evolution(
        store,
        skill_name=parent_spec.name,
        parent_version=parent_spec.version,
        new_version=new_version,
        metric_before=metric_before,
        metric_after=metric_after,
        notes=f"auto={auto}",
    )

    return EvolutionResult(
        skill_name=parent_spec.name,
        parent_version=parent_spec.version,
        new_version=new_version,
        new_skill_path=new_path,
        metric_before=metric_before,
        metric_after=metric_after,
        evolution_log_id=log_id,
    )
