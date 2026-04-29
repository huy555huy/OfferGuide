"""GEPA evolution orchestrator — compose trainset + metric + LM, run, write back.

This module is the only one that imports DSPy. Tests outside this file run
without a DSPy install. The runner is split into three pieces so each can be
tested in isolation:

1. ``build_trainset(...)`` — pure: GoldenExample → list[dspy.Example]
2. ``make_metric(...)``    — pure-ish: returns a closure that GEPA calls
3. ``run_gepa(...)``       — the only path that hits real LLMs
4. ``write_evolved_skill(...)`` + ``log_evolution(...)`` — pure file IO + DB IO

A full end-to-end run is composed in ``evolve_skill(...)`` which is what the
CLI invokes.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dspy

from ..memory import Store
from ..skills import load_skill
from ..skills._spec import SkillSpec
from .dspy_module import build_student, read_instructions
from .golden_trainset import GOLDEN_EXAMPLES, GoldenExample, split_train_val
from .metrics import MetricBreakdown, aggregate, score_match_metric


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


def _golden_by_name(examples: list[GoldenExample]) -> dict[str, GoldenExample]:
    return {ex.name: ex for ex in examples}


# ---------- 2. Metric closure ------------------------------------------


def make_score_match_metric(
    examples: list[GoldenExample],
) -> Callable[..., dspy.Prediction]:
    """Return a GEPA-compatible metric callable.

    The callable looks up the GoldenExample by name (round-tripped through
    ``_golden_name`` on dspy.Example) and computes
    :func:`score_match_metric`, returning a ``dspy.Prediction(score, feedback)``.
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
        breakdown: MetricBreakdown = score_match_metric(golden, raw)
        return dspy.Prediction(score=breakdown.total, feedback=breakdown.feedback)

    return metric


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
    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO evolution_log(skill_name, parent_version, new_version, "
            "metric_name, metric_before, metric_after, notes) VALUES (?,?,?,?,?,?,?)",
            (
                skill_name,
                parent_version,
                new_version,
                "score_match_total",
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
    examples: list[GoldenExample],
    input_names: list[str],
) -> dict[str, float]:
    """Run a student on all examples and aggregate per-axis metric scores.

    Used for both the pre-evolution baseline and the post-evolution comparison.
    A forward() exception is treated as a hard parse failure (score 0) rather
    than a soft signal — the model didn't even produce output, so partial
    credit on recall/anti-FP would be misleading.
    """
    breakdowns: list[MetricBreakdown] = []
    for ex in examples:
        kwargs = {name: getattr(ex, name) for name in input_names}
        try:
            pred = student(**kwargs)
            raw: str = getattr(pred, "response_json", None) or ""
        except Exception:
            # Empty string forces score_match_metric into OUTPUT_PARSE_FAILURE branch
            # which returns a hard 0 across all axes.
            raw = ""
        breakdowns.append(score_match_metric(ex, raw))
    return aggregate(breakdowns)


# ---------- 6. End-to-end orchestration --------------------------------


def evolve_skill(
    skill_dir: Path,
    *,
    store: Store,
    main_lm: dspy.LM,
    reflection_lm: dspy.LM,
    auto: str = "light",
    examples: list[GoldenExample] | None = None,
    val_fraction: float = 0.4,
    backup: bool = True,
    log_dir: str | None = None,
) -> EvolutionResult:
    """Full evolution pipeline: load → baseline eval → GEPA → write back → log.

    Returns an EvolutionResult with the metric delta — the headline number for
    the W6 deliverable.
    """
    parent_spec = load_skill(skill_dir)
    if not parent_spec.inputs:
        raise ValueError(f"{parent_spec.name} has no declared inputs; cannot evolve.")

    if examples is None:
        examples = list(GOLDEN_EXAMPLES)
    train_examples, val_examples = split_train_val(val_fraction=val_fraction)

    input_names = list(parent_spec.inputs)

    # Baseline: how well does the original prompt do on val?
    dspy.configure(lm=main_lm)
    baseline_student = build_student(parent_spec)
    metric_before = evaluate_module(
        student=baseline_student, examples=val_examples, input_names=input_names
    )

    # GEPA evolution
    student = build_student(parent_spec)
    trainset = build_trainset(train_examples, input_names=input_names)
    valset = build_trainset(val_examples, input_names=input_names)
    metric = make_score_match_metric(examples)
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
        student=optimized, examples=val_examples, input_names=input_names
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
