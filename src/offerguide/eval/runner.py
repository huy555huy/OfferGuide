"""W13.7 eval runner — load JSON dataset, invoke SKILL, score, report.

Eval shape (one JSON file per SKILL, in evals/datasets/<skill_name>.json):

    [
      {
        "name": "high_match_bytedance_agent",
        "inputs": {"job_text": "...", "user_profile": "..."},
        "expected": {
          "json_valid": true,
          "must_contain_keys": ["probability", "reasoning"],
          "must_contain_phrases_in_output": ["LangGraph"],
          "score_range": {"probability": [0.55, 0.95]}
        },
        "notes": "high-match should score moderately high"
      },
      ...
    ]

Run with::

    python -m offerguide.eval                # all SKILLs, all cases
    python -m offerguide.eval score_match    # one SKILL
    python -m offerguide.eval --help

The runner is **deliberately not** a full DeepEval clone — we only check
the assertions we care about. Anything fancier should be added on demand,
not pre-built.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm import LLMClient
from ..memory import Store
from ..skills import SkillRuntime, SkillSpec, discover_skills

log = logging.getLogger(__name__)

DEFAULT_SKILLS_ROOT = Path(__file__).parent.parent / "skills"
DEFAULT_DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "evals" / "datasets"


@dataclass
class EvalCase:
    """One test case for a SKILL."""
    name: str
    inputs: dict[str, Any]
    expected: dict[str, Any]
    notes: str = ""


@dataclass
class EvalResult:
    case_name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: int = 0
    raw_output: str = ""


@dataclass
class EvalReport:
    skill_name: str
    skill_version: str
    total: int
    passed: int
    results: list[EvalResult]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def render_summary(self) -> str:
        lines = [
            f"## {self.skill_name} v{self.skill_version}",
            f"   {self.passed}/{self.total} passed ({self.pass_rate:.0%})",
        ]
        total_cost = sum(r.cost_usd for r in self.results)
        if total_cost > 0:
            lines.append(f"   total cost: ${total_cost:.4f}")
        for r in self.results:
            mark = "✓" if r.passed else "✗"
            lines.append(f"   {mark} {r.case_name}")
            for f in r.failures:
                lines.append(f"      - {f}")
        return "\n".join(lines)


# ─────────────────────── case loading ───────────────────────


def load_cases(skill_name: str, datasets_dir: Path = DEFAULT_DATASETS_DIR) -> list[EvalCase]:
    """Load eval cases for a SKILL from evals/datasets/<skill>.json. Returns []
    if the file doesn't exist (skill simply has no eval set yet)."""
    fp = datasets_dir / f"{skill_name}.json"
    if not fp.exists():
        return []
    try:
        raw = json.loads(fp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log.error("eval dataset %s is malformed: %s", fp, e)
        return []
    cases: list[EvalCase] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        cases.append(EvalCase(
            name=entry.get("name", "<unnamed>"),
            inputs=entry.get("inputs", {}),
            expected=entry.get("expected", {}),
            notes=entry.get("notes", ""),
        ))
    return cases


# ─────────────────────── case scoring ───────────────────────


def _score_case(case: EvalCase, parsed: dict[str, Any] | None, raw_text: str) -> list[str]:
    """Run all assertions in case.expected; return list of failure messages."""
    failures: list[str] = []
    expected = case.expected

    # JSON validity
    if expected.get("json_valid", False) and parsed is None:
        failures.append("json_valid required but parse failed")

    # Required keys at top level of parsed JSON
    must_keys = expected.get("must_contain_keys", [])
    if must_keys:
        if parsed is None:
            failures.append("must_contain_keys requires JSON, got non-parsed output")
        else:
            for k in must_keys:
                if k not in parsed:
                    failures.append(f"missing required key '{k}'")

    # Phrases that must appear in raw output
    must_phrases = expected.get("must_contain_phrases_in_output", [])
    for phrase in must_phrases:
        if phrase not in raw_text:
            failures.append(f"missing required phrase {phrase!r}")

    # Forbidden phrases
    forbidden_phrases = expected.get("must_not_contain_phrases", [])
    for phrase in forbidden_phrases:
        if phrase in raw_text:
            failures.append(f"forbidden phrase {phrase!r} present")

    # Numeric range checks on parsed fields
    score_ranges = expected.get("score_range", {})
    if score_ranges and parsed is None:
        failures.append("score_range requires JSON, got non-parsed output")
    else:
        for key, range_pair in (score_ranges or {}).items():
            if not isinstance(range_pair, list | tuple) or len(range_pair) != 2:
                failures.append(f"score_range[{key}] malformed (need [lo, hi])")
                continue
            lo, hi = range_pair
            val = (parsed or {}).get(key)
            try:
                # parsed.get returns Any|None; float(None) → TypeError caught below.
                fval = float(val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                failures.append(f"score_range[{key}] expected numeric, got {val!r}")
                continue
            if not (lo <= fval <= hi):
                failures.append(f"{key}={fval} outside [{lo}, {hi}]")

    return failures


# ─────────────────────── runner ───────────────────────


def run_eval_for_skill(
    *,
    runtime: SkillRuntime,
    spec: SkillSpec,
    cases: list[EvalCase],
    consult_variant_registry: bool = True,
) -> EvalReport:
    """Run all cases against one SKILL spec. Cost + latency aggregated."""
    results: list[EvalResult] = []
    for case in cases:
        try:
            sr = runtime.invoke(
                spec, case.inputs,
                strict_inputs=False,  # eval datasets sometimes carry extra context
                consult_variant_registry=consult_variant_registry,
            )
        except Exception as e:
            results.append(EvalResult(
                case_name=case.name, passed=False,
                failures=[f"SKILL invocation raised {type(e).__name__}: {e}"],
            ))
            continue
        failures = _score_case(case, sr.parsed, sr.raw_text)
        results.append(EvalResult(
            case_name=case.name,
            passed=not failures,
            failures=failures,
            cost_usd=sr.cost_usd,
            latency_ms=sr.latency_ms,
            raw_output=sr.raw_text[:500],
        ))
    return EvalReport(
        skill_name=spec.name,
        skill_version=spec.version,
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        results=results,
    )


def run_eval(
    *,
    runtime: SkillRuntime,
    skill_specs: list[SkillSpec],
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    only_skill: str | None = None,
) -> list[EvalReport]:
    """Run eval across all SKILLs that have a dataset (or just one if filtered).

    Returns one EvalReport per SKILL with at least 1 loaded case.
    """
    reports: list[EvalReport] = []
    for spec in skill_specs:
        if only_skill and spec.name != only_skill:
            continue
        cases = load_cases(spec.name, datasets_dir)
        if not cases:
            continue
        report = run_eval_for_skill(runtime=runtime, spec=spec, cases=cases)
        reports.append(report)
    return reports


# ─────────────────────── CLI ───────────────────────


def _cli() -> int:
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(prog="offerguide.eval")
    parser.add_argument("skill", nargs="?",
                         help="Run eval for this SKILL only (default: all SKILLs with a dataset)")
    parser.add_argument("--datasets-dir", default=str(DEFAULT_DATASETS_DIR))
    parser.add_argument("--db", default="/tmp/offerguide_eval.db",
                         help="SQLite DB path for skill_runs persistence (default: /tmp)")
    args = parser.parse_args()

    api_key = os.environ.get("OFFERGUIDE_LLM_API_KEY")
    base_url = os.environ.get("OFFERGUIDE_LLM_BASE_URL")
    model = os.environ.get("OFFERGUIDE_LLM_MODEL")
    if not api_key:
        print("[err] OFFERGUIDE_LLM_API_KEY not set", file=sys.stderr)
        return 1

    store = Store(args.db)
    store.init_schema()
    llm = LLMClient(api_key=api_key, base_url=base_url, default_model=model)
    runtime = SkillRuntime(llm=llm, store=store)
    skills = discover_skills(DEFAULT_SKILLS_ROOT)

    reports = run_eval(
        runtime=runtime, skill_specs=skills,
        datasets_dir=Path(args.datasets_dir),
        only_skill=args.skill,
    )

    if not reports:
        target = args.skill or "any SKILL"
        print(f"No eval cases found for {target} in {args.datasets_dir}")
        return 0

    overall_pass = sum(r.passed for r in reports)
    overall_total = sum(r.total for r in reports)
    overall_cost = sum(sum(c.cost_usd for c in r.results) for r in reports)

    print("# Eval Report\n")
    for r in reports:
        print(r.render_summary() + "\n")
    print(f"## TOTAL: {overall_pass}/{overall_total} pass · ${overall_cost:.4f}")

    return 0 if overall_pass == overall_total else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
