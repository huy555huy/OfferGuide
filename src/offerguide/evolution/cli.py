"""CLI: ``python -m offerguide.evolution evolve <skill_name>``.

Wires the runner to environment-driven LLM config. Cheap default (`auto=light`)
keeps a real run around the $2 mark per the user's W6 brief.

Examples:

    # Evolve the score_match SKILL with default 'light' GEPA budget
    python -m offerguide.evolution evolve score_match

    # Heavier budget (more rollouts, more API spend)
    python -m offerguide.evolution evolve score_match --auto medium

    # Reflection LM = Claude (env var must be set separately when wiring Anthropic)
    python -m offerguide.evolution evolve score_match --reflection-model claude-sonnet-4-5

The output:
- New SKILL.md content overwrites the original (parent kept in `*.bak` next to it)
- One row appended to `evolution_log`
- A short before/after metric table printed to stdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dspy

from ..config import Settings
from ..memory import Store
from .runner import evolve_skill


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="offerguide.evolution")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_evolve = sub.add_parser("evolve", help="Run GEPA on a SKILL")
    p_evolve.add_argument("skill_name", help="SKILL name (matches the directory under src/offerguide/skills/)")
    p_evolve.add_argument("--auto", default="light", choices=("light", "medium", "heavy"))
    p_evolve.add_argument(
        "--reflection-model",
        default=None,
        help="Override reflection LM. Defaults to deepseek-v4-pro (stronger reasoning).",
    )
    p_evolve.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't write SKILL.md.v<old>.bak alongside the parent.",
    )
    p_evolve.add_argument(
        "--log-dir",
        default=None,
        help="Optional directory for GEPA's per-trial logs.",
    )

    args = parser.parse_args(argv)
    if args.cmd != "evolve":
        parser.error(f"unknown command: {args.cmd}")

    settings = Settings.from_env()
    if not settings.deepseek_api_key:
        print(
            "ERROR: DEEPSEEK_API_KEY not set. The evolve command needs LLM access.\n"
            "  export DEEPSEEK_API_KEY=sk-...",
            file=sys.stderr,
        )
        return 2

    skills_root = Path(__file__).parent.parent / "skills"
    skill_dir = skills_root / args.skill_name
    if not skill_dir.exists():
        print(f"ERROR: no SKILL directory at {skill_dir}", file=sys.stderr)
        return 2

    main_model = settings.default_model  # deepseek-v4-flash by default
    reflection_model = args.reflection_model or "deepseek-v4-pro"

    main_lm = dspy.LM(
        model=f"openai/{main_model}",
        api_base=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
    )
    reflection_lm = dspy.LM(
        model=f"openai/{reflection_model}",
        api_base=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
    )

    store = Store(settings.db_path)
    store.init_schema()

    print(
        f"\n✦ Evolving SKILL '{args.skill_name}'\n"
        f"  main_lm       = {main_model}\n"
        f"  reflection_lm = {reflection_model}\n"
        f"  auto          = {args.auto}\n"
    )

    result = evolve_skill(
        skill_dir,
        store=store,
        main_lm=main_lm,
        reflection_lm=reflection_lm,
        auto=args.auto,
        backup=not args.no_backup,
        log_dir=args.log_dir,
    )

    print(f"✦ Evolution complete: {result.parent_version} → {result.new_version}")
    print(f"  evolution_log.id = {result.evolution_log_id}")
    print(f"  SKILL.md         = {result.new_skill_path}")
    print()
    print(_format_metric_table(result.metric_before, result.metric_after))
    return 0


def _format_metric_table(before: dict[str, float], after: dict[str, float]) -> str:
    lines = ["  metric  before   after    Δ"]
    lines.append("  " + "─" * 32)
    for axis in ("total", "prob", "recall", "anti"):
        b = before.get(axis, 0.0)
        a = after.get(axis, 0.0)
        d = a - b
        sign = "+" if d >= 0 else ""
        lines.append(f"  {axis:7s} {b:5.2f}    {a:5.2f}   {sign}{d:5.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
