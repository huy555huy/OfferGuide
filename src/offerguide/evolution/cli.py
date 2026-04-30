"""CLI for evolution commands.

Two subcommands::

    python -m offerguide.evolution evolve <skill_name>   # run GEPA
    python -m offerguide.evolution diff   <skill_name>   # before/after report

The cheap default ``auto=light`` keeps a real evolve run around the $2 mark
per the W6 brief.

Examples::

    # Evolve the score_match SKILL with default 'light' GEPA budget
    python -m offerguide.evolution evolve score_match

    # Heavier budget (more rollouts, more API spend)
    python -m offerguide.evolution evolve score_match --auto medium

    # Print the before/after diff for the latest score_match evolution
    python -m offerguide.evolution diff score_match

    # Same, but as raw markdown — paste into README / blog
    python -m offerguide.evolution diff score_match --markdown > evolution.md

The evolve command output:
- New SKILL.md content overwrites the original (parent kept in `*.bak` next to it)
- One row appended to `evolution_log`
- A short before/after metric table printed to stdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import Settings
from ..memory import Store
from .diff import build_diff_report, render_markdown


def _skills_root() -> Path:
    return Path(__file__).parent.parent / "skills"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="offerguide.evolution")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── evolve ──────────────────────────────────────────────────────
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

    # ── diff ────────────────────────────────────────────────────────
    p_diff = sub.add_parser("diff", help="Show the latest GEPA before/after report for a SKILL")
    p_diff.add_argument("skill_name")
    p_diff.add_argument(
        "--markdown",
        action="store_true",
        help="Output raw markdown (suitable for piping into a file) instead of the pretty header.",
    )

    args = parser.parse_args(argv)
    if args.cmd == "evolve":
        return _cmd_evolve(args)
    if args.cmd == "diff":
        return _cmd_diff(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2  # unreachable


def _cmd_evolve(args: argparse.Namespace) -> int:
    """Run GEPA evolution on a SKILL — the only path that touches LLMs."""
    # Lazy imports so that `diff` works even when dspy isn't installed.
    import dspy

    from .runner import evolve_skill

    settings = Settings.from_env()
    if not settings.deepseek_api_key:
        print(
            "ERROR: DEEPSEEK_API_KEY not set. The evolve command needs LLM access.\n"
            "  export DEEPSEEK_API_KEY=sk-...",
            file=sys.stderr,
        )
        return 2

    # Validate the skill has an evolution adapter before any expensive setup
    from .adapters import get_adapter, list_evolvable_skills

    try:
        adapter = get_adapter(args.skill_name)
    except KeyError:
        print(
            f"ERROR: no evolution adapter for SKILL '{args.skill_name}'.\n"
            f"  Evolvable SKILLs: {', '.join(list_evolvable_skills())}",
            file=sys.stderr,
        )
        return 2

    skill_dir = _skills_root() / args.skill_name
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
    print(
        _format_metric_table(
            result.metric_before, result.metric_after, axes=adapter.METRIC_AXES
        )
    )
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    """Print the before/after diff for the latest evolution of a SKILL."""
    settings = Settings.from_env()
    skill_dir = _skills_root() / args.skill_name
    if not skill_dir.exists():
        print(f"ERROR: no SKILL directory at {skill_dir}", file=sys.stderr)
        return 2

    store = Store(settings.db_path)
    store.init_schema()

    report = build_diff_report(store, _skills_root(), args.skill_name)
    if report is None:
        print(
            f"No evolution_log entries for '{args.skill_name}' yet.\n"
            "Run `python -m offerguide.evolution evolve "
            f"{args.skill_name}` first.",
            file=sys.stderr,
        )
        return 1

    md = render_markdown(report)
    if args.markdown:
        # Pure markdown to stdout — for redirecting into a file
        print(md)
    else:
        print()
        print(md)
        print()
    return 0


def _format_metric_table(
    before: dict[str, float],
    after: dict[str, float],
    *,
    axes: list[str] | None = None,
) -> str:
    """Pretty-print a 4-column table over the given metric axes.

    ``axes`` defaults to score_match's axes; pass the adapter's
    ``METRIC_AXES`` for SKILL-specific axis naming.
    """
    if axes is None:
        axes = ["total", "prob", "recall", "anti"]
    width = max(8, max(len(a) for a in axes) + 2)
    lines = [f"  {'metric':{width}} before   after    Δ"]
    lines.append("  " + "─" * (width + 24))
    for axis in axes:
        b = before.get(axis, 0.0)
        a = after.get(axis, 0.0)
        d = a - b
        sign = "+" if d >= 0 else ""
        lines.append(f"  {axis:{width}} {b:5.2f}    {a:5.2f}   {sign}{d:5.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
