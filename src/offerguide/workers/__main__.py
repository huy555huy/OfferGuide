"""CLI entry point for OfferGuide background workers.

Subcommands::

    python -m offerguide.workers tracker run        # one-shot silence sweep
    python -m offerguide.workers scout nowcoder     # crawl 牛客 sitemap
    python -m offerguide.workers scout nowcoder --limit 50

The CLI reads ``Settings.from_env()`` for DB path + notifier config; pass
``--once`` on tracker for explicit single-pass intent (the default).
A persistent scheduling loop (APScheduler) is intentionally not wired —
use cron or a launchd plist to invoke ``run`` periodically.
"""

from __future__ import annotations

import argparse
import sys

from ..config import Settings
from ..memory import Store
from ..ui.notify import make_notifier
from . import scout, tracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="offerguide.workers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── tracker ──────────────────────────────────────────────────────
    p_tracker = sub.add_parser("tracker", help="Tracker worker (silence detection)")
    p_tracker.add_argument(
        "action",
        choices=("run",),
        help="`run` performs one sweep + records events + sends notifications",
    )
    p_tracker.add_argument(
        "--once",
        action="store_true",
        default=True,
        help="(default) one-shot pass; reserved for future scheduler mode",
    )

    # ── scout ────────────────────────────────────────────────────────
    p_scout = sub.add_parser("scout", help="JD ingestion worker")
    p_scout.add_argument("source", choices=("nowcoder",), help="Platform to crawl")
    p_scout.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of JD URLs to fetch (default: no cap)",
    )

    args = parser.parse_args(argv)
    settings = Settings.from_env()
    store = Store(settings.db_path)
    store.init_schema()

    if args.cmd == "tracker":
        return _cmd_tracker(args, settings, store)
    if args.cmd == "scout":
        return _cmd_scout(args, store)

    parser.error(f"unknown subcommand: {args.cmd}")
    return 2  # unreachable


def _cmd_tracker(args: argparse.Namespace, settings: Settings, store: Store) -> int:
    notifier = make_notifier(settings)
    print(
        f"\n✦ Tracker run\n"
        f"  db       = {settings.db_path}\n"
        f"  notifier = {settings.notify_channel}"
        f" ({'ready' if settings.notify_ready() else 'fallback console'})\n"
    )
    counters = tracker.tracker_run(store, notifier=notifier)
    print(_format_counters(counters))
    return 0


def _cmd_scout(args: argparse.Namespace, store: Store) -> int:
    print(
        f"\n✦ Scout {args.source}\n"
        f"  limit = {args.limit if args.limit else '(no cap)'}\n"
    )
    if args.source == "nowcoder":
        counters = scout.crawl_nowcoder(store, limit=args.limit)
        print(_format_counters(counters))
        return 0
    print(f"ERROR: unknown source {args.source!r}", file=sys.stderr)
    return 2


def _format_counters(counters: dict[str, int]) -> str:
    """Two-column counter dump."""
    if not counters:
        return "  (no counters)"
    width = max(len(k) for k in counters)
    return "\n".join(f"  {k.ljust(width)} = {v}" for k, v in counters.items())


if __name__ == "__main__":
    sys.exit(main())
