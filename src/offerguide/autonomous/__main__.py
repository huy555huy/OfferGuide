"""CLI for the autonomous daemon.

Subcommands::

    python -m offerguide.autonomous run          # block forever, run cron triggers
    python -m offerguide.autonomous run-once <job>   # fire one job immediately, exit
    python -m offerguide.autonomous list           # show registered jobs

Designed to run under launchd (macOS) / systemd (Linux) / a tmux session
on a small VPS. APScheduler's misfire_grace_time (300s) means a sleeping
laptop catches up on missed triggers when it wakes.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="offerguide.autonomous")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="Block and run all scheduled jobs on cron triggers")

    p_once = sub.add_parser(
        "run-once", help="Fire one job immediately and exit (good for cron-driven setups)"
    )
    p_once.add_argument(
        "job",
        choices=(
            "extract_facts",
            "discover_jobs",
            "jd_enrich",
            "corpus_classify",
            "silence_check",
            "corpus_refresh",
            "brief_update",
        ),
        help="Which job to run",
    )

    sub.add_parser("list", help="Print registered jobs and exit")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from .scheduler import build_default_scheduler

    sched = build_default_scheduler()

    if args.cmd == "list":
        for name in sched.list_jobs():
            print(name)
        return 0

    if args.cmd == "run-once":
        result = sched.trigger_once(args.job)
        print(f"\n✦ {args.job} → {result}")
        return 0

    if args.cmd == "run":
        import contextlib
        with contextlib.suppress(KeyboardInterrupt, SystemExit):
            sched.run_blocking()
        return 0

    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
