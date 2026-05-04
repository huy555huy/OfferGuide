"""CLI for the autonomous daemon (W13.2 redesign).

Subcommands::

    python -m offerguide.autonomous run          # block forever; cron wakes the agent
    python -m offerguide.autonomous run-once     # wake the agent once, exit
    python -m offerguide.autonomous list         # show what jobs are registered

In W13.2 the scheduler registers a single job ``wake_agent`` (default cron:
every 4 hours from 08:00 to 22:00 Asia/Shanghai). Each fire wakes the
central AgentLoop with a "巡检" goal — the agent reads the current system
state and decides which maintenance tools (discover/enrich/classify/etc)
are worth running this tick.

Designed to run under launchd (macOS) / systemd (Linux) / a tmux session
on a small VPS. APScheduler's misfire_grace_time means a sleeping laptop
catches up on missed wakes when it next wakes.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="offerguide.autonomous")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="Block forever; the cron wakes the agent on schedule")
    sub.add_parser(
        "run-once", help="Wake the agent loop once immediately and exit",
    )
    sub.add_parser("list", help="Print registered jobs and exit")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from .scheduler import build_agent_wake_scheduler

    sched = build_agent_wake_scheduler()

    if args.cmd == "list":
        for name in sched.list_jobs():
            print(name)
        return 0

    if args.cmd == "run-once":
        result = sched.trigger_once("wake_agent")
        print(f"\n✦ wake_agent → {result}")
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
