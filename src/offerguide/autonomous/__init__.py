"""Autonomous layer — APScheduler-based daemon.

Borrowed patterns:
- **APScheduler** (MIT, Alex Grönholm) — cron-style trigger + jobstore
  https://github.com/agronholm/apscheduler
- **OpenHands** (MIT, All Hands AI) — observation-action-feedback agent
  loop pattern (we adapt the structure, not the code)
  https://github.com/All-Hands-AI/OpenHands
- **LangChain AgentExecutor** (MIT) — tool-budgeted iteration shape

Three recurring jobs:
- ``silence_check``  daily   — wraps tracker.tracker_run
- ``corpus_refresh`` weekly  — for each tracked company, run the agentic
                                corpus collector (search + LLM filter +
                                ingest)
- ``brief_update``   daily   — agent reads recent observations and
                                regenerates company_briefs

The user runs ``python -m offerguide.autonomous`` as a long-lived
process (or under launchd/cron). The daemon blocks until killed.
"""

from .jobs import brief_update, corpus_refresh, silence_check
from .scheduler import AutonomousScheduler, build_default_scheduler

__all__ = [
    "AutonomousScheduler",
    "brief_update",
    "build_default_scheduler",
    "corpus_refresh",
    "silence_check",
]
