"""Recurring jobs run by the autonomous scheduler.

Each module exports a JobSpec instance the scheduler can register:

- ``silence_check.SILENCE_CHECK_JOB``       daily 09:00 — tracker sweep
- ``corpus_refresh.CORPUS_REFRESH_JOB``     weekly Mon 08:00 — agentic 面经
- ``brief_update.BRIEF_UPDATE_JOB``         daily 23:00 — refresh briefs

Each job's main function takes a ``JobContext`` and returns a dict of
counters for logging.
"""

from . import brief_update, corpus_refresh, silence_check

__all__ = ["brief_update", "corpus_refresh", "silence_check"]
