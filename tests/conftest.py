"""Shared pytest configuration.

W14.10: set ``OFFERGUIDE_SKIP_DOTENV=1`` *before* any test imports so the
``Settings.from_env`` auto-loader (added so users don't have to ``source
.env``) doesn't pull the developer's real LLM key into the test session.
"""

from __future__ import annotations

import os

# Must run before pytest collects any test module that imports offerguide.
os.environ.setdefault("OFFERGUIDE_SKIP_DOTENV", "1")
