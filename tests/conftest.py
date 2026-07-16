"""Shared pytest configuration.

W14.10: set ``OFFERGUIDE_SKIP_DOTENV=1`` *before* any test imports so the
``Settings.from_env`` auto-loader (added so users don't have to ``source
.env``) doesn't pull the developer's real LLM key into the test session.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable

import pytest

# Must run before pytest collects any test module that imports offerguide.
os.environ.setdefault("OFFERGUIDE_SKIP_DOTENV", "1")

from offerguide.resume import MasterResumeSource


@pytest.fixture
def master_resume_source_factory(
    tmp_path,
) -> Callable[[str], MasterResumeSource]:
    """Build master sources backed by real, test-local PDF paths."""
    counter = 0

    def build(extracted_text: str = "测试简历文本") -> MasterResumeSource:
        nonlocal counter
        counter += 1
        path = tmp_path / f"master-resume-{counter}.pdf"
        payload = f"%PDF-1.4\n% OfferGuide test fixture {counter}\n".encode()
        path.write_bytes(payload)
        return MasterResumeSource(
            source_path=str(path.resolve()),
            sha256=hashlib.sha256(payload).hexdigest(),
            extracted_text=extracted_text,
        )

    return build
