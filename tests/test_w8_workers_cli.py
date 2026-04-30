"""W8' — Tests for the workers CLI ``python -m offerguide.workers``.

Verifies argparse wiring + that each subcommand reaches the right
underlying function. The actual scout / tracker logic is exercised by
``test_w2_scout.py`` and ``test_w7_tracker.py`` — these tests just
prove the CLI dispatch works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from offerguide.workers.__main__ import main


def test_no_args_errors(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0


def test_unknown_subcommand_errors(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["unknown_cmd"])


def test_tracker_run_invokes_tracker_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """The CLI ``tracker run`` path should reach ``tracker_run`` with a Store + Notifier."""
    monkeypatch.setenv("OFFERGUIDE_DB", str(tmp_path / "tracker.db"))

    seen: list[dict[str, Any]] = []

    from offerguide.workers import __main__ as cli_mod

    def fake_tracker_run(store, *, notifier):
        seen.append({"store": store, "notifier_name": getattr(notifier, "name", "?")})
        return {"silences_found": 0, "events_recorded": 0, "notify_ok": 0}

    monkeypatch.setattr(cli_mod.tracker, "tracker_run", fake_tracker_run)

    rc = main(["tracker", "run"])
    assert rc == 0
    assert len(seen) == 1
    assert seen[0]["store"] is not None

    out = capsys.readouterr().out
    assert "Tracker run" in out
    assert "silences_found" in out
    assert "= 0" in out


def test_scout_nowcoder_invokes_crawl_nowcoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """The CLI ``scout nowcoder`` path should reach ``crawl_nowcoder``."""
    monkeypatch.setenv("OFFERGUIDE_DB", str(tmp_path / "scout.db"))

    seen: list[dict[str, Any]] = []

    from offerguide.workers import __main__ as cli_mod

    def fake_crawl(store, *, limit=None):
        seen.append({"store": store, "limit": limit})
        return {"discovered": 0, "fetched": 0, "ingested_new": 0, "dup": 0, "errors": 0}

    monkeypatch.setattr(cli_mod.scout, "crawl_nowcoder", fake_crawl)

    rc = main(["scout", "nowcoder", "--limit", "5"])
    assert rc == 0
    assert seen[0]["limit"] == 5

    out = capsys.readouterr().out
    assert "Scout nowcoder" in out
    assert "discovered" in out


def test_scout_nowcoder_no_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OFFERGUIDE_DB", str(tmp_path / "x.db"))

    seen: list[dict[str, Any]] = []

    from offerguide.workers import __main__ as cli_mod

    def fake_crawl(store, *, limit=None):
        seen.append({"limit": limit})
        return {"discovered": 0}

    monkeypatch.setattr(cli_mod.scout, "crawl_nowcoder", fake_crawl)

    rc = main(["scout", "nowcoder"])
    assert rc == 0
    assert seen[0]["limit"] is None


def test_tracker_help_shows_subcommands(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "tracker" in out
    assert "scout" in out
