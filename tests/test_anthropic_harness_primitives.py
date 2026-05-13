from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_hook(
    name: str,
    payload: dict,
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        ["bash", str(ROOT / ".claude" / "hooks" / name)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=cwd,
        env=merged_env,
        check=False,
    )


def test_default_fail_contract_exists_with_explicit_false_work_items() -> None:
    contract = json.loads((ROOT / "test-results.json").read_text())

    assert contract
    assert all(isinstance(row["passes"], bool) for row in contract.values())
    assert "project-vault-truthfulness-loop" in contract
    assert "resume-tailor-no-fake-claims-loop" in contract
    assert contract["project-vault-truthfulness-loop"]["passes"] is False
    assert contract["resume-tailor-no-fake-claims-loop"]["passes"] is False
    assert contract["agent-chat-artifact-loop"]["passes"] is False


def test_claude_settings_points_to_existing_harness_primitives() -> None:
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    commands: list[str] = []
    for hook_group in settings["hooks"].values():
        for entry in hook_group:
            for hook in entry["hooks"]:
                commands.append(hook["command"])

    assert ".claude/hooks/verify-gate.sh" in commands
    assert ".claude/hooks/track-read.sh" in commands
    for command in commands:
        assert (ROOT / command).exists(), command


def test_verify_gate_blocks_contract_write_without_evidence(tmp_path: Path) -> None:
    result = _run_hook(
        "verify-gate.sh",
        {"tool_input": {"file_path": "test-results.json"}},
        cwd=tmp_path,
        env={
            "RESULTS_FILE": "test-results.json",
            "VERIFY_READ_LOG": str(tmp_path / ".evidence-reads"),
        },
    )

    assert result.returncode == 0
    decision = json.loads(result.stdout)
    assert decision["decision"] == "block"
    assert "no screenshot or console-log evidence" in decision["reason"]


def test_evidence_read_unlocks_one_contract_write(tmp_path: Path) -> None:
    screenshots = tmp_path / "screenshots"
    screenshots.mkdir()
    evidence = screenshots / "agent-proof.png"
    evidence.write_bytes(b"not a real png; hook only checks that the file exists")
    read_log = tmp_path / ".evidence-reads"
    env = {
        "RESULTS_FILE": "test-results.json",
        "VERIFY_READ_LOG": str(read_log),
    }

    read = _run_hook(
        "track-read.sh",
        {"tool_input": {"file_path": str(evidence)}},
        cwd=tmp_path,
        env=env,
    )
    assert read.returncode == 0
    assert str(evidence) in read_log.read_text()

    first_write = _run_hook(
        "verify-gate.sh",
        {"tool_input": {"file_path": "test-results.json"}},
        cwd=tmp_path,
        env=env,
    )
    assert first_write.returncode == 0
    assert first_write.stdout == ""
    assert read_log.read_text() == ""

    second_write = _run_hook(
        "verify-gate.sh",
        {"tool_input": {"file_path": "test-results.json"}},
        cwd=tmp_path,
        env=env,
    )
    assert json.loads(second_write.stdout)["decision"] == "block"


def test_kill_switch_blocks_when_stop_file_exists(tmp_path: Path) -> None:
    stop_file = tmp_path / "AGENT_STOP"
    stop_file.touch()

    result = _run_hook(
        "kill-switch.sh",
        {"tool_input": {"name": "anything"}},
        cwd=tmp_path,
        env={"AGENT_STOP_FILE": str(stop_file)},
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["decision"] == "block"
