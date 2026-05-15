"""Anthropic-style long-running harness primitives for OfferGuide.

This package owns the project-level quality loop:

- default-fail contract: ``test-results.json``
- evidence reads: concrete files opened before a contract item can pass
- agent-maintained handoff: ``PROGRESS.md``
- fresh-context evaluator prompt: ``.claude/agents/evaluator.md``

The in-product job-search agent runtime lives in ``offerguide.agent_runtime``.
Keep chat/tools/memory/scheduled-wake code out of this package so the word
"harness" means the same thing here as in Anthropic's CWC reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTRACT_FILE = "test-results.json"
PROGRESS_FILE = "PROGRESS.md"
EVIDENCE_READ_LOG = ".claude/.evidence-reads"
EVALUATOR_FILE = ".claude/agents/evaluator.md"


@dataclass(frozen=True, slots=True)
class ContractItem:
    """One default-fail criterion in ``test-results.json``."""

    name: str
    passes: bool
    evidence: str
    notes: str


@dataclass(frozen=True, slots=True)
class HarnessState:
    """Snapshot of the repo-level long-running harness state."""

    root: Path
    contract_items: tuple[ContractItem, ...]
    progress_text: str
    evidence_reads: tuple[str, ...]
    evaluator_exists: bool

    @property
    def failing_items(self) -> tuple[ContractItem, ...]:
        return tuple(item for item in self.contract_items if not item.passes)

    @property
    def passing_items(self) -> tuple[ContractItem, ...]:
        return tuple(item for item in self.contract_items if item.passes)


def load_contract(root: str | Path = ".") -> dict[str, ContractItem]:
    """Load the default-fail contract from ``test-results.json``."""

    base = Path(root)
    data = json.loads((base / CONTRACT_FILE).read_text(encoding="utf-8"))
    return {
        name: ContractItem(
            name=name,
            passes=bool(raw.get("passes", False)),
            evidence=str(raw.get("evidence", "")),
            notes=str(raw.get("notes", "")),
        )
        for name, raw in data.items()
    }


def write_contract(
    items: dict[str, ContractItem],
    root: str | Path = ".",
) -> None:
    """Write ``test-results.json`` in the contract schema.

    Callers should only mark items passing after evidence has been read. The
    Claude Code ``verify-gate`` hook enforces that for agent writes; this helper
    keeps programmatic writes in the same shape.
    """

    base = Path(root)
    payload: dict[str, dict[str, Any]] = {
        name: {
            "passes": item.passes,
            "evidence": item.evidence,
            "notes": item.notes,
        }
        for name, item in items.items()
    }
    (base / CONTRACT_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def record_evidence_read(path: str | Path, root: str | Path = ".") -> None:
    """Record that an evidence file was opened.

    Mirrors the Claude Code ``track-read.sh`` primitive for Python callers.
    """

    base = Path(root)
    log_path = base / EVIDENCE_READ_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        (log_path.read_text(encoding="utf-8") if log_path.exists() else "")
        + f"{Path(path)}\n",
        encoding="utf-8",
    )


def consume_evidence_reads(root: str | Path = ".") -> tuple[str, ...]:
    """Return and clear evidence reads, matching ``verify-gate.sh`` semantics."""

    base = Path(root)
    log_path = base / EVIDENCE_READ_LOG
    if not log_path.exists():
        return ()
    reads = tuple(
        line.strip()
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    log_path.write_text("", encoding="utf-8")
    return reads


def read_progress(root: str | Path = ".") -> str:
    """Read the agent-maintained handoff file."""

    return (Path(root) / PROGRESS_FILE).read_text(encoding="utf-8")


def snapshot(root: str | Path = ".") -> HarnessState:
    """Collect the current long-running harness state."""

    base = Path(root)
    evidence_log = base / EVIDENCE_READ_LOG
    evidence_reads: tuple[str, ...] = ()
    if evidence_log.exists():
        evidence_reads = tuple(
            line.strip()
            for line in evidence_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return HarnessState(
        root=base,
        contract_items=tuple(load_contract(base).values()),
        progress_text=read_progress(base),
        evidence_reads=evidence_reads,
        evaluator_exists=(base / EVALUATOR_FILE).exists(),
    )


__all__ = [
    "CONTRACT_FILE",
    "EVALUATOR_FILE",
    "EVIDENCE_READ_LOG",
    "PROGRESS_FILE",
    "ContractItem",
    "HarnessState",
    "consume_evidence_reads",
    "load_contract",
    "read_progress",
    "record_evidence_read",
    "snapshot",
    "write_contract",
]
