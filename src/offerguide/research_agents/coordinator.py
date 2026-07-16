"""Persistent transport for domain-agent invocations.

The coordinator deliberately does not know how either agent researches or
publishes. It only keeps a browser request from owning a long model run,
deduplicates the same subject revision, and exposes an honest run state.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from ..memory import Store
from .runner import AgentRunResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_agent_invocations (
    id                  TEXT PRIMARY KEY,
    agent_name          TEXT NOT NULL,
    subject_kind        TEXT NOT NULL,
    subject_id          TEXT NOT NULL,
    subject_revision    INTEGER NOT NULL CHECK (subject_revision >= 0),
    status              TEXT NOT NULL CHECK (
                            status IN (
                                'queued', 'running', 'published', 'unchanged',
                                'blocked', 'stale', 'failed'
                            )
                        ),
    trigger_reason      TEXT NOT NULL,
    domain_run_id       TEXT,
    message             TEXT NOT NULL DEFAULT '',
    unresolved_json     TEXT NOT NULL DEFAULT '[]'
                            CHECK (json_valid(unresolved_json)
                               AND json_type(unresolved_json) = 'array'),
    error_text          TEXT,
    owner_pid           INTEGER,
    owner_token         TEXT,
    created_at          REAL NOT NULL DEFAULT (julianday('now')),
    started_at          REAL,
    updated_at          REAL NOT NULL DEFAULT (julianday('now')),
    finished_at         REAL
);
CREATE INDEX IF NOT EXISTS idx_research_invocations_subject
    ON research_agent_invocations(
        agent_name, subject_kind, subject_id, created_at DESC
    );
CREATE UNIQUE INDEX IF NOT EXISTS idx_research_invocations_active_revision
    ON research_agent_invocations(
        agent_name, subject_kind, subject_id, subject_revision
    ) WHERE status IN ('queued', 'running');
"""


class DomainAgentCall(Protocol):
    def __call__(self) -> AgentRunResult: ...


@dataclass(frozen=True, slots=True)
class AgentInvocation:
    id: str
    agent_name: str
    subject_kind: str
    subject_id: str
    subject_revision: int
    status: str
    trigger_reason: str
    domain_run_id: str | None
    message: str
    unresolved: tuple[str, ...]
    error_text: str | None
    created_at: float
    started_at: float | None
    updated_at: float
    finished_at: float | None


def init_agent_invocation_schema(store: Store) -> None:
    with store.connect() as conn:
        conn.executescript(_SCHEMA)
        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(research_agent_invocations)"
            ).fetchall()
        }
        if "owner_pid" not in columns:
            conn.execute(
                "ALTER TABLE research_agent_invocations ADD COLUMN owner_pid INTEGER"
            )
        if "owner_token" not in columns:
            conn.execute(
                "ALTER TABLE research_agent_invocations ADD COLUMN owner_token TEXT"
            )


class ResearchAgentCoordinator:
    """Run accepted domain-agent calls outside the request thread."""

    def __init__(self, store: Store, *, max_workers: int = 2) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.store = store
        init_agent_invocation_schema(store)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="offerguide-research-agent",
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Future[None]] = {}
        self._owner_pid = os.getpid()
        self._owner_token = uuid.uuid4().hex
        # Only reclaim invocations whose owning process is actually gone. A
        # second live Web/CLI process must not invalidate the first one's run.
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT id, owner_pid FROM research_agent_invocations "
                "WHERE status IN ('queued', 'running')"
            ).fetchall()
            stale_ids = [
                str(row[0])
                for row in rows
                if row[1] is None or not _process_is_alive(int(row[1]))
            ]
            for invocation_id in stale_ids:
                conn.execute(
                    "UPDATE research_agent_invocations SET status = 'failed', "
                    "message = '应用重启前 Agent 尚未完成', "
                    "error_text = 'owner process is no longer running', "
                    "updated_at = julianday('now'), finished_at = julianday('now') "
                    "WHERE id = ? AND status IN ('queued', 'running')",
                    (invocation_id,),
                )

    def enqueue(
        self,
        *,
        agent_name: str,
        subject_kind: str,
        subject_id: str | int,
        subject_revision: int,
        trigger_reason: str,
        call: DomainAgentCall,
    ) -> tuple[AgentInvocation, bool]:
        if subject_revision < 0:
            raise ValueError("subject_revision must be non-negative")
        values = {
            "agent_name": agent_name.strip(),
            "subject_kind": subject_kind.strip(),
            "subject_id": str(subject_id),
            "trigger_reason": trigger_reason.strip(),
        }
        if any(not value for value in values.values()):
            raise ValueError("invocation identity and trigger reason must not be blank")

        invocation_id = uuid.uuid4().hex
        created = False
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM research_agent_invocations "
                "WHERE agent_name = ? AND subject_kind = ? AND subject_id = ? "
                "AND subject_revision = ? AND status IN ('queued', 'running') "
                "ORDER BY created_at DESC LIMIT 1",
                (
                    values["agent_name"],
                    values["subject_kind"],
                    values["subject_id"],
                    subject_revision,
                ),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO research_agent_invocations("
                    "id, agent_name, subject_kind, subject_id, subject_revision, "
                    "status, trigger_reason, message, owner_pid, owner_token"
                    ") VALUES (?, ?, ?, ?, ?, 'queued', ?, '等待 Agent 运行', ?, ?)",
                    (
                        invocation_id,
                        values["agent_name"],
                        values["subject_kind"],
                        values["subject_id"],
                        subject_revision,
                        values["trigger_reason"],
                        self._owner_pid,
                        self._owner_token,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM research_agent_invocations WHERE id = ?",
                    (invocation_id,),
                ).fetchone()
                created = True

        invocation = _invocation_from_row(row)
        if created:
            future = self._executor.submit(self._execute, invocation.id, call)
            with self._lock:
                self._futures[invocation.id] = future
            future.add_done_callback(lambda _future, iid=invocation.id: self._forget(iid))
        return invocation, created

    def get(self, invocation_id: str) -> AgentInvocation | None:
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM research_agent_invocations WHERE id = ?",
                (invocation_id,),
            ).fetchone()
        return _invocation_from_row(row) if row is not None else None

    def latest(
        self,
        *,
        agent_name: str,
        subject_kind: str,
        subject_id: str | int,
    ) -> AgentInvocation | None:
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM research_agent_invocations "
                "WHERE agent_name = ? AND subject_kind = ? AND subject_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (agent_name, subject_kind, str(subject_id)),
            ).fetchone()
        return _invocation_from_row(row) if row is not None else None

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _execute(self, invocation_id: str, call: DomainAgentCall) -> None:
        with self.store.connect() as conn:
            updated = conn.execute(
                "UPDATE research_agent_invocations SET status = 'running', "
                "message = 'Agent 正在检查真实来源', started_at = julianday('now'), "
                "updated_at = julianday('now') WHERE id = ? AND status = 'queued' "
                "AND owner_token = ?",
                (invocation_id, self._owner_token),
            )
            if updated.rowcount != 1:
                return
        try:
            result = call()
            message, unresolved = _result_details(result)
            if result.error_text:
                unresolved.append(result.error_text)
            with self.store.connect() as conn:
                conn.execute(
                    "UPDATE research_agent_invocations SET status = ?, domain_run_id = ?, "
                    "message = ?, unresolved_json = ?, error_text = ?, "
                    "updated_at = julianday('now'), finished_at = julianday('now') "
                    "WHERE id = ? AND owner_token = ?",
                    (
                        result.status.value,
                        result.run_id,
                        message,
                        json.dumps(unresolved, ensure_ascii=False),
                        result.error_text,
                        invocation_id,
                        self._owner_token,
                    ),
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            with self.store.connect() as conn:
                conn.execute(
                    "UPDATE research_agent_invocations SET status = 'failed', "
                    "message = 'Agent 运行失败', error_text = ?, "
                    "updated_at = julianday('now'), finished_at = julianday('now') "
                    "WHERE id = ? AND owner_token = ?",
                    (error, invocation_id, self._owner_token),
                )

    def _forget(self, invocation_id: str) -> None:
        with self._lock:
            self._futures.pop(invocation_id, None)


def _result_details(result: AgentRunResult) -> tuple[str, list[str]]:
    labels = {
        "published": "Agent 已发布新的当前结果",
        "unchanged": "Agent 已检查，当前结果无需更新",
        "blocked": "Agent 暂时无法取得足够证据",
        "stale": "运行期间上下文已更新，本次结果未发布",
        "failed": "Agent 运行失败",
    }
    message = labels[result.status.value]
    unresolved: list[str] = []
    output = result.terminal_output
    if isinstance(output, Mapping):
        reason = str(output.get("reason") or "").strip()
        checks_value = output.get("checks_completed", output.get("completed_checks"))
        checks = (
            [str(item).strip() for item in checks_value if str(item).strip()]
            if isinstance(checks_value, list)
            else []
        )
        gaps_value = output.get("evidence_gaps")
        gaps = (
            [str(item).strip() for item in gaps_value if str(item).strip()]
            if isinstance(gaps_value, list)
            else []
        )
        if reason:
            message += f"：{reason}"
        if checks:
            message += "；已检查：" + "、".join(checks)
        unresolved.extend(gaps or ([reason] if result.status.value == "blocked" and reason else []))
    elif result.status.value == "blocked":
        reason = {
            "model_stopped_without_terminal_tool": (
                "模型停止前未调用发布、确认无变化或阻塞报告工具"
            ),
            "iteration_limit_without_terminal_tool": (
                "达到研究轮次上限时仍未调用发布、确认无变化或阻塞报告工具"
            ),
        }.get(result.reason, result.reason)
        message = f"Agent 未按完成协议结束：{reason}"
        completed_tools = list(dict.fromkeys(result.tool_calls))
        if completed_tools:
            message += "；已调用工具：" + "、".join(completed_tools)
        else:
            message += "；未记录已完成的来源检查"
        unresolved.append(reason)
        if result.tool_errors:
            recent_errors = list(result.tool_errors[-3:])
            message += "；最近工具错误：" + "；".join(recent_errors)
            unresolved.extend(recent_errors)
    return message, unresolved


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _invocation_from_row(row: sqlite3.Row) -> AgentInvocation:
    raw_unresolved = json.loads(str(row["unresolved_json"] or "[]"))
    unresolved = tuple(str(item) for item in raw_unresolved if str(item).strip())
    return AgentInvocation(
        id=str(row["id"]),
        agent_name=str(row["agent_name"]),
        subject_kind=str(row["subject_kind"]),
        subject_id=str(row["subject_id"]),
        subject_revision=int(row["subject_revision"]),
        status=str(row["status"]),
        trigger_reason=str(row["trigger_reason"]),
        domain_run_id=str(row["domain_run_id"]) if row["domain_run_id"] else None,
        message=str(row["message"] or ""),
        unresolved=unresolved,
        error_text=str(row["error_text"]) if row["error_text"] else None,
        created_at=float(row["created_at"]),
        started_at=float(row["started_at"]) if row["started_at"] is not None else None,
        updated_at=float(row["updated_at"]),
        finished_at=float(row["finished_at"]) if row["finished_at"] is not None else None,
    )


__all__ = [
    "AgentInvocation",
    "ResearchAgentCoordinator",
    "init_agent_invocation_schema",
]
