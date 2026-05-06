"""Memory tool — file-based persistent storage for the agent.

Implementation of Anthropic-spec memory tool with 6 commands:
- view, create, str_replace, insert, delete, rename

All operations are scoped to ``.offerguide/worldview/`` (path traversal
blocked at the harness boundary).

The agent treats this as its **own brain on disk** — markdown files that
it writes/reads/restructures itself. Harness only provides the file I/O
primitives + safety (path scoping, atomic writes); it never decides what
goes in or imposes schema beyond the bootstrap template.

Anthropic reference (from research):
> Each session injects first 200 lines of MEMORY.md into system context.
> Agent calls memory tool during session to read/write specific topics.
> Memory survives context compaction (it's not in the message history).

W15: replaces W14.20's ``agent_self_notes`` SQL table. Markdown files
are the natural medium for an agent to think in (vs SQL rows that
require schema discipline).
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Bootstrap files written when worldview/ is empty. The agent owns these
# files after creation — feel free to restructure / add new files.
_BOOTSTRAP_FILES: dict[str, str] = {
    "MEMORY.md": """# 你的主页 (auto-injected first 200 lines each wake)

## 我对用户的理解 (摘要)
[启动时为空 — 你 ask_user 后回填到 candidate.md, 这里写摘要]

## 当前阶段
[校招日历位置 + 用户当前节奏]

## 当前策略
[你自己定的, 可改]

## 紧急事
[deadline 临近 / sleep N 天的应用 / pending question 等]

## 你给自己的备忘
[跨 wake 的小事]
""",
    "candidate.md": """# 用户全貌

## CV 摘要
[空 — 启动时 ask_user 拿 cv]

## 偏好
[职能 / 公司类型 / 地点 / 薪资]

## 雷区
[绝对不去的]

## 目标演化
[用户的目标随时间变化的痕迹]
""",
    "tracked-jobs.md": """# 跟进中的岗位

[每个岗位一段, 包含: 公司 / 职位 / 状态 / 你的判断 / 关键日期]

例:
- **字节跳动 / Algo Intern (NLP)**
  状态: 投了 5 天 (2026-04-30 投)
  我的判断: 高匹配, 但简历定向不够
  下一步: 7 天没回应建议 followup
""",
    "upcoming-events.md": """# 即将到来的事件

[面试 / deadline / 每个一行]
""",
    "reflections.md": """# 复盘

[每次 wake 结束前简单写: 我做了啥 / 用户反应 / 我学到啥]
""",
    "strategy.md": """# 当前策略

## 这周聚焦
[你自己定]

## 未解疑问
[你想找时机问 user 的事]
""",
}


# Memory tool input schema (OpenAI function-calling format).
# The 6 commands map to the Anthropic memory tool spec.
MEMORY_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": (
            "Read/write your worldview markdown files in `.offerguide/worldview/`. "
            "This is your persistent brain — files survive across wakes. "
            "6 commands: view (read file or list dir), create (new file or overwrite), "
            "str_replace (find/replace text), insert (insert at line N), "
            "delete (remove file), rename (move file)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": [
                        "view", "create", "str_replace",
                        "insert", "delete", "rename",
                    ],
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Path relative to `.offerguide/worldview/`. "
                        "Use 'MEMORY.md' or 'tracked-jobs.md' etc. "
                        "Or '/' to list directory."
                    ),
                },
                # view-specific
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "[start, end] 1-indexed lines for view. "
                        "Optional; default = whole file."
                    ),
                },
                # create-specific
                "file_text": {
                    "type": "string",
                    "description": "Full file content for create.",
                },
                # str_replace-specific
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
                # insert-specific
                "insert_line": {
                    "type": "integer",
                    "description": "0 = insert at top; N = insert after line N.",
                },
                "insert_text": {"type": "string"},
                # rename-specific
                "new_path": {"type": "string"},
            },
            "required": ["command"],
        },
    },
}


@dataclass
class MemoryStore:
    """File-system backed memory store. Single instance per agent harness.

    Path scoping: every operation is resolved relative to ``root`` and
    must stay inside it. ``..`` traversal is rejected at the boundary.

    Atomic writes: create/str_replace/insert write to a temp file then
    os.replace() — no half-written files if the process crashes.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._bootstrap_if_empty()

    # ── Public dispatch ─────────────────────────────────────────────

    def execute(self, args: dict[str, Any]) -> str:
        """Dispatch a memory tool call. Returns string for the LLM.

        On error, returns a message starting with 'ERROR:' so the model
        can read it and self-correct (no exception bubbling).
        """
        command = args.get("command", "")
        try:
            if command == "view":
                return self._cmd_view(args)
            if command == "create":
                return self._cmd_create(args)
            if command == "str_replace":
                return self._cmd_str_replace(args)
            if command == "insert":
                return self._cmd_insert(args)
            if command == "delete":
                return self._cmd_delete(args)
            if command == "rename":
                return self._cmd_rename(args)
            return f"ERROR: unknown command {command!r}"
        except _MemoryError as e:
            return f"ERROR: {e}"
        except Exception as e:
            log.exception("memory tool crashed: %s", args)
            return f"ERROR: {type(e).__name__}: {e}"

    # ── Cross-wake helpers (used by Context assembly) ────────────────

    def auto_load_text(self, max_lines: int = 200) -> str:
        """Return first ``max_lines`` of MEMORY.md as a single string.

        Empty string if MEMORY.md is missing. Injected into the system
        context every wake so the agent always sees its 'home page'
        without needing to call view explicitly.
        """
        memory_md = self.root / "MEMORY.md"
        if not memory_md.exists():
            return ""
        try:
            lines = memory_md.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            log.warning("auto_load failed: %s", e)
            return ""
        kept = lines[:max_lines]
        if len(lines) > max_lines:
            kept.append(f"... (truncated; {len(lines) - max_lines} more lines)")
        return "\n".join(kept)

    def list_files(self) -> list[str]:
        """List all .md files in worldview, sorted."""
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*.md")
        )

    # ── Internals: command implementations ──────────────────────────

    def _cmd_view(self, args: dict[str, Any]) -> str:
        path_str = (args.get("path") or "").strip()
        if path_str in ("", "/", "."):
            files = self.list_files()
            if not files:
                return "OK (empty worldview): no .md files yet"
            return "OK files:\n" + "\n".join(f"  - {p}" for p in files)

        full = self._resolve(path_str)
        if not full.exists():
            return f"ERROR: {path_str} not found"
        if full.is_dir():
            inner = sorted(p.name for p in full.iterdir() if p.is_file())
            return f"OK dir {path_str}:\n" + "\n".join(f"  - {n}" for n in inner)

        text = full.read_text(encoding="utf-8")
        view_range = args.get("view_range")
        if view_range:
            if (
                not isinstance(view_range, list)
                or len(view_range) != 2
                or not all(isinstance(x, int) for x in view_range)
            ):
                return "ERROR: view_range must be [start, end] integers"
            start, end = view_range[0], view_range[1]
            lines = text.splitlines()
            # 1-indexed; -1 means end-of-file
            start_idx = max(0, start - 1)
            end_idx = len(lines) if end == -1 else min(len(lines), end)
            sliced = lines[start_idx:end_idx]
            numbered = _number_lines(sliced, start=start_idx + 1)
            return f"OK {path_str} (lines {start}-{end}):\n{numbered}"
        numbered = _number_lines(text.splitlines(), start=1)
        return f"OK {path_str} ({len(text.splitlines())} lines):\n{numbered}"

    def _cmd_create(self, args: dict[str, Any]) -> str:
        path_str = self._require_path(args)
        file_text = args.get("file_text")
        if file_text is None:
            return "ERROR: create requires file_text"
        full = self._resolve(path_str)
        full.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(full, str(file_text))
        n_lines = len(str(file_text).splitlines())
        return f"OK created {path_str} ({n_lines} lines)"

    def _cmd_str_replace(self, args: dict[str, Any]) -> str:
        path_str = self._require_path(args)
        old_str = args.get("old_str")
        new_str = args.get("new_str", "")
        if old_str is None:
            return "ERROR: str_replace requires old_str"
        full = self._resolve(path_str)
        if not full.exists():
            return f"ERROR: {path_str} not found"
        text = full.read_text(encoding="utf-8")
        count = text.count(old_str)
        if count == 0:
            return f"ERROR: old_str not found in {path_str}"
        if count > 1:
            return (
                f"ERROR: old_str matches {count} times in {path_str} "
                "(must be unique — add more context to old_str)"
            )
        new_text = text.replace(old_str, str(new_str), 1)
        _atomic_write(full, new_text)
        return f"OK replaced 1 occurrence in {path_str}"

    def _cmd_insert(self, args: dict[str, Any]) -> str:
        path_str = self._require_path(args)
        line_no = args.get("insert_line")
        insert_text = args.get("insert_text", "")
        if line_no is None:
            return "ERROR: insert requires insert_line"
        if not isinstance(line_no, int) or line_no < 0:
            return "ERROR: insert_line must be non-negative integer"
        full = self._resolve(path_str)
        if not full.exists():
            return f"ERROR: {path_str} not found"
        text = full.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        if line_no > len(lines):
            return (
                f"ERROR: insert_line {line_no} > file length {len(lines)}"
            )
        new_block = str(insert_text)
        if not new_block.endswith("\n"):
            new_block += "\n"
        lines.insert(line_no, new_block)
        _atomic_write(full, "".join(lines))
        return f"OK inserted at line {line_no} of {path_str}"

    def _cmd_delete(self, args: dict[str, Any]) -> str:
        path_str = self._require_path(args)
        full = self._resolve(path_str)
        if not full.exists():
            return f"ERROR: {path_str} not found"
        if full.is_dir():
            shutil.rmtree(full)
        else:
            full.unlink()
        return f"OK deleted {path_str}"

    def _cmd_rename(self, args: dict[str, Any]) -> str:
        old_path = self._require_path(args)
        new_path = (args.get("new_path") or "").strip()
        if not new_path:
            return "ERROR: rename requires new_path"
        old_full = self._resolve(old_path)
        new_full = self._resolve(new_path)
        if not old_full.exists():
            return f"ERROR: {old_path} not found"
        if new_full.exists():
            return f"ERROR: {new_path} already exists (won't overwrite)"
        new_full.parent.mkdir(parents=True, exist_ok=True)
        old_full.rename(new_full)
        return f"OK renamed {old_path} → {new_path}"

    # ── Path safety + bootstrap ─────────────────────────────────────

    def _resolve(self, rel_path: str) -> Path:
        """Resolve rel_path under root. Reject path traversal."""
        cleaned = rel_path.lstrip("/").strip()
        if not cleaned:
            raise _MemoryError("path is empty")
        candidate = (self.root / cleaned).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise _MemoryError(
                f"path {rel_path!r} escapes worldview/ (no traversal allowed)"
            ) from exc
        return candidate

    def _require_path(self, args: dict[str, Any]) -> str:
        path_str = (args.get("path") or "").strip()
        if not path_str:
            raise _MemoryError("path is required")
        return path_str

    def _bootstrap_if_empty(self) -> None:
        """First-run: write the MEMORY.md / candidate.md / etc. templates.

        Only writes files that don't exist — so re-running on an
        established worldview is a no-op and never overwrites the
        agent's own writing.
        """
        for name, body in _BOOTSTRAP_FILES.items():
            path = self.root / name
            if not path.exists():
                _atomic_write(path, body)
                log.debug("bootstrap wrote %s", name)


# ── helpers ─────────────────────────────────────────────────────────


class _MemoryError(RuntimeError):
    """Internal — caught by execute() and turned into 'ERROR: ...' string."""


def _atomic_write(path: Path, text: str) -> None:
    """Write text atomically: write to .tmp, then os.replace().

    Prevents half-written files if the process crashes mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _number_lines(lines: Iterable[str], *, start: int = 1) -> str:
    """Render lines with 1-indexed line numbers (cat -n style)."""
    out = []
    for i, line in enumerate(lines, start=start):
        out.append(f"{i:5d}\t{line}")
    return "\n".join(out)
