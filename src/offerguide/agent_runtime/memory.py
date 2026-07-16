"""Optional file-based notes for facts useful across job-search sessions.

Implementation of Anthropic-spec memory tool with 6 commands:
- view, create, str_replace, insert, delete, rename

All operations are scoped to ``.offerguide/worldview/`` (path traversal
blocked at the runtime boundary).

The runtime provides scoped, atomic file operations. Notes are useful only when
they preserve confirmed user preferences, application state, or future events;
the agent is not required to maintain a diary or internal identity.
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


# Default view truncation keeps an accidentally large note from consuming the
# whole model context.
DEFAULT_VIEW_LIMIT = 500


# Small optional note set written when the directory is empty.
_BOOTSTRAP_FILES: dict[str, str] = {
    "MEMORY.md": """# 已确认的求职上下文

只记录用户明确确认、且会影响后续找岗、投递或面试的信息。
""",
    "candidate.md": """# 用户偏好与边界

尚无额外记录。主简历仍以配置的原始文件为准。
""",
    "tracked-jobs.md": """# 需要跨会话跟进的岗位

暂无。
""",
    "upcoming-events.md": """# 已确认的面试与截止时间

暂无。
""",
}


# Memory tool input schema (OpenAI function-calling format).
# The 6 commands map to the Anthropic memory tool spec.
MEMORY_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": (
            "Read/write optional persistent job-search notes in `.offerguide/worldview/`. "
            "Store only confirmed preferences, tracked applications, and future events. "
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
                        "view",
                        "create",
                        "str_replace",
                        "insert",
                        "delete",
                        "rename",
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
                        "[start, end] 1-indexed lines for view. Optional; default = whole file."
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
    """File-system backed memory store. Single instance per agent runtime.

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
        """Return the first ``max_lines`` of MEMORY.md, or an empty string."""
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
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*.md"))

    def _build_file_index(self, *, skip: str | None = None) -> list[str]:
        """One line per non-skipped .md file: ``- name (N lines): first heading``.

        Helps agent decide if a file is empty (still bootstrap text), has
        agent-written content, or recently changed — without `view`-ing.
        """
        out: list[str] = []
        for fname in self.list_files():
            if fname == skip:
                continue
            try:
                content = (self.root / fname).read_text(encoding="utf-8")
            except OSError:
                continue
            file_lines = content.splitlines()
            n = len(file_lines)
            # First non-empty line that's a heading or content marker
            first = ""
            for line in file_lines:
                stripped = line.strip()
                if stripped:
                    first = stripped[:80]
                    break
            out.append(f"- {fname} ({n} lines): {first}")
        return out

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
        all_lines = text.splitlines()
        view_range = args.get("view_range")
        if view_range:
            if (
                not isinstance(view_range, list)
                or len(view_range) != 2
                or not all(isinstance(x, int) for x in view_range)
            ):
                return "ERROR: view_range must be [start, end] integers"
            start, end = view_range[0], view_range[1]
            # 1-indexed; -1 means end-of-file
            start_idx = max(0, start - 1)
            end_idx = len(all_lines) if end == -1 else min(len(all_lines), end)
            sliced = all_lines[start_idx:end_idx]
            numbered = _number_lines(sliced, start=start_idx + 1)
            return f"OK {path_str} (lines {start}-{end}):\n{numbered}"

        # Smell 8 fix (W15.12 review): default view truncates large files
        # to DEFAULT_VIEW_LIMIT lines so a single command doesn't blow up
        # the model's context. Agent can use view_range=[1,N] to override.
        n = len(all_lines)
        if n > DEFAULT_VIEW_LIMIT:
            sliced = all_lines[:DEFAULT_VIEW_LIMIT]
            numbered = _number_lines(sliced, start=1)
            return (
                f"OK {path_str} ({n} lines, showing first {DEFAULT_VIEW_LIMIT} — "
                f"use view_range=[start,end] for the rest):\n{numbered}"
            )
        numbered = _number_lines(all_lines, start=1)
        return f"OK {path_str} ({n} lines):\n{numbered}"

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
            return f"ERROR: insert_line {line_no} > file length {len(lines)}"
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
