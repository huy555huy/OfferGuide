"""Markdown block helpers — small dataclass + renderer for assembling
structured markdown documents from typed pieces.

This is just markdown formatting. It is not "context engineering" — the
actual LLM-context-window engineering (token budget, compaction,
clearing) lives in ``harness/context.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MarkdownBlock:
    """A small markdown section with an optional heading."""

    heading: str | None = None
    body: str | None = None
    lines: Sequence[str] = ()
    level: int = 2
    enabled: bool = True

    def render(self) -> str:
        if not self.enabled:
            return ""

        parts: list[str] = []
        heading = (self.heading or "").strip()
        body = (self.body or "").strip()
        lines = [line.rstrip() for line in self.lines if line and line.strip()]

        if heading:
            parts.append(f"{'#' * max(1, self.level)} {heading}")
        if body:
            if heading:
                parts.append("")
            parts.append(body)
        if lines:
            if heading or body:
                parts.append("")
            parts.extend(lines)

        return "\n".join(parts).strip()


def render_markdown_document(
    *blocks: MarkdownBlock | str,
    title: str | None = None,
) -> str:
    """Render a canonical markdown document from blocks.

    Strings are preserved verbatim (after trimming), while ``MarkdownBlock``
    instances render headings + bodies in a consistent shape.
    """
    rendered: list[str] = []
    if title:
        cleaned_title = title.strip()
        if cleaned_title:
            rendered.append(f"# {cleaned_title}")

    for block in blocks:
        text = block.render() if isinstance(block, MarkdownBlock) else str(block).strip()
        if text:
            rendered.append(text)

    return "\n\n".join(rendered).strip()


def render_block(
    heading: str | None = None,
    *,
    body: str | None = None,
    lines: Sequence[str] = (),
    level: int = 2,
    enabled: bool = True,
) -> MarkdownBlock:
    """Convenience factory for a markdown block."""
    return MarkdownBlock(
        heading=heading,
        body=body,
        lines=lines,
        level=level,
        enabled=enabled,
    )
