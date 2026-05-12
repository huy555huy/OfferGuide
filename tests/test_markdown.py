from __future__ import annotations

from offerguide._markdown import MarkdownBlock, render_markdown_document


class TestMarkdownContext:
    def test_markdown_block_renders_heading_body_and_lines(self):
        block = MarkdownBlock(
            heading="系统事实",
            body="今天是 2026-05-11",
            lines=("- a", "- b"),
        )
        rendered = block.render()
        assert rendered.startswith("## 系统事实")
        assert "今天是 2026-05-11" in rendered
        assert "- a" in rendered
        assert "- b" in rendered

    def test_render_markdown_document_skips_empty_blocks(self):
        rendered = render_markdown_document(
            MarkdownBlock(heading="A", body="first"),
            "",
            MarkdownBlock(enabled=False, body="ignored"),
            "second",
            title="Snapshot",
        )
        assert rendered.startswith("# Snapshot")
        assert "first" in rendered
        assert "second" in rendered
        assert "ignored" not in rendered
