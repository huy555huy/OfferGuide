from __future__ import annotations

from pathlib import Path

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1] / "src" / "offerguide" / "ui" / "templates"
)


def test_pipeline_cards_keep_real_actions_visible_and_reachable() -> None:
    page = (TEMPLATE_ROOT / "pipeline.html").read_text(encoding="utf-8")
    card = (TEMPLATE_ROOT / "_kanban_card.html").read_text(encoding="utf-8")

    assert 'class="card-primary-hit"' in card
    assert 'href="{{ primary_url }}"' in card
    assert ".kcard .card-primary-hit::after" in page
    assert ".kcard .card-primary-hit:focus-visible::after" in page
    assert "准备投递" in card
    assert "历史线索 · 未进入当前选择集" in card
    assert "opacity: 0" not in page


def test_application_results_are_outside_terminal_transition_guard() -> None:
    card = (TEMPLATE_ROOT / "_kanban_card.html").read_text(encoding="utf-8")

    resources = card.index("pipeline_application_resources.get")
    apply_pack = card.index('data-resource="apply-pack"')
    interview = card.index('data-resource="interview"')
    terminal_guard = card.index("{% if not is_terminal and card.application_id is not none %}")

    assert resources < apply_pack < terminal_guard
    assert resources < interview < terminal_guard
    assert "查看投递材料" in card
    assert "真实面经问答" in card
    assert 'aria-label="更新投递状态"' in card
