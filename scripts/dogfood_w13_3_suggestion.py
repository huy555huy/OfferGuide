"""W13.3 dogfood — agent writes suggestion → simulated user approve → signal lands.

Seeds DB with: a high-score job that's not yet tailored.
Run agent. Agent should call write_suggestion (not auto-tailor).
Then simulate user approving and verify the user_thumbs signal lands."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ENV = Path(__file__).parent.parent / ".env"
if ENV.exists():
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from offerguide.agent import AgentLoop  # noqa: E402
from offerguide import inbox as inbox_mod  # noqa: E402
from offerguide.evolution.signals import fetch_signals  # noqa: E402
from offerguide.llm import LLMClient  # noqa: E402
from offerguide.memory import Store  # noqa: E402
from offerguide.profile import load_resume_pdf  # noqa: E402
from offerguide.skills import SkillRuntime, discover_skills  # noqa: E402

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


def seed(store: Store) -> None:
    with store.connect() as conn:
        # 1 well-formed JD that hasn't been scored or tailored yet
        full_jd = (
            "字节跳动 · AI Lab · LLM Agent 工程师实习\n\n"
            "工作地点: 北京·海淀\n\n"
            "岗位职责:\n"
            "1. 参与字节内部 AI Agent 平台核心模块开发\n"
            "2. 设计并实现 LLM agent runtime, 包括 tool registry / 长期记忆 / RAG\n"
            "3. 探索 GEPA / DSPy / Constitutional AI 等论文方法的工程化落地\n\n"
            "要求:\n"
            "1. 熟练 Python, 有 LangGraph 或类似 agent 框架使用经验\n"
            "2. 熟悉 LLM 应用开发: prompt engineering / RAG / tool calling\n"
            "3. 数学/统计基础, 能阅读论文实现\n"
            "4. 在校生硕士/博士, 2026/27 届优先\n"
        )
        conn.execute(
            "INSERT INTO jobs(source, source_id, url, title, company, location, "
            "  raw_text, content_hash) VALUES (?,?,?,?,?,?,?,?)",
            ("manual", "bytedance_agent", "https://example.com",
             "LLM Agent 工程师实习", "字节跳动", "北京",
             full_jd, "h1"),
        )
        # user_facts indicating user is interested
        conn.execute(
            "INSERT INTO user_facts(fact_text, kind, confidence, used_count) "
            "VALUES ('用户目标 AI Agent / LLM 应用方向, 字节是首选', 'preference', 0.95, 8)"
        )
        conn.execute(
            "INSERT INTO user_facts(fact_text, kind, confidence, used_count) "
            "VALUES ('用户已有 LangGraph + 双层 agent 架构实战经验 (Deep Research 项目)', 'experience', 0.9, 5)"
        )
    print("[seed] 1 high-relevance JD + 2 user_facts")


def main():
    db_path = "/tmp/offerguide_w13_3_dogfood.db"
    if Path(db_path).exists():
        Path(db_path).unlink()
    store = Store(db_path)
    store.init_schema()
    seed(store)

    api_key = os.environ.get("OFFERGUIDE_LLM_API_KEY")
    base_url = os.environ.get("OFFERGUIDE_LLM_BASE_URL")
    model = os.environ.get("OFFERGUIDE_LLM_MODEL")
    print(f"[llm] {base_url} | {model}\n")

    llm = LLMClient(api_key=api_key, base_url=base_url, default_model=model)
    skills = discover_skills(SKILLS_ROOT)
    runtime = SkillRuntime(llm=llm, store=store)

    resume_path = os.environ.get("OFFERGUIDE_RESUME_PDF")
    master_resume = ""
    if resume_path and Path(resume_path).exists():
        try:
            master_resume = load_resume_pdf(Path(resume_path)).raw_resume_text
        except Exception:
            pass

    agent = AgentLoop(
        llm=llm, runtime=runtime, store=store, skills=skills,
        master_resume_text=master_resume,
        max_iterations=6, critic_enabled=True,
    )

    goal = (
        "看 snapshot 有没有值得让用户知道的事。如果有, 调 write_suggestion "
        "把建议写到 inbox 让用户决定 (不要直接执行有副作用的 SKILL)。如果没有, 直接 final 收工。"
    )
    print("=" * 70)
    print(f"GOAL: {goal}")
    print("=" * 70)

    def on_event(ev):
        kind = ev.get("kind")
        if kind == "thinking":
            text = ev.get("text", "")
            if text:
                print(f"[think] {text[:200]}")
        elif kind == "tool_call":
            args = ev.get("arguments", {})
            args_str = ", ".join(f"{k}={str(v)[:50]}" for k, v in args.items())[:120]
            print(f"[call ] {ev.get('name')}({args_str})")
        elif kind == "tool_result":
            preview = ev.get("result_preview", "")[:200].replace("\n", " | ")
            print(f"[ret  ] {preview}")
        elif kind == "final":
            print(f"\n[final]\n{ev.get('text')}")
        elif kind == "critique":
            print(f"\n[critic] {ev.get('score')} — {ev.get('notes', '')[:200]}")

    result = agent.run(goal=goal, trigger_kind="dogfood_w13_3", on_event=on_event)

    # Verify what landed in inbox
    print(f"\n{'=' * 70}\nINBOX STATE\n{'=' * 70}")
    items = inbox_mod.list_items(store, status="pending")
    print(f"pending suggestions: {len(items)}")
    for it in items:
        print(f"\n  inbox#{it.id} [{it.kind}]")
        print(f"    title:  {it.title}")
        print(f"    body:   {it.body[:200] if it.body else ''}")
        print(f"    source: {it.source_skill_name} v{it.source_skill_version}, agent_run#{it.source_agent_run_id}")
        if it.proposed_action:
            print(f"    action: {it.proposed_action}")

    # Simulate user approval on the most recent agent_suggestion
    if items:
        first_suggestion = next(
            (i for i in items if i.kind == "agent_suggestion"), None,
        )
        if first_suggestion:
            print(f"\n--- Simulating user 👍 on inbox#{first_suggestion.id} ---")
            inbox_mod.decide(store, first_suggestion.id, decision="approved",
                             note="dogfood approve")

            sigs = fetch_signals(
                store, skill_name=first_suggestion.source_skill_name or "",
            )
            print(f"\nevolution_signals for {first_suggestion.source_skill_name}:")
            for s in sigs:
                print(f"  - {s.signal_kind}: value={s.signal_value} weight={s.signal_weight}")
                print(f"    notes: {s.notes}")

    print(f"\nagent_run summary: id={result.run_id} iters={result.iterations} "
          f"latency={result.latency_ms/1000:.1f}s critic={result.critic_score}")


if __name__ == "__main__":
    main()
