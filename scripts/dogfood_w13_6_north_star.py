"""W13.6 dogfood — agent reasons against north star goal + writes self-observation.

Seeds: an off-track goal (45 days left, 1 active app, no offers).
Expectation: agent's snapshot includes 🎯 North Star section. Agent's final
should reference the goal explicitly + suggest concrete actions to recover.
Then we check if agent considered using meta_reflect (this is harder to test;
agent only uses meta_reflect when it sees a true behavior pattern across runs)."""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
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

from offerguide import goals as _goals  # noqa: E402
from offerguide.agent import AgentLoop  # noqa: E402
from offerguide.llm import LLMClient  # noqa: E402
from offerguide.memory import Store  # noqa: E402
from offerguide.profile import load_resume_pdf  # noqa: E402
from offerguide.skills import SkillRuntime, discover_skills  # noqa: E402

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


def seed(store: Store) -> None:
    # 1. Set a real north star
    _goals.add_goal(
        store,
        title="2026 暑期拿到 1 个 AI Agent 实习 offer",
        target_date=date.today() + timedelta(days=45),
        target_metric="1 offer",
        description="字节/腾讯/小红书/MiniMax 优先; 不投纯算法刷题型 OD; 期望地点北京或上海",
    )
    # 2. Some state — 1 in-flight app, 0 offers (off track)
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, source_id, url, title, company, "
            "  raw_text, content_hash) VALUES (?,?,?,?,?,?,?)",
            ("manual", "x1", "https://example.com",
             "AI Agent 后端实习", "字节跳动",
             "完整 JD" * 100, "h1"),
        )
        conn.execute(
            "INSERT INTO applications(job_id, status, applied_at, last_status_change) "
            "VALUES (1, 'applied', julianday('now') - 5, julianday('now') - 5)"
        )
        # Pre-existing self-observation (simulate agent has reflected once before)
        conn.execute(
            "INSERT INTO agent_self_observations(observation, pattern_kind) "
            "VALUES ('用户深夜很少回应建议, 优先建议放在白天', 'tone')"
        )
        # user_facts hints
        conn.execute(
            "INSERT INTO user_facts(fact_text, kind, confidence, used_count) "
            "VALUES ('用户偏好 AI Agent / LLM 应用方向', 'preference', 0.95, 5)"
        )
    print("[seed] 1 active goal, 1 active app (off track), 1 prior self-observation")


def main():
    db_path = "/tmp/offerguide_w13_6_dogfood.db"
    if Path(db_path).exists():
        Path(db_path).unlink()
    store = Store(db_path)
    store.init_schema()
    seed(store)

    api_key = os.environ.get("OFFERGUIDE_LLM_API_KEY")
    base_url = os.environ.get("OFFERGUIDE_LLM_BASE_URL")
    model = os.environ.get("OFFERGUIDE_LLM_MODEL")

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

    goal_text = (
        "你被定时唤醒。看 snapshot 顶部的 north star + 当前进度, 判断:\n"
        "1) 用户朝目标走得怎么样\n"
        "2) 当下最该做什么 (或不做什么) 才能让用户离 target_date 近一步\n"
        "3) 你过去的 self_observations 有没有相关教训要遵守\n"
        "做你认为对的, 给一段诚实的现状评估。"
    )
    print("=" * 70)
    print("GOAL:", goal_text)
    print("=" * 70)

    def on_event(ev):
        kind = ev.get("kind")
        if kind == "thinking":
            text = ev.get("text", "")
            if text:
                print(f"[think] {text[:250]}")
        elif kind == "tool_call":
            args = ev.get("arguments", {})
            args_str = ", ".join(f"{k}={str(v)[:60]}" for k, v in args.items())[:120]
            print(f"[call ] {ev.get('name')}({args_str})")
        elif kind == "tool_result":
            preview = ev.get("result_preview", "")[:200].replace("\n", " | ")
            print(f"[ret  ] {preview}")
        elif kind == "final":
            print(f"\n[FINAL]\n{ev.get('text')}")
        elif kind == "critique":
            print(f"\n[critic] {ev.get('score')} — {ev.get('notes', '')[:300]}")

    result = agent.run(goal=goal_text, trigger_kind="dogfood_w13_6", on_event=on_event)

    # Inspect resulting state
    print(f"\n{'=' * 70}\nDB STATE AFTER\n{'=' * 70}")
    new_obs = _goals.list_active_self_observations(store)
    print(f"self_observations now: {len(new_obs)} (was 1)")
    for o in new_obs:
        print(f"  - [{o.pattern_kind}] {o.observation}")

    print(f"\nagent_run: id={result.run_id} iters={result.iterations} "
          f"latency={result.latency_ms/1000:.1f}s critic={result.critic_score}")


if __name__ == "__main__":
    main()
