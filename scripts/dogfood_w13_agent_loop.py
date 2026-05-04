"""W13 dogfood — run the central agent loop against real ccvibe Claude.

Run from repo root with .env loaded:

    OFFERGUIDE_LLM_API_KEY=... OFFERGUIDE_LLM_BASE_URL=... \\
    OFFERGUIDE_LLM_MODEL=... python scripts/dogfood_w13_agent_loop.py

Pass --use-real-db to use ~/.offerguide/store.db (if the user's already-
populated DB exists), otherwise creates a fresh DB in /tmp with a few
seeded rows.

Prints trajectory + final + critic score so you can eyeball whether the
agent actually exhibited "model in main position" behavior, not just
called tools mechanically.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env if present (dotenv may not be installed, fall back to manual parse)
ENV_FILE = Path(__file__).parent.parent / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)

from offerguide.agent import AgentLoop  # noqa: E402
from offerguide.llm import LLMClient  # noqa: E402
from offerguide.memory import Store  # noqa: E402
from offerguide.profile import load_resume_pdf  # noqa: E402
from offerguide.skills import SkillRuntime, discover_skills  # noqa: E402

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


def seed_minimal_store(store: Store) -> None:
    """Insert a few real-shape rows so the snapshot has something to show."""
    with store.connect() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if existing > 0:
            print(f"[seed] DB already has {existing} jobs; not seeding")
            return
        # 2 jobs, one short (filtered out of snapshot) one full JD
        full_jd = (
            "字节跳动 · AI Lab · AI Agent 后端实习生\n"
            "工作地点: 北京 · 海淀\n\n"
            "岗位要求:\n"
            "1. 熟悉 Python, 有 LangGraph / DSPy 等 agent 框架使用经验优先\n"
            "2. 熟悉 LLM 应用开发, 包括 prompt engineering / RAG / tool calling\n"
            "3. 有大模型应用项目落地经验, 能从 0 到 1 设计 agent state machine\n"
            "4. 数学统计基础扎实, 能阅读论文 (DSPy, GEPA, Constitutional AI 等)\n"
            "5. 在校生硕士 / 博士, 2026/27 届优先\n\n"
            "工作内容: 参与字节内部 AI Agent 平台核心模块开发, 包括 agent runtime、"
            "tool registry、长期记忆 (mem0 / vector store) 等子系统。"
        )
        conn.execute(
            "INSERT INTO jobs(source, source_id, url, title, company, location, "
            "raw_text, content_hash) VALUES (?,?,?,?,?,?,?,?)",
            ("manual", "test1", "https://example.com/job1",
             "AI Agent 后端实习生", "字节跳动", "北京",
             full_jd, "h_test1"),
        )
        # Add a user_fact to prove the agent reads it
        conn.execute(
            "INSERT INTO user_facts(fact_text, kind, confidence, used_count) "
            "VALUES (?,?,?,?)",
            ("胡阳的 RemeDi 项目用过 LangGraph + 双层 agent 架构", "experience", 0.95, 3),
        )
        conn.execute(
            "INSERT INTO user_facts(fact_text, kind, confidence, used_count) "
            "VALUES (?,?,?,?)",
            ("胡阳偏好 AI Agent / LLM 应用方向, 不投纯算法刷题岗", "preference", 0.9, 5),
        )
    print("[seed] inserted 1 job + 2 user_facts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-real-db", action="store_true",
                        help="Use ~/.offerguide/store.db if it exists")
    parser.add_argument("--goal", default=(
        "检查当前系统状态, 找出最近一个还没 score_match 评分过的 job, "
        "给它跑 score_match. 然后用 1-3 句话告诉我结果如何。"
    ))
    parser.add_argument("--max-iter", type=int, default=5)
    args = parser.parse_args()

    if args.use_real_db:
        db_path = Path.home() / ".offerguide/store.db"
        print(f"[db] using real DB: {db_path}")
    else:
        db_path = Path("/tmp/offerguide_w13_dogfood.db")
        if db_path.exists():
            db_path.unlink()
        print(f"[db] using fresh DB: {db_path}")

    store = Store(db_path)
    store.init_schema()
    if not args.use_real_db:
        seed_minimal_store(store)

    api_key = os.environ.get("OFFERGUIDE_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("OFFERGUIDE_LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL")
    model = os.environ.get("OFFERGUIDE_LLM_MODEL") or os.environ.get("OFFERGUIDE_DEFAULT_MODEL")
    if not api_key:
        print("[err] OFFERGUIDE_LLM_API_KEY not set; aborting")
        sys.exit(1)
    print(f"[llm] base={base_url} model={model}")

    llm = LLMClient(api_key=api_key, base_url=base_url, default_model=model)
    skills = discover_skills(SKILLS_ROOT)
    print(f"[skills] discovered {len(skills)} SKILLs (will be exposed as tools)")
    runtime = SkillRuntime(llm=llm, store=store)

    # Load user's master resume so read_user_resume tool works
    resume_path = os.environ.get("OFFERGUIDE_RESUME_PDF")
    master_resume = ""
    if resume_path and Path(resume_path).exists():
        try:
            profile = load_resume_pdf(Path(resume_path))
            master_resume = profile.raw_resume_text
            print(f"[resume] loaded {len(master_resume)} chars from {resume_path}")
        except Exception as e:
            print(f"[resume] WARN: failed to load {resume_path}: {e}")
    else:
        print(f"[resume] WARN: OFFERGUIDE_RESUME_PDF not set or missing")

    agent = AgentLoop(
        llm=llm, runtime=runtime, store=store, skills=skills,
        master_resume_text=master_resume,
        max_iterations=args.max_iter, critic_enabled=True,
    )

    print(f"\n{'=' * 70}")
    print(f"GOAL: {args.goal}")
    print(f"{'=' * 70}\n")

    # Stream events to stdout as they happen
    def on_event(ev):
        kind = ev.get("kind")
        if kind == "state_snapshot":
            snapshot_lines = ev.get("snapshot", "").split("\n")
            print(f"[{kind}] {len(snapshot_lines)} 行 snapshot")
        elif kind == "thinking":
            text = ev.get("text", "")
            tools = ev.get("tool_call_names", [])
            preview = (text[:160] + "…") if len(text) > 160 else text
            print(f"[{kind}] iter={ev.get('iteration')} preamble={preview!r} tools={tools}")
        elif kind == "tool_call":
            print(f"[{kind}] {ev.get('name')}({list(ev.get('arguments', {}).keys())})")
        elif kind == "tool_result":
            preview = ev.get("result_preview", "")[:120]
            print(f"[{kind}] {ev.get('name')}: {preview}…  (full {ev.get('result_full_len')} chars)")
        elif kind == "critique":
            print(f"[{kind}] score={ev.get('score')} notes={ev.get('notes', '')[:120]}")
        elif kind == "final":
            print(f"[{kind}] {ev.get('text')}")
        elif kind == "error":
            print(f"[{kind}] ⚠ {ev.get('message')}")

    result = agent.run(
        goal=args.goal,
        trigger_kind="dogfood_w13",
        on_event=on_event,
    )

    print(f"\n{'=' * 70}")
    print("RESULT SUMMARY")
    print(f"{'=' * 70}")
    print(f"  run_id:        {result.run_id}")
    print(f"  iterations:    {result.iterations}")
    print(f"  latency:       {result.latency_ms / 1000:.2f}s")
    print(f"  critic_score:  {result.critic_score}")
    print(f"  critic_notes:  {result.critic_notes}")
    print(f"  error:         {result.error}")
    print(f"\n  final_answer:")
    for line in (result.final_answer or "").splitlines():
        print(f"    {line}")

    print(f"\n  trajectory ({len(result.events)} events):")
    for ev in result.events:
        print(f"    - {ev.kind:20s}  payload_keys={list(ev.payload.keys())}")

    # Verify persistence
    print(f"\n  DB verification (agent_runs#{result.run_id}):")
    with store.connect() as conn:
        row = conn.execute(
            "SELECT trigger_kind, goal, status, iterations, critic_score, "
            "       length(trajectory_json), latency_ms "
            "FROM agent_runs WHERE id = ?", (result.run_id,),
        ).fetchone()
        if row:
            print(f"    persisted: trigger={row[0]} status={row[2]} "
                  f"iters={row[3]} critic={row[4]} traj={row[5]}字 latency={row[6]}ms")
        else:
            print("    ⚠ NOT PERSISTED")


if __name__ == "__main__":
    main()
