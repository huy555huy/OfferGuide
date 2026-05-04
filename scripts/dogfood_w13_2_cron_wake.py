"""W13.2 dogfood — real LLM cron-wake.

Seeds a DB with multiple maintenance "needs" (thin JDs, unclassified corpus,
silent applications), then triggers wake_agent ONCE and watches what the
agent picks to do.

Expected agent behavior: read snapshot → see maintenance hints showing
multiple pending items → pick the highest-ROI 1-3 to invoke → don't run
all 7 (that would be the old cron model)."""

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

from offerguide.autonomous.scheduler import build_agent_wake_scheduler  # noqa: E402
from offerguide.config import Settings  # noqa: E402
from offerguide.memory import Store  # noqa: E402


def seed_pending_maintenance(db_path: str) -> None:
    """Seed a DB with several conditions that should make agent pick maintenance."""
    s = Store(db_path)
    s.init_schema()
    with s.connect() as conn:
        # Condition 1: 4 thin JDs (need enrich)
        for i in range(4):
            conn.execute(
                "INSERT INTO jobs(source, source_id, url, title, company, "
                "  raw_text, content_hash) VALUES (?,?,?,?,?,?,?)",
                ("manual", f"thin{i}", "http://x", "AI Engineer", "字节跳动",
                 f"thin JD #{i} body 80 chars only", f"thin{i}"),
            )
        # Condition 2: 1 full JD with raw_text >= 200 (snapshot will list it)
        conn.execute(
            "INSERT INTO jobs(source, source_id, url, title, company, "
            "  raw_text, content_hash) VALUES (?,?,?,?,?,?,?)",
            ("manual", "full", "http://full", "AI Agent 后端", "字节跳动",
             ("AI Agent 后端实习 · 字节跳动 · 北京。"
              "要求: Python 熟练 + LangGraph 经验 + 大模型应用项目。" * 4),
             "fullj"),
        )
        # Condition 3: 2 unclassified corpus items
        for i in range(2):
            conn.execute(
                "INSERT INTO interview_experiences(company, raw_text, source, "
                "  content_hash) VALUES ('字节跳动', ?, 'manual', ?)",
                (f"面经 #{i} body" * 5, f"corpus{i}"),
            )
        # Condition 4: 1 silent application past 14 days
        conn.execute(
            "INSERT INTO applications(job_id, status, last_status_change) "
            "VALUES (1, 'applied', julianday('now') - 14)"
        )
        # Condition 5: a user_fact so the snapshot shows preferences
        conn.execute(
            "INSERT INTO user_facts(fact_text, kind, confidence, used_count) "
            "VALUES ('胡阳目标 AI Agent / LLM 应用岗', 'preference', 0.9, 5)"
        )
    print(f"[seed] DB at {db_path} seeded with: 4 thin JDs + 1 full JD + 2 unclassified corpus + 1 silent app")


def main():
    db_path = "/tmp/offerguide_w13_2_dogfood.db"
    if Path(db_path).exists():
        Path(db_path).unlink()
    seed_pending_maintenance(db_path)

    api_key = os.environ.get("OFFERGUIDE_LLM_API_KEY")
    base_url = os.environ.get("OFFERGUIDE_LLM_BASE_URL")
    model = os.environ.get("OFFERGUIDE_LLM_MODEL")
    if not api_key:
        print("[err] OFFERGUIDE_LLM_API_KEY not set"); sys.exit(1)

    settings = Settings(
        deepseek_api_key=api_key,
        deepseek_base_url=base_url,
        default_model=model,
        db_path=db_path,
        resume_pdf=os.environ.get("OFFERGUIDE_RESUME_PDF"),
    )

    print(f"\n[llm] {base_url} | {model}")
    print(f"[goal] cron-wake the agent and let it decide maintenance")
    print("=" * 70)

    sched = build_agent_wake_scheduler(settings=settings)
    print(f"jobs registered: {sched.list_jobs()}")
    print()

    result = sched.trigger_once("wake_agent")
    sched.shutdown()

    print(f"\n{'=' * 70}\nWAKE_AGENT RESULT\n{'=' * 70}")
    for k, v in (result or {}).items():
        if k == "final":
            print(f"  {k}:")
            for line in str(v).splitlines():
                print(f"    {line}")
        else:
            print(f"  {k}: {v}")

    # Inspect what changed in the DB
    print(f"\n{'=' * 70}\nWHAT THE AGENT DID (DB diff)\n{'=' * 70}")
    s = Store(db_path)
    with s.connect() as conn:
        n_thin_now = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE length(raw_text) < 200"
        ).fetchone()[0]
        n_unclass_now = conn.execute(
            "SELECT COUNT(*) FROM interview_experiences "
            "WHERE quality_classified_at IS NULL"
        ).fetchone()[0]
        n_silent_events = conn.execute(
            "SELECT COUNT(*) FROM application_events WHERE kind = 'silent_check'"
        ).fetchone()[0]
        n_skill_runs = conn.execute(
            "SELECT COUNT(*) FROM skill_runs"
        ).fetchone()[0]
        n_agent_runs = conn.execute(
            "SELECT COUNT(*) FROM agent_runs"
        ).fetchone()[0]

        # Show every tool the agent invoked from this run
        agent_run = conn.execute(
            "SELECT id, iterations, critic_score, latency_ms, trajectory_json "
            "FROM agent_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()

    print(f"  thin JDs remaining: {n_thin_now} (was 4)")
    print(f"  unclassified corpus remaining: {n_unclass_now} (was 2)")
    print(f"  silent_check events: {n_silent_events}")
    print(f"  total skill_runs after wake: {n_skill_runs}")
    print(f"  total agent_runs: {n_agent_runs}")

    if agent_run:
        rid, iters, critic, lat, traj_json = agent_run
        import json as _json
        traj = _json.loads(traj_json or "[]")
        print(f"\n  agent_run#{rid}: {iters} iters, critic={critic}, {lat/1000:.1f}s")
        print(f"  tool_calls in this trajectory:")
        for ev in traj:
            if ev.get("kind") == "tool_call":
                args = ev.get("payload", {}).get("arguments", {})
                args_str = ", ".join(f"{k}={v}" for k, v in args.items())[:80]
                print(f"    - {ev.get('payload', {}).get('name')}({args_str})")


if __name__ == "__main__":
    main()
