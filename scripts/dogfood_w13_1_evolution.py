"""W13.1 evolution end-to-end dogfood with real ccvibe Claude.

Sets up: a SKILL with 12+ low-fitness signals → triggers evolve_skill via
the agent loop → verifies a shadow variant gets generated and persisted.

Then runs the gray-release cycle to promote the shadow → live.
"""

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
from offerguide.evolution.signals import record_critic_signal  # noqa: E402
from offerguide.llm import LLMClient  # noqa: E402
from offerguide.memory import Store  # noqa: E402
from offerguide.skills import SkillRuntime, discover_skills  # noqa: E402

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


def main():
    # Use a fresh DB so the test is reproducible
    db_path = Path("/tmp/offerguide_w13_1_evo.db")
    if db_path.exists():
        db_path.unlink()
    store = Store(db_path)
    store.init_schema()

    # 1. Seed: pretend score_match has been performing badly for a while.
    # Single connection to avoid SQLite "database is locked" from nested writes.
    print("[seed] inserting 14 low-quality critic signals on score_match v0.1.0...")
    with store.connect() as conn:
        run_ids: list[int] = []
        for i in range(14):
            cur = conn.execute(
                "INSERT INTO skill_runs(skill_name, skill_version, input_hash, "
                "  input_json, output_json, latency_ms) VALUES (?,?,?,?,?,?)",
                (
                    "score_match", "0.1.0", f"h{i}",
                    f'{{"job_text": "fake JD #{i} for AI agent role", "user_profile": "胡阳..."}}',
                    f'{{"probability": {0.3 + i*0.01:.2f}, "reasoning": "weak match"}}',
                    1000 + i * 50,
                ),
            )
            run_ids.append(int(cur.lastrowid or 0))
        # Insert evolution_signals in the same connection
        for i, rid in enumerate(run_ids):
            conn.execute(
                "INSERT INTO evolution_signals(skill_name, skill_version, "
                "  skill_run_id, signal_kind, signal_value, signal_weight, notes) "
                "VALUES (?,?,?,?,?,?,?)",
                ("score_match", "0.1.0", rid, "critic",
                 0.3 + (i % 5) * 0.05, 1.0, f"seed run {i}"),
            )
        # Live variant pointer so detect_evolution_candidates picks it up
        conn.execute(
            "INSERT INTO skill_variants(skill_name, version, body_md, status, "
            "  promoted_at) VALUES (?,?,?,?, julianday('now') - 30)",
            ("score_match", "0.1.0", "(seed body, will be replaced from disk)", "live"),
        )
    # Verify the seed actually landed
    with store.connect() as conn:
        n_signals = conn.execute(
            "SELECT COUNT(*) FROM evolution_signals WHERE skill_name='score_match'"
        ).fetchone()[0]
        n_variants = conn.execute(
            "SELECT COUNT(*) FROM skill_variants WHERE skill_name='score_match'"
        ).fetchone()[0]
    print(f"[seed] done — {n_signals} signals, {n_variants} variants in DB")

    # 2. Set up real LLM
    api_key = os.environ.get("OFFERGUIDE_LLM_API_KEY")
    base_url = os.environ.get("OFFERGUIDE_LLM_BASE_URL")
    model = os.environ.get("OFFERGUIDE_LLM_MODEL")
    if not api_key:
        print("[err] OFFERGUIDE_LLM_API_KEY not set")
        sys.exit(1)
    print(f"[llm] {base_url} | {model}")
    llm = LLMClient(api_key=api_key, base_url=base_url, default_model=model)
    skills = discover_skills(SKILLS_ROOT)
    runtime = SkillRuntime(llm=llm, store=store)

    agent = AgentLoop(
        llm=llm, runtime=runtime, store=store, skills=skills,
        max_iterations=6, critic_enabled=True,
    )

    # 3. Trigger the agent with an evolution-flavored goal
    goal = (
        "看 detect_evolution_candidates 找出该进化的 SKILL, "
        "然后给最差的那个 evolve_skill 生成 2 个变种 (num_variants=2)。"
        "完事后用 1-2 句话告诉我做了啥。"
    )
    print(f"\n{'=' * 70}\nGOAL: {goal}\n{'=' * 70}\n")

    def on_event(ev):
        kind = ev.get("kind")
        if kind == "thinking":
            text = ev.get("text", "")
            if text:
                print(f"[think] {text[:180]}...")
        elif kind == "tool_call":
            args = ev.get("arguments", {})
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())[:80]
            print(f"[call ] {ev.get('name')}({args_str})")
        elif kind == "tool_result":
            preview = ev.get("result_preview", "")[:200].replace("\n", " | ")
            print(f"[ret  ] {preview}")
        elif kind == "final":
            print(f"\n[final] {ev.get('text')}")
        elif kind == "critique":
            print(f"\n[critic] {ev.get('score')} — {ev.get('notes', '')[:120]}")

    result = agent.run(goal=goal, trigger_kind="dogfood_evolution", on_event=on_event)

    # 4. Verify a shadow variant was created
    print(f"\n{'=' * 70}\nVERIFICATION\n{'=' * 70}")
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT version, status, parent_version, length(body_md) AS bodylen "
            "FROM skill_variants WHERE skill_name='score_match' "
            "ORDER BY created_at"
        ).fetchall()
    print("score_match variants in DB:")
    for v, st, pv, bl in rows:
        print(f"  - v{v} (parent={pv}) status={st} body={bl}字")

    shadow_count = sum(1 for _, st, _, _ in rows if st == "shadow")
    print(f"\nShadow variants generated: {shadow_count}")
    print(f"agent run: id={result.run_id} iters={result.iterations} "
          f"latency={result.latency_ms/1000:.1f}s critic={result.critic_score}")

    # 5. Run gray-release once: shadow → canary
    if shadow_count > 0:
        print("\n--- Running gray-release cycle ---")
        from offerguide.evolution.release import run_release_cycle
        cycle = run_release_cycle(store)
        print(cycle.render_summary())

        # Now the shadow should be canary
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT version, status, canary_traffic_pct FROM skill_variants "
                "WHERE skill_name='score_match' ORDER BY created_at"
            ).fetchall()
        print("\nAfter release cycle:")
        for v, st, pct in rows:
            print(f"  - v{v} status={st}" + (f" ({pct:.0%} canary)" if pct else ""))


if __name__ == "__main__":
    main()
