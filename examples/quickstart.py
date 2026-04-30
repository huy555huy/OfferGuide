"""End-to-end quickstart — exercises every layer through W8.

Usage::

    pip install -e ".[dev,evolution]"
    python examples/quickstart.py /path/to/resume.pdf

Optional flags::

    --crawl-nowcoder N     Pull N real JDs from nowcoder.com via the public
                           sitemap (rate-limited 1 req/s). Default: skipped.
    --invoke-skills        Actually call the LLM for the 3 evolvable SKILLs on
                           the canned JD + this resume. Requires DEEPSEEK_API_KEY.
                           Without this flag, prints the rendered prompts.
    --invoke-agent         Build the LangGraph agent, run with action='everything',
                           print the rendered Chinese summary. Implies --invoke-skills.

After the quickstart runs you have a working ``.offerguide/store.db`` ready
for::

    python -m offerguide.ui.web                          # FastAPI + HTMX UI
    python -m offerguide.workers tracker run             # silence-detection sweep
    python -m offerguide.workers scout nowcoder --limit 10
    python -m offerguide.evolution evolve score_match    # GEPA evolution
    python -m offerguide.evolution diff   score_match    # before/after report

Without any flags it ingests one canned JD via the manual-paste path so the
quickstart runs offline (no API calls).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import offerguide
from offerguide.llm import LLMClient, LLMError
from offerguide.memory.vec import attach_vec, vec_version
from offerguide.platforms import manual
from offerguide.skills import SkillRuntime, SkillSpec
from offerguide.skills._runtime import _render_inputs
from offerguide.workers import scout

DEMO_JD = """前端实习生
负责中后台 H5 页面开发与维护，与设计师/产品经理协作，承担前端架构演进的小任务。

任职要求：
1. 计算机或相关专业本科及以上，2026 届实习
2. 熟练 React + TypeScript，理解 Webpack/Vite 构建流程
3. 关注 Web 性能与可访问性
4. 主动、有责任心
"""

# Order matters: score → gaps → prep. The agent respects the same order
# in 'everything' mode — score gates whether to invest in gap analysis,
# which gates whether to invest in interview prep.
SKILL_DEMO_ORDER = ("score_match", "analyze_gaps", "prepare_interview")

DEMO_COMPANY = "(demo company)"  # used by prepare_interview when invoked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OfferGuide quickstart (W1-W8)")
    parser.add_argument("resume_pdf", type=Path, help="Path to a resume PDF to parse")
    parser.add_argument("--db", type=Path, default=Path(".offerguide/store.db"))
    parser.add_argument(
        "--crawl-nowcoder",
        type=int,
        default=0,
        metavar="N",
        help="Crawl N real JDs from nowcoder (1 req/s). Default 0 = skipped.",
    )
    parser.add_argument(
        "--invoke-skills",
        action="store_true",
        help="Actually call the LLM for the demo SKILLs (needs DEEPSEEK_API_KEY).",
    )
    parser.add_argument(
        "--invoke-agent",
        action="store_true",
        help="Build LangGraph agent and run with action='everything'. Implies --invoke-skills.",
    )
    args = parser.parse_args(argv)

    print(f"== OfferGuide v{offerguide.__version__} quickstart ==\n")

    skills_root = Path(__file__).parent.parent / "src/offerguide/skills"
    skills = offerguide.discover_skills(skills_root)
    skills_by_name = {s.name: s for s in skills}
    print(f"[1/6] Loaded {len(skills)} SKILL(s):")
    for s in skills:
        print(f"      • {s.name} v{s.version} — {s.description[:60]}")

    store = offerguide.Store(args.db)
    store.init_schema()
    with store.connect() as conn:
        attach_vec(conn)
        v = vec_version(conn)
    print(f"\n[2/6] Memory at {store.db_path}; sqlite-vec={v}")

    profile = offerguide.load_resume_pdf(args.resume_pdf)
    print(
        f"\n[3/6] Resume parsed: {len(profile.raw_resume_text)} chars "
        f"from {args.resume_pdf.name}"
    )

    print("\n[4/6] Scout:")
    rj = manual.from_text(DEMO_JD, company="(demo company)", location="Shanghai")
    was_new, jid = scout.ingest(store, rj)
    print(f"      manual JD → was_new={was_new}, job_id={jid}")
    if args.crawl_nowcoder > 0:
        print(f"      crawling {args.crawl_nowcoder} real JDs from nowcoder...")
        counters = scout.crawl_nowcoder(store, limit=args.crawl_nowcoder)
        print(f"      {counters}")
    else:
        print("      nowcoder crawl skipped (--crawl-nowcoder 0)")

    invoke_skills = args.invoke_skills or args.invoke_agent
    print("\n[5/6] SKILL runtime:")
    base_inputs = {
        "job_text": rj.raw_text,
        "user_profile": profile.raw_resume_text,
    }
    llm = LLMClient() if invoke_skills else None
    rt = SkillRuntime(llm, store) if llm else None  # type: ignore[arg-type]

    for skill_name in SKILL_DEMO_ORDER:
        spec = skills_by_name.get(skill_name)
        if spec is None:
            print(f"      • {skill_name}: SKILL not found, skipping")
            continue
        # prepare_interview needs extra inputs the others don't
        if skill_name == "prepare_interview":
            inputs = {**base_inputs, "company": DEMO_COMPANY, "past_experiences": ""}
        else:
            inputs = base_inputs
        print(f"\n      ── {skill_name} v{spec.version} ──")
        if invoke_skills and rt is not None:
            try:
                _invoke_and_print(rt, spec, inputs)
            except LLMError as e:
                print(f"      ⚠ LLM call failed: {e}")
                print("      (set DEEPSEEK_API_KEY to actually call the API)")
        else:
            _render_and_print(spec, inputs)

    print("\n[6/6] Agent (LangGraph dispatcher):")
    if args.invoke_agent and rt is not None:
        from offerguide.agent import build_graph

        graph = build_graph(skills=skills, runtime=rt, store=store)
        try:
            result = graph.invoke(
                {
                    "messages": [{"role": "user", "content": "evaluate this JD"}],
                    "requested_action": "everything",
                    "job_text": rj.raw_text,
                    "user_profile_text": profile.raw_resume_text,
                    "company": DEMO_COMPANY,
                }
            )
            response = result.get("final_response") or "(empty response)"
            print(_indent(response))
        except LLMError as e:
            print(f"      ⚠ Agent call failed: {e}")
    else:
        print("      (skipped — pass --invoke-agent to build & run the LangGraph)")
        print("      Routing topology:")
        print("        START → start_router → {score|gaps|prep|summarize}")
        print("        score → after_score_router → {gaps|summarize}")
        print("        gaps → after_gaps_router → {prep|summarize}")
        print("        prep → summarize → END")

    print(
        "\nAcceptance: ✓ scaffold + 3 SKILLs + memory + profile + Scout"
        " + agent + Inbox + Tracker + GEPA infra wired up."
    )
    return 0


def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _invoke_and_print(rt: SkillRuntime, spec: SkillSpec, inputs: dict) -> None:
    result = rt.invoke(spec, inputs)
    snippet = result.raw_text[:400] + ("..." if len(result.raw_text) > 400 else "")
    print(f"      output ({len(result.raw_text)} chars):")
    print("\n".join(f"        {line}" for line in snippet.splitlines()))
    print(f"      latency={result.latency_ms}ms, persisted skill_runs.id={result.skill_run_id}")


def _render_and_print(spec: SkillSpec, inputs: dict) -> None:
    rendered = _render_inputs(spec, inputs)
    head = "\n".join(rendered.splitlines()[:14])
    print("      (offline: showing the user message that WOULD be sent)")
    print("\n".join("      " + line for line in head.splitlines()))
    print("      … (truncated)")
    print(f"      system msg = {len(spec.body)} chars of SKILL body")


if __name__ == "__main__":
    sys.exit(main())
