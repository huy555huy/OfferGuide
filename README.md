# OfferGuide

**国内校招 Ambient 求职 Copilot + GEPA-style Self-Evolution.**

> Status: **early development (W1)** — scaffold + SKILL loader + memory + profile + agent skeleton landing.

## Why exists

Existing job-hunt agents (AIHawk, ApplyPilot, get_jobs) all chase **automated submission**.
But the data says automated submission is a dead end:

- **97%** of companies use AI-driven ATS to filter applicants
  ([source](https://boterview.com/a/ai-recruitment-statistics))
- **49%** auto-dismiss resumes suspected of being AI-written
  ([source](https://www.gettailor.ai/blog/ai-resume-detection))
- LazyApply Trustpilot rating: **2.1 stars**, 52% give the lowest score; one user reports
  submitting 14,000 applications and receiving only "hundreds of skills-mismatch rejections"
  ([source](https://www.trustpilot.com/review/lazyapply.com))

OfferGuide takes the **opposite** position: don't click apply. Instead, do the things that
actually move reply rate — quality matching, targeted resume gap-tailoring (not full rewrites),
post-application tracking, interview prep — and use the user's own dogfood data to
**self-evolve** the agent's SKILL prompts via GEPA.

## How (target architecture)

| Layer | What it does |
|---|---|
| **Conversational Agent** | Single LangGraph agent — Anthropic-aligned, [start simple](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them) |
| **SKILLs** | YAML-frontmatter + markdown content + helper scripts (Hermes-pattern). 5 SKILLs: `score_match` / `analyze_gaps` / `prepare_interview` / `update_status` / `query_history` |
| **Background Workers** | `Scout` (牛客 sitemap + browser-extension push) and `Tracker` (state machine + reminders) |
| **Memory** | SQLite + sqlite-vec, local-first |
| **Self-Evolution** | DSPy [GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026 Oral) reflectively evolves the 3 ★ SKILLs from dogfood reply-rate signal |
| **Notify** | Feishu webhook + Telegram bot, runtime-selectable |
| **Inbox UI** | FastAPI + HTMX (HITL: approve / reject / edit before any action lands) |

## Roadmap

- [x] **W1** — scaffold + SKILL.md loader + memory + profile + agent skeleton  ← **you are here**
- [ ] **W2** — Scout v1 (牛客 sitemap, manual paste) + `score_match` SKILL v1
- [ ] **W3** — `analyze_gaps` SKILL
- [ ] **W4** — Conversational agent + Inbox + Feishu/Telegram dual-rail notify
- [ ] **W5** — `prepare_interview` SKILL + DSPy GEPA wiring
- [ ] **W6** — Tracker worker + Boss browser-extension v1 + first GEPA evolution of `score_match`
- [ ] **W7** — GEPA evolution of `analyze_gaps` + `prepare_interview`
- [ ] **W8** — Pre/post-evolution diff report + dogfood numbers + 30s demo + blog

## Quick start (W1)

```bash
pip install -e ".[dev]"
python examples/quickstart.py /path/to/your_resume.pdf
```

You'll see your resume parsed into a Profile, a SKILL loaded from disk, the SQLite + sqlite-vec
store initialized, and a stub LangGraph agent run.

## License

MIT. See also [ATTRIBUTION.md](ATTRIBUTION.md) — OfferGuide borrows the SKILL.md design from
Hermes Agent and uses DSPy GEPA, both MIT.
