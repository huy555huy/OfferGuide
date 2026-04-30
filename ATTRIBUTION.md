# Attribution

OfferGuide stands on the shoulders of several MIT-licensed open source projects.
This file records what we borrowed and why — both as required by the MIT licenses
of upstream projects and as a record of design lineage.

## Design pattern: SKILL.md format

Borrowed from **[Hermes Agent](https://github.com/nousresearch/hermes-agent)** by Nous Research
(95.6k stars, ICLR 2026 Oral, MIT). The YAML-frontmatter-plus-markdown skill format
(`name / description / version / triggers / ...`) is a near-direct adaptation of
Hermes' skill system. We do *not* depend on the Hermes runtime — we re-implemented
a slim loader (`src/offerguide/skills/_loader.py`) that parses the same on-disk shape.

## Self-evolution algorithm: GEPA

Used **[DSPy GEPA](https://dspy.ai/api/optimizers/GEPA/overview/)** (MIT) — the
production implementation of *Genetic-Pareto Prompt Evolution* (Agrawal et al., 2025;
[arXiv:2507.19457](https://arxiv.org/abs/2507.19457); ICLR 2026 Oral). Imported as
a dependency; not vendored. The reflective evolutionary search drives our self-improving
SKILL prompts.

## Agent orchestration

**[LangGraph](https://github.com/langchain-ai/langgraph)** (MIT) for the single-agent
state machine. The ambient + HITL pattern is informed by LangChain's
[agents-from-scratch](https://github.com/langchain-ai/agents-from-scratch) reference.

## Vector storage

**[sqlite-vec](https://github.com/asg017/sqlite-vec)** (Apache 2.0) — chosen over
Qdrant because OfferGuide is single-user local-first, and sqlite-vec runs anywhere
SQLite does with zero deployment.

## PDF parsing

**[pypdf](https://github.com/py-pdf/pypdf)** (BSD-3-Clause) for extracting text
from resume PDFs.

## Autonomous scheduler (W8 +1)

**[APScheduler](https://github.com/agronholm/apscheduler)** (MIT, by Alex Grönholm)
provides cron-style triggers + misfire-grace-time + max-instances semantics for
the daily / weekly autonomous job loop in `src/offerguide/autonomous/scheduler.py`.
Imported as a dependency (`scheduling` extra); not vendored.

## Agent loop architecture (inspiration)

The observation-action-feedback structure of our autonomous jobs is inspired by:

- **[OpenHands](https://github.com/All-Hands-AI/OpenHands)** (MIT, by All Hands AI) —
  the formerly-named OpenDevin project. We adapt the observation-action-feedback
  loop pattern; we do not import or vendor any OpenHands code. Their event-driven
  state machine inspired our job-context + idempotent-job approach.
- **[LangChain AgentExecutor](https://github.com/langchain-ai/langchain)** (MIT) —
  the tool-budgeted iteration shape: each agent run has a fixed budget of tool
  calls, and the agent self-terminates when the goal is reached or the budget
  is exhausted. We borrow the *shape*, not the code, in our scheduled-job
  per-run budgets (`MAX_COMPANIES_PER_RUN` etc.).
- **[AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)** (MIT) — the
  task-decomposition idea: a high-level goal ("refresh briefs") breaks into
  per-company atomic refreshes that each succeed or fail independently. Our
  `brief_update` job uses this shape directly.

None of these projects' code lives in our tree. We borrow design patterns + cite
each here per their MIT licenses' attribution requirements.

## Web search backend

**[DuckDuckGo HTML SERP](https://html.duckduckgo.com)** (no license needed for
HTML scraping, but ToS-fragile) is the no-API-key default in
`src/offerguide/agentic/search.py`. Production deployments should swap in
[Tavily](https://tavily.com), Google Custom Search, or another commercial
search API. The ``SearchBackend`` Protocol abstraction means switching
backends is a one-class change.
