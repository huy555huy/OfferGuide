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

## Career-Ops — STAR+Reflection story bank + cover letter framework (W8 +2)

**[Career-Ops](https://github.com/santifer/career-ops)** (MIT, by santifer,
13.3k★) — an open-source Claude-Code-driven job-hunt agent. We adopt two
high-leverage patterns:

1. **STAR + Reflection story bank** — Career-Ops' insight that 5-10 master
   behavioral narratives accumulated across interview evaluations beats
   regenerating answers each time. We implement this in
   ``src/offerguide/story_bank.py`` with the ``behavioral_stories`` SQLite
   table; ``prepare_interview`` and ``deep_project_prep`` SKILLs query by
   tag at retrieval time. The ``+R`` (Reflection) extension to STAR is
   directly from Career-Ops' methodology.

2. **6-block evaluation framework for cover letters** — Career-Ops grades
   cover letters across opening_hook / narrative / customization / ATS-keyword
   density / closing-CTA / personalization-score. We adopt the six-block
   schema in ``src/offerguide/skills/write_cover_letter/helpers.py``
   (``CoverLetterResult`` Pydantic model). The ``ai_risk_warnings`` self-audit
   field is our addition addressing the 49% AI-detection auto-dismiss rate.

We do **not** vendor Career-Ops code; we adopt patterns and cite per its MIT
license attribution requirement.

## Resume-Matcher — cover letter + multi-provider LLM inspiration (W8 +2)

**[Resume-Matcher](https://github.com/srbhr/Resume-Matcher)** (Apache 2.0,
by srbhr) — open-source ATS resume matcher. We adopt:

- **Cover letter generation as a primary feature** — confirmed our decision
  to add ``write_cover_letter`` SKILL.
- **Multi-provider LLM (LiteLLM)** — validates our DeepSeek-V4 default;
  the abstraction we'd swap to if multi-provider became important.

No code vendored. Pattern + product-decision references only.

## Pytai / GPTInterviewer — interview question dynamics (W8 +2)

**[Pytai](https://github.com/getFrontend/app-ai-interviews)** and
**[GPTInterviewer](https://github.com/jiatastic/GPTInterviewer)** (MIT) —
open-source AI mock interview platforms. We adopt the pattern of real-time
question generation conditioned on role + tech stack (validates our
``prepare_interview`` / ``deep_project_prep`` contracts) and comprehensive
post-interview feedback (inspires a future ``post_interview_reflection``
SKILL — W9 candidate).

Reference / pattern-validation only — no code adopted.

## mem0 — memory layer architecture validation

**[mem0](https://github.com/mem0ai/mem0)** (Apache 2.0, by mem0ai) — universal
memory layer for AI agents. Our ``company_briefs`` + ``briefs.refresh_brief``
LLM-driven synthesis follows mem0's "agent reads observations → produces
structured memory → upserts" pipeline. Reference paper:
[Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)
by Chhikara et al.

We don't import mem0 (single-user case doesn't justify their multi-tenant
infra), but their architectural shape inspired our
``effective_app_limit`` override-when-confident pattern.

## Pytai / GPTInterviewer — post-interview transcript analysis (W8 +3)

**[Pytai](https://github.com/getFrontend/app-ai-interviews)** and
**[GPTInterviewer](https://github.com/jiatastic/GPTInterviewer)** (MIT) —
open-source AI mock interview platforms whose **transcript-based feedback
loop** we adopt for the seventh SKILL ``post_interview_reflection``:

The user submits a free-form transcript of an actual interview + the
prep run id from before. The SKILL outputs:
- ``hit_rate`` of agent's predictions vs reality
- ``surprises`` (questions asked but not predicted, with honest
  ``why_we_missed`` analysis)
- ``suggested_stories`` (STAR moments in the transcript worth adding
  to ``behavioral_stories``)
- ``brief_delta`` (proposed update to ``company_briefs.interview_style``)
- ``weak_spots_to_practice``

When the user opts in via UI checkboxes, the suggested stories
auto-insert into the story bank and the brief delta auto-applies. This
is the **dogfood feedback loop** that closes the agent's evolution —
without it, predictions never get audited.

Our addition over Pytai: the ``calibration_score()`` helper computes
mean abs error between ``predicted_likelihood`` and 1-if-matched-else-0
across all matched_predictions, surfacing prompt miscalibration as a
direct GEPA optimization target.

## Career-Ops PDF / Playwright export (W8 +3)

We adopt Career-Ops' "render HTML with print-optimized CSS, let user
⌘P → Save as PDF" pattern in ``src/offerguide/ui/templates/cover_letter_print.html``
+ ``GET /cover-letter/{run_id}.html`` route. We skip the Playwright
dependency: a standalone HTML page with ``@page`` rules and a
``no-print`` audit panel hits the same UX without the 200MB Chromium
download. Career-Ops' typography choice (Source Serif 4 for headers
+ system sans for body) carried over directly.

## namewyf/Campus2026 — 校招清单数据源 (W10)

**[namewyf/Campus2026](https://github.com/namewyf/Campus2026)** (393 ⭐) —
社区维护的 2026 届校招 + 实习信息汇总仓库。``AwesomeJobsSpider``
(``src/offerguide/spiders/awesome_jobs.py``) 默认抓这个 README 的 markdown
表格，把每一行（公司 · 投递入口 · 更新日期 · 地点 · 备注）作为一个候选
``RawJob`` 入库。

为什么这是合法且最佳的做法：

1. 公开 raw 内容 —— ``raw.githubusercontent.com`` 无 cookie / API key、无反爬压力
2. 维护者期望被消费 —— awesome-style 列表的存在意义就是被引用，仓库 README
   明确写着「欢迎共创、欢迎引用」
3. Git 有版本历史 —— 失效 JD 会被维护者下架，我们每天定时拉就是「跟着维护者的节奏」

我们只**消费**这个仓库；不 fork、不 vendor、不 mirror 内容。每条入库的
``RawJob.extras['github_repo']`` 都保留 ``namewyf/Campus2026`` 作为来源。

## ccvibe.cc — Claude API 网关（开发期使用）

**[ccvibe.cc](https://ccvibe.cc)** —— 第三方 OpenAI-compatible API 网关，
代理 Claude 模型 (Opus 4.5/4.6/4.7、Sonnet 4.5/4.6、Haiku 4.5)。开发期
默认走这个端点是出于成本与可达性考虑；``offerguide.config.Settings``
也支持原生 DeepSeek、OpenAI、OpenRouter 等任意 OpenAI-compat 端点。
``llm.client._normalize_base_url`` 会自动处理 ``/v1`` 路径补全，所以用户
只需在 ``.env`` 写 ``BASE_URL=https://my-proxy``、``TOKEN=sk-...`` 就可以。
