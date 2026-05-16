# OfferGuide Long-Running Harness Progress

## Current Goal

把 OfferGuide 优化成一个真实好用的求职 agent: 能围绕用户目标持续推进 search /
draft / track / prep，不自动替用户投递，不编造证据，并把过程和验证结果留在项目里。

项目级目标与过程记录: `docs/goal_real_agent_2026-05-16.md`。

## Done

- Added the Anthropic CWC long-running harness primitives to the project root:
  default-fail contract, evidence gate hooks, fresh evaluator, handoff file,
  kill switch, steer hook, and stop-time checkpoint hook.
- Aligned README wording so `src/offerguide/harness` is described as the
  application agent loop, while `.claude/` + `test-results.json` + `PROGRESS.md`
  are the Anthropic-style long-running harness primitives.
- Verified the harness-related slice with
  `pytest tests/test_anthropic_harness_primitives.py tests/test_project_vault.py tests/test_evidence_first_context.py`
  and recorded the output in `harness-related-tests-result.txt`.
- W21 Mission Control UI/test slice was completed before this goal pass; latest
  recorded full-suite result from that pass: `858 passed, 2 skipped`.
- 2026-05-16 goal pass:
  - Defined the “real useful agent” success criteria and non-goals in
    `docs/goal_real_agent_2026-05-16.md`.
  - Wired active goals, factual funnel progress, heuristic assessment labels,
    and active agent self-observations into the agent runtime system context.
  - Added regression coverage that active goal progress appears in the system
    prompt.
  - Updated worldview strategy/memory to reflect the product-level goal.
  - Verified with `uv run pytest -q`: `859 passed, 2 skipped`.
- 2026-05-16 UI connection pass:
  - Verified `/Users/huy/Downloads/offer.zip` was already copied into
    `src/offerguide/ui/static/redesign/*`; the issue was integration, not
    missing assets.
  - Replaced the old global top navigation with the redesign-style left rail,
    topbar, editorial navy/bone tokens, breadcrumb, and bracket-tag status
    language.
  - Reworked the visible Pipeline and Tailor workspaces toward the supplied
    hi-fi screens, and aligned Interview/Lab/Settings/Portfolio/Applications/
    Recommended/Apply Pack pages to the new shell and IA.
  - Removed visible old UI leftovers from the core paths: `Workspace` fallback,
    old emoji-heavy action labels, stale “待点列表” naming, and “critic” display
    language in user-facing run details.
  - Browser-verified the updated app on `http://127.0.0.1:8770/` across
    `/`, `/pipeline`, `/tailor`, `/recommended`, `/portfolio`, `/interviews`,
    `/evolution`, and `/applications`; active rail state and breadcrumbs matched
    the new IA and no old top nav appeared.
  - Verified with focused UI tests (`136 passed`) and full suite
    `uv run pytest -q`: `859 passed, 2 skipped`.
- 2026-05-16 agent agenda pass:
  - Reframed the runtime away from fixed “one input -> one tool chain”
    behavior after the user clarified that this is workflow automation, not an
    agent.
  - Replaced the old chat-first workflow prompt with an
    Observe -> Agenda -> Decide -> Act -> Verify -> Sleep decision contract.
  - Added `agenda.md` as the agent's persistent open-loop ledger and now inject
    it into every runtime system prompt alongside active goals and MEMORY.md.
  - Updated the current `.offerguide/worldview/agenda.md` with the real
    open loops, blockers, opportunities, and quiet-wait items for this project.
  - Verified focused agenda tests (`4 passed`) and runtime/context/evidence
    tests (`29 passed`).
- 2026-05-16 agent work-items pass:
  - Reverted the `record_decision` / decision-guard approach after the user
    correctly called it formalism.
  - Added `agent_work_items` as durable agent-owned work state: open,
    in_progress, blocked, waiting, done, dismissed.
  - User input, lifecycle events, and scheduled wakes now materialize into
    work items before the LLM runs; loop runs attach those items and mark
    open items in_progress.
  - Runtime context now injects Agent Work Items alongside active goals,
    agenda.md, and MEMORY.md, so the model wakes to real work state.
  - Verified focused runtime/work-item tests: `21 passed`.
- 2026-05-16 feature necessity audit:
  - Paused further feature/UI changes after the user pointed out that the
    remaining question is whether these features should exist as product
    surfaces at all.
  - Added `docs/feature_necessity_audit_2026-05-16.md` with a keep / merge /
    demote / freeze classification for UI pages, agent tools, and SKILLs.
  - Main conclusion: most capabilities are useful, but too many are exposed as
    first-class entrances. The agent should own the state and action selection;
    users should intervene at boundary decisions.
- 2026-05-16 minimal IA consolidation:
  - Kept routes/data intact, but renamed visible surfaces so `/recommended`,
    `/applications`, and `/funnel` read as Pipeline views rather than separate
    products.
  - Mission Control now says “新进候选”; Pipeline links say “候选视图” and
    “应用时间线”; funnel page says “Pipeline · 转化概览”.
  - Apply pack and post-apply pack now return to Pipeline; post-apply pack no
    longer promotes `/mock` as a primary CTA and routes to the Interview
    workspace instead.
  - Verified focused IA/UI tests: `32 passed`.
- 2026-05-16 work-item closure pass:
  - After the user reiterated that the project must first be an agent, audited
    the runtime gap: work items could be created and attached to runs, but the
    agent could not explicitly close, defer, or block them.
  - Added `_schema.update_work_item()` and the main-agent
    `update_work_item` tool. The tool updates durable work state directly,
    not a separate formal decision log.
  - `done` / `dismissed` work items now get `closed_at` and drop out of the
    active work-item context; `waiting` / `blocked` stay active with required
    `next_action`.
  - Updated `instructions.md` so Verify requires real work-item state updates
    after actual progress, and explicitly forbids calling the tool merely for
    proof-of-agent theater.
  - Verified focused runtime tests: `38 passed`.
  - Verified full suite: `866 passed, 2 skipped`.

## In Progress

- Agent-quality slice, agent agenda pass, work-items pass, UI connection pass,
  feature necessity audit, and the first minimal IA consolidation slice are
  complete at the focused level. Every runtime wake now receives active-goal
  progress, durable work items, agenda.md, and MEMORY.md, and the agent now
  has a real tool to close/defer/block work items it actually advanced.

## Next

- Next product/runtime slice: run real end-to-end dogfood for a JD/user-input run:
  user input/event creates a work item, agent advances it with minimal
  fetch/score/tailor/read_artifact actions, then updates `agenda.md`,
  run/skill/event state references, and closes or defers the work item with
  `update_work_item`.
- Then continue IA consolidation. Demote or merge remaining standalone
  surfaces (`/jobs`, `/compare`, `/mock`, `/reflect`, `/stories`) into
  Pipeline / Tailor / Interview state views or detail routes.
- Continue filling `candidate.md` from user-confirmed CV, preferences, and
  deal-breakers before widening proactive discovery.
- The old `http://127.0.0.1:8766/` process may still be serving pre-restart
  code; the verified UI server for this pass is `http://127.0.0.1:8770/`.

## Notes

- The official primitive names are kept: `test-results.json`, `PROGRESS.md`,
  `AGENT_STOP`, `STEER.md`, `.claude/hooks/*`, and `.claude/agents/evaluator.md`.
- Do not revert the existing W21 Mission Control UI edits in
  `src/offerguide/ui/templates/mission_control.html`, `src/offerguide/ui/web.py`,
  or their test updates; they are intentional prior work.
- “记录所有思路与过程” is implemented as audit-friendly decisions, assumptions,
  process logs, and verification notes, not hidden chain-of-thought.
