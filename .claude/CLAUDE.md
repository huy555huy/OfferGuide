<!-- Copyright 2026 Anthropic PBC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Modified for OfferGuide: added project-specific tests and truthfulness rules. -->

# Long-running conventions for OfferGuide

These conventions are the Anthropic CWC long-running harness primitives adapted
to this repository. They are intentionally mechanical: default-fail contract,
fresh evaluator, evidence before completion, and a written handoff.

## Always start here

Before doing anything else, read `PROGRESS.md`. It is your handoff note from the
previous session. If it does not exist yet, create it with four sections
(`## Done`, `## In progress`, `## Next`, `## Notes`) and leave them empty.
Then run `git log --oneline -10` to see what was just committed.

Run a small smoke test before changing behavior so you know whether the handoff
is already broken. For this project, prefer:

```bash
pytest tests/test_anthropic_harness_primitives.py tests/test_project_vault.py tests/test_evidence_first_context.py
```

## One feature at a time

Work on exactly one item from `PROGRESS.md` per session. Finish it with evidence
before starting another. If the user gives a new task mid-session, add it to
`PROGRESS.md`, then make a deliberate call about whether it supersedes the
current item.

## Proof before passing

`test-results.json` is the default-fail contract. A criterion may move to
`"passes": true` only after the agent has opened concrete evidence first:

1. Run the relevant test, browser check, or CLI verification.
2. Save or locate the evidence file, such as a screenshot, console log, or
   `*-result.txt`.
3. Open that evidence with the Read tool and check what it actually shows.
4. Only then update `test-results.json`.

The `verify-gate` hook blocks writes to `test-results.json` until evidence has
been read. Do not work around it.

## Keep `PROGRESS.md` current

After each completed item, update `PROGRESS.md`: what changed, what evidence
was checked, and what should happen next. Future sessions read this cold.

## Commit often

The Stop hook commits tracked changes at session end, but also `git add` new
source/config files yourself and commit at meaningful checkpoints with
descriptive messages.

## Truthfulness rule for this project

OfferGuide is a job-search agent, so false claims are product bugs. Do not
invent resume metrics, project impact, ranking, users, deployment status,
benchmark gains, or test coverage. If evidence is missing, leave the contract
failing or ask for facts.

## If you are told to stop

`OPERATOR STEERING:` messages come from a human via the steer hook. Treat them
as higher priority than the current plan.
