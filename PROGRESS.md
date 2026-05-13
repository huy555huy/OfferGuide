# OfferGuide Long-Running Harness Progress

## Done

- Added the Anthropic CWC long-running harness primitives to the project root:
  default-fail contract, evidence gate hooks, fresh evaluator, handoff file,
  kill switch, steer hook, and stop-time checkpoint hook.
- Align README wording so `src/offerguide/harness` is described as the
  application agent loop, while `.claude/` + `test-results.json` + `PROGRESS.md`
  are the Anthropic-style long-running harness primitives.
- Verified the harness-related slice with
  `pytest tests/test_anthropic_harness_primitives.py tests/test_project_vault.py tests/test_evidence_first_context.py`
  and recorded the output in `harness-related-tests-result.txt`.

## In progress

- Full-suite cleanup is separate: `pytest` currently has 9 legacy
  Home/Mission-Control assertion failures against the newer chat-first page.

## Next

- Use the contract loop for the next product slice: project-vault truthfulness
  and resume tailoring must pass only with concrete evidence, not invented
  metrics.

## Notes

- The official primitive names are kept: `test-results.json`, `PROGRESS.md`,
  `AGENT_STOP`, `STEER.md`, `.claude/hooks/*`, and `.claude/agents/evaluator.md`.
