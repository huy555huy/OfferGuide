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
