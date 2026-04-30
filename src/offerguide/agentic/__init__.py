"""Agentic layer — components that USE LLM + WebSearch to do real work,
not regex / hardcoded heuristics.

Modules:

- ``email_classifier_llm`` — DeepSeek-driven email classifier (replaces
  the earlier regex one). Extracts kind + structured info from arbitrary
  HR/recruiter emails.
- ``corpus_collector`` — Agent that searches the web for 面经 about a
  specific company, LLM-filters them for quality, dedups, and ingests
  to ``interview_experiences``. The user no longer has to manually
  paste; the agent does it.
- ``meta_agent`` — Top-level orchestrator that runs all of the above
  on a per-company sweep.
- ``search`` — Abstract search backend with a DuckDuckGo HTML
  default implementation (no API key required for prototyping).

Design principle: every component here uses the LLM where the LLM is
the right tool. Regex is reserved for *parsing* structured formats
(ICS, JSON), not classification.
"""

from .corpus_collector import CorpusCollector
from .email_classifier_llm import LLMEmailClassification, classify_email_llm
from .meta_agent import CompanySweepResult, sweep_company
from .search import SearchHit, build_default_search

__all__ = [
    "CompanySweepResult",
    "CorpusCollector",
    "LLMEmailClassification",
    "SearchHit",
    "build_default_search",
    "classify_email_llm",
    "sweep_company",
]
