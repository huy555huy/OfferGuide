"""LLM-backed helpers used outside the main agent loop.

Despite the package name, these are NOT agents in the ReAct-loop sense.
They are procedural functions that call the LLM where the LLM is the
right tool (parsing arbitrary email text, filtering interview-corpus
results), plus a thin search backend.

Modules:

- ``email_classifier_llm`` — DeepSeek-driven email classifier (replaces
  the earlier regex one). Extracts kind + structured info from arbitrary
  HR/recruiter emails.
- ``corpus_collector`` — Searches the web for 面经 about a specific
  company, LLM-filters them for quality, dedups, ingests to
  ``interview_experiences``.
- ``company_sweep`` — Top-level procedural helper that runs the above on
  one company. NOT an agent loop — pure orchestration.
- ``search`` — Abstract search backend with a Tavily default.

The W21 SubAgent infrastructure (`offerguide.agents.*`) is the real agent
layer. This package is utility code.
"""

from .company_sweep import CompanySweepResult, sweep_company
from .corpus_collector import CorpusCollector
from .email_classifier_llm import LLMEmailClassification, classify_email_llm
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
