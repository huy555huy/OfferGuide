"""LLM email classification and reusable web-search backends.

The actual job and interview research agents live under
``offerguide.research_agents``.  This package only contains utilities that
remain shared with email ingestion and source search.
"""

from .email_classifier_llm import LLMEmailClassification, classify_email_llm
from .search import SearchHit, build_default_search

__all__ = [
    "LLMEmailClassification",
    "SearchHit",
    "build_default_search",
    "classify_email_llm",
]
