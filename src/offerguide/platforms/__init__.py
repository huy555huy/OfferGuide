"""Job-source adapters."""

from . import manual, nowcoder, official_jobs
from ._spec import RawJob, canonical_text, content_hash

__all__ = [
    "RawJob",
    "canonical_text",
    "content_hash",
    "manual",
    "nowcoder",
    "official_jobs",
]
