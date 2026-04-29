"""Job-source adapters. Today: nowcoder, manual. W6 adds Boss browser-extension endpoint."""

from . import manual, nowcoder
from ._spec import RawJob, canonical_text, content_hash

__all__ = ["RawJob", "canonical_text", "content_hash", "manual", "nowcoder"]
