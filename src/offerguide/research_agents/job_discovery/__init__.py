"""Public job-discovery domain API."""

from .agent import JobDiscoveryAgent, SourceEvidenceVerifier
from .models import (
    CandidateEvidence,
    CandidateEvidenceDocument,
    JobDiscoverySnapshot,
    JobEvidenceCatalogItem,
    JobEvidenceCatalogPage,
    JobPostingEvidence,
    JobSearchContext,
    JobSelectionItem,
    JobSelectionSet,
    SelectionGroundingQuote,
    platform_job_reference,
)
from .repository import (
    JobDiscoveryError,
    JobDiscoveryEvidenceError,
    JobDiscoveryNotConfiguredError,
    JobDiscoveryRepository,
    JobDiscoveryRevisionConflict,
    canonicalize_job_url,
    init_job_discovery_schema,
    job_stable_key,
)
from .source_tools import (
    StoredJobSourceEvidenceVerifier,
    build_job_source_tools,
    standard_source_dependencies,
)

__all__ = [
    "CandidateEvidence",
    "CandidateEvidenceDocument",
    "JobDiscoveryAgent",
    "JobDiscoveryError",
    "JobDiscoveryEvidenceError",
    "JobDiscoveryNotConfiguredError",
    "JobDiscoveryRepository",
    "JobDiscoveryRevisionConflict",
    "JobDiscoverySnapshot",
    "JobEvidenceCatalogItem",
    "JobEvidenceCatalogPage",
    "JobPostingEvidence",
    "JobSearchContext",
    "JobSelectionItem",
    "JobSelectionSet",
    "SelectionGroundingQuote",
    "SourceEvidenceVerifier",
    "StoredJobSourceEvidenceVerifier",
    "build_job_source_tools",
    "canonicalize_job_url",
    "init_job_discovery_schema",
    "job_stable_key",
    "platform_job_reference",
    "standard_source_dependencies",
]
