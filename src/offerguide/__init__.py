"""OfferGuide public API."""

from .memory import Store
from .resume import (
    MasterResumeDocument,
    MasterResumeSource,
    ResumeContext,
    ResumeDocument,
    build_resume_context,
    load_resume_pdf,
)
from .skills import SkillSpec, discover_skills, load_skill

__version__ = "0.0.1"

__all__ = [
    "MasterResumeDocument",
    "MasterResumeSource",
    "ResumeContext",
    "ResumeDocument",
    "SkillSpec",
    "Store",
    "__version__",
    "build_resume_context",
    "discover_skills",
    "load_resume_pdf",
    "load_skill",
]
