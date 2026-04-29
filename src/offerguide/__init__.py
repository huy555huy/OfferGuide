"""OfferGuide — Ambient job-hunt copilot for Chinese campus recruitment.

Top-level public API is intentionally small. Sub-packages expose the surface
area that's reasonable to consume from outside.
"""

from .agent import AgentState, build_graph
from .memory import Store
from .profile import UserProfile, load_resume_pdf
from .skills import SkillSpec, discover_skills, load_skill

__version__ = "0.0.1"

__all__ = [
    "AgentState",
    "SkillSpec",
    "Store",
    "UserProfile",
    "__version__",
    "build_graph",
    "discover_skills",
    "load_resume_pdf",
    "load_skill",
]
