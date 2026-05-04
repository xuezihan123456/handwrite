"""Temporal handwriting style module.

Simulates age-dependent handwriting characteristics and historical writing
instrument styles (brush pen, fountain pen, ballpoint pen).
"""

from .age_profiles import AgeGroup, get_age_profile, list_age_groups
from .historical_style import HistoricalInstrument, apply_historical_style
from .skill_simulator import SkillSimulator
from .temporal_engine import generate_historical, generate_with_age
from .temporal_renderer import TemporalRenderer

__all__ = [
    "AgeGroup",
    "HistoricalInstrument",
    "SkillSimulator",
    "TemporalRenderer",
    "apply_historical_style",
    "generate_historical",
    "generate_with_age",
    "get_age_profile",
    "list_age_groups",
]
