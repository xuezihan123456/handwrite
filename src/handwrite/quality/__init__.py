"""Handwriting quality evaluation and feedback module.

Provides tools for assessing the authenticity and naturalness of
generated handwriting images, with actionable improvement suggestions.
"""

from handwrite.quality.authenticity_scorer import score_authenticity
from handwrite.quality.improvement_advisor import generate_char_advice, generate_page_advice
from handwrite.quality.naturalness_scorer import score_naturalness
from handwrite.quality.quality_engine import evaluate_char, evaluate_page
from handwrite.quality.quality_report import (
    CharacterQualityReport,
    DimensionScore,
    PageQualityReport,
)

__all__ = [
    "evaluate_page",
    "evaluate_char",
    "score_authenticity",
    "score_naturalness",
    "generate_page_advice",
    "generate_char_advice",
    "PageQualityReport",
    "CharacterQualityReport",
    "DimensionScore",
]
