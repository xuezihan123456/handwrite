"""Handwrite grading module -- simulate teacher-style annotation and scoring.

Components:
    error_detector       -- Error detection (typos, grammar, punctuation, format)
    annotation_renderer  -- Render red annotations (strikethrough, circles, waves, margin notes)
    score_calculator     -- Score calculation (percentage and grade-based)
    feedback_generator   -- Generate feedback reports
    grading_engine       -- Main entry point: ``grade()`` and ``annotate()``
"""

from handwrite.grading.annotation_renderer import (
    Annotation,
    AnnotationRenderer,
    AnnotationType,
)
from handwrite.grading.error_detector import ErrorDetector, ErrorInfo, ErrorType
from handwrite.grading.feedback_generator import FeedbackGenerator
from handwrite.grading.grading_engine import GradingEngine, annotate, grade
from handwrite.grading.score_calculator import GradeLevel, ScoreCalculator

__all__ = [
    "ErrorDetector",
    "ErrorInfo",
    "ErrorType",
    "AnnotationRenderer",
    "Annotation",
    "AnnotationType",
    "ScoreCalculator",
    "GradeLevel",
    "FeedbackGenerator",
    "GradingEngine",
    "grade",
    "annotate",
]
