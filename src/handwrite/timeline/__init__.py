"""Handwriting Timeline module (笔迹时光机)."""

from __future__ import annotations

from .curve_fitting import Curve, fit_curve
from .feature_extractor import AgeFeatures, HandwritingFeatureExtractor
from .timeline_model import (
    TimelineModel,
    fit_timeline,
    generate_at_age,
)

__all__ = [
    "AgeFeatures",
    "Curve",
    "HandwritingFeatureExtractor",
    "TimelineModel",
    "fit_curve",
    "fit_timeline",
    "generate_at_age",
]
