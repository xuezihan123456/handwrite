"""Extract style vectors from handwriting features for glyph synthesis.

Converts raw HandwritingFeatures into a normalized StyleVector that
controls glyph rendering parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from handwrite.personalization.sample_analyzer import HandwritingFeatures


@dataclass(frozen=True)
class StyleVector:
    """Normalized style parameters for glyph synthesis.

    All values are in [0, 1] range unless noted.
    """

    # Stroke thickness factor (0=thin, 1=thick)
    stroke_thickness: float
    # Slant angle in degrees (can be negative)
    slant_angle: float
    # Cursiveness factor (0=print, 1=cursive)
    cursiveness: float
    # Ink darkness factor (0=light, 1=dark)
    ink_darkness: float
    # Stroke smoothness (0=rough, 1=smooth)
    smoothness: float
    # Overall scale factor derived from ink coverage
    ink_density: float


class StyleExtractor:
    """Extract a normalized StyleVector from handwriting features."""

    # Reference ranges for normalization
    _STROKE_WIDTH_MIN = 1.0
    _STROKE_WIDTH_MAX = 12.0
    _INK_MEAN_MIN = 0.0
    _INK_MEAN_MAX = 200.0
    _INK_STD_MIN = 0.0
    _INK_STD_MAX = 80.0
    _COVERAGE_MIN = 0.01
    _COVERAGE_MAX = 0.40

    def extract(self, features: HandwritingFeatures) -> StyleVector:
        """Convert HandwritingFeatures into a StyleVector.

        Args:
            features: Raw features from SampleAnalyzer.

        Returns:
            Normalized StyleVector for glyph synthesis.
        """
        stroke_thickness = self._clamp(
            (features.stroke_width_mean - self._STROKE_WIDTH_MIN)
            / (self._STROKE_WIDTH_MAX - self._STROKE_WIDTH_MIN)
        )

        # Slant: normalize to [-1, 1] range from [-30, 30] degrees
        slant_angle = max(-30.0, min(30.0, features.slant_angle))

        cursiveness = self._clamp(features.connectivity)

        # Ink darkness: lower pixel value = darker ink
        # Convert so 0 (light) -> 0, very dark ink -> 1
        darkness = self._clamp(
            1.0
            - (features.ink_intensity_mean - self._INK_MEAN_MIN)
            / (self._INK_MEAN_MAX - self._INK_MEAN_MIN)
        )

        # Smoothness: lower std of ink intensity means more uniform strokes
        smoothness = self._clamp(
            1.0
            - (features.ink_intensity_std - self._INK_STD_MIN)
            / (self._INK_STD_MAX - self._INK_STD_MIN)
        )

        ink_density = self._clamp(
            (features.ink_coverage - self._COVERAGE_MIN)
            / (self._COVERAGE_MAX - self._COVERAGE_MIN)
        )

        return StyleVector(
            stroke_thickness=round(stroke_thickness, 4),
            slant_angle=round(slant_angle, 2),
            cursiveness=round(cursiveness, 4),
            ink_darkness=round(darkness, 4),
            smoothness=round(smoothness, 4),
            ink_density=round(ink_density, 4),
        )

    def average_vectors(self, vectors: Sequence[StyleVector]) -> StyleVector:
        """Average multiple StyleVectors into one.

        Args:
            vectors: Sequence of StyleVectors to average.

        Returns:
            Averaged StyleVector.

        Raises:
            ValueError: If vectors is empty.
        """
        if not vectors:
            raise ValueError("Cannot average empty sequence of StyleVectors")

        n = len(vectors)
        return StyleVector(
            stroke_thickness=sum(v.stroke_thickness for v in vectors) / n,
            slant_angle=sum(v.slant_angle for v in vectors) / n,
            cursiveness=sum(v.cursiveness for v in vectors) / n,
            ink_darkness=sum(v.ink_darkness for v in vectors) / n,
            smoothness=sum(v.smoothness for v in vectors) / n,
            ink_density=sum(v.ink_density for v in vectors) / n,
        )

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))
