"""Extract handwriting style features from segmented character images.

Features extracted:
  - Stroke width distribution (via distance transform)
  - Character aspect ratio statistics
  - Ink density distribution
  - Stroke curvature statistics (via contour analysis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from handwrite.ocr_style.character_segmenter import CharBox


@dataclass(frozen=True)
class StyleFeatures:
    """Aggregated style features extracted from a set of characters."""

    # Stroke width
    stroke_width_mean: float = 0.0
    stroke_width_std: float = 0.0
    stroke_width_median: float = 0.0

    # Aspect ratio (width / height)
    aspect_ratio_mean: float = 0.0
    aspect_ratio_std: float = 0.0

    # Ink density (fraction of white pixels in bounding box)
    ink_density_mean: float = 0.0
    ink_density_std: float = 0.0

    # Curvature (sum of absolute curvature along contours)
    curvature_mean: float = 0.0
    curvature_std: float = 0.0

    # General
    num_characters: int = 0
    avg_char_height: float = 0.0
    avg_char_width: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        """Return features as a plain dictionary."""
        return {
            "stroke_width_mean": self.stroke_width_mean,
            "stroke_width_std": self.stroke_width_std,
            "stroke_width_median": self.stroke_width_median,
            "aspect_ratio_mean": self.aspect_ratio_mean,
            "aspect_ratio_std": self.aspect_ratio_std,
            "ink_density_mean": self.ink_density_mean,
            "ink_density_std": self.ink_density_std,
            "curvature_mean": self.curvature_mean,
            "curvature_std": self.curvature_std,
            "num_characters": self.num_characters,
            "avg_char_height": self.avg_char_height,
            "avg_char_width": self.avg_char_width,
        }


class StyleExtractor:
    """Extract handwriting style features from segmented characters."""

    def __init__(self, *, curvature_sample_step: int = 3) -> None:
        self.curvature_sample_step = curvature_sample_step

    def extract(self, chars: Sequence[CharBox]) -> StyleFeatures:
        """Extract aggregated style features from a collection of characters.

        Parameters
        ----------
        chars:
            Segmented character boxes with cropped images.

        Returns
        -------
        StyleFeatures
            Aggregated statistical features describing the handwriting style.
        """
        if not chars:
            return StyleFeatures()

        stroke_widths: list[float] = []
        aspect_ratios: list[float] = []
        ink_densities: list[float] = []
        curvatures: list[float] = []
        heights: list[float] = []
        widths: list[float] = []

        for cb in chars:
            img = cb.image
            if img.size == 0:
                continue

            h, w = img.shape[:2]
            heights.append(float(h))
            widths.append(float(w))

            # Aspect ratio
            aspect_ratios.append(w / max(h, 1))

            # Ink density
            density = float(np.sum(img > 0)) / max(img.size, 1)
            ink_densities.append(density)

            # Stroke width via distance transform
            sw = self._estimate_stroke_width(img)
            if sw > 0:
                stroke_widths.append(sw)

            # Curvature
            curv = self._estimate_curvature(img)
            if curv >= 0:
                curvatures.append(curv)

        return StyleFeatures(
            stroke_width_mean=_safe_mean(stroke_widths),
            stroke_width_std=_safe_std(stroke_widths),
            stroke_width_median=_safe_median(stroke_widths),
            aspect_ratio_mean=_safe_mean(aspect_ratios),
            aspect_ratio_std=_safe_std(aspect_ratios),
            ink_density_mean=_safe_mean(ink_densities),
            ink_density_std=_safe_std(ink_densities),
            curvature_mean=_safe_mean(curvatures),
            curvature_std=_safe_std(curvatures),
            num_characters=len(chars),
            avg_char_height=_safe_mean(heights),
            avg_char_width=_safe_mean(widths),
        )

    # ------------------------------------------------------------------
    # Feature computation helpers
    # ------------------------------------------------------------------

    def _estimate_stroke_width(self, img: np.ndarray) -> float:
        """Estimate stroke width using distance transform.

        Computes the distance from each foreground pixel to the nearest
        background pixel, then returns the median distance * 2 as the
        estimated stroke width.
        """
        # Ensure binary
        binary = (img > 0).astype(np.uint8) * 255
        if np.sum(binary) == 0:
            return 0.0

        # Distance transform: distance from each foreground pixel to nearest background pixel
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

        # Only consider distances at foreground pixels
        fg_dists = dist[binary > 0]
        if len(fg_dists) == 0:
            return 0.0

        # Median distance * 2 ≈ stroke width
        return float(np.median(fg_dists) * 2)

    def _estimate_curvature(self, img: np.ndarray) -> float:
        """Estimate average curvature from contour analysis.

        Returns the mean absolute curvature sampled along contours.
        Returns -1 if no valid contours are found.
        """
        binary = (img > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            return -1.0

        all_curvatures: list[float] = []
        step = self.curvature_sample_step

        for contour in contours:
            pts = contour.reshape(-1, 2).astype(np.float64)
            if len(pts) < step * 2 + 1:
                continue

            for i in range(step, len(pts) - step, step):
                p_prev = pts[i - step]
                p_curr = pts[i]
                p_next = pts[i + step]

                # Curvature via Menger curvature
                curvature = _menger_curvature(p_prev, p_curr, p_next)
                if curvature >= 0:
                    all_curvatures.append(curvature)

        if not all_curvatures:
            return -1.0

        return float(np.mean(all_curvatures))


def _menger_curvature(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
    """Compute the Menger curvature of three 2D points.

    Returns 0 for collinear points, -1 on degenerate input.
    """
    # Signed area * 2
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    d12 = np.linalg.norm(p2 - p1)
    d23 = np.linalg.norm(p3 - p2)
    d31 = np.linalg.norm(p1 - p3)
    denom = d12 * d23 * d31
    if denom < 1e-12:
        return -1.0
    return abs(cross) / denom


# ------------------------------------------------------------------
# Statistical helpers
# ------------------------------------------------------------------

def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


def _safe_median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else 0.0


__all__ = ["StyleExtractor", "StyleFeatures"]
