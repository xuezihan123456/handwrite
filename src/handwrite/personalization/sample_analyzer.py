"""Analyze handwriting sample images and extract stroke features.

Extracts: stroke width distribution, character slant angle,
stroke connectivity, ink intensity distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class HandwritingFeatures:
    """Quantitative features extracted from a handwriting sample."""

    # Stroke width: (mean, std) in pixels
    stroke_width_mean: float
    stroke_width_std: float

    # Character slant angle in degrees (positive = right-leaning)
    slant_angle: float

    # Stroke connectivity ratio (0..1, higher = more connected/cursive)
    connectivity: float

    # Ink intensity distribution: (mean, std) of ink pixels (0=black, 255=white)
    ink_intensity_mean: float
    ink_intensity_std: float

    # Image dimensions
    image_width: int
    image_height: int

    # Ink coverage ratio (fraction of pixels that are ink)
    ink_coverage: float


class SampleAnalyzer:
    """Analyze handwriting sample images to extract quantitative features."""

    def __init__(self, ink_threshold: int = 128) -> None:
        """Initialize analyzer.

        Args:
            ink_threshold: Pixel value below which a pixel is considered ink (0-255).
        """
        self._ink_threshold = ink_threshold

    def analyze(self, image: Union[str, Path, Image.Image]) -> HandwritingFeatures:
        """Analyze a handwriting sample image.

        Args:
            image: Path to image file or PIL Image.

        Returns:
            Extracted HandwritingFeatures.
        """
        img = self._load_image(image).convert("L")
        arr = np.array(img, dtype=np.uint8)

        ink_mask = arr < self._ink_threshold

        stroke_widths = self._extract_stroke_widths(arr, ink_mask)
        slant = max(-30.0, min(30.0, self._estimate_slant(arr, ink_mask)))
        connectivity = self._estimate_connectivity(ink_mask)
        ink_pixels = arr[ink_mask]
        ink_mean = float(np.mean(ink_pixels)) if ink_pixels.size > 0 else 255.0
        ink_std = float(np.std(ink_pixels)) if ink_pixels.size > 0 else 0.0
        ink_coverage = float(np.sum(ink_mask)) / ink_mask.size if ink_mask.size > 0 else 0.0

        sw_mean = float(np.mean(stroke_widths)) if stroke_widths.size > 0 else 0.0
        sw_std = float(np.std(stroke_widths)) if stroke_widths.size > 0 else 0.0

        return HandwritingFeatures(
            stroke_width_mean=round(sw_mean, 2),
            stroke_width_std=round(sw_std, 2),
            slant_angle=round(slant, 2),
            connectivity=round(connectivity, 4),
            ink_intensity_mean=round(ink_mean, 2),
            ink_intensity_std=round(ink_std, 2),
            image_width=img.width,
            image_height=img.height,
            ink_coverage=round(ink_coverage, 4),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    def _extract_stroke_widths(
        self, arr: np.ndarray, ink_mask: np.ndarray
    ) -> np.ndarray:
        """Estimate stroke widths via horizontal and vertical run-lengths.

        Uses vectorized numpy diff operations to find run boundaries.
        """
        widths: list[int] = []

        # Horizontal runs
        for row in range(arr.shape[0]):
            row_mask = ink_mask[row]
            widths.extend(self._run_lengths(row_mask))

        # Vertical runs
        for col in range(arr.shape[1]):
            col_mask = ink_mask[:, col]
            widths.extend(self._run_lengths(col_mask))

        return np.array(widths, dtype=np.float64)

    @staticmethod
    def _run_lengths(line: np.ndarray) -> list[int]:
        """Extract run lengths of True values in a 1D boolean array."""
        if not np.any(line):
            return []
        # Pad with False at boundaries to catch edge runs
        padded = np.concatenate(([False], line, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return (ends - starts).tolist()

    def _estimate_slant(self, arr: np.ndarray, ink_mask: np.ndarray) -> float:
        """Estimate character slant angle using projection profile analysis.

        Tests shear angles from -30 to +30 degrees and finds the angle
        that maximizes horizontal stroke density variance (higher variance
        = more consistent vertical strokes = correct slant).
        """
        ink_coords = np.argwhere(ink_mask)
        if ink_coords.size < 10:
            return 0.0

        best_angle = 0.0
        best_variance = -1.0

        for angle_deg in range(-30, 31):
            angle_rad = np.deg2rad(angle_deg)
            shear_factor = -np.tan(angle_rad)
            sheared_y = ink_coords[:, 0] + shear_factor * ink_coords[:, 1]
            # Bin sheared y-coordinates
            y_min, y_max = sheared_y.min(), sheared_y.max()
            if y_max - y_min < 1:
                continue
            bins = max(20, int(y_max - y_min))
            hist, _ = np.histogram(sheared_y, bins=bins)
            variance = float(np.var(hist))
            if variance > best_variance:
                best_variance = variance
                best_angle = float(angle_deg)

        return best_angle

    def _estimate_connectivity(self, ink_mask: np.ndarray) -> float:
        """Estimate stroke connectivity using OpenCV morphological operations.

        Connectivity = ratio of connected ink pixels after dilation
        to total ink pixels. Higher means more cursive/connected writing.
        """
        import cv2

        if not np.any(ink_mask):
            return 0.0

        mask_u8 = ink_mask.astype(np.uint8) * 255

        # cv2.connectedComponents returns (num_labels, labels) where
        # num_labels includes the background (label 0).
        num_labels_orig, _ = cv2.connectedComponents(mask_u8)
        num_orig = num_labels_orig - 1  # Exclude background

        # Dilate to bridge small gaps (simulating cursive connections)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(mask_u8, kernel, iterations=2)
        num_labels_dilated, _ = cv2.connectedComponents(dilated)
        num_dilated = num_labels_dilated - 1

        if num_orig <= 1:
            return 1.0

        if num_dilated <= 0:
            return 0.0
        component_ratio = num_dilated / num_orig
        connectivity = max(0.0, 1.0 - (component_ratio - 1.0) / max(num_orig - 1, 1))
        return min(1.0, max(0.0, connectivity))
