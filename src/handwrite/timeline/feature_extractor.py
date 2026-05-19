"""Feature extraction for the handwriting timeline model.

Extracts a compact :class:`AgeFeatures` record describing how strokes look
at a given developmental age: average stroke width, dominant slant, stroke
connectivity, legibility, and ink density.

The extractor is intentionally lightweight - it relies only on
``numpy`` and ``Pillow`` so that the timeline module can be exercised in
test environments where ``opencv`` may not be installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


PathOrImage = Union[str, Path, Image.Image]


@dataclass(frozen=True)
class AgeFeatures:
    """Quantitative handwriting features captured at a specific age.

    All values are non-negative and roughly normalized so they can be
    used as inputs to a curve-fitting routine.

    Attributes:
        stroke_width: Average ink-run width in pixels (>= 0).
        slant: Estimated character slant in degrees, clamped to [-30, 30].
        connectivity: Estimate of how connected/cursive strokes are (0..1).
        legibility: Heuristic legibility score (0..1, higher is cleaner).
        density: Ratio of ink pixels over the whole image (0..1).
    """

    stroke_width: float
    slant: float
    connectivity: float
    legibility: float
    density: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable dictionary representation."""
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "AgeFeatures":
        """Reconstruct AgeFeatures from a dictionary."""
        return cls(
            stroke_width=float(data["stroke_width"]),
            slant=float(data["slant"]),
            connectivity=float(data["connectivity"]),
            legibility=float(data["legibility"]),
            density=float(data["density"]),
        )


class HandwritingFeatureExtractor:
    """Extract :class:`AgeFeatures` from a handwriting sample image.

    The extractor is pure-numpy: it converts the image to grayscale, finds
    ink pixels via a fixed threshold, and computes stroke-run statistics,
    a shear-based slant estimate, a dilation-style connectivity proxy,
    a legibility score from the ink/empty contrast, and the ink density.
    """

    def __init__(self, ink_threshold: int = 160) -> None:
        """Initialize the extractor.

        Args:
            ink_threshold: Pixel value below which a pixel is considered ink
                (0-255). Defaults to 160 to be tolerant of light strokes.
        """
        self._ink_threshold = int(ink_threshold)

    @property
    def ink_threshold(self) -> int:
        """Return the active ink threshold."""
        return self._ink_threshold

    def extract(self, image: PathOrImage) -> AgeFeatures:
        """Extract :class:`AgeFeatures` from a handwriting sample.

        Args:
            image: Path to a PNG/JPEG file or an in-memory PIL Image.

        Returns:
            The extracted :class:`AgeFeatures`.
        """
        img = self._load_image(image).convert("L")
        arr = np.asarray(img, dtype=np.uint8)

        ink_mask = arr < self._ink_threshold

        density = float(np.mean(ink_mask)) if ink_mask.size > 0 else 0.0
        stroke_width = self._estimate_stroke_width(ink_mask)
        slant = self._estimate_slant(ink_mask)
        connectivity = self._estimate_connectivity(ink_mask)
        legibility = self._estimate_legibility(arr, ink_mask)

        return AgeFeatures(
            stroke_width=round(float(stroke_width), 4),
            slant=round(float(slant), 4),
            connectivity=round(float(connectivity), 4),
            legibility=round(float(legibility), 4),
            density=round(float(density), 4),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: PathOrImage) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        return Image.open(str(image))

    @staticmethod
    def _run_lengths(line: np.ndarray) -> list[int]:
        """Return run-lengths of True values in a 1D boolean array."""
        if not np.any(line):
            return []
        padded = np.concatenate(([False], line, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return (ends - starts).tolist()

    def _estimate_stroke_width(self, ink_mask: np.ndarray) -> float:
        """Estimate the mean ink run-length over rows and columns."""
        if not np.any(ink_mask):
            return 0.0

        widths: list[int] = []
        rows, cols = ink_mask.shape
        # Subsample rows/cols to keep cost bounded for large images.
        row_step = max(1, rows // 64)
        col_step = max(1, cols // 64)
        for row_idx in range(0, rows, row_step):
            widths.extend(self._run_lengths(ink_mask[row_idx]))
        for col_idx in range(0, cols, col_step):
            widths.extend(self._run_lengths(ink_mask[:, col_idx]))
        if not widths:
            return 0.0
        return float(np.mean(widths))

    @staticmethod
    def _estimate_slant(ink_mask: np.ndarray) -> float:
        """Estimate slant in degrees in the range [-30, 30].

        Uses a coarse shear search that maximizes the variance of the
        projection histogram - the angle that best aligns vertical strokes
        produces the sharpest peaks.
        """
        coords = np.argwhere(ink_mask)
        if coords.shape[0] < 10:
            return 0.0

        ys = coords[:, 0].astype(np.float64)
        xs = coords[:, 1].astype(np.float64)

        best_angle = 0.0
        best_variance = -1.0
        for angle_deg in range(-30, 31, 2):
            shear = -np.tan(np.deg2rad(angle_deg))
            projected = ys + shear * xs
            spread = projected.max() - projected.min()
            if spread < 1.0:
                continue
            bins = max(10, int(spread))
            hist, _ = np.histogram(projected, bins=bins)
            variance = float(np.var(hist))
            if variance > best_variance:
                best_variance = variance
                best_angle = float(angle_deg)
        return float(max(-30.0, min(30.0, best_angle)))

    @staticmethod
    def _estimate_connectivity(ink_mask: np.ndarray) -> float:
        """Estimate connectivity in [0, 1].

        Approximates the number of connected components by counting
        transitions in the ink mask. Fewer transitions per ink pixel means
        more continuous (cursive) strokes.
        """
        if not np.any(ink_mask):
            return 0.0

        total_ink = float(np.sum(ink_mask))
        if total_ink < 1.0:
            return 0.0

        # Count horizontal + vertical "run starts" - this approximates the
        # number of distinct ink segments without requiring scipy/cv2.
        row_starts = 0
        for row in ink_mask:
            if not np.any(row):
                continue
            padded = np.concatenate(([False], row))
            row_starts += int(np.sum(np.diff(padded.astype(np.int8)) == 1))

        col_starts = 0
        for col in ink_mask.T:
            if not np.any(col):
                continue
            padded = np.concatenate(([False], col))
            col_starts += int(np.sum(np.diff(padded.astype(np.int8)) == 1))

        starts = max(1, row_starts + col_starts)
        # Heuristic: fewer starts per ink pixel => more connected.
        ratio = total_ink / (starts + total_ink)
        return float(max(0.0, min(1.0, ratio)))

    def _estimate_legibility(
        self,
        arr: np.ndarray,
        ink_mask: np.ndarray,
    ) -> float:
        """Estimate legibility in [0, 1].

        Higher is cleaner: strong contrast between ink and background and a
        moderate density tend to indicate readable strokes.
        """
        if not np.any(ink_mask):
            return 0.0

        ink_pixels = arr[ink_mask].astype(np.float32)
        bg_pixels = arr[~ink_mask].astype(np.float32)

        if ink_pixels.size == 0 or bg_pixels.size == 0:
            return 0.0

        contrast = float(np.mean(bg_pixels) - np.mean(ink_pixels))
        contrast_score = max(0.0, min(1.0, contrast / 255.0))

        density = float(np.mean(ink_mask))
        # Density "sweet spot" around 0.12; either extreme reduces legibility.
        density_score = max(0.0, 1.0 - abs(density - 0.12) / 0.4)

        return float(max(0.0, min(1.0, 0.5 * contrast_score + 0.5 * density_score)))


__all__ = ["AgeFeatures", "HandwritingFeatureExtractor"]
