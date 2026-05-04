"""Paper detection using edge detection and contour finding.

Detects paper boundaries in a photograph by applying Canny edge detection,
finding contours, and selecting the largest quadrilateral as the paper region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class PaperDetectionResult:
    """Result of paper boundary detection.

    Attributes:
        corners: Ordered four corner points of the detected paper,
            shape ``(4, 2)`` in ``(x, y``) format, ordered as
            top-left, top-right, bottom-right, bottom-left.
        contour: The raw contour array from OpenCV.
        confidence: Detection confidence score in ``[0.0, 1.0]``.
        mask: Binary mask of the detected paper region, same size as input.
    """

    corners: np.ndarray
    contour: np.ndarray
    confidence: float
    mask: np.ndarray


class PaperDetector:
    """Detects paper boundaries in photographs.

    Uses adaptive Canny edge detection with morphological closing to find
    the largest quadrilateral contour that likely represents a paper sheet.

    Parameters:
        min_area_ratio: Minimum contour area as a fraction of image area
            to consider as a valid paper candidate.
        canny_low: Lower threshold for Canny edge detection.
        canny_high: Upper threshold for Canny edge detection.
        morph_kernel_size: Kernel size for morphological closing.
        approx_epsilon_factor: Contour approximation precision factor
            relative to contour perimeter.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.05,
        canny_low: int = 50,
        canny_high: int = 150,
        morph_kernel_size: int = 5,
        approx_epsilon_factor: float = 0.02,
    ) -> None:
        if not 0.0 < min_area_ratio < 1.0:
            raise ValueError("min_area_ratio must be in (0, 1)")
        if canny_low < 0 or canny_high < 0 or canny_low >= canny_high:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high")
        if morph_kernel_size < 1:
            raise ValueError("morph_kernel_size must be >= 1")
        if approx_epsilon_factor <= 0:
            raise ValueError("approx_epsilon_factor must be > 0")

        self._min_area_ratio = min_area_ratio
        self._canny_low = canny_low
        self._canny_high = canny_high
        self._morph_kernel_size = morph_kernel_size
        self._approx_epsilon = approx_epsilon_factor

    def detect(self, image: np.ndarray) -> Optional[PaperDetectionResult]:
        """Detect paper boundaries in the given image.

        Parameters:
            image: Input BGR image as a numpy array of shape ``(H, W, 3)``.

        Returns:
            A ``PaperDetectionResult`` if a paper region is found, or ``None``
            if detection fails.

        Raises:
            ValueError: If the input image is empty or has an unexpected shape.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be BGR with shape (H, W, 3)")

        h, w = image.shape[:2]
        min_area = h * w * self._min_area_ratio

        # Preprocess: grayscale, blur, edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self._canny_low, self._canny_high)

        # Morphological closing to connect nearby edges
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self._morph_kernel_size, self._morph_kernel_size),
        )
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Sort by area descending, try candidates
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                break  # remaining contours are too small

            # Approximate to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self._approx_epsilon * peri, True)

            # We want a quadrilateral
            if len(approx) != 4:
                continue

            corners = self._order_corners(approx.reshape(4, 2).astype(np.float64))
            confidence = self._compute_confidence(corners, area, h * w)
            mask = self._build_mask(corners, h, w)

            return PaperDetectionResult(
                corners=corners,
                contour=contour,
                confidence=confidence,
                mask=mask,
            )

        return None

    @staticmethod
    def _order_corners(corners: np.ndarray) -> np.ndarray:
        """Order corners as top-left, top-right, bottom-right, bottom-left.

        Uses the sum and difference of coordinates to identify each corner.
        """
        s = corners.sum(axis=1)
        d = np.diff(corners, axis=1).flatten()

        ordered = np.empty((4, 2), dtype=corners.dtype)
        ordered[0] = corners[np.argmin(s)]   # top-left: smallest sum
        ordered[2] = corners[np.argmax(s)]   # bottom-right: largest sum
        ordered[1] = corners[np.argmin(d)]   # top-right: smallest diff
        ordered[3] = corners[np.argmax(d)]   # bottom-left: largest diff
        return ordered

    @staticmethod
    def _compute_confidence(
        corners: np.ndarray, contour_area: float, image_area: float
    ) -> float:
        """Compute a heuristic confidence score for the detection.

        Combines area coverage ratio with how close the shape is to a rectangle
        (measured by the ratio of area to bounding-rect area).
        """
        # Area coverage factor
        coverage = min(contour_area / image_area * 4, 1.0)

        # Rectangularity: ratio of contour area to its bounding rect area
        rect = cv2.boundingRect(corners.astype(np.int32))
        rect_area = rect[2] * rect[3]
        rectangularity = contour_area / rect_area if rect_area > 0 else 0.0

        return float(np.clip(0.4 * coverage + 0.6 * rectangularity, 0.0, 1.0))

    @staticmethod
    def _build_mask(corners: np.ndarray, height: int, width: int) -> np.ndarray:
        """Build a binary mask for the detected paper region."""
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
        return mask
