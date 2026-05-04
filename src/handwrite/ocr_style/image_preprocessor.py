"""Scanned image preprocessing for handwriting extraction.

Pipeline:
  1. Grayscale conversion
  2. Perspective correction (detect paper edges via contour)
  3. Skew correction (Hough line transform)
  4. Binarization (Otsu threshold)
  5. Denoising (median filter)
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessResult:
    """Output of the preprocessing pipeline."""

    image: np.ndarray  # binarized, deskewed; foreground (ink) = 255
    original_shape: tuple[int, int]
    skew_angle: float  # degrees, positive = clockwise
    perspective_corrected: bool


class ImagePreprocessor:
    """Preprocess scanned handwriting images for character segmentation."""

    def __init__(
        self,
        *,
        target_dpi_scale: float = 1.0,
        median_kernel: int = 3,
        min_paper_ratio: float = 0.15,
    ) -> None:
        self.target_dpi_scale = target_dpi_scale
        self.median_kernel = median_kernel
        self.min_paper_ratio = min_paper_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, image_path: str | os.PathLike[str]) -> PreprocessResult:
        """Run the full preprocessing pipeline on a scanned image.

        Parameters
        ----------
        image_path:
            Path to the scanned image file.

        Returns
        -------
        PreprocessResult
            Contains the binarized image and metadata.
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        original_shape = img.shape[:2]  # (H, W)

        # Step 1: grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 2: perspective correction
        corrected, perspective_applied = self._perspective_correction(gray)

        # Step 3: skew correction
        deskewed, skew_angle = self._skew_correction(corrected)

        # Step 4: binarize (Otsu)
        binary = self._binarize(deskewed)

        # Step 5: denoise
        denoised = self._denoise(binary)

        return PreprocessResult(
            image=denoised,
            original_shape=original_shape,
            skew_angle=skew_angle,
            perspective_corrected=perspective_applied,
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _perspective_correction(
        self, gray: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Detect paper edges and apply perspective warp.

        Falls back to returning the original image if no suitable
        quadrilateral contour is found.
        """
        h, w = gray.shape[:2]
        img_area = h * w

        # Adaptive threshold to find paper region
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2,
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return gray, False

        # Find the largest contour that looks like paper
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours[:5]:
            area = cv2.contourArea(contour)
            if area < img_area * self.min_paper_ratio:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                return self._four_point_transform(gray, approx.reshape(4, 2))

        return gray, False

    @staticmethod
    def _four_point_transform(
        gray: np.ndarray, pts: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Apply perspective transform given four corner points."""
        # Order: top-left, top-right, bottom-right, bottom-left
        rect = _order_points(pts)
        tl, tr, br, bl = rect

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_w = int(max(width_a, width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_h = int(max(height_a, height_b))

        if max_w < 50 or max_h < 50:
            return gray, False

        dst = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1],
        ], dtype=np.float32)

        m = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
        warped = cv2.warpPerspective(gray, m, (max_w, max_h))
        return warped, True

    @staticmethod
    def _skew_correction(gray: np.ndarray) -> tuple[np.ndarray, float]:
        """Detect and correct text skew using Hough transform."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100,
            minLineLength=gray.shape[1] // 8,
            maxLineGap=20,
        )

        if lines is None:
            return gray, 0.0

        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines
            if abs(angle) < 15:
                angles.append(float(angle))

        if not angles:
            return gray, 0.0

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return gray, median_angle

        h, w = gray.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            gray, m, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, median_angle

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """Otsu binarization, ensuring text (ink) is white (255)."""
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        # Ensure foreground (ink) is white
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        return binary

    def _denoise(self, binary: np.ndarray) -> np.ndarray:
        """Remove salt-and-pepper noise with median filter."""
        k = self.median_kernel
        if k % 2 == 0:
            k += 1
        return cv2.medianBlur(binary, k)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as [top-left, top-right, bottom-right, bottom-left]."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


__all__ = ["ImagePreprocessor", "PreprocessResult"]
