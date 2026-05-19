"""Quality assessment for handwriting photographs.

Inspects a single photo (path / PIL / numpy) and reports whether the image
is suitable for the photo-to-style pipeline. The check is intentionally
lightweight: resolution, ink coverage, blur (variance of Laplacian) and a
rough character-count estimate based on connected components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image


# Minimum required pixel dimensions for a usable photograph.
_MIN_SIDE_PX = 200
# Below this Laplacian variance the image is treated as blurry.
_BLUR_THRESHOLD = 80.0
# Pixel value below which a pixel is considered ink (on greyscale).
_INK_PIXEL_THRESHOLD = 160
# Image area must be at least this fraction of ink to be considered handwriting.
_MIN_INK_RATIO = 0.0008
# Maximum sensible ink ratio (otherwise the photo is mostly dark/over-exposed).
_MAX_INK_RATIO = 0.6


@dataclass(frozen=True)
class PhotoQualityReport:
    """Outcome of :func:`assess_photo_quality`.

    Attributes:
        resolution_ok: Whether the image meets the minimum side-length.
        blur_score: Variance of the Laplacian (higher means sharper).
        estimated_char_count: Rough character-component count.
        ink_ratio: Fraction of dark pixels in the image (0..1).
        recommendation: Human-readable Chinese guidance string.
        is_usable: True when the pipeline can proceed with this photo.
        image_width: Pixel width of the inspected image.
        image_height: Pixel height of the inspected image.
    """

    resolution_ok: bool
    blur_score: float
    estimated_char_count: int
    ink_ratio: float
    recommendation: str
    is_usable: bool
    image_width: int
    image_height: int


def _load_grayscale(image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """Load *image* into a single-channel uint8 array."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return arr.astype(np.uint8)
    if isinstance(image, Image.Image):
        return np.array(image.convert("L"), dtype=np.uint8)
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Photo not found: {path}")
    loaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if loaded is None:
        raise FileNotFoundError(f"Cannot decode photo: {path}")
    return loaded


def _estimate_blur(gray: np.ndarray) -> float:
    """Variance of the Laplacian; a common sharpness proxy."""
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _estimate_char_count(gray: np.ndarray) -> tuple[int, float]:
    """Rough character count + ink coverage via connected components."""
    if gray.size == 0:
        return 0, 0.0
    # Otsu binarisation; invert so ink is white for component analysis.
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    ink_ratio = float(np.count_nonzero(binary)) / float(binary.size)
    if ink_ratio <= 0.0:
        return 0, 0.0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    h, w = gray.shape[:2]
    min_area = max(20, (h * w) // 5000)
    max_area = max(min_area * 50, (h * w) // 4)
    valid = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        cw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if cw < 4 or ch < 4:
            continue
        valid += 1
    return valid, ink_ratio


def _build_recommendation(
    resolution_ok: bool,
    blur_score: float,
    estimated_char_count: int,
    ink_ratio: float,
) -> tuple[str, bool]:
    """Compose user-facing guidance and overall usability flag."""
    notes: list[str] = []
    usable = True

    if not resolution_ok:
        notes.append("分辨率偏低，建议在 200px 以上重新拍摄")
        usable = False
    if blur_score < _BLUR_THRESHOLD:
        notes.append("画面发糊，建议保持手机稳定、光线充足后重拍")
        usable = False
    if ink_ratio < _MIN_INK_RATIO:
        notes.append("墨迹太少，请确保照片中包含完整的手写内容")
        usable = False
    if ink_ratio > _MAX_INK_RATIO:
        notes.append("画面过暗或被遮挡，请调整白底拍摄")
        usable = False
    if estimated_char_count <= 0 and usable:
        notes.append("未检测到字符组件，建议拍摄更清晰的字迹")
        usable = False

    if usable and not notes:
        notes.append("照片质量良好，可直接进入风格提取流程")
    elif usable and notes:
        notes.insert(0, "照片基本可用，建议改进以下点以获得更好的效果")

    return "；".join(notes), usable


def assess_photo_quality(
    image: Union[str, Path, Image.Image, np.ndarray]
) -> PhotoQualityReport:
    """Assess a handwriting photograph and return a :class:`PhotoQualityReport`.

    Args:
        image: File path, PIL image or numpy array.

    Returns:
        Detailed quality report including a recommendation string.
    """
    gray = _load_grayscale(image)
    height, width = gray.shape[:2]
    resolution_ok = min(width, height) >= _MIN_SIDE_PX

    blur_score = _estimate_blur(gray)
    char_count, ink_ratio = _estimate_char_count(gray)

    recommendation, is_usable = _build_recommendation(
        resolution_ok=resolution_ok,
        blur_score=blur_score,
        estimated_char_count=char_count,
        ink_ratio=ink_ratio,
    )

    return PhotoQualityReport(
        resolution_ok=resolution_ok,
        blur_score=round(blur_score, 2),
        estimated_char_count=int(char_count),
        ink_ratio=round(ink_ratio, 4),
        recommendation=recommendation,
        is_usable=is_usable,
        image_width=int(width),
        image_height=int(height),
    )


__all__ = ["PhotoQualityReport", "assess_photo_quality"]
