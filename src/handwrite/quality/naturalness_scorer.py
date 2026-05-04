"""Naturalness scoring for handwriting images.

Evaluates how natural the handwriting looks by analyzing line alignment,
character size variation, tilt consistency, and ink density patterns.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from handwrite.quality.quality_report import DimensionScore

_WEIGHT_LINE_ALIGNMENT = 0.25
_WEIGHT_SIZE_VARIATION = 0.25
_WEIGHT_TILT_CONSISTENCY = 0.25
_WEIGHT_INK_DENSITY = 0.25

_INK_THRESHOLD = 128


def _to_binary_array(image: Image.Image) -> np.ndarray:
    """Convert image to binary array (True where ink is present)."""
    grayscale = image.convert("L")
    arr = np.array(grayscale, dtype=np.float32)
    return arr < _INK_THRESHOLD


def score_naturalness(image: Image.Image) -> DimensionScore:
    """Compute overall naturalness score for a handwriting image.

    Args:
        image: Grayscale or RGBA handwriting image.

    Returns:
        DimensionScore with weighted sub-dimension scores and suggestions.
    """
    binary = _to_binary_array(image)
    if not binary.any():
        return DimensionScore(
            name="自然度",
            score=0.0,
            weight=1.0,
            details="图像无墨迹",
            suggestions=("图像中未检测到笔画，请检查输入",),
        )

    alignment = _score_line_alignment(binary)
    size_var = _score_size_variation(binary)
    tilt = _score_tilt_consistency(binary)
    ink_density = _score_ink_density(binary)

    weighted = (
        alignment.weighted_score
        + size_var.weighted_score
        + tilt.weighted_score
        + ink_density.weighted_score
    )
    total_weight = (
        _WEIGHT_LINE_ALIGNMENT
        + _WEIGHT_SIZE_VARIATION
        + _WEIGHT_TILT_CONSISTENCY
        + _WEIGHT_INK_DENSITY
    )
    overall = weighted / total_weight if total_weight > 0 else 0.0

    all_suggestions: list[str] = []
    for dim in (alignment, size_var, tilt, ink_density):
        all_suggestions.extend(dim.suggestions)

    return DimensionScore(
        name="自然度",
        score=round(overall, 1),
        weight=1.0,
        details=(
            f"行对齐={alignment.score:.0f}, "
            f"大小变化={size_var.score:.0f}, "
            f"倾斜一致性={tilt.score:.0f}, "
            f"墨迹浓淡={ink_density.score:.0f}"
        ),
        suggestions=tuple(all_suggestions),
    )


def _score_line_alignment(binary: np.ndarray) -> DimensionScore:
    """Evaluate baseline alignment consistency.

    Real handwriting follows a natural baseline with slight variation.
    Too straight or too wavy baselines look unnatural.
    """
    height, width = binary.shape

    # Find connected components via horizontal projection to identify text lines
    row_projection = binary.any(axis=1).astype(np.int32)
    if row_projection.sum() == 0:
        return DimensionScore(
            name="行对齐", score=50.0, weight=_WEIGHT_LINE_ALIGNMENT
        )

    # Find text line regions (contiguous rows with ink)
    in_text = False
    line_ranges: list[tuple[int, int]] = []
    start = 0
    for y in range(height):
        if row_projection[y]:
            if not in_text:
                start = y
                in_text = True
        else:
            if in_text:
                line_ranges.append((start, y))
                in_text = False
    if in_text:
        line_ranges.append((start, height))

    if len(line_ranges) < 2:
        # Single line: check baseline straightness
        return _score_single_line_alignment(binary, line_ranges[0] if line_ranges else (0, height))

    # Multiple lines: check inter-line spacing consistency
    spacings = []
    for i in range(1, len(line_ranges)):
        gap = line_ranges[i][0] - line_ranges[i - 1][1]
        spacings.append(gap)

    if len(spacings) < 1:
        return DimensionScore(
            name="行对齐", score=70.0, weight=_WEIGHT_LINE_ALIGNMENT
        )

    spacing_arr = np.array(spacings, dtype=np.float64)
    mean_spacing = spacing_arr.mean()
    if mean_spacing < 1:
        mean_spacing = 1.0

    cv = spacing_arr.std() / mean_spacing

    # Ideal CV: 0.05-0.2 for natural line spacing
    if cv < 0.02:
        score = 60.0  # Too uniform
    elif cv < 0.1:
        score = 80.0 + (cv - 0.02) / 0.08 * 15
    elif cv < 0.25:
        score = 95.0 - (cv - 0.1) / 0.15 * 10
    else:
        score = max(30.0, 85.0 - (cv - 0.25) * 100)

    score = max(0.0, min(100.0, score))

    suggestions: list[str] = []
    if cv < 0.03:
        suggestions.append("行间距过于均匀，建议增加自然的行距波动")
    elif cv > 0.3:
        suggestions.append("行间距不一致，建议注意行距的稳定性")

    return DimensionScore(
        name="行对齐",
        score=round(score, 1),
        weight=_WEIGHT_LINE_ALIGNMENT,
        suggestions=tuple(suggestions),
    )


def _score_single_line_alignment(
    binary: np.ndarray, line_range: tuple[int, int]
) -> DimensionScore:
    """Score baseline straightness for a single line of text."""
    start_y, end_y = line_range
    height, width = binary.shape

    # Find bottom-most ink pixel per column
    baselines: list[int] = []
    for x in range(width):
        col_slice = binary[start_y:end_y, x]
        ink_rows = np.where(col_slice)[0]
        if len(ink_rows) > 0:
            baselines.append(start_y + ink_rows[-1])

    if len(baselines) < 5:
        return DimensionScore(
            name="行对齐", score=70.0, weight=_WEIGHT_LINE_ALIGNMENT
        )

    baseline_arr = np.array(baselines, dtype=np.float64)
    # Fit a line to the baseline
    x_coords = np.arange(len(baseline_arr), dtype=np.float64)
    coeffs = np.polyfit(x_coords, baseline_arr, 1)
    fitted = np.polyval(coeffs, x_coords)
    residuals = baseline_arr - fitted

    # Residual std relative to character height
    line_height = end_y - start_y
    if line_height < 1:
        line_height = 1.0
    residual_std = residuals.std() / line_height

    # Ideal residual_std: 0.02-0.08
    if residual_std < 0.01:
        score = 60.0  # Too straight
    elif residual_std < 0.05:
        score = 85.0 + (residual_std - 0.01) / 0.04 * 10
    elif residual_std < 0.1:
        score = 95.0 - (residual_std - 0.05) / 0.05 * 15
    else:
        score = max(30.0, 80.0 - (residual_std - 0.1) * 200)

    score = max(0.0, min(100.0, score))

    suggestions: list[str] = []
    if residual_std < 0.01:
        suggestions.append("基线过于平直，建议增加自然的手写波动")
    elif residual_std > 0.12:
        suggestions.append("基线波动过大，建议保持书写的稳定性")

    return DimensionScore(
        name="行对齐",
        score=round(score, 1),
        weight=_WEIGHT_LINE_ALIGNMENT,
        suggestions=tuple(suggestions),
    )


def _score_size_variation(binary: np.ndarray) -> DimensionScore:
    """Evaluate character size variation.

    Natural handwriting has moderate size variation; too uniform or
    too erratic looks unnatural.
    """
    # Find connected components via column projection
    height, width = binary.shape
    col_projection = binary.any(axis=0).astype(np.int32)

    # Find character-width regions
    in_char = False
    char_widths: list[int] = []
    char_heights: list[int] = []
    current_width = 0
    char_start_x = 0

    for x in range(width):
        if col_projection[x]:
            if not in_char:
                char_start_x = x
                current_width = 1
                in_char = True
            else:
                current_width += 1
        else:
            if in_char:
                char_widths.append(current_width)
                # Find height of this character region
                region = binary[:, char_start_x:char_start_x + current_width]
                ink_rows = np.where(region.any(axis=1))[0]
                if len(ink_rows) > 0:
                    char_heights.append(ink_rows[-1] - ink_rows[0] + 1)
                in_char = False

    if in_char:
        char_widths.append(current_width)
        region = binary[:, char_start_x:char_start_x + current_width]
        ink_rows = np.where(region.any(axis=1))[0]
        if len(ink_rows) > 0:
            char_heights.append(ink_rows[-1] - ink_rows[0] + 1)

    if len(char_heights) < 3:
        return DimensionScore(
            name="大小变化",
            score=65.0,
            weight=_WEIGHT_SIZE_VARIATION,
            details="字符样本不足",
        )

    h_arr = np.array(char_heights, dtype=np.float64)
    mean_h = h_arr.mean()
    if mean_h < 1:
        mean_h = 1.0
    cv = h_arr.std() / mean_h

    # Ideal CV: 0.05-0.2
    if cv < 0.02:
        score = 50.0  # Too uniform
    elif cv < 0.1:
        score = 70.0 + (cv - 0.02) / 0.08 * 20
    elif cv < 0.25:
        score = 90.0 + (cv - 0.1) / 0.15 * 10
    else:
        score = max(25.0, 100.0 - (cv - 0.25) * 120)

    score = max(0.0, min(100.0, score))

    suggestions: list[str] = []
    if cv < 0.03:
        suggestions.append("字大小过于一致，建议增加自然的大小变化")
    elif cv > 0.3:
        suggestions.append("字大小差异过大，建议保持书写的匀称性")

    return DimensionScore(
        name="大小变化",
        score=round(score, 1),
        weight=_WEIGHT_SIZE_VARIATION,
        suggestions=tuple(suggestions),
    )


def _score_tilt_consistency(binary: np.ndarray) -> DimensionScore:
    """Evaluate stroke angle/tilt consistency.

    Natural handwriting has a consistent tilt angle with slight variation.
    """
    height, width = binary.shape

    # Find connected components and compute their orientation
    col_projection = binary.any(axis=0).astype(np.int32)

    in_char = False
    char_angles: list[float] = []
    char_start_x = 0
    current_width = 0

    for x in range(width):
        if col_projection[x]:
            if not in_char:
                char_start_x = x
                current_width = 1
                in_char = True
            else:
                current_width += 1
        else:
            if in_char and current_width > 3:
                region = binary[:, char_start_x:char_start_x + current_width]
                angle = _compute_region_angle(region)
                if angle is not None:
                    char_angles.append(angle)
                in_char = False
                current_width = 0
            else:
                in_char = False
                current_width = 0

    if in_char and current_width > 3:
        region = binary[:, char_start_x:char_start_x + current_width]
        angle = _compute_region_angle(region)
        if angle is not None:
            char_angles.append(angle)

    if len(char_angles) < 3:
        return DimensionScore(
            name="倾斜一致性",
            score=70.0,
            weight=_WEIGHT_TILT_CONSISTENCY,
            details="倾斜样本不足",
        )

    angle_arr = np.array(char_angles)
    # Use circular statistics for angle consistency
    mean_cos = np.mean(np.cos(2 * angle_arr))
    mean_sin = np.mean(np.sin(2 * angle_arr))
    consistency = math.sqrt(mean_cos ** 2 + mean_sin ** 2)

    # consistency in [0, 1], 1 = perfectly consistent
    # Ideal: 0.7-0.95
    if consistency < 0.5:
        score = 40.0 + consistency * 60
    elif consistency < 0.8:
        score = 70.0 + (consistency - 0.5) / 0.3 * 20
    elif consistency < 0.95:
        score = 90.0 + (consistency - 0.8) / 0.15 * 10
    else:
        score = 95.0  # Slightly suspicious if too consistent

    # Penalize if too consistent (looks mechanical)
    if consistency > 0.98:
        score = min(score, 85.0)

    score = max(0.0, min(100.0, score))

    suggestions: list[str] = []
    if consistency < 0.5:
        suggestions.append("字体倾斜角度不一致，建议保持统一的书写角度")
    elif consistency > 0.98:
        suggestions.append("倾斜角度过于一致，建议增加自然的微小角度变化")

    return DimensionScore(
        name="倾斜一致性",
        score=round(score, 1),
        weight=_WEIGHT_TILT_CONSISTENCY,
        suggestions=tuple(suggestions),
    )


def _compute_region_angle(region: np.ndarray) -> float | None:
    """Compute the dominant stroke angle of a character region using moments."""
    ink_y, ink_x = np.where(region)
    if len(ink_y) < 5:
        return None

    # Center the points
    cx = ink_x.mean()
    cy = ink_y.mean()
    dx = ink_x - cx
    dy = ink_y - cy

    # Compute covariance matrix
    mu20 = np.mean(dx * dx)
    mu02 = np.mean(dy * dy)
    mu11 = np.mean(dx * dy)

    # Angle of the principal axis
    if abs(mu20 - mu02) < 1e-10 and abs(mu11) < 1e-10:
        return 0.0

    angle = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
    return angle


def _score_ink_density(binary: np.ndarray) -> DimensionScore:
    """Evaluate ink density variation.

    Natural handwriting has varying ink density due to pressure changes.
    Uniform density looks printed.
    """
    height, width = binary.shape

    # Divide image into grid cells and measure ink density per cell
    cell_size = max(8, min(height, width) // 8)
    densities: list[float] = []

    for y in range(0, height - cell_size + 1, cell_size):
        for x in range(0, width - cell_size + 1, cell_size):
            cell = binary[y:y + cell_size, x:x + cell_size]
            ink_ratio = cell.mean()
            if ink_ratio > 0.01:  # Only count cells with some ink
                densities.append(ink_ratio)

    if len(densities) < 3:
        return DimensionScore(
            name="墨迹浓淡",
            score=65.0,
            weight=_WEIGHT_INK_DENSITY,
            details="墨迹样本不足",
        )

    density_arr = np.array(densities)
    mean_d = density_arr.mean()
    if mean_d < 1e-6:
        return DimensionScore(
            name="墨迹浓淡", score=50.0, weight=_WEIGHT_INK_DENSITY
        )

    cv = density_arr.std() / mean_d

    # Ideal CV: 0.1-0.4
    if cv < 0.05:
        score = 45.0  # Too uniform
    elif cv < 0.15:
        score = 65.0 + (cv - 0.05) / 0.10 * 20
    elif cv < 0.4:
        score = 85.0 + (cv - 0.15) / 0.25 * 15
    else:
        score = max(35.0, 100.0 - (cv - 0.4) * 80)

    score = max(0.0, min(100.0, score))

    suggestions: list[str] = []
    if cv < 0.05:
        suggestions.append("墨迹浓淡过于均匀，建议模拟真实书写的压笔变化")
    elif cv > 0.5:
        suggestions.append("墨迹浓淡变化过大，建议保持书写的稳定性")

    return DimensionScore(
        name="墨迹浓淡",
        score=round(score, 1),
        weight=_WEIGHT_INK_DENSITY,
        suggestions=tuple(suggestions),
    )


__all__ = ["score_naturalness"]
