"""Authenticity scoring for handwriting images.

Evaluates how closely generated handwriting resembles genuine handwriting
by analyzing stroke continuity, thickness variation, spacing consistency,
and detecting print-like artifacts.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from handwrite.quality.quality_report import DimensionScore

# Scoring weights for each sub-dimension
_WEIGHT_STROKE_CONTINUITY = 0.30
_WEIGHT_THICKNESS_VARIATION = 0.25
_WEIGHT_SPACING_CONSISTENCY = 0.20
_WEIGHT_PRINT_FEEL = 0.25

_INK_THRESHOLD = 128


def _to_binary_array(image: Image.Image) -> np.ndarray:
    """Convert image to binary array (True where ink is present)."""
    grayscale = image.convert("L")
    arr = np.array(grayscale, dtype=np.float32)
    return arr < _INK_THRESHOLD


def score_authenticity(image: Image.Image) -> DimensionScore:
    """Compute overall authenticity score for a handwriting image.

    Args:
        image: Grayscale or RGBA handwriting image.

    Returns:
        DimensionScore with weighted sub-dimension scores and suggestions.
    """
    binary = _to_binary_array(image)
    if not binary.any():
        return DimensionScore(
            name="真实性",
            score=0.0,
            weight=1.0,
            details="图像无墨迹",
            suggestions=("图像中未检测到笔画，请检查输入",),
        )

    continuity = _score_stroke_continuity(binary)
    thickness = _score_thickness_variation(binary)
    spacing = _score_spacing_consistency(binary)
    print_feel = _score_print_feel(binary)

    weighted = (
        continuity.weighted_score
        + thickness.weighted_score
        + spacing.weighted_score
        + print_feel.weighted_score
    )
    total_weight = (
        _WEIGHT_STROKE_CONTINUITY
        + _WEIGHT_THICKNESS_VARIATION
        + _WEIGHT_SPACING_CONSISTENCY
        + _WEIGHT_PRINT_FEEL
    )
    overall = weighted / total_weight if total_weight > 0 else 0.0

    all_suggestions: list[str] = []
    for dim in (continuity, thickness, spacing, print_feel):
        all_suggestions.extend(dim.suggestions)

    return DimensionScore(
        name="真实性",
        score=round(overall, 1),
        weight=1.0,
        details=(
            f"笔画连续性={continuity.score:.0f}, "
            f"粗细变化={thickness.score:.0f}, "
            f"间距一致性={spacing.score:.0f}, "
            f"打印感={print_feel.score:.0f}"
        ),
        suggestions=tuple(all_suggestions),
    )


def _score_stroke_continuity(binary: np.ndarray) -> DimensionScore:
    """Evaluate stroke continuity by analyzing edge smoothness.

    Real handwriting has smooth, continuous strokes. Discontinuities
    (jagged edges, broken strokes) reduce the score.
    """
    height, width = binary.shape
    ink_pixels = int(binary.sum())
    if ink_pixels == 0:
        return DimensionScore(
            name="笔画连续性", score=0.0, weight=_WEIGHT_STROKE_CONTINUITY
        )

    # Find edge pixels: ink pixels adjacent to non-ink pixels
    shifted_right = np.zeros_like(binary)
    shifted_right[:, 1:] = binary[:, :-1]
    shifted_down = np.zeros_like(binary)
    shifted_down[1:, :] = binary[:-1, :]

    edge_horizontal = binary ^ shifted_right
    edge_vertical = binary ^ shifted_down
    edge_pixels = edge_horizontal | edge_vertical

    if not edge_pixels.any():
        return DimensionScore(
            name="笔画连续性", score=100.0, weight=_WEIGHT_STROKE_CONTINUITY
        )

    # Measure edge smoothness via gradient direction consistency
    # Compute gradients using Sobel-like approach
    gy, gx = np.gradient(binary.astype(np.float32))
    gradient_mag = np.sqrt(gx ** 2 + gy ** 2)

    edge_indices = np.argwhere(edge_pixels)
    if len(edge_indices) < 2:
        return DimensionScore(
            name="笔画连续性", score=80.0, weight=_WEIGHT_STROKE_CONTINUITY
        )

    # Sample edge points for direction consistency
    sample_size = min(500, len(edge_indices))
    sampled = edge_indices[
        np.linspace(0, len(edge_indices) - 1, sample_size, dtype=int)
    ]

    angles = []
    for y, x in sampled:
        mag = gradient_mag[y, x]
        if mag > 0.1:
            angle = math.atan2(gy[y, x], gx[y, x])
            angles.append(angle)

    if len(angles) < 2:
        return DimensionScore(
            name="笔画连续性", score=70.0, weight=_WEIGHT_STROKE_CONTINUITY
        )

    # Compute angular coherence (how consistent are neighboring edge directions)
    angle_arr = np.array(angles)
    # Circular variance: 1 - |mean unit vector|
    mean_cos = np.mean(np.cos(2 * angle_arr))
    mean_sin = np.mean(np.sin(2 * angle_arr))
    coherence = math.sqrt(mean_cos ** 2 + mean_sin ** 2)

    # Higher coherence = smoother edges = more authentic
    score = min(100.0, coherence * 120)

    suggestions: list[str] = []
    if score < 60:
        suggestions.append("笔画边缘锯齿较多，建议增大渲染分辨率或添加抗锯齿处理")
    elif score < 75:
        suggestions.append("笔画边缘略有粗糙，可适当平滑处理")

    return DimensionScore(
        name="笔画连续性",
        score=round(score, 1),
        weight=_WEIGHT_STROKE_CONTINUITY,
        suggestions=tuple(suggestions),
    )


def _score_thickness_variation(binary: np.ndarray) -> DimensionScore:
    """Evaluate stroke thickness variation.

    Real handwriting has natural thickness variation; overly uniform
    strokes look printed.
    """
    ink_pixels = int(binary.sum())
    if ink_pixels == 0:
        return DimensionScore(
            name="粗细变化", score=0.0, weight=_WEIGHT_THICKNESS_VARIATION
        )

    # Measure stroke width at multiple horizontal cross-sections
    heights = binary.shape[0]
    widths_at_rows: list[int] = []

    for y in range(heights):
        row = binary[y, :]
        if not row.any():
            continue
        ink_cols = np.where(row)[0]
        if len(ink_cols) >= 2:
            # Count contiguous ink segments
            diffs = np.diff(ink_cols)
            segments = np.split(ink_cols, np.where(diffs > 2)[0] + 1)
            for seg in segments:
                if len(seg) >= 2:
                    widths_at_rows.append(seg[-1] - seg[0] + 1)

    if len(widths_at_rows) < 3:
        return DimensionScore(
            name="粗细变化",
            score=60.0,
            weight=_WEIGHT_THICKNESS_VARIATION,
            suggestions=("笔画样本不足，难以评估粗细变化",),
        )

    width_arr = np.array(widths_at_rows, dtype=np.float64)
    mean_w = width_arr.mean()
    if mean_w < 1:
        mean_w = 1.0

    cv = width_arr.std() / mean_w  # coefficient of variation

    # Ideal CV for handwriting is around 0.2-0.5
    # Too low (<0.1) = too uniform (printed), too high (>0.8) = chaotic
    if cv < 0.05:
        score = 30.0
    elif cv < 0.15:
        score = 50.0 + (cv - 0.05) / 0.10 * 25
    elif cv < 0.5:
        score = 75.0 + (cv - 0.15) / 0.35 * 20
    else:
        score = max(40.0, 95.0 - (cv - 0.5) * 80)

    score = max(0.0, min(100.0, score))

    suggestions: list[str] = []
    if cv < 0.1:
        suggestions.append("笔画粗细过于均匀，建议增加自然的压力变化")
    elif cv > 0.6:
        suggestions.append("笔画粗细变化过大，建议控制书写力度的稳定性")

    return DimensionScore(
        name="粗细变化",
        score=round(score, 1),
        weight=_WEIGHT_THICKNESS_VARIATION,
        suggestions=tuple(suggestions),
    )


def _score_spacing_consistency(binary: np.ndarray) -> DimensionScore:
    """Evaluate character/component spacing consistency.

    Moderate spacing variation is natural; too uniform or too erratic
    looks artificial.
    """
    # Find connected components (characters or strokes) via horizontal projection
    col_projection = binary.any(axis=0).astype(np.int32)
    if col_projection.sum() == 0:
        return DimensionScore(
            name="间距一致性", score=50.0, weight=_WEIGHT_SPACING_CONSISTENCY
        )

    # Find gaps between ink regions
    in_ink = False
    gaps: list[int] = []
    current_gap = 0
    ink_regions: list[int] = []
    current_ink = 0

    for val in col_projection:
        if val:
            if not in_ink:
                if current_gap > 0:
                    gaps.append(current_gap)
                in_ink = True
                current_ink = 1
            else:
                current_ink += 1
            current_gap = 0
        else:
            if in_ink:
                ink_regions.append(current_ink)
                in_ink = False
                current_gap = 1
            else:
                current_gap += 1

    if in_ink:
        ink_regions.append(current_ink)

    if len(gaps) < 2:
        return DimensionScore(
            name="间距一致性",
            score=70.0,
            weight=_WEIGHT_SPACING_CONSISTENCY,
            details="间距样本不足",
        )

    gap_arr = np.array(gaps, dtype=np.float64)
    # Filter out very large gaps (likely between words/lines)
    median_gap = np.median(gap_arr)
    if median_gap > 0:
        reasonable = gap_arr[gap_arr < median_gap * 3]
        if len(reasonable) >= 2:
            gap_arr = reasonable

    if len(gap_arr) < 2:
        return DimensionScore(
            name="间距一致性", score=70.0, weight=_WEIGHT_SPACING_CONSISTENCY
        )

    mean_gap = gap_arr.mean()
    if mean_gap < 1:
        mean_gap = 1.0
    cv = gap_arr.std() / mean_gap

    # Ideal CV for spacing: 0.1-0.3
    if cv < 0.05:
        score = 55.0  # Too uniform
    elif cv < 0.15:
        score = 75.0 + (cv - 0.05) / 0.10 * 15
    elif cv < 0.35:
        score = 90.0 + (cv - 0.15) / 0.20 * 10
    else:
        score = max(30.0, 100.0 - (cv - 0.35) * 100)

    score = max(0.0, min(100.0, score))

    suggestions: list[str] = []
    if cv < 0.05:
        suggestions.append("字间距过于均匀，建议增加自然的间距波动")
    elif cv > 0.4:
        suggestions.append("字间距不一致，建议注意控制字距的稳定性")

    return DimensionScore(
        name="间距一致性",
        score=round(score, 1),
        weight=_WEIGHT_SPACING_CONSISTENCY,
        suggestions=tuple(suggestions),
    )


def _score_print_feel(binary: np.ndarray) -> DimensionScore:
    """Detect print-like artifacts: overly straight edges and uniform strokes.

    Returns a higher score for more natural (less printed) looking text.
    """
    ink_pixels = int(binary.sum())
    if ink_pixels == 0:
        return DimensionScore(
            name="打印感检测", score=0.0, weight=_WEIGHT_PRINT_FEEL
        )

    straight_penalty = _detect_straight_edges(binary)
    uniformity_penalty = _detect_stroke_uniformity(binary)

    # Start from 100 and subtract penalties
    score = 100.0 - straight_penalty * 50 - uniformity_penalty * 50
    score = max(0.0, min(100.0, score))

    suggestions: list[str] = []
    if straight_penalty > 0.5:
        suggestions.append("笔画边缘过于笔直，建议增加微小的手抖效果")
    if uniformity_penalty > 0.5:
        suggestions.append("笔画宽度过于均匀，建议模拟真实书写的压力变化")

    return DimensionScore(
        name="打印感检测",
        score=round(score, 1),
        weight=_WEIGHT_PRINT_FEEL,
        suggestions=tuple(suggestions),
    )


def _detect_straight_edges(binary: np.ndarray) -> float:
    """Detect overly straight edges.

    Returns penalty in [0, 1] where 1 means extremely straight (printed).
    """
    # Use Hough-like approach: check how many edge pixels lie on straight lines
    height, width = binary.shape

    # Find edge pixels
    shifted_right = np.zeros_like(binary)
    shifted_right[:, 1:] = binary[:, :-1]
    shifted_down = np.zeros_like(binary)
    shifted_down[1:, :] = binary[:-1, :]
    edges = binary ^ (shifted_right | shifted_down)

    if not edges.any():
        return 0.0

    edge_y, edge_x = np.argwhere(edges).T
    n_edges = len(edge_y)
    if n_edges < 10:
        return 0.0

    # Check horizontal and vertical line alignment
    # Count edge pixels that align in rows/columns
    row_counts = np.bincount(edge_y, minlength=height)
    col_counts = np.bincount(edge_x, minlength=width)

    # Pixels on lines with many aligned edges suggest straight segments
    straight_row_pixels = sum(
        row_counts[y] for y in range(height) if row_counts[y] > width * 0.1
    )
    straight_col_pixels = sum(
        col_counts[x] for x in range(width) if col_counts[x] > height * 0.1
    )

    total_edge = n_edges
    straight_ratio = (straight_row_pixels + straight_col_pixels) / max(total_edge, 1)

    # Also check diagonal alignment (45-degree lines)
    diag_sum = np.zeros(height + width, dtype=np.int32)
    diag_diff = np.zeros(height + width, dtype=np.int32)
    for y, x in zip(edge_y, edge_x):
        diag_sum[y + x] += 1
        diag_diff[y - x + width] += 1

    max_diag_aligned = max(diag_sum.max(), diag_diff.max())
    diag_ratio = max_diag_aligned / max(total_edge, 1)

    penalty = min(1.0, straight_ratio * 0.7 + diag_ratio * 0.3)
    return penalty


def _detect_stroke_uniformity(binary: np.ndarray) -> float:
    """Detect overly uniform stroke widths.

    Returns penalty in [0, 1] where 1 means extremely uniform (printed).
    """
    heights, widths = binary.shape
    stroke_widths: list[int] = []

    # Sample horizontal and vertical cross-sections
    step = max(1, heights // 20)
    for y in range(0, heights, step):
        row = binary[y, :]
        ink_cols = np.where(row)[0]
        if len(ink_cols) >= 2:
            diffs = np.diff(ink_cols)
            segments = np.split(ink_cols, np.where(diffs > 2)[0] + 1)
            for seg in segments:
                if len(seg) >= 2:
                    stroke_widths.append(len(seg))

    step = max(1, widths // 20)
    for x in range(0, widths, step):
        col = binary[:, x]
        ink_rows = np.where(col)[0]
        if len(ink_rows) >= 2:
            diffs = np.diff(ink_rows)
            segments = np.split(ink_rows, np.where(diffs > 2)[0] + 1)
            for seg in segments:
                if len(seg) >= 2:
                    stroke_widths.append(len(seg))

    if len(stroke_widths) < 3:
        return 0.0

    sw = np.array(stroke_widths, dtype=np.float64)
    mean_sw = sw.mean()
    if mean_sw < 1:
        return 0.0

    cv = sw.std() / mean_sw
    # Lower CV = more uniform = higher penalty
    # CV < 0.05 is suspiciously uniform
    if cv < 0.05:
        return 0.9
    elif cv < 0.15:
        return 0.5 + (0.15 - cv) / 0.10 * 0.4
    elif cv < 0.3:
        return max(0.0, (0.3 - cv) / 0.15 * 0.3)
    return 0.0


__all__ = ["score_authenticity"]
