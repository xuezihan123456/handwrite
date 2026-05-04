"""Bezier curve trajectory generation for stroke animation.

Takes raw stroke point sequences and generates smooth Bezier curve
trajectories suitable for animation rendering.
"""

from __future__ import annotations

import math

import numpy as np


def generate_trajectories(
    strokes: list[list[tuple[int, int]]],
    samples_per_stroke: int = 40,
) -> list[list[tuple[float, float]]]:
    """Generate smooth trajectories from raw stroke points.

    For each stroke, fits a series of cubic Bezier segments and resamples
    the result to produce evenly-spaced points.

    Args:
        strokes: Ordered list of strokes, each a list of (x, y) points.
        samples_per_stroke: Number of output points per stroke.

    Returns:
        List of smooth trajectories, each a list of (x, y) float tuples.
    """
    trajectories: list[list[tuple[float, float]]] = []

    for stroke in strokes:
        if len(stroke) < 2:
            continue

        # Convert to float
        points = [(float(x), float(y)) for x, y in stroke]

        # Simplify path to reduce noise
        simplified = _simplify_path(points, epsilon=1.5)

        if len(simplified) < 2:
            continue

        # Fit Bezier segments
        bezier_segments = _fit_bezier_spline(simplified)

        # Sample the Bezier spline
        sampled = _sample_bezier_spline(bezier_segments, samples_per_stroke)

        if len(sampled) >= 2:
            trajectories.append(sampled)

    return trajectories


def evaluate_bezier(
    control_points: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    t: float,
) -> tuple[float, float]:
    """Evaluate a cubic Bezier curve at parameter t in [0, 1].

    B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
    """
    p0, p1, p2, p3 = control_points
    u = 1.0 - t
    u2 = u * u
    u3 = u2 * u
    t2 = t * t
    t3 = t2 * t

    x = u3 * p0[0] + 3 * u2 * t * p1[0] + 3 * u * t2 * p2[0] + t3 * p3[0]
    y = u3 * p0[1] + 3 * u2 * t * p1[1] + 3 * u * t2 * p2[1] + t3 * p3[1]

    return (x, y)


def _simplify_path(
    points: list[tuple[float, float]],
    epsilon: float,
) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker path simplification."""
    if len(points) < 3:
        return points

    # Find the point with maximum distance from the line start->end
    start = points[0]
    end = points[-1]

    max_dist = 0.0
    max_index = 0

    for i in range(1, len(points) - 1):
        dist = _point_to_line_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            max_index = i

    if max_dist > epsilon:
        # Recursive simplification
        left = _simplify_path(points[: max_index + 1], epsilon)
        right = _simplify_path(points[max_index:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def _point_to_line_distance(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    """Calculate the perpendicular distance from a point to a line segment."""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq == 0:
        return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    # Project point onto line
    t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return math.sqrt((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2)


def _fit_bezier_spline(
    points: list[tuple[float, float]],
    max_segment_length: int = 30,
) -> list[
    tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
]:
    """Fit a series of cubic Bezier segments to a polyline.

    Breaks the path into segments and fits each one independently.
    """
    n = len(points)
    if n < 2:
        return []

    segments = []
    i = 0
    while i < n - 1:
        end = min(i + max_segment_length, n - 1)
        segment_points = points[i : end + 1]
        bezier = _fit_single_cubic_bezier(segment_points)
        segments.append(bezier)
        i = end

    return segments


def _fit_single_cubic_bezier(
    points: list[tuple[float, float]],
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Fit a single cubic Bezier curve to a set of points using least squares.

    P0 and P3 are fixed as the first and last points.
    P1 and P2 are solved via least squares.
    """
    n = len(points)
    p0 = points[0]
    p3 = points[-1]

    if n < 3:
        # Not enough points for a fit; use default control points
        return (p0, _lerp(p0, p3, 1.0 / 3), _lerp(p0, p3, 2.0 / 3), p3)

    # Chord-length parameterization
    distances = [0.0]
    for i in range(1, n):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        distances.append(distances[-1] + math.sqrt(dx * dx + dy * dy))

    total_length = distances[-1]
    if total_length < 1e-10:
        return (p0, p0, p3, p3)

    t_values = [d / total_length for d in distances]

    # Set up least squares: solve for P1 and P2
    # B(t) = b0*P0 + b1*P1 + b2*P2 + b3*P3
    # => b1*P1 + b2*P2 = point - b0*P0 - b3*P3
    A = np.zeros((n, 2))
    bx = np.zeros(n)
    by = np.zeros(n)

    for i in range(n):
        t = t_values[i]
        u = 1.0 - t
        b0 = u * u * u
        b1 = 3 * u * u * t
        b2 = 3 * u * t * t
        b3 = t * t * t

        A[i, 0] = b1
        A[i, 1] = b2
        bx[i] = points[i][0] - b0 * p0[0] - b3 * p3[0]
        by[i] = points[i][1] - b0 * p0[1] - b3 * p3[1]

    # Solve using least squares
    try:
        result_x, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
        result_y, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
        p1 = (float(result_x[0]), float(result_y[0]))
        p2 = (float(result_x[1]), float(result_y[1]))
    except np.linalg.LinAlgError:
        p1 = _lerp(p0, p3, 1.0 / 3)
        p2 = _lerp(p0, p3, 2.0 / 3)

    return (p0, p1, p2, p3)


def _lerp(
    a: tuple[float, float],
    b: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Linear interpolation between two points."""
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def _sample_bezier_spline(
    segments: list[
        tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ]
    ],
    num_samples: int,
) -> list[tuple[float, float]]:
    """Sample evenly-spaced points along a Bezier spline.

    First computes the total arc length, then distributes samples
    proportional to arc length across segments.
    """
    if not segments or num_samples < 2:
        if segments:
            return [segments[0][0], segments[-1][3]]
        return []

    # Compute arc length of each segment
    segment_lengths = []
    for seg in segments:
        length = _estimate_bezier_length(seg)
        segment_lengths.append(length)

    total_length = sum(segment_lengths)
    if total_length < 1e-10:
        return [segments[0][0]] * num_samples

    # Distribute samples proportional to arc length
    result: list[tuple[float, float]] = []
    for i in range(num_samples):
        target_length = (i / (num_samples - 1)) * total_length

        # Find which segment contains this length
        cumulative = 0.0
        for seg_idx, seg_length in enumerate(segment_lengths):
            if cumulative + seg_length >= target_length or seg_idx == len(segments) - 1:
                # Within this segment
                if seg_length > 1e-10:
                    local_t = (target_length - cumulative) / seg_length
                    local_t = max(0.0, min(1.0, local_t))
                else:
                    local_t = 0.0
                point = evaluate_bezier(segments[seg_idx], local_t)
                result.append(point)
                break
            cumulative += seg_length

    return result


def _estimate_bezier_length(
    control_points: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    num_steps: int = 20,
) -> float:
    """Estimate the arc length of a cubic Bezier curve by sampling."""
    length = 0.0
    prev = control_points[0]
    for i in range(1, num_steps + 1):
        t = i / num_steps
        curr = evaluate_bezier(control_points, t)
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        length += math.sqrt(dx * dx + dy * dy)
        prev = curr
    return length


__all__ = ["generate_trajectories", "evaluate_bezier"]
