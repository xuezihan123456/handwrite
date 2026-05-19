"""Render 3D pen strokes back to a 2D image with tilt-aware stroke width.

The replay routine walks every adjacent pair of ``PenSample3D`` and
draws a line segment whose half-width is a function of the pen's
pressure **and** tilt magnitude. Higher pressure and larger tilt both
result in a wider stroke, mirroring how a real pen leaves a broader
mark when held at an angle.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from PIL import Image, ImageDraw

from .samples import PenSample3D, PenStroke3D


def _tilt_width_modulation(sample: PenSample3D) -> float:
    """Combine pressure and tilt into a width modulation factor.

    Output range is roughly ``[0.5, 2.0]``: low pressure + low tilt
    yields the thinnest stroke; high pressure + steep tilt yields the
    widest.
    """
    tilt_magnitude = math.hypot(sample.tilt_x, sample.tilt_y)
    # Normalize tilt to [0, 1] using pi/2 as the practical maximum.
    tilt_norm = min(tilt_magnitude / (math.pi / 2.0), 1.0)
    pressure_term = 0.5 + 1.0 * sample.pressure  # 0.5 .. 1.5
    tilt_term = 0.85 + 0.45 * tilt_norm  # 0.85 .. 1.30
    return float(pressure_term * tilt_term)


def replay_to_image(
    strokes: Sequence[PenStroke3D] | PenStroke3D,
    *,
    canvas_size: tuple[int, int] = (256, 256),
    pen_color: tuple[int, int, int] = (0, 0, 0),
    background: tuple[int, int, int] = (255, 255, 255),
    base_width: float = 2.0,
) -> Image.Image:
    """Render 3D strokes to a 2D RGB image.

    Args:
        strokes: A single ``PenStroke3D`` or a sequence of them.
        canvas_size: Output ``(width, height)`` in pixels.
        pen_color: RGB pen color.
        background: RGB background color.
        base_width: Reference stroke width (pixels) before tilt /
            pressure modulation.

    Returns:
        A new PIL ``Image`` in ``RGB`` mode.

    Raises:
        ValueError: If ``strokes`` is empty or contains no samples.
    """
    if isinstance(strokes, PenStroke3D):
        strokes_list: list[PenStroke3D] = [strokes]
    else:
        strokes_list = list(strokes)

    if not strokes_list:
        raise ValueError("replay_to_image requires at least one stroke")

    total_samples = sum(len(s.samples) for s in strokes_list)
    if total_samples == 0:
        raise ValueError("replay_to_image requires strokes with samples")

    width, height = canvas_size
    if width <= 0 or height <= 0:
        raise ValueError("canvas_size dimensions must be positive")

    image = Image.new("RGB", (int(width), int(height)), color=background)
    draw = ImageDraw.Draw(image)

    def _clamp_xy(x: float, y: float) -> tuple[float, float]:
        cx = max(0.0, min(float(width) - 1.0, float(x)))
        cy = max(0.0, min(float(height) - 1.0, float(y)))
        return cx, cy

    for stroke in strokes_list:
        samples = stroke.samples
        if len(samples) == 1:
            sample = samples[0]
            radius = max(1.0, 0.5 * base_width * _tilt_width_modulation(sample))
            cx, cy = _clamp_xy(sample.x, sample.y)
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=pen_color,
            )
            continue

        for i in range(len(samples) - 1):
            a = samples[i]
            b = samples[i + 1]
            mod = 0.5 * (_tilt_width_modulation(a) + _tilt_width_modulation(b))
            seg_width = max(1, int(round(base_width * mod)))
            ax, ay = _clamp_xy(a.x, a.y)
            bx, by = _clamp_xy(b.x, b.y)
            draw.line(
                ((ax, ay), (bx, by)),
                fill=pen_color,
                width=seg_width,
            )
            # Also stamp a pen-tip dot so the color is visible even when the
            # stroke runs along a canvas edge with width=1 antialiasing.
            radius = max(1.0, 0.5 * seg_width)
            draw.ellipse(
                (ax - radius, ay - radius, ax + radius, ay + radius),
                fill=pen_color,
            )

    return image


__all__ = ["replay_to_image"]
