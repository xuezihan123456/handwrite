"""Geometric figure primitives drawn onto transparent overlays.

Every drawing helper returns a transparent ``RGBA`` :class:`PIL.Image.Image`
of the requested ``size``.  The figures are rendered in black ink on the
alpha channel so they can be pasted onto any background.
"""

from __future__ import annotations

import math
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_INK = (20, 20, 20, 255)
_DEFAULT_STROKE_WIDTH = 3
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(8, int(size))
    cached = _FONT_CACHE.get(size)
    if cached is not None:
        return cached
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _create_overlay(size: tuple[int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Create a transparent RGBA overlay of the given size."""
    width = max(1, int(size[0]))
    height = max(1, int(size[1]))
    overlay = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
    return overlay, ImageDraw.Draw(overlay)


def _clamp_color(color: object | None) -> tuple[int, int, int, int]:
    if color is None:
        return _DEFAULT_INK
    if isinstance(color, tuple):
        if len(color) == 3:
            r, g, b = color
            return (int(r), int(g), int(b), 255)
        if len(color) == 4:
            r, g, b, a = color
            return (int(r), int(g), int(b), int(a))
    if isinstance(color, int):
        v = int(color)
        return (v, v, v, 255)
    return _DEFAULT_INK


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def draw_circle(
    size: tuple[int, int],
    center: tuple[int, int],
    radius: int,
    *,
    color: object | None = None,
    width: int = _DEFAULT_STROKE_WIDTH,
    label: str | None = None,
    label_offset: tuple[int, int] = (8, 8),
) -> Image.Image:
    """Draw a circle outline onto a transparent overlay."""
    overlay, draw = _create_overlay(size)
    ink = _clamp_color(color)
    cx, cy = int(center[0]), int(center[1])
    r = max(1, int(radius))
    bbox = (cx - r, cy - r, cx + r, cy + r)
    draw.ellipse(bbox, outline=ink, width=max(1, int(width)))
    # mark the centre with a small dot so the figure is unambiguous.
    draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=ink)
    if label:
        font = _load_font(max(14, r // 4))
        draw.text((cx + label_offset[0], cy + label_offset[1]), label, fill=ink, font=font)
    return overlay


def draw_triangle(
    size: tuple[int, int],
    vertices: Sequence[tuple[int, int]],
    *,
    color: object | None = None,
    width: int = _DEFAULT_STROKE_WIDTH,
    labels: Sequence[str] | None = None,
) -> Image.Image:
    """Draw a triangle outline through the provided three vertices."""
    if len(vertices) != 3:
        raise ValueError("triangle requires exactly three vertices")

    overlay, draw = _create_overlay(size)
    ink = _clamp_color(color)
    pts = [(int(x), int(y)) for x, y in vertices]
    line_width = max(1, int(width))
    draw.line([pts[0], pts[1]], fill=ink, width=line_width)
    draw.line([pts[1], pts[2]], fill=ink, width=line_width)
    draw.line([pts[2], pts[0]], fill=ink, width=line_width)
    # Mark each vertex with a small dot for visibility.
    for x, y in pts:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=ink)

    if labels:
        font = _load_font(20)
        for (vx, vy), name in zip(pts, labels):
            if not name:
                continue
            draw.text((vx + 6, vy - 22), name, fill=ink, font=font)

    return overlay


def draw_rectangle(
    size: tuple[int, int],
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    *,
    color: object | None = None,
    width: int = _DEFAULT_STROKE_WIDTH,
    label: str | None = None,
) -> Image.Image:
    """Draw an axis-aligned rectangle outline."""
    overlay, draw = _create_overlay(size)
    ink = _clamp_color(color)
    x0, y0 = int(top_left[0]), int(top_left[1])
    x1, y1 = int(bottom_right[0]), int(bottom_right[1])
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    draw.rectangle((x0, y0, x1, y1), outline=ink, width=max(1, int(width)))
    if label:
        font = _load_font(18)
        draw.text((x0 + 6, y0 + 6), label, fill=ink, font=font)
    return overlay


def draw_line(
    size: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: object | None = None,
    width: int = _DEFAULT_STROKE_WIDTH,
    label: str | None = None,
) -> Image.Image:
    """Draw a straight line segment between ``start`` and ``end``."""
    overlay, draw = _create_overlay(size)
    ink = _clamp_color(color)
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    draw.line([(x0, y0), (x1, y1)], fill=ink, width=max(1, int(width)))
    if label:
        font = _load_font(16)
        mx = (x0 + x1) // 2
        my = (y0 + y1) // 2
        draw.text((mx + 4, my - 18), label, fill=ink, font=font)
    return overlay


def draw_axes(
    size: tuple[int, int],
    origin: tuple[int, int] | None = None,
    *,
    color: object | None = None,
    width: int = _DEFAULT_STROKE_WIDTH,
    x_label: str = "x",
    y_label: str = "y",
    tick_step: int = 40,
) -> Image.Image:
    """Draw an x/y coordinate axis pair with labels and tick marks."""
    overlay, draw = _create_overlay(size)
    ink = _clamp_color(color)
    canvas_w, canvas_h = overlay.size
    if origin is None:
        origin = (canvas_w // 2, canvas_h // 2)
    ox, oy = int(origin[0]), int(origin[1])

    line_width = max(1, int(width))
    draw.line([(0, oy), (canvas_w - 1, oy)], fill=ink, width=line_width)
    draw.line([(ox, 0), (ox, canvas_h - 1)], fill=ink, width=line_width)

    # Arrowheads.
    draw.polygon(
        [(canvas_w - 1, oy), (canvas_w - 14, oy - 7), (canvas_w - 14, oy + 7)],
        fill=ink,
    )
    draw.polygon(
        [(ox, 0), (ox - 7, 14), (ox + 7, 14)],
        fill=ink,
    )

    # Ticks.
    step = max(8, int(tick_step))
    for x in range(ox + step, canvas_w - 14, step):
        draw.line([(x, oy - 5), (x, oy + 5)], fill=ink, width=1)
    for x in range(ox - step, 14, -step):
        draw.line([(x, oy - 5), (x, oy + 5)], fill=ink, width=1)
    for y in range(oy + step, canvas_h - 14, step):
        draw.line([(ox - 5, y), (ox + 5, y)], fill=ink, width=1)
    for y in range(oy - step, 14, -step):
        draw.line([(ox - 5, y), (ox + 5, y)], fill=ink, width=1)

    font = _load_font(18)
    draw.text((canvas_w - 24, oy + 6), x_label, fill=ink, font=font)
    draw.text((ox + 8, 4), y_label, fill=ink, font=font)
    draw.text((ox + 6, oy + 4), "O", fill=ink, font=font)

    return overlay


def draw_angle_arc(
    size: tuple[int, int],
    vertex: tuple[int, int],
    radius: int,
    start_angle: float,
    end_angle: float,
    *,
    color: object | None = None,
    width: int = _DEFAULT_STROKE_WIDTH,
    label: str | None = None,
) -> Image.Image:
    """Draw an arc representing an angle at ``vertex``.

    Angles are expressed in degrees following Pillow's convention
    (0 = east, 90 = south).  ``start_angle == end_angle`` produces a
    degenerate arc which is rendered as a small dot for visibility.
    A full 360-degree sweep produces a closed circle.
    """
    overlay, draw = _create_overlay(size)
    ink = _clamp_color(color)
    vx, vy = int(vertex[0]), int(vertex[1])
    r = max(1, int(radius))
    bbox = (vx - r, vy - r, vx + r, vy + r)

    span = float(end_angle) - float(start_angle)
    line_width = max(1, int(width))

    if abs(span) < 0.5:
        # Degenerate sweep — draw a small marker at the vertex so the
        # overlay still contains visible pixels.
        draw.ellipse((vx - 3, vy - 3, vx + 3, vy + 3), fill=ink)
    elif abs(span) >= 359.5:
        draw.ellipse(bbox, outline=ink, width=line_width)
    else:
        draw.arc(bbox, start=float(start_angle), end=float(end_angle), fill=ink, width=line_width)

    if label:
        # Place the label near the bisector of the angle.
        mid_deg = (float(start_angle) + float(end_angle)) / 2.0
        rad = math.radians(mid_deg)
        lx = int(vx + (r + 8) * math.cos(rad))
        ly = int(vy + (r + 8) * math.sin(rad))
        font = _load_font(16)
        draw.text((lx, ly), label, fill=ink, font=font)
    return overlay


def draw_arrow(
    size: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: object | None = None,
    width: int = _DEFAULT_STROKE_WIDTH,
    head_size: int = 12,
    label: str | None = None,
) -> Image.Image:
    """Draw an arrow from ``start`` to ``end`` with a triangular head."""
    overlay, draw = _create_overlay(size)
    ink = _clamp_color(color)
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    line_width = max(1, int(width))
    draw.line([(int(x0), int(y0)), (int(x1), int(y1))], fill=ink, width=line_width)

    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length >= 1.0:
        ux = dx / length
        uy = dy / length
        hs = max(4, int(head_size))
        left = (
            int(x1 - hs * ux - hs * 0.55 * uy),
            int(y1 - hs * uy + hs * 0.55 * ux),
        )
        right = (
            int(x1 - hs * ux + hs * 0.55 * uy),
            int(y1 - hs * uy - hs * 0.55 * ux),
        )
        draw.polygon([(int(x1), int(y1)), left, right], fill=ink)

    if label:
        font = _load_font(14)
        mx = int((x0 + x1) // 2)
        my = int((y0 + y1) // 2)
        draw.text((mx + 4, my - 18), label, fill=ink, font=font)
    return overlay


def draw_labeled_point(
    size: tuple[int, int],
    position: tuple[int, int],
    label: str = "",
    *,
    color: object | None = None,
    radius: int = 5,
    label_offset: tuple[int, int] = (8, -18),
) -> Image.Image:
    """Draw a filled dot with an optional text label."""
    overlay, draw = _create_overlay(size)
    ink = _clamp_color(color)
    px, py = int(position[0]), int(position[1])
    r = max(1, int(radius))
    draw.ellipse((px - r, py - r, px + r, py + r), fill=ink)
    if label:
        font = _load_font(16)
        draw.text((px + label_offset[0], py + label_offset[1]), label, fill=ink, font=font)
    return overlay
