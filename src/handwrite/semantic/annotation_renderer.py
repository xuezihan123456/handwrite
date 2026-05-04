"""Visual annotation rendering for semantic-aware typesetting.

Draws decorations (underline, wavy line, circle, background highlight)
onto a PIL Image at specified character positions. The renderer operates
on a per-line basis, receiving bounding boxes of characters that belong
to an annotated segment.
"""

from __future__ import annotations

import math
from typing import Sequence

from PIL import Image, ImageDraw

from handwrite.semantic.layout_planner import Decoration, INK_BLACK

# Default alpha for highlight backgrounds (0-255)
_HIGHLIGHT_ALPHA = 60


def _resolve_color(
    color: tuple[int, int, int], mode: str
) -> tuple[int, int, int] | int:
    """Return a fill value compatible with the given image mode.

    PIL ``ImageDraw`` on *L* (grayscale) images requires an **int** fill,
    whereas *RGB*/*RGBA* images accept tuples.  When the page is in *L*
    mode the RGB colour is converted to a perceptual-grey value.
    """
    if mode == "L":
        r, g, b = color
        return int(0.299 * r + 0.587 * g + 0.114 * b)
    return color


def render_annotations(
    page: Image.Image,
    char_boxes: Sequence[tuple[int, int, int, int]],
    decoration: Decoration,
    color: tuple[int, int, int] = INK_BLACK,
    line_width: int = 3,
) -> Image.Image:
    """Draw the specified decoration over the given character bounding boxes.

    Parameters
    ----------
    page:
        The page image to annotate (modified in-place and returned).
    char_boxes:
        Sequence of ``(x, y, width, height)`` rectangles for each character
        that belongs to the annotated segment. All boxes should be on the
        same line (similar ``y`` values).
    decoration:
        The kind of decoration to draw.
    color:
        RGB colour for the decoration.
    line_width:
        Stroke width in pixels.

    Returns
    -------
    Image.Image
        The annotated page image (same object as *page*).
    """
    if not char_boxes or decoration == Decoration.NONE:
        return page

    # Resolve colour to match the image mode.  PIL ImageDraw on "L" mode
    # images requires an *int* fill, not an RGB tuple.  The highlight
    # path works in RGBA internally so it keeps the original tuple.
    draw_color: tuple[int, int, int] | int = _resolve_color(color, page.mode)

    draw = ImageDraw.Draw(page)

    if decoration == Decoration.UNDERLINE:
        _draw_underline(draw, char_boxes, draw_color, line_width)
    elif decoration == Decoration.WAVE_UNDERLINE:
        _draw_wave_underline(draw, char_boxes, draw_color, line_width)
    elif decoration == Decoration.CIRCLE:
        _draw_circle(draw, char_boxes, draw_color, line_width)
    elif decoration == Decoration.HIGHLIGHT:
        page = _draw_highlight(page, char_boxes, color)

    return page


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _line_extents(
    char_boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[int, int, int]:
    """Compute ``(left_x, right_x, baseline_y)`` from character boxes."""
    left = min(box[0] for box in char_boxes)
    right = max(box[0] + box[2] for box in char_boxes)
    bottom = max(box[1] + box[3] for box in char_boxes)
    return left, right, bottom


def _draw_underline(
    draw: ImageDraw.ImageDraw,
    char_boxes: Sequence[tuple[int, int, int, int]],
    color: tuple[int, int, int] | int,
    line_width: int,
) -> None:
    """Draw a straight underline beneath the character boxes."""
    left, right, bottom = _line_extents(char_boxes)
    y = bottom + 2
    draw.line([(left, y), (right, y)], fill=color, width=line_width)


def _draw_wave_underline(
    draw: ImageDraw.ImageDraw,
    char_boxes: Sequence[tuple[int, int, int, int]],
    color: tuple[int, int, int] | int,
    line_width: int,
) -> None:
    """Draw a wavy underline beneath the character boxes."""
    left, right, bottom = _line_extents(char_boxes)
    y_base = bottom + 2
    amplitude = max(3, line_width)
    wavelength = max(12, line_width * 4)

    points: list[tuple[float, float]] = []
    for x in range(left, right + 1, 2):
        phase = (x - left) / wavelength * 2 * math.pi
        y = y_base + amplitude * math.sin(phase)
        points.append((x, y))

    if len(points) >= 2:
        draw.line(points, fill=color, width=max(1, line_width - 1))


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    char_boxes: Sequence[tuple[int, int, int, int]],
    color: tuple[int, int, int] | int,
    line_width: int,
) -> None:
    """Draw an ellipse around the character boxes (used for list bullets)."""
    padding = max(4, line_width * 2)
    left = min(box[0] for box in char_boxes) - padding
    top = min(box[1] for box in char_boxes) - padding
    right = max(box[0] + box[2] for box in char_boxes) + padding
    bottom = max(box[1] + box[3] for box in char_boxes) + padding
    draw.ellipse([(left, top), (right, bottom)], outline=color, width=line_width)


def _draw_highlight(
    page: Image.Image,
    char_boxes: Sequence[tuple[int, int, int, int]],
    base_color: tuple[int, int, int],
) -> Image.Image:
    """Draw a semi-transparent background highlight behind the text.

    Because the page may be grayscale ("L" mode), we overlay a separate
    RGBA layer and composite it.
    """
    if page.mode != "RGBA":
        overlay = page.convert("RGBA")
    else:
        overlay = page.copy()

    highlight = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    h_draw = ImageDraw.Draw(highlight)

    padding = 2
    for box in char_boxes:
        x, y, w, h = box
        h_draw.rectangle(
            (x - padding, y - padding, x + w + padding, y + h + padding),
            fill=(*base_color, _HIGHLIGHT_ALPHA),
        )

    result = Image.alpha_composite(overlay, highlight)
    return result.convert(page.mode)


__all__ = ["render_annotations"]
