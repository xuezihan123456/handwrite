"""Paper renderer: draws paper templates from JSON-driven definitions.

Supported region types:
  - line:           single straight line
  - hline_group:    a group of horizontal lines
  - vline_group:    a group of vertical lines
  - staff_group:    music staff (groups of 5 lines)
  - four_line_group: English four-line three-grid groups
  - text:           label or annotation text
  - ellipse:        ellipse or circle outline
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_DEFAULT_SIZE = (2480, 3508)


def render_paper(definition: dict[str, Any]) -> Image.Image:
    """Render a paper definition dict into a PIL Image.

    Args:
        definition: Paper definition with 'size' and 'regions' keys.

    Returns:
        Grayscale PIL Image of the paper.
    """
    # If the definition carries a background image, load and return it directly.
    image_path = definition.get("_image")
    if image_path:
        return Image.open(image_path).convert("L")

    size = tuple(definition.get("size", _DEFAULT_SIZE))
    regions = definition.get("regions", [])

    paper = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(paper)

    for region in regions:
        _render_region(draw, paper, region)

    return paper


def _render_region(draw: ImageDraw.ImageDraw, paper: Image.Image, region: dict[str, Any]) -> None:
    """Dispatch rendering for a single region definition."""
    rtype = region.get("type", "")
    handler = _REGION_HANDLERS.get(rtype)
    if handler is not None:
        handler(draw, paper, region)


# ---------------------------------------------------------------------------
# Individual region renderers
# ---------------------------------------------------------------------------


def _render_line(draw: ImageDraw.ImageDraw, _paper: Image.Image, r: dict[str, Any]) -> None:
    x1, y1 = int(r["x1"]), int(r["y1"])
    x2, y2 = int(r["x2"]), int(r["y2"])
    color = int(r.get("color", 200))
    width = int(r.get("width", 1))
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)


def _render_hline_group(draw: ImageDraw.ImageDraw, _paper: Image.Image, r: dict[str, Any]) -> None:
    y_start = int(r["y_start"])
    y_end = int(r["y_end"])
    spacing = int(r.get("spacing", 80))
    color = int(r.get("color", 214))
    width = int(r.get("width", 1))
    x1 = int(r.get("x1", 0))
    x2 = int(r.get("x2", _DEFAULT_SIZE[0]))
    y = y_start
    while y <= y_end:
        draw.line([(x1, y), (x2, y)], fill=color, width=width)
        y += spacing


def _render_vline_group(draw: ImageDraw.ImageDraw, _paper: Image.Image, r: dict[str, Any]) -> None:
    x_start = int(r["x_start"])
    x_end = int(r["x_end"])
    spacing = int(r.get("spacing", 80))
    color = int(r.get("color", 214))
    width = int(r.get("width", 1))
    y1 = int(r.get("y1", 0))
    y2 = int(r.get("y2", _DEFAULT_SIZE[1]))
    x = x_start
    while x <= x_end:
        draw.line([(x, y1), (x, y2)], fill=color, width=width)
        x += spacing


def _render_staff_group(draw: ImageDraw.ImageDraw, _paper: Image.Image, r: dict[str, Any]) -> None:
    """Render music staff groups (5 lines per staff)."""
    y_start = int(r["y_start"])
    staff_count = int(r.get("staff_count", 8))
    line_spacing = int(r.get("line_spacing", 24))
    staff_gap = int(r.get("staff_gap", 260))
    color = int(r.get("color", 180))
    width = int(r.get("width", 1))
    x1 = int(r.get("x1", 200))
    x2 = int(r.get("x2", _DEFAULT_SIZE[0] - 200))

    y = y_start
    for _ in range(staff_count):
        for i in range(5):
            line_y = y + i * line_spacing
            draw.line([(x1, line_y), (x2, line_y)], fill=color, width=width)
        y += 5 * line_spacing + staff_gap


def _render_four_line_group(draw: ImageDraw.ImageDraw, _paper: Image.Image, r: dict[str, Any]) -> None:
    """Render English four-line three-grid groups.

    Each group has 4 lines: top (solid), upper-middle (dashed), lower-middle (dashed), bottom (solid).
    """
    y_start = int(r["y_start"])
    y_end = int(r["y_end"])
    group_height = int(r.get("group_height", 200))
    color = int(r.get("color", 200))
    width = int(r.get("width", 1))
    x1 = int(r.get("x1", 200))
    x2 = int(r.get("x2", _DEFAULT_SIZE[0] - 200))
    dash_middle = bool(r.get("dash_middle", False))
    dash_color = int(r.get("dash_color", 220))

    y = y_start
    while y + group_height <= y_end:
        # Top line (solid)
        draw.line([(x1, y), (x2, y)], fill=color, width=width)
        # Upper-middle line
        mid1 = y + group_height // 3
        if dash_middle:
            _draw_dashed_line(draw, x1, x2, mid1, dash_color, width)
        else:
            draw.line([(x1, mid1), (x2, mid1)], fill=color, width=width)
        # Lower-middle line
        mid2 = y + 2 * group_height // 3
        if dash_middle:
            _draw_dashed_line(draw, x1, x2, mid2, dash_color, width)
        else:
            draw.line([(x1, mid2), (x2, mid2)], fill=color, width=width)
        # Bottom line (solid)
        bottom = y + group_height
        draw.line([(x1, bottom), (x2, bottom)], fill=color, width=width)
        y = bottom + 40  # gap between groups


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x1: int,
    x2: int,
    y: int,
    color: int,
    width: int,
    dash_len: int = 20,
    gap_len: int = 12,
) -> None:
    """Draw a horizontal dashed line."""
    x = x1
    while x < x2:
        end = min(x + dash_len, x2)
        draw.line([(x, y), (end, y)], fill=color, width=width)
        x = end + gap_len


def _render_text(draw: ImageDraw.ImageDraw, _paper: Image.Image, r: dict[str, Any]) -> None:
    x = int(r["x"])
    y = int(r["y"])
    text = str(r.get("text", ""))
    size = int(r.get("size", 60))
    color = int(r.get("color", 160))
    font = _get_font(size)
    draw.text((x, y), text, fill=color, font=font)


def _render_ellipse(draw: ImageDraw.ImageDraw, _paper: Image.Image, r: dict[str, Any]) -> None:
    cx = int(r["cx"])
    cy = int(r["cy"])
    rx = int(r["rx"])
    ry = int(r["ry"])
    color = int(r.get("color", 200))
    width = int(r.get("width", 2))
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(bbox, outline=color, width=width)


# Font cache
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype("msyh.ttc", size)
        except (OSError, IOError):
            try:
                _FONT_CACHE[size] = ImageFont.truetype("simhei.ttf", size)
            except (OSError, IOError):
                _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


# Handler dispatch table
_REGION_HANDLERS: dict[str, Any] = {
    "line": _render_line,
    "hline_group": _render_hline_group,
    "vline_group": _render_vline_group,
    "staff_group": _render_staff_group,
    "four_line_group": _render_four_line_group,
    "text": _render_text,
    "ellipse": _render_ellipse,
}
