"""Annotation overlay primitives for the error-notebook module.

These helpers draw red-pen style markings on top of a page image: a
strike-through over wrong tokens, an underline beneath them, and a small
red-pen correction note next to a position.  Every function returns a NEW
``PIL.Image`` (RGB mode) — the input image is never mutated in-place.
"""

from __future__ import annotations

from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "strike_through",
    "underline",
    "red_correction",
]


# Type aliases ---------------------------------------------------------------

BBox = Tuple[int, int, int, int]
Color = Tuple[int, int, int] | Tuple[int, int, int, int]
Position = Tuple[int, int]


_DEFAULT_INK: Tuple[int, int, int, int] = (220, 30, 30, 200)
_MIN_STROKE_WIDTH = 2


def _to_rgb(image: Image.Image) -> Image.Image:
    """Return a copy of *image* in RGB mode."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if image.mode == "RGB":
        return image.copy()
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, color=(255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def _normalize_color(color: Color) -> Tuple[int, int, int, int]:
    if len(color) == 3:
        r, g, b = color
        return (int(r), int(g), int(b), 200)
    if len(color) == 4:
        r, g, b, a = color
        return (int(r), int(g), int(b), int(a))
    raise ValueError("color must be an RGB or RGBA tuple")


def _validate_bbox(bbox: BBox, size: Tuple[int, int]) -> BBox:
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        raise ValueError("bbox must be a 4-tuple (left, top, right, bottom)")
    left, top, right, bottom = (int(v) for v in bbox)
    if right < left or bottom < top:
        raise ValueError("bbox must satisfy right>=left and bottom>=top")
    width, height = size
    # Clamp to image bounds so drawing never silently misses the canvas.
    left = max(0, min(left, width - 1))
    right = max(left + 1, min(right, width))
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    return (left, top, right, bottom)


def _blend_overlay(base_rgb: Image.Image, overlay_rgba: Image.Image) -> Image.Image:
    """Alpha-composite *overlay_rgba* onto *base_rgb* and return RGB."""
    canvas = base_rgb.convert("RGBA")
    canvas.alpha_composite(overlay_rgba)
    return canvas.convert("RGB")


# ---------------------------------------------------------------------------
# Strike-through
# ---------------------------------------------------------------------------

def strike_through(
    image: Image.Image,
    bbox: BBox,
    color: Color = _DEFAULT_INK,
) -> Image.Image:
    """Draw a horizontal strike-through line across the centre of *bbox*.

    Args:
        image: Source page image.  Treated as RGB.
        bbox: ``(left, top, right, bottom)`` of the wrong tokens to cross out.
        color: RGB or RGBA stroke colour.

    Returns:
        A new RGB ``PIL.Image`` with the strike-through painted on top.
    """
    base = _to_rgb(image)
    rgba_color = _normalize_color(color)
    left, top, right, bottom = _validate_bbox(bbox, base.size)
    height = bottom - top

    # Use a transparent overlay so the alpha channel composites smoothly.
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    mid_y = (top + bottom) // 2
    stroke_width = max(_MIN_STROKE_WIDTH, height // 8)
    draw.line(
        [(left, mid_y), (right, mid_y)],
        fill=rgba_color,
        width=stroke_width,
    )
    # Add a light "scribble" double-strike for visual weight.
    secondary_offset = max(1, stroke_width // 2)
    draw.line(
        [(left, mid_y - secondary_offset), (right, mid_y + secondary_offset)],
        fill=(rgba_color[0], rgba_color[1], rgba_color[2], max(60, rgba_color[3] // 2)),
        width=max(1, stroke_width // 2),
    )

    return _blend_overlay(base, overlay)


# ---------------------------------------------------------------------------
# Underline
# ---------------------------------------------------------------------------

def underline(
    image: Image.Image,
    bbox: BBox,
    color: Color = _DEFAULT_INK,
) -> Image.Image:
    """Draw a wavy-ish underline beneath *bbox*.

    Returns a new RGB ``PIL.Image``.
    """
    base = _to_rgb(image)
    rgba_color = _normalize_color(color)
    left, top, right, bottom = _validate_bbox(bbox, base.size)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    underline_y = min(base.size[1] - 1, bottom + max(2, (bottom - top) // 12))
    stroke_width = max(_MIN_STROKE_WIDTH, (bottom - top) // 10)

    # Primary underline.
    draw.line(
        [(left, underline_y), (right, underline_y)],
        fill=rgba_color,
        width=stroke_width,
    )
    # Faint secondary line for a "wavy" feel; cheap and avoids deps.
    if right - left > 8:
        light = (rgba_color[0], rgba_color[1], rgba_color[2], max(50, rgba_color[3] // 3))
        draw.line(
            [(left, underline_y + stroke_width + 1), (right, underline_y + stroke_width + 1)],
            fill=light,
            width=max(1, stroke_width // 2),
        )

    return _blend_overlay(base, overlay)


# ---------------------------------------------------------------------------
# Red correction text
# ---------------------------------------------------------------------------

def _load_default_font(size: int = 24) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def red_correction(
    image: Image.Image,
    position: Position,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None = None,
    color: Color = _DEFAULT_INK,
) -> Image.Image:
    """Draw small red text at *position* as a teacher-style correction.

    Args:
        image: Source page image (treated as RGB).
        position: ``(x, y)`` top-left coordinate of the correction note.
        text: Annotation text.
        font: Optional PIL font; if ``None`` a sensible default is loaded.
        color: RGB or RGBA stroke colour.

    Returns:
        A new RGB ``PIL.Image``.
    """
    base = _to_rgb(image)
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if not isinstance(position, (tuple, list)) or len(position) != 2:
        raise ValueError("position must be a 2-tuple (x, y)")
    x, y = int(position[0]), int(position[1])
    rgba_color = _normalize_color(color)
    pen_font = font if font is not None else _load_default_font()

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if text:
        draw.text((x, y), text, fill=rgba_color, font=pen_font)
    return _blend_overlay(base, overlay)
