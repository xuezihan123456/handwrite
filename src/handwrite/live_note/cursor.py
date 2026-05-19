"""Pen-tip cursor overlay for the live-note writer.

Draws a small visual indicator at the current writing position to make the
animation feel like a real classroom recording. The overlay operates on
PIL grayscale images and never mutates the source frame.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

# Channel value (0..255) used to draw the cursor on a grayscale frame.
# Use a clearly darker-than-paper value so tests can detect the change.
_CURSOR_GRAY = 32
_HALO_GRAY = 96


def draw_pen_cursor(
    frame: Image.Image,
    position: tuple[int, int],
    *,
    radius: int = 6,
    halo: bool = True,
) -> Image.Image:
    """Return a copy of ``frame`` with a pen-tip cursor drawn on top.

    The cursor is rendered as a small filled circle (the nib) with an optional
    softer halo behind it. The function is pure: it never mutates ``frame``.

    Args:
        frame: Source frame (any PIL mode; will be coerced to L).
        position: (x, y) coordinates of the pen tip.
        radius: Radius of the pen-tip circle in pixels.
        halo: When ``True`` draw a softer halo around the nib.

    Returns:
        A new PIL Image with the cursor overlaid.
    """
    if frame.mode != "L":
        canvas = frame.convert("L")
    else:
        canvas = frame.copy()

    width, height = canvas.size
    x, y = position
    x = max(0, min(width - 1, int(x)))
    y = max(0, min(height - 1, int(y)))
    nib_radius = max(1, int(radius))

    draw = ImageDraw.Draw(canvas)
    if halo:
        halo_radius = nib_radius + max(2, nib_radius)
        draw.ellipse(
            (x - halo_radius, y - halo_radius, x + halo_radius, y + halo_radius),
            outline=_HALO_GRAY,
            width=1,
        )

    draw.ellipse(
        (x - nib_radius, y - nib_radius, x + nib_radius, y + nib_radius),
        fill=_CURSOR_GRAY,
    )
    return canvas


def overlay_cursor_on_frames(
    frames: list[Image.Image],
    positions: list[tuple[int, int]],
    *,
    radius: int = 6,
    halo: bool = True,
) -> list[Image.Image]:
    """Apply :func:`draw_pen_cursor` to every frame in lock-step.

    Args:
        frames: Animation frames to overlay.
        positions: One ``(x, y)`` cursor position per frame. The list is padded
            or truncated to match ``frames`` automatically.
        radius: Pen-tip radius.
        halo: Whether to render the surrounding halo.

    Returns:
        New list of frames with the cursor drawn.
    """
    if not frames:
        return []
    if not positions:
        return [frame.copy() if frame.mode == "L" else frame.convert("L") for frame in frames]

    out: list[Image.Image] = []
    last_position = positions[-1]
    for index, frame in enumerate(frames):
        position = positions[index] if index < len(positions) else last_position
        out.append(draw_pen_cursor(frame, position, radius=radius, halo=halo))
    return out


__all__ = ["draw_pen_cursor", "overlay_cursor_on_frames"]
