"""Render laid-out formula items into a PIL image with handwritten style.

Fraction lines, radical signs, integral curves, and matrix brackets are drawn
with slight randomness to simulate hand-drawn appearance.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from handwrite.formula.formula_layout import BBox, FormulaLayout, LayoutConfig, LayoutItem
from handwrite.formula.latex_parser import GREEK_MAP, ParseNode


# ---------------------------------------------------------------------------
# Render configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RenderConfig:
    """Configuration for formula rendering."""

    ink_color: int = 30
    """Grayscale ink intensity (0 = black, 255 = white)."""

    bg_color: int = 255
    """Background grayscale value."""

    padding: tuple[int, int, int, int] = (16, 16, 16, 16)
    """Padding around the formula: (top, right, bottom, left)."""

    jitter_pixels: int = 1
    """Max random pixel offset for hand-drawn effect on text."""

    line_wobble: float = 0.8
    """Max perpendicular wobble (pixels) for straight lines."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""

    font_path: Optional[str] = None
    """Path to a TTF font file. If None, uses the default PIL font."""


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class FormulaRenderer:
    """Render layout items into a PIL grayscale image.

    Usage::

        renderer = FormulaRenderer(config)
        image = renderer.render(items, total_bbox)
    """

    def __init__(self, config: RenderConfig | None = None) -> None:
        self._cfg = config or RenderConfig()
        self._rng = random.Random(self._cfg.seed)
        self._font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    # -- public --

    def render(self, items: list[LayoutItem], canvas_bbox: BBox) -> Image.Image:
        """Render layout items into a PIL image.

        Args:
            items: Laid-out renderable items.
            canvas_bbox: The total bounding box encompassing all items.

        Returns:
            Grayscale PIL Image.
        """
        cfg = self._cfg
        top, right, bottom, left = cfg.padding
        width = canvas_bbox.width + left + right
        height = canvas_bbox.height + top + bottom

        img = Image.new("L", (max(1, width), max(1, height)), color=cfg.bg_color)
        draw = ImageDraw.Draw(img)

        origin_x = left - canvas_bbox.x
        origin_y = top - canvas_bbox.y

        for item in items:
            self._render_item(draw, img, item, origin_x=origin_x, origin_y=origin_y)

        return img

    # -- item dispatch --

    def _render_item(
        self, draw: ImageDraw.ImageDraw, img: Image.Image,
        item: LayoutItem, *, origin_x: int, origin_y: int,
    ) -> None:
        kind = item.kind
        if kind == "text":
            self._render_text(draw, item, origin_x=origin_x, origin_y=origin_y)
        elif kind == "fraction_line":
            self._render_fraction_line(draw, item, origin_x=origin_x, origin_y=origin_y)
        elif kind == "radical_sign":
            self._render_radical_sign(draw, item, origin_x=origin_x, origin_y=origin_y)
        elif kind == "integral_sign":
            self._render_integral_sign(draw, item, origin_x=origin_x, origin_y=origin_y)
        elif kind == "sum_sign":
            self._render_sum_sign(draw, item, origin_x=origin_x, origin_y=origin_y)
        elif kind == "matrix_bracket":
            self._render_matrix_bracket(draw, item, origin_x=origin_x, origin_y=origin_y)

    # -- text --

    def _render_text(
        self, draw: ImageDraw.ImageDraw, item: LayoutItem,
        *, origin_x: int, origin_y: int,
    ) -> None:
        cfg = self._cfg
        x = item.bbox.x + origin_x
        y = item.bbox.y + origin_y
        font_size = item.font_size or 40

        # Resolve display text: Greek names -> Unicode symbols.
        display_text = self._resolve_display_text(item.text, font_size)

        font = self._get_font(font_size)

        # Apply handwriting jitter.
        jx = self._rng.randint(-cfg.jitter_pixels, cfg.jitter_pixels)
        jy = self._rng.randint(-cfg.jitter_pixels, cfg.jitter_pixels)
        draw.text((x + jx, y + jy), display_text, fill=cfg.ink_color, font=font)

    def _resolve_display_text(self, text: str, font_size: int) -> str:
        """Convert Greek command names to their Unicode glyphs."""
        result = []
        for part in text.split():
            if part in GREEK_MAP:
                result.append(GREEK_MAP[part])
            else:
                result.append(part)
        resolved = " ".join(result)
        if not resolved:
            resolved = text
        return resolved

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if size in self._font_cache:
            return self._font_cache[size]

        cfg = self._cfg
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        if cfg.font_path:
            font = ImageFont.truetype(cfg.font_path, size)
        else:
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except (OSError, IOError):
                font = ImageFont.load_default()
        self._font_cache[size] = font
        return font

    # -- fraction line --

    def _render_fraction_line(
        self, draw: ImageDraw.ImageDraw, item: LayoutItem,
        *, origin_x: int, origin_y: int,
    ) -> None:
        cfg = self._cfg
        x0 = item.bbox.x + origin_x
        y0 = item.bbox.baseline_y + origin_y
        x1 = x0 + item.bbox.width
        self._draw_wobbly_line(draw, x0, y0, x1, y0, cfg.ink_color, cfg.line_wobble, width=2)

    # -- radical sign --

    def _render_radical_sign(
        self, draw: ImageDraw.ImageDraw, item: LayoutItem,
        *, origin_x: int, origin_y: int,
    ) -> None:
        cfg = self._cfg
        x = item.bbox.x + origin_x
        y = item.bbox.y + origin_y
        w = item.bbox.width
        h = item.bbox.height
        content_w = item.extra.get("content_width", w // 2)
        content_h = item.extra.get("content_height", h)

        # Radical sign: a short horizontal tick, a descending diagonal, a rising
        # diagonal to the content top, and then an overbar across the content.
        tick_x = x + int(h * 0.15)
        mid_y = y + int(h * 0.6)
        bottom_y = y + h
        top_y = y + 2
        bar_start_x = tick_x + int(h * 0.2)
        bar_end_x = bar_start_x + content_w + 4

        ink = cfg.ink_color
        wobble = cfg.line_wobble

        # Small tick.
        self._draw_wobbly_line(draw, x, mid_y, tick_x, bottom_y, ink, wobble, width=2)
        # Descending stroke.
        self._draw_wobbly_line(draw, tick_x, bottom_y, bar_start_x, top_y, ink, wobble, width=2)
        # Overbar.
        self._draw_wobbly_line(draw, bar_start_x, top_y, bar_end_x, top_y, ink, wobble, width=2)

    # -- integral sign --

    def _render_integral_sign(
        self, draw: ImageDraw.ImageDraw, item: LayoutItem,
        *, origin_x: int, origin_y: int,
    ) -> None:
        cfg = self._cfg
        x = item.bbox.x + origin_x
        y = item.bbox.y + origin_y
        w = item.bbox.width
        h = item.bbox.height
        ink = cfg.ink_color

        # Draw integral sign as an elongated S-curve.
        cx = x + w // 2
        points: list[tuple[int, int]] = []
        num_segments = 20
        for i in range(num_segments + 1):
            t = i / num_segments
            # S-curve parametric.
            py = y + int(t * h)
            px = cx + int(w * 0.35 * math.sin(t * math.pi * 1.2 - 0.3))
            jx = self._rng.randint(-1, 1)
            jy = self._rng.randint(-1, 1)
            points.append((px + jx, py + jy))

        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=ink, width=2)

        # Top and bottom serifs.
        self._draw_wobbly_line(draw, cx - 3, y, cx + 5, y, ink, cfg.line_wobble, width=2)
        self._draw_wobbly_line(draw, cx - 5, y + h, cx + 3, y + h, ink, cfg.line_wobble, width=2)

    # -- sum sign --

    def _render_sum_sign(
        self, draw: ImageDraw.ImageDraw, item: LayoutItem,
        *, origin_x: int, origin_y: int,
    ) -> None:
        cfg = self._cfg
        x = item.bbox.x + origin_x
        y = item.bbox.y + origin_y
        w = item.bbox.width
        h = item.bbox.height
        ink = cfg.ink_color
        wobble = cfg.line_wobble

        # Top horizontal line.
        self._draw_wobbly_line(draw, x, y, x + w, y, ink, wobble, width=2)
        # Left descending diagonal.
        self._draw_wobbly_line(draw, x, y, x + w // 4, y + h // 2, ink, wobble, width=2)
        # Left bottom diagonal.
        self._draw_wobbly_line(draw, x + w // 4, y + h // 2, x, y + h, ink, wobble, width=2)
        # Bottom horizontal line.
        self._draw_wobbly_line(draw, x, y + h, x + w, y + h, ink, wobble, width=2)

    # -- matrix bracket --

    def _render_matrix_bracket(
        self, draw: ImageDraw.ImageDraw, item: LayoutItem,
        *, origin_x: int, origin_y: int,
    ) -> None:
        cfg = self._cfg
        x = item.bbox.x + origin_x
        y = item.bbox.y + origin_y
        w = item.bbox.width
        h = item.bbox.height
        ink = cfg.ink_color
        wobble = cfg.line_wobble
        side = item.extra.get("side", "left")
        kind = item.extra.get("kind", "matrix")

        if kind in ("pmatrix", "bmatrix", "Bmatrix"):
            if side == "left":
                # Curved or square left bracket.
                self._draw_wobbly_line(draw, x + w, y, x, y + h * 0.1, ink, wobble, width=2)
                self._draw_wobbly_line(draw, x, y + h * 0.1, x, y + h * 0.9, ink, wobble, width=2)
                self._draw_wobbly_line(draw, x, y + h * 0.9, x + w, y + h, ink, wobble, width=2)
            else:
                self._draw_wobbly_line(draw, x, y, x + w, y + h * 0.1, ink, wobble, width=2)
                self._draw_wobbly_line(draw, x + w, y + h * 0.1, x + w, y + h * 0.9, ink, wobble, width=2)
                self._draw_wobbly_line(draw, x + w, y + h * 0.9, x, y + h, ink, wobble, width=2)
        else:
            # Plain vertical line for matrix.
            if side == "left":
                self._draw_wobbly_line(draw, x + w // 2, y, x + w // 2, y + h, ink, wobble, width=2)
            else:
                self._draw_wobbly_line(draw, x + w // 2, y, x + w // 2, y + h, ink, wobble, width=2)

    # -- wobbly line helper --

    def _draw_wobbly_line(
        self, draw: ImageDraw.ImageDraw,
        x0: int, y0: int, x1: int, y1: int,
        fill: int, wobble: float, width: int = 1,
    ) -> None:
        """Draw a line with slight perpendicular wobble for handwritten effect."""
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length < 2:
            draw.line([(x0, y0), (x1, y1)], fill=fill, width=width)
            return

        # Perpendicular unit vector.
        nx = -dy / length
        ny = dx / length

        num_segments = max(3, int(length / 8))
        points: list[tuple[int, int]] = []
        for i in range(num_segments + 1):
            t = i / num_segments
            px = x0 + dx * t
            py = y0 + dy * t
            offset = self._rng.uniform(-wobble, wobble)
            points.append((int(px + nx * offset), int(py + ny * offset)))

        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=fill, width=width)


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def render_formula_to_image(
    nodes: list[ParseNode],
    *,
    layout_config: LayoutConfig | None = None,
    render_config: RenderConfig | None = None,
) -> Image.Image:
    """Parse nodes -> layout -> render -> PIL Image.

    Args:
        nodes: Parsed LaTeX AST nodes.
        layout_config: Optional layout configuration.
        render_config: Optional render configuration.

    Returns:
        Grayscale PIL Image of the formula.
    """
    layout = FormulaLayout(layout_config)
    items = layout.layout(nodes)

    # Compute total bounding box.
    if not items:
        return Image.new("L", (1, 1), color=255)

    min_x = min(item.bbox.x for item in items)
    min_y = min(item.bbox.y for item in items)
    max_x = max(item.bbox.right for item in items)
    max_y = max(item.bbox.bottom for item in items)

    canvas_bbox = BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)

    renderer = FormulaRenderer(render_config)
    return renderer.render(items, canvas_bbox)
