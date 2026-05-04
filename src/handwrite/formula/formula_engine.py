"""Formula handwriting engine.

Main entry points:
- ``render_latex_formula()``: render a LaTeX math string to a handwritten image.
- ``render_chemistry()``: render a chemical equation to a handwritten image.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from handwrite.formula.chemistry_parser import (
    ArrowType,
    ChemArrow,
    ChemCompound,
    ChemEquation,
    ChemToken,
    parse_chemistry,
)
from handwrite.formula.formula_layout import BBox, FormulaLayout, LayoutConfig
from handwrite.formula.formula_renderer import FormulaRenderer, RenderConfig
from handwrite.formula.latex_parser import parse_latex


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormulaConfig:
    """Unified configuration for formula rendering."""

    font_size: int = 40
    """Base font size in pixels."""

    output_width: int = 0
    """Desired output width. 0 = auto-fit to content."""

    output_height: int = 0
    """Desired output height. 0 = auto-fit to content."""

    ink_color: int = 30
    """Grayscale ink intensity (0 = black)."""

    bg_color: int = 255
    """Background grayscale value (255 = white)."""

    padding: tuple[int, int, int, int] = (20, 20, 20, 20)
    """Padding (top, right, bottom, left) in pixels."""

    seed: Optional[int] = None
    """Random seed for reproducible hand-drawn effects."""

    font_path: Optional[str] = None
    """Path to a TTF font. None = auto-detect."""


# ---------------------------------------------------------------------------
# LaTeX formula rendering
# ---------------------------------------------------------------------------

def render_latex_formula(
    latex: str,
    config: FormulaConfig | None = None,
) -> Image.Image:
    r"""Render a LaTeX math formula to a handwritten image.

    Args:
        latex: LaTeX math string, e.g. ``r"\frac{a}{b} + \sqrt{x}"``.
        config: Optional rendering configuration.

    Returns:
        Grayscale PIL Image of the handwritten formula.

    Examples::

        >>> img = render_latex_formula(r"\frac{1}{2}")
        >>> img.size
        (100, 80)
    """
    cfg = config or FormulaConfig()

    nodes = parse_latex(latex)
    if not nodes:
        return Image.new("L", (1, 1), color=cfg.bg_color)

    layout_config = LayoutConfig(
        base_font_size=cfg.font_size,
    )
    render_config = RenderConfig(
        ink_color=cfg.ink_color,
        bg_color=cfg.bg_color,
        padding=cfg.padding,
        seed=cfg.seed,
        font_path=cfg.font_path,
    )

    layout = FormulaLayout(layout_config)
    items = layout.layout(nodes)

    if not items:
        return Image.new("L", (1, 1), color=cfg.bg_color)

    # Compute canvas bbox.
    min_x = min(item.bbox.x for item in items)
    min_y = min(item.bbox.y for item in items)
    max_x = max(item.bbox.right for item in items)
    max_y = max(item.bbox.bottom for item in items)
    canvas_bbox = BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)

    renderer = FormulaRenderer(render_config)
    image = renderer.render(items, canvas_bbox)

    # Resize to desired output dimensions if specified.
    if cfg.output_width > 0 or cfg.output_height > 0:
        image = _resize_to_fit(image, cfg.output_width, cfg.output_height)

    return image


# ---------------------------------------------------------------------------
# Chemistry equation rendering
# ---------------------------------------------------------------------------

def render_chemistry(
    equation: str,
    config: FormulaConfig | None = None,
) -> Image.Image:
    r"""Render a chemical equation to a handwritten image.

    Args:
        equation: Chemical equation string, e.g. ``"2H2 + O2 -> 2H2O"``.
        config: Optional rendering configuration.

    Returns:
        Grayscale PIL Image of the handwritten equation.

    Examples::

        >>> img = render_chemistry("H2 + O2 -> H2O")
        >>> isinstance(img, Image.Image)
        True
    """
    cfg = config or FormulaConfig()

    parsed = parse_chemistry(equation)

    # Render using a dedicated chemistry drawing pipeline.
    renderer = _ChemistryRenderer(cfg)
    return renderer.render(parsed)


# ---------------------------------------------------------------------------
# Chemistry renderer internals
# ---------------------------------------------------------------------------

class _ChemistryRenderer:
    """Render a parsed chemical equation to a PIL image."""

    def __init__(self, config: FormulaConfig) -> None:
        self._cfg = config
        self._rng = random.Random(config.seed)
        self._font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def render(self, eq: ChemEquation) -> Image.Image:
        cfg = self._cfg
        font_size = cfg.font_size
        subscript_size = max(10, int(font_size * 0.65))
        arrow_height = max(int(font_size * 1.2), 30)

        # Measure components.
        parts: list[tuple[str, int, int, str]] = []  # (text, width, font_size, kind)
        for compound in eq.reactants:
            text = self._compound_to_display(compound)
            w = self._measure_text(text, font_size)
            parts.append((text, w, font_size, "compound"))

        # Plus signs between reactants.
        for i in range(len(eq.reactants) - 1):
            plus_w = self._measure_text(" + ", font_size)
            parts.append((" + ", plus_w, font_size, "operator"))

        # Arrow.
        arrow_display = self._arrow_display(eq.arrow)
        arrow_w = self._measure_arrow_width(eq.arrow, font_size)
        parts.append((arrow_display, arrow_w, font_size, "arrow"))

        # Plus signs between products.
        for i in range(len(eq.products) - 1):
            plus_w = self._measure_text(" + ", font_size)
            parts.append((" + ", plus_w, font_size, "operator"))

        for compound in eq.products:
            text = self._compound_to_display(compound)
            w = self._measure_text(text, font_size)
            parts.append((text, w, font_size, "compound"))

        # Compute total width.
        total_w = sum(p[1] for p in parts) + cfg.padding[1] + cfg.padding[3]
        total_h = font_size + cfg.padding[0] + cfg.padding[2]
        if eq.arrow.condition_above or eq.arrow.condition_below:
            total_h += int(font_size * 0.8)

        total_w = max(1, total_w)
        total_h = max(1, total_h)

        img = Image.new("L", (total_w, total_h), color=cfg.bg_color)
        draw = ImageDraw.Draw(img)

        x = cfg.padding[3]
        y = cfg.padding[0]
        ink = cfg.ink_color

        for text, w, fs, kind in parts:
            if kind == "compound":
                self._draw_compound(draw, text, x, y, fs, ink)
                x += w
            elif kind == "operator":
                font = self._get_font(fs)
                jx = self._rng.randint(-1, 1)
                jy = self._rng.randint(-1, 1)
                draw.text((x + jx, y + jy), text, fill=ink, font=font)
                x += w
            elif kind == "arrow":
                self._draw_arrow(draw, eq.arrow, x, y, w, fs, ink)
                x += w

        return img

    def _compound_to_display(self, compound: ChemCompound) -> str:
        """Convert compound tokens to display string with subscripts."""
        parts: list[str] = []
        for tok in compound.tokens:
            if tok.kind == "number":
                # Render as subscript unicode.
                parts.append(tok.text.translate(_SUBSCRIPT_MAP))
            elif tok.kind == "charge":
                # Render as superscript unicode.
                parts.append(tok.text.translate(_SUPERSCRIPT_MAP))
            elif tok.kind == "state":
                parts.append(tok.text)
            else:
                parts.append(tok.text)
        return "".join(parts)

    def _measure_text(self, text: str, font_size: int) -> int:
        font = self._get_font(font_size)
        try:
            bbox = font.getbbox(text)
            return int(bbox[2] - bbox[0]) + 4
        except Exception:
            return max(1, int(len(text) * font_size * 0.6))

    def _measure_arrow_width(self, arrow: ChemArrow, font_size: int) -> int:
        base_w = int(font_size * 2.5)
        if arrow.condition_above:
            cond_w = self._measure_text(arrow.condition_above, int(font_size * 0.5))
            base_w = max(base_w, cond_w + 20)
        if arrow.condition_below:
            cond_w = self._measure_text(arrow.condition_below, int(font_size * 0.5))
            base_w = max(base_w, cond_w + 20)
        return base_w

    def _arrow_display(self, arrow: ChemArrow) -> str:
        mapping = {
            ArrowType.FORWARD: "->",
            ArrowType.YIELD: "\u2192",
            ArrowType.REVERSE: "\u2194",
            ArrowType.EQUILIBRIUM: "\u21cc",
            ArrowType.NOT_YIELD: "\u2192/",
            ArrowType.EQUALS: "=",
        }
        return mapping.get(arrow.arrow_type, "->")

    def _draw_compound(self, draw: ImageDraw.ImageDraw, text: str, x: int, y: int, font_size: int, ink: int) -> None:
        """Draw compound text, handling subscript unicode characters."""
        font = self._get_font(font_size)
        jx = self._rng.randint(-1, 1)
        jy = self._rng.randint(-1, 1)
        draw.text((x + jx, y + jy), text, fill=ink, font=font)

    def _draw_arrow(
        self, draw: ImageDraw.ImageDraw, arrow: ChemArrow,
        x: int, y: int, width: int, font_size: int, ink: int,
    ) -> None:
        """Draw a chemical reaction arrow with optional conditions."""
        rng = self._rng
        wobble = 0.6

        mid_y = y + font_size // 2
        arrow_y = mid_y
        x0 = x + 8
        x1 = x + width - 8

        if arrow.arrow_type == ArrowType.EQUILIBRIUM:
            # Two parallel arrows.
            self._draw_wobbly_line(draw, x0, arrow_y - 3, x1, arrow_y - 3, ink, wobble, width=2)
            self._draw_wobbly_line(draw, x0, arrow_y + 3, x1, arrow_y + 3, ink, wobble, width=2)
            # Arrowheads.
            self._draw_arrowhead(draw, x1, arrow_y - 3, ink, direction="right")
            self._draw_arrowhead(draw, x0, arrow_y + 3, ink, direction="left")
        elif arrow.arrow_type == ArrowType.NOT_YIELD:
            # Arrow with a cross through it.
            self._draw_wobbly_line(draw, x0, arrow_y, x1, arrow_y, ink, wobble, width=2)
            self._draw_arrowhead(draw, x1, arrow_y, ink, direction="right")
            # Cross.
            mid_x = (x0 + x1) // 2
            self._draw_wobbly_line(draw, mid_x - 5, arrow_y - 6, mid_x + 5, arrow_y + 6, ink, wobble, width=2)
        else:
            # Standard forward arrow.
            self._draw_wobbly_line(draw, x0, arrow_y, x1, arrow_y, ink, wobble, width=2)
            self._draw_arrowhead(draw, x1, arrow_y, ink, direction="right")

        # Conditions above / below.
        cond_font_size = max(10, int(font_size * 0.5))
        cond_font = self._get_font(cond_font_size)
        if arrow.condition_above:
            cond_w = self._measure_text(arrow.condition_above, cond_font_size)
            cx = x + (width - cond_w) // 2
            cy = arrow_y - cond_font_size - 2
            draw.text((cx, cy), arrow.condition_above, fill=ink, font=cond_font)
        if arrow.condition_below:
            cond_w = self._measure_text(arrow.condition_below, cond_font_size)
            cx = x + (width - cond_w) // 2
            cy = arrow_y + 6
            draw.text((cx, cy), arrow.condition_below, fill=ink, font=cond_font)

    def _draw_arrowhead(
        self, draw: ImageDraw.ImageDraw, x: int, y: int, ink: int, direction: str = "right",
    ) -> None:
        size = 6
        if direction == "right":
            draw.line([(x, y), (x - size, y - size)], fill=ink, width=2)
            draw.line([(x, y), (x - size, y + size)], fill=ink, width=2)
        else:
            draw.line([(x, y), (x + size, y - size)], fill=ink, width=2)
            draw.line([(x, y), (x + size, y + size)], fill=ink, width=2)

    def _draw_wobbly_line(
        self, draw: ImageDraw.ImageDraw,
        x0: int, y0: int, x1: int, y1: int,
        fill: int, wobble: float, width: int = 1,
    ) -> None:
        """Draw a line with slight perpendicular wobble."""
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length < 2:
            draw.line([(x0, y0), (x1, y1)], fill=fill, width=width)
            return

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


# Unicode subscript / superscript maps (shared with chemistry_parser).
_SUBSCRIPT_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUPERSCRIPT_MAP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


# ---------------------------------------------------------------------------
# Resize helper
# ---------------------------------------------------------------------------

def _resize_to_fit(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize image to fit target dimensions while preserving aspect ratio."""
    src_w, src_h = image.size
    if target_w <= 0 and target_h <= 0:
        return image

    if target_w <= 0:
        scale = target_h / src_h
        target_w = int(src_w * scale)
    elif target_h <= 0:
        scale = target_w / src_w
        target_h = int(src_h * scale)
    else:
        scale = min(target_w / src_w, target_h / src_h)
        target_w = int(src_w * scale)
        target_h = int(src_h * scale)

    return image.resize((max(1, target_w), max(1, target_h)), Image.Resampling.LANCZOS)
