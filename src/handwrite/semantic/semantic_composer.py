"""Semantic-aware page composer.

Orchestrates the full pipeline:
1. Analyse text for semantic structure.
2. Plan per-segment layout (font size, colour, decoration).
3. Compose characters onto paper with typography overrides.
4. Render decorations (underlines, highlights, etc.).

The main entry point is :func:`compose_semantic_page`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PIL import Image

from handwrite.composer import _paste_char
from handwrite.semantic.annotation_renderer import render_annotations
from handwrite.semantic.layout_planner import (
    Decoration,
    LayoutPlanner,
    SegmentLayout,
)
from handwrite.semantic.text_analyzer import SemanticRole, TextAnalyzer, TextSegment


# ---------------------------------------------------------------------------
# Segment layout result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentRenderInfo:
    """Rendering metadata produced during composition for one segment."""

    role: SemanticRole
    clean_text: str
    decoration: Decoration
    ink_color: tuple[int, int, int]
    char_boxes: list[tuple[int, int, int, int]]
    use_grid_paper: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compose_semantic_page(
    chars: Sequence[Image.Image],
    text: str,
    *,
    page_size: tuple[int, int] = (2480, 3508),
    font_size: int = 80,
    margins: tuple[int, int, int, int] = (200, 200, 200, 200),
    title_scale: float = 1.5,
    analyzer: TextAnalyzer | None = None,
    planner: LayoutPlanner | None = None,
) -> Image.Image:
    """Compose a handwriting page with semantic-aware typography.

    Parameters
    ----------
    chars:
        Pre-rendered character images (one per non-whitespace character in
        *text*).  The images are consumed in reading order.
    text:
        The source text containing Markdown-style markup.
    page_size:
        Output image dimensions ``(width, height)`` in pixels.
    font_size:
        Base font size in pixels. Titles use ``font_size * title_scale``.
    margins:
        ``(top, right, bottom, left)`` margins in pixels.
    title_scale:
        Font size multiplier for title segments.
    analyzer:
        Optional custom :class:`TextAnalyzer`. Uses the default when *None*.
    planner:
        Optional custom :class:`LayoutPlanner`. Uses the default when *None*.

    Returns
    -------
    Image.Image
        The composed page image.
    """
    analyzer = analyzer or TextAnalyzer()
    planner = planner or LayoutPlanner(title_scale=title_scale)

    segments = analyzer.analyze(text)
    layout_plan = planner.plan(segments, base_font_size=font_size)

    page, render_infos = _compose_layout(
        chars,
        layout_plan,
        page_size=page_size,
        font_size=font_size,
        margins=margins,
    )

    # Post-pass: draw decorations
    for info in render_infos:
        if info.char_boxes and info.decoration != Decoration.NONE:
            render_annotations(
                page,
                info.char_boxes,
                info.decoration,
                color=info.ink_color,
            )

    return page


# ---------------------------------------------------------------------------
# Internal composition logic
# ---------------------------------------------------------------------------

def _compose_layout(
    chars: Sequence[Image.Image],
    plan: Sequence[SegmentLayout],
    *,
    page_size: tuple[int, int],
    font_size: int,
    margins: tuple[int, int, int, int],
) -> tuple[Image.Image, list[SegmentRenderInfo]]:
    """Core layout engine. Returns the page and per-segment render info."""
    top, right, bottom, left = margins
    page_width, page_height = page_size

    page = Image.new("L", page_size, color=255)
    render_infos: list[SegmentRenderInfo] = []

    char_idx = 0
    x = left
    y = top

    for layout in plan:
        clean = layout.clean_text
        if not clean.strip():
            continue

        segment_font_size = max(8, int(font_size * layout.font_size_multiplier))
        char_gap = max(6, segment_font_size // 10)
        line_gap = max(12, segment_font_size // 3)
        line_height = segment_font_size + line_gap
        column_step = segment_font_size + char_gap

        char_boxes: list[tuple[int, int, int, int]] = []

        for ch in clean:
            if ch == "\n":
                # Newline: advance to next line
                x = left
                y += line_height
                continue

            if ch.isspace():
                x += column_step
                continue

            if char_idx >= len(chars):
                break

            # Line wrap
            if x + segment_font_size > page_width - right:
                x = left
                y += line_height

            if y + segment_font_size > page_height - bottom:
                break

            # Paste character
            char_img = chars[char_idx]
            _paste_char(page, char_img, (x, y), segment_font_size)
            char_boxes.append((x, y, segment_font_size, segment_font_size))
            char_idx += 1
            x += column_step

        if char_boxes:
            render_infos.append(
                SegmentRenderInfo(
                    role=layout.role,
                    clean_text=clean,
                    decoration=layout.decoration,
                    ink_color=layout.ink_color,
                    char_boxes=char_boxes,
                    use_grid_paper=layout.use_grid_paper,
                )
            )

    return page, render_infos


__all__ = ["compose_semantic_page", "SegmentRenderInfo"]
