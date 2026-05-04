"""Semantic-aware layout planning.

Takes a sequence of :class:`~text_analyzer.TextSegment` objects and
produces a layout plan that specifies per-segment typography overrides
(font size multiplier, paper type, decoration, ink colour).

The plan is consumed by the annotation renderer and the page composer
to produce visually differentiated output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique

from handwrite.semantic.text_analyzer import SemanticRole, TextSegment


# ---------------------------------------------------------------------------
# Colour constants  (RGB tuples)
# ---------------------------------------------------------------------------

INK_BLACK: tuple[int, int, int] = (0, 0, 0)
INK_RED: tuple[int, int, int] = (200, 30, 30)
INK_BLUE: tuple[int, int, int] = (30, 60, 200)


@unique
class Decoration(str, Enum):
    """Visual decoration applied over text."""

    NONE = "none"
    UNDERLINE = "underline"
    WAVE_UNDERLINE = "wave_underline"
    CIRCLE = "circle"
    HIGHLIGHT = "highlight"


@dataclass(frozen=True)
class SegmentLayout:
    """Typography specification for a single text segment."""

    text: str
    clean_text: str
    role: SemanticRole
    font_size_multiplier: float = 1.0
    decoration: Decoration = Decoration.NONE
    ink_color: tuple[int, int, int] = INK_BLACK
    use_grid_paper: bool = False
    level: int = 0
    start: int = 0
    end: int = 0


class LayoutPlanner:
    """Map semantic segments to layout specifications.

    Default rules:
    - **Title**: 1.5x font size, blue ink, underline decoration.
    - **Emphasis**: red ink, wave underline decoration.
    - **Formula**: grid paper, highlight background.
    - **List item**: black ink, circle decoration (for bullet).
    - **Body**: standard black ink, no decoration.
    """

    def __init__(
        self,
        *,
        title_scale: float = 1.5,
        title_color: tuple[int, int, int] = INK_BLUE,
        title_decoration: Decoration = Decoration.UNDERLINE,
        emphasis_color: tuple[int, int, int] = INK_RED,
        emphasis_decoration: Decoration = Decoration.WAVE_UNDERLINE,
        formula_decoration: Decoration = Decoration.HIGHLIGHT,
        list_decoration: Decoration = Decoration.CIRCLE,
        body_color: tuple[int, int, int] = INK_BLACK,
    ) -> None:
        self._title_scale = title_scale
        self._title_color = title_color
        self._title_decoration = title_decoration
        self._emphasis_color = emphasis_color
        self._emphasis_decoration = emphasis_decoration
        self._formula_decoration = formula_decoration
        self._list_decoration = list_decoration
        self._body_color = body_color

    def plan(
        self,
        segments: list[TextSegment],
        *,
        base_font_size: int = 80,
    ) -> list[SegmentLayout]:
        """Produce a :class:`SegmentLayout` for each segment.

        Parameters
        ----------
        segments:
            Ordered segments from :class:`TextAnalyzer`.
        base_font_size:
            The base (body) font size in pixels. Title segments will
            use ``base_font_size * title_scale``.
        """
        from handwrite.semantic.text_analyzer import extract_clean_text

        plan: list[SegmentLayout] = []
        for seg in segments:
            clean = extract_clean_text(seg)
            if not clean.strip():
                continue

            layout = self._segment_layout(seg, clean, base_font_size)
            plan.append(layout)

        return plan

    def _segment_layout(
        self,
        seg: TextSegment,
        clean_text: str,
        base_font_size: int,
    ) -> SegmentLayout:
        role = seg.role

        if role == SemanticRole.TITLE:
            return SegmentLayout(
                text=seg.text,
                clean_text=clean_text,
                role=role,
                font_size_multiplier=self._title_scale,
                decoration=self._title_decoration,
                ink_color=self._title_color,
                level=seg.level,
                start=seg.start,
                end=seg.end,
            )

        if role == SemanticRole.EMPHASIS:
            return SegmentLayout(
                text=seg.text,
                clean_text=clean_text,
                role=role,
                decoration=self._emphasis_decoration,
                ink_color=self._emphasis_color,
                start=seg.start,
                end=seg.end,
            )

        if role == SemanticRole.FORMULA:
            return SegmentLayout(
                text=seg.text,
                clean_text=clean_text,
                role=role,
                decoration=self._formula_decoration,
                use_grid_paper=True,
                level=seg.level,
                start=seg.start,
                end=seg.end,
            )

        if role == SemanticRole.LIST_ITEM:
            return SegmentLayout(
                text=seg.text,
                clean_text=clean_text,
                role=role,
                decoration=self._list_decoration,
                ink_color=self._body_color,
                start=seg.start,
                end=seg.end,
            )

        # BODY
        return SegmentLayout(
            text=seg.text,
            clean_text=clean_text,
            role=role,
            ink_color=self._body_color,
            start=seg.start,
            end=seg.end,
        )


__all__ = [
    "INK_BLACK",
    "INK_RED",
    "INK_BLUE",
    "Decoration",
    "SegmentLayout",
    "LayoutPlanner",
]
