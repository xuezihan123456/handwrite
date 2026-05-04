"""Semantic-aware typesetting for handwriting generation.

This module analyses text for semantic structure (headings, emphasis,
formulae, lists) and applies typography-aware layout rules such as
scaled font sizes, coloured ink, and visual annotations.
"""

from handwrite.semantic.text_analyzer import (
    SemanticRole,
    TextAnalyzer,
    TextSegment,
    extract_clean_text,
)
from handwrite.semantic.layout_planner import (
    INK_BLACK,
    INK_BLUE,
    INK_RED,
    Decoration,
    LayoutPlanner,
    SegmentLayout,
)
from handwrite.semantic.annotation_renderer import render_annotations
from handwrite.semantic.semantic_composer import (
    SegmentRenderInfo,
    compose_semantic_page,
)

__all__ = [
    # text_analyzer
    "SemanticRole",
    "TextAnalyzer",
    "TextSegment",
    "extract_clean_text",
    # layout_planner
    "INK_BLACK",
    "INK_BLUE",
    "INK_RED",
    "Decoration",
    "LayoutPlanner",
    "SegmentLayout",
    # annotation_renderer
    "render_annotations",
    # semantic_composer
    "SegmentRenderInfo",
    "compose_semantic_page",
]
