"""Summary engine: main public API for smart summary handwriting.

Provides three primary functions:
- extract_summary(): Extract structured summary from raw text.
- render_mind_map(): Generate a mind map handwriting image.
- render_outline(): Generate an outline handwriting image.
"""

from __future__ import annotations

from PIL import Image

from handwrite.summary.mind_map_layout import MindMapLayout, compute_mind_map_layout
from handwrite.summary.outline_layout import OutlineLayout, compute_outline_layout
from handwrite.summary.summary_renderer import (
    render_mind_map_image,
    render_outline_image,
)
from handwrite.summary.text_summarizer import SummaryResult, extract_summary as _extract


def extract_summary(
    text: str,
    *,
    max_key_sentences: int = 8,
    max_bullet_points: int = 12,
    max_keywords: int = 15,
) -> SummaryResult:
    """Extract a structured summary from raw text.

    Uses rule-based extraction (regex + keyword frequency) to identify
    title, key sentences, bullet points, keywords, and sections.
    Supports Chinese and English mixed content.

    Args:
        text: Input text to summarize.
        max_key_sentences: Maximum key sentences to extract.
        max_bullet_points: Maximum bullet points to extract.
        max_keywords: Maximum keywords to extract.

    Returns:
        SummaryResult with all extracted information.
    """
    return _extract(
        text,
        max_key_sentences=max_key_sentences,
        max_bullet_points=max_bullet_points,
        max_keywords=max_keywords,
    )


def render_mind_map(
    text_or_summary: str | SummaryResult,
    *,
    font_size: int = 24,
    page_size: tuple[int, int] = (2480, 3508),
    margins: tuple[int, int, int, int] = (120, 120, 120, 120),
    paper: str = "\u767d\u7eb8",
    max_key_sentences: int = 8,
    max_bullet_points: int = 12,
    max_keywords: int = 15,
    generate_char_fn=None,
) -> Image.Image:
    """Generate a mind map handwriting page from text.

    Extracts a summary (if raw text is provided) and renders it as a
    mind map with a central topic and radiating branches.

    Args:
        text_or_summary: Raw text string or pre-computed SummaryResult.
        font_size: Base font size for rendering.
        page_size: Output page dimensions (width, height).
        margins: Page margins (top, right, bottom, left).
        paper: Paper type for background.
        max_key_sentences: Maximum key sentences to extract.
        max_bullet_points: Maximum bullet points to extract.
        max_keywords: Maximum keywords to extract.
        generate_char_fn: Optional callable(char, font_size) -> PIL.Image
            for handwriting character rendering.

    Returns:
        PIL Image of the rendered mind map.
    """
    summary = _ensure_summary(
        text_or_summary,
        max_key_sentences=max_key_sentences,
        max_bullet_points=max_bullet_points,
        max_keywords=max_keywords,
    )

    canvas_width, canvas_height = page_size
    layout = compute_mind_map_layout(
        summary,
        font_size=font_size,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )

    return render_mind_map_image(
        layout,
        page_size=page_size,
        font_size=font_size,
        margins=margins,
        paper=paper,
        generate_char_fn=generate_char_fn,
    )


def render_outline(
    text_or_summary: str | SummaryResult,
    *,
    font_size: int = 36,
    page_size: tuple[int, int] = (2480, 3508),
    margins: tuple[int, int, int, int] = (120, 120, 120, 120),
    paper: str = "\u767d\u7eb8",
    max_key_sentences: int = 8,
    max_bullet_points: int = 12,
    max_keywords: int = 15,
    generate_char_fn=None,
) -> Image.Image:
    """Generate an outline handwriting page from text.

    Extracts a summary (if raw text is provided) and renders it as a
    3-level hierarchical outline with indentation and bullet symbols.

    Args:
        text_or_summary: Raw text string or pre-computed SummaryResult.
        font_size: Base font size for rendering.
        page_size: Output page dimensions (width, height).
        margins: Page margins (top, right, bottom, left).
        paper: Paper type for background.
        max_key_sentences: Maximum key sentences to extract.
        max_bullet_points: Maximum bullet points to extract.
        max_keywords: Maximum keywords to extract.
        generate_char_fn: Optional callable(char, font_size) -> PIL.Image
            for handwriting character rendering.

    Returns:
        PIL Image of the rendered outline.
    """
    summary = _ensure_summary(
        text_or_summary,
        max_key_sentences=max_key_sentences,
        max_bullet_points=max_bullet_points,
        max_keywords=max_keywords,
    )

    layout = compute_outline_layout(summary)

    return render_outline_image(
        layout,
        page_size=page_size,
        font_size=font_size,
        margins=margins,
        paper=paper,
        generate_char_fn=generate_char_fn,
    )


def _ensure_summary(
    text_or_summary: str | SummaryResult,
    *,
    max_key_sentences: int,
    max_bullet_points: int,
    max_keywords: int,
) -> SummaryResult:
    """Ensure we have a SummaryResult, extracting one if needed."""
    if isinstance(text_or_summary, SummaryResult):
        return text_or_summary
    return _extract(
        text_or_summary,
        max_key_sentences=max_key_sentences,
        max_bullet_points=max_bullet_points,
        max_keywords=max_keywords,
    )
