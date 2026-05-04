"""Outline-style layout engine with 3-level indentation and bullet symbols.

Produces a structured text layout suitable for handwriting rendering,
with hierarchical indentation matching the summary structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from handwrite.summary.text_summarizer import SummaryResult


@dataclass(frozen=True)
class OutlineItem:
    """A single item in the outline."""

    text: str
    level: int  # 0 = title, 1 = heading, 2 = sub-item, 3 = detail
    bullet: str = ""  # Display bullet character


@dataclass(frozen=True)
class OutlineLayout:
    """Complete outline layout with ordered items."""

    items: list[OutlineItem]
    total_lines: int


# Bullet symbols for each level
_BULLET_LEVEL_1 = "\u25cf"  # Filled circle: ●
_BULLET_LEVEL_2 = "\u25cb"  # Open circle: ○
_BULLET_LEVEL_3 = "\u25aa"  # Small square: ▪

# Max characters per line at each indent level (for text wrapping)
_MAX_LINE_WIDTH = 40


def compute_outline_layout(
    summary: SummaryResult,
    *,
    max_line_width: int = _MAX_LINE_WIDTH,
) -> OutlineLayout:
    """Compute an outline layout from a summary.

    Creates a 3-level hierarchical outline:
    - Level 0: Title
    - Level 1: Section headings / key points (with bullet)
    - Level 2: Sub-items / details (with bullet)
    - Level 3: Supporting details (with bullet)

    Args:
        summary: The extracted summary result.
        max_line_width: Maximum characters per line for wrapping.

    Returns:
        OutlineLayout with ordered items.
    """
    items: list[OutlineItem] = []

    # Title
    if summary.title:
        items.append(OutlineItem(text=summary.title, level=0, bullet=""))

    # Sections with their items
    if summary.sections:
        for section in summary.sections:
            items.append(
                OutlineItem(text=section.heading, level=1, bullet=_BULLET_LEVEL_1)
            )
            for sub_item in section.items:
                wrapped = _wrap_text(sub_item, level=2, max_width=max_line_width)
                for line in wrapped:
                    items.append(
                        OutlineItem(text=line, level=2, bullet=_BULLET_LEVEL_2)
                    )

    # Key sentences as level-1 items (if no sections)
    if not summary.sections and summary.key_sentences:
        for sentence in summary.key_sentences:
            wrapped = _wrap_text(sentence, level=1, max_width=max_line_width)
            for i, line in enumerate(wrapped):
                bullet = _BULLET_LEVEL_1 if i == 0 else ""
                items.append(OutlineItem(text=line, level=1, bullet=bullet))

    # Bullet points as level-2 items
    if summary.bullet_points:
        # Only add a "要点" section if we already have sections
        if summary.sections:
            for bp in summary.bullet_points:
                wrapped = _wrap_text(bp, level=2, max_width=max_line_width)
                for line in wrapped:
                    items.append(
                        OutlineItem(text=line, level=2, bullet=_BULLET_LEVEL_2)
                    )
        else:
            for bp in summary.bullet_points:
                wrapped = _wrap_text(bp, level=2, max_width=max_line_width)
                for line in wrapped:
                    items.append(
                        OutlineItem(text=line, level=2, bullet=_BULLET_LEVEL_2)
                    )

    # Keywords as level-3 items
    if summary.keywords:
        # Group keywords into a compact display
        keyword_groups = _group_keywords(summary.keywords, max_width=max_line_width)
        if keyword_groups:
            items.append(OutlineItem(text="关键词", level=1, bullet=_BULLET_LEVEL_1))
            for group in keyword_groups:
                items.append(
                    OutlineItem(text=group, level=2, bullet=_BULLET_LEVEL_2)
                )

    return OutlineLayout(items=items, total_lines=len(items))


def _wrap_text(
    text: str,
    *,
    level: int,
    max_width: int,
) -> list[str]:
    """Wrap text to fit within the given width, accounting for indentation.

    Args:
        text: The text to wrap.
        level: Indentation level (affects available width).
        max_width: Maximum characters per line.

    Returns:
        List of wrapped lines.
    """
    # Indent prefix lengths (in characters)
    indent_chars = {0: 0, 1: 2, 2: 4, 3: 6}
    indent = indent_chars.get(level, 0)
    available = max_width - indent

    if available < 10:
        available = 10

    if len(text) <= available:
        return [text]

    lines: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= available:
            lines.append(remaining)
            break

        # Find a good break point
        break_point = _find_break_point(remaining, available)
        lines.append(remaining[:break_point])
        remaining = remaining[break_point:].lstrip()

    return lines


def _find_break_point(text: str, max_width: int) -> int:
    """Find the best position to break a line of text.

    Prefers breaking at punctuation, spaces, or between CJK characters.
    """
    if len(text) <= max_width:
        return len(text)

    # Look for punctuation break points (Chinese punctuation)
    zh_punct = set("，。！？；：、,")
    for i in range(min(max_width, len(text)) - 1, max(0, max_width // 2) - 1, -1):
        if text[i] in zh_punct:
            return i + 1

    # Look for space break
    for i in range(min(max_width, len(text)) - 1, max(0, max_width // 2) - 1, -1):
        if text[i] == " ":
            return i + 1

    # CJK characters can break between any two characters
    for i in range(min(max_width, len(text)) - 1, max(0, max_width // 2) - 1, -1):
        if ord(text[i]) > 0x2E7F:
            return i + 1

    # Hard break at max width
    return max_width


def _group_keywords(keywords: list[str], *, max_width: int) -> list[str]:
    """Group keywords into compact lines separated by commas."""
    if not keywords:
        return []

    groups: list[str] = []
    current = ""
    separator = " / "

    for kw in keywords:
        test = current + (separator if current else "") + kw
        if len(test) > max_width - 4 and current:
            groups.append(current)
            current = kw
        else:
            current = test

    if current:
        groups.append(current)

    return groups
