"""Semantic text analysis using regex-based rules.

Identifies structural elements in text: titles, emphasis markers,
formula regions, and list items. Each segment is tagged with a
semantic role that downstream layout and rendering components use
to adjust typography.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, unique
from typing import Sequence


@unique
class SemanticRole(str, Enum):
    """Semantic role assigned to a text segment."""

    BODY = "body"
    TITLE = "title"
    EMPHASIS = "emphasis"
    FORMULA = "formula"
    LIST_ITEM = "list_item"


@dataclass(frozen=True)
class TextSegment:
    """A contiguous piece of text tagged with a semantic role."""

    text: str
    role: SemanticRole
    level: int = 0
    start: int = 0
    end: int = 0


# ---------------------------------------------------------------------------
# Internal pattern definitions
# ---------------------------------------------------------------------------

# Markdown-style headings:  # Title, ## Title, ...
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Chinese underline emphasis markers:  __text__  or  **text**
_EMPHASIS_RE = re.compile(r"(?:\*\*(.+?)\*\*|__(.+?)__)")

# LaTeX-style inline formula:  $...$
_INLINE_FORMULA_RE = re.compile(r"\$(.+?)\$")

# LaTeX-style block formula:  $$...$$
_BLOCK_FORMULA_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

# Chinese formula brackets:  \[...\]  or  \(...\)
_BRACKET_FORMULA_RE = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)")

# Ordered list: 1. / 1) / (1)
_ORDERED_LIST_RE = re.compile(r"^\s*(?:\d+[.)]\s|\(\d+\)\s)(.+)$", re.MULTILINE)

# Unordered list: - / * / +
_UNORDERED_LIST_RE = re.compile(r"^\s*[-*+]\s+(.+)$", re.MULTILINE)


class TextAnalyzer:
    """Analyse plain text and produce a sequence of semantic segments.

    The analyzer applies regex rules in priority order so that more
    specific patterns (headings, formulas) take precedence over
    generic body text.
    """

    def analyze(self, text: str) -> list[TextSegment]:
        """Return an ordered list of :class:`TextSegment` objects.

        Segments are non-overlapping and cover the full input text.
        """
        if not text:
            return []

        # Phase 1: collect all matches with priority
        annotations: list[tuple[int, int, SemanticRole, int]] = []
        annotations.extend(self._find_headings(text))
        annotations.extend(self._find_block_formulas(text))
        annotations.extend(self._find_inline_formulas(text))
        annotations.extend(self._find_emphasis(text))
        annotations.extend(self._find_ordered_lists(text))
        annotations.extend(self._find_unordered_lists(text))

        # Phase 2: resolve overlaps (higher priority wins)
        resolved = self._resolve_overlaps(annotations)

        # Phase 3: fill gaps with BODY
        return self._fill_gaps(text, resolved)

    # ------------------------------------------------------------------
    # Pattern finders
    # ------------------------------------------------------------------

    def _find_headings(self, text: str) -> list[tuple[int, int, SemanticRole, int]]:
        results: list[tuple[int, int, SemanticRole, int]] = []
        for match in _HEADING_RE.finditer(text):
            level = len(match.group(1))
            start = match.start()
            end = match.end()
            results.append((start, end, SemanticRole.TITLE, level))
        return results

    def _find_emphasis(self, text: str) -> list[tuple[int, int, SemanticRole, int]]:
        results: list[tuple[int, int, SemanticRole, int]] = []
        for match in _EMPHASIS_RE.finditer(text):
            results.append((match.start(), match.end(), SemanticRole.EMPHASIS, 0))
        return results

    def _find_inline_formulas(self, text: str) -> list[tuple[int, int, SemanticRole, int]]:
        results: list[tuple[int, int, SemanticRole, int]] = []
        for match in _INLINE_FORMULA_RE.finditer(text):
            results.append((match.start(), match.end(), SemanticRole.FORMULA, 0))
        return results

    def _find_block_formulas(self, text: str) -> list[tuple[int, int, SemanticRole, int]]:
        results: list[tuple[int, int, SemanticRole, int]] = []
        for match in _BLOCK_FORMULA_RE.finditer(text):
            results.append((match.start(), match.end(), SemanticRole.FORMULA, 1))
        for match in _BRACKET_FORMULA_RE.finditer(text):
            results.append((match.start(), match.end(), SemanticRole.FORMULA, 1))
        return results

    def _find_ordered_lists(self, text: str) -> list[tuple[int, int, SemanticRole, int]]:
        results: list[tuple[int, int, SemanticRole, int]] = []
        for match in _ORDERED_LIST_RE.finditer(text):
            results.append((match.start(), match.end(), SemanticRole.LIST_ITEM, 0))
        return results

    def _find_unordered_lists(self, text: str) -> list[tuple[int, int, SemanticRole, int]]:
        results: list[tuple[int, int, SemanticRole, int]] = []
        for match in _UNORDERED_LIST_RE.finditer(text):
            results.append((match.start(), match.end(), SemanticRole.LIST_ITEM, 0))
        return results

    # ------------------------------------------------------------------
    # Overlap resolution
    # ------------------------------------------------------------------

    _ROLE_PRIORITY: dict[SemanticRole, int] = {
        SemanticRole.FORMULA: 0,
        SemanticRole.TITLE: 1,
        SemanticRole.EMPHASIS: 2,
        SemanticRole.LIST_ITEM: 3,
        SemanticRole.BODY: 99,
    }

    def _resolve_overlaps(
        self,
        annotations: Sequence[tuple[int, int, SemanticRole, int]],
    ) -> list[tuple[int, int, SemanticRole, int]]:
        """Keep higher-priority annotations when spans overlap.

        Priority order (lower number wins): FORMULA > TITLE > EMPHASIS > LIST.
        """
        if not annotations:
            return []

        sorted_anns = sorted(annotations, key=lambda a: (a[0], self._ROLE_PRIORITY.get(a[2], 99)))
        result: list[tuple[int, int, SemanticRole, int]] = []

        for ann in sorted_anns:
            start, end, role, level = ann
            if start >= end:
                continue

            # Check overlap with existing result entries
            overlapping = False
            for i, (rs, re_, rr, rl) in enumerate(result):
                if start < re_ and end > rs:
                    # Overlap detected - keep the one with higher priority
                    if self._ROLE_PRIORITY.get(role, 99) < self._ROLE_PRIORITY.get(rr, 99):
                        result[i] = ann
                    overlapping = True
                    break

            if not overlapping:
                result.append(ann)

        result.sort(key=lambda a: a[0])
        return result

    # ------------------------------------------------------------------
    # Gap filling
    # ------------------------------------------------------------------

    def _fill_gaps(
        self,
        text: str,
        annotations: Sequence[tuple[int, int, SemanticRole, int]],
    ) -> list[TextSegment]:
        segments: list[TextSegment] = []
        cursor = 0

        for start, end, role, level in annotations:
            if start > cursor:
                body_text = text[cursor:start]
                if body_text:
                    segments.append(
                        TextSegment(text=body_text, role=SemanticRole.BODY, start=cursor, end=start)
                    )
            match_text = text[start:end]
            segments.append(TextSegment(text=match_text, role=role, level=level, start=start, end=end))
            cursor = end

        if cursor < len(text):
            remaining = text[cursor:]
            if remaining:
                segments.append(
                    TextSegment(text=remaining, role=SemanticRole.BODY, start=cursor, end=len(text))
                )

        return segments


def extract_clean_text(segment: TextSegment) -> str:
    """Strip markup delimiters from a segment, returning plain content.

    For example ``"## Hello"`` becomes ``"Hello"``, and ``"**bold**"``
    becomes ``"bold"``.
    """
    text = segment.text
    role = segment.role

    if role == SemanticRole.TITLE:
        return _HEADING_RE.sub(r"\2", text)

    if role == SemanticRole.EMPHASIS:
        return _EMPHASIS_RE.sub(lambda m: m.group(1) or m.group(2), text)

    if role == SemanticRole.FORMULA:
        cleaned = text
        cleaned = re.sub(r"^\$\$|\$\$$", "", cleaned)
        cleaned = re.sub(r"^\$|\$$", "", cleaned)
        cleaned = re.sub(r"^\\\[|\\\]$", "", cleaned)
        cleaned = re.sub(r"^\\\(|\\\)$", "", cleaned)
        return cleaned.strip()

    if role == SemanticRole.LIST_ITEM:
        cleaned = _ORDERED_LIST_RE.sub(r"\1", text)
        cleaned = _UNORDERED_LIST_RE.sub(r"\1", cleaned)
        return cleaned.strip()

    return text


__all__ = [
    "SemanticRole",
    "TextSegment",
    "TextAnalyzer",
    "extract_clean_text",
]
