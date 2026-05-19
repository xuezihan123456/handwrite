"""Rule-based natural-language style picker (LLM-free).

The :class:`NaturalLanguageStyler` accepts a short bilingual description such
as "紧张焦虑的高三学生赶时间的字" or "calm, neat elementary teacher" and
produces a deterministic :class:`StyleVector` that downstream rendering code
can feed into the composer.

The pipeline is intentionally simple:
    1. Tokenise the description into overlapping Chinese substrings and
       whitespace-separated English words (lowercased).
    2. Walk the tokens left-to-right, tracking pending intensity modifiers.
    3. For every token that matches an emotion or style keyword apply its
       parameter delta multiplied by the active modifier.
    4. Resolve the most-voted layout, prefer the most-confident style name,
       and assemble a :class:`StyleVector`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .keywords import (
    EMOTION_TO_PARAMS,
    INTENSITY_MODIFIERS,
    KEYWORD_TO_STYLE,
    normalize_keyword,
)
from .style_vector import (
    CURSIVE_LAYOUT,
    NATURAL_LAYOUT,
    NEAT_LAYOUT,
    StyleVector,
)


# Maximum length of a Chinese keyword we will look up. Longest entry in the
# keyword tables is 5 characters; we cap at 6 for safety.
_MAX_CHINESE_KEYWORD_LEN = 6

# A token is either a run of CJK characters or a run of ASCII letters.
_TOKEN_PATTERN = re.compile(
    r"[\u4e00-\u9fff]+|[A-Za-z]+",
)


@dataclass
class _Hit:
    """One keyword match found while scanning the description."""

    name: str
    params: dict[str, float | str]
    modifier: float


class NaturalLanguageStyler:
    """Map a natural-language description to a :class:`StyleVector`.

    The implementation is fully deterministic: given the same input string,
    :meth:`parse` always returns an equal vector. No randomness, no external
    services.
    """

    def parse(self, description: str) -> StyleVector:
        """Convert ``description`` to a concrete :class:`StyleVector`.

        Empty or whitespace-only descriptions return ``StyleVector()`` with
        all defaults intact.
        """
        if not description or not description.strip():
            return StyleVector()

        hits = list(self._collect_hits(description))
        if not hits:
            # The text was non-empty but we did not recognise any keywords;
            # still return the default vector so callers get something usable.
            return StyleVector()

        return self._assemble(hits)

    # ------------------------------------------------------------------
    # Tokenisation / scanning
    # ------------------------------------------------------------------

    def _collect_hits(self, description: str) -> Iterable[_Hit]:
        text = description.strip()
        pending_modifier = 1.0
        position = 0
        length = len(text)

        while position < length:
            char = text[position]

            # Skip whitespace and punctuation between tokens.
            if not (char.isalpha() or self._is_cjk(char)):
                position += 1
                continue

            if self._is_cjk(char):
                consumed, hit, modifier_update = self._scan_cjk(
                    text, position, pending_modifier
                )
                position += consumed
            else:
                consumed, hit, modifier_update = self._scan_latin(
                    text, position, pending_modifier
                )
                position += consumed

            if hit is not None:
                yield hit
                pending_modifier = 1.0
            elif modifier_update is not None:
                pending_modifier = modifier_update

    def _scan_cjk(
        self,
        text: str,
        start: int,
        pending_modifier: float,
    ) -> tuple[int, _Hit | None, float | None]:
        """Try to match the longest CJK keyword starting at ``start``.

        Returns ``(consumed, hit, modifier_update)``.
        """
        max_len = min(_MAX_CHINESE_KEYWORD_LEN, len(text) - start)
        # Look-up longest match first.
        for size in range(max_len, 0, -1):
            candidate = text[start : start + size]
            normalised = normalize_keyword(candidate)

            if normalised in INTENSITY_MODIFIERS:
                return size, None, INTENSITY_MODIFIERS[normalised]

            if normalised in EMOTION_TO_PARAMS:
                params = EMOTION_TO_PARAMS[normalised]
                return size, _Hit(normalised, params, pending_modifier), None

            if normalised in KEYWORD_TO_STYLE:
                params = KEYWORD_TO_STYLE[normalised]
                return size, _Hit(normalised, params, pending_modifier), None

        # No match - consume a single CJK char and move on.
        return 1, None, None

    def _scan_latin(
        self,
        text: str,
        start: int,
        pending_modifier: float,
    ) -> tuple[int, _Hit | None, float | None]:
        """Match an English token (single word or short bigram modifier)."""
        match = _TOKEN_PATTERN.match(text, start)
        if not match:
            return 1, None, None
        token = match.group(0)
        consumed = len(token)
        normalised = normalize_keyword(token)

        # Try a two-word intensity modifier (e.g. "a bit", "kind of").
        peek_end = start + consumed
        bigram = self._peek_bigram(text, peek_end, normalised)
        if bigram is not None and bigram in INTENSITY_MODIFIERS:
            extra = self._whitespace_run(text, peek_end)
            second = _TOKEN_PATTERN.match(text, peek_end + extra)
            if second is not None:
                total = (peek_end + extra + len(second.group(0))) - start
                return total, None, INTENSITY_MODIFIERS[bigram]

        if normalised in INTENSITY_MODIFIERS:
            return consumed, None, INTENSITY_MODIFIERS[normalised]

        if normalised in EMOTION_TO_PARAMS:
            params = EMOTION_TO_PARAMS[normalised]
            return consumed, _Hit(normalised, params, pending_modifier), None

        if normalised in KEYWORD_TO_STYLE:
            params = KEYWORD_TO_STYLE[normalised]
            return consumed, _Hit(normalised, params, pending_modifier), None

        return consumed, None, None

    @staticmethod
    def _peek_bigram(text: str, start: int, first: str) -> str | None:
        offset = start
        while offset < len(text) and text[offset].isspace():
            offset += 1
        match = _TOKEN_PATTERN.match(text, offset)
        if match is None:
            return None
        return f"{first} {normalize_keyword(match.group(0))}"

    @staticmethod
    def _whitespace_run(text: str, start: int) -> int:
        offset = start
        while offset < len(text) and text[offset].isspace():
            offset += 1
        return offset - start

    @staticmethod
    def _is_cjk(char: str) -> bool:
        return "\u4e00" <= char <= "\u9fff"

    # ------------------------------------------------------------------
    # Vector assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble(hits: list[_Hit]) -> StyleVector:
        numeric_totals: dict[str, float] = {
            "rotation_jitter": 0.0,
            "scale_jitter": 0.0,
            "ink_density": 0.0,
            "baseline_jitter": 0.0,
            "char_spacing": 0.0,
            "line_spacing": 0.0,
        }
        # Defaults that we offset from.
        base = {
            "rotation_jitter": 1.5,
            "scale_jitter": 0.08,
            "ink_density": 1.0,
            "baseline_jitter": 0.1,
            "char_spacing": 1.0,
            "line_spacing": 1.0,
        }
        layout_votes: Counter[str] = Counter()
        style_votes: Counter[str] = Counter()
        mood_tags: list[str] = []

        for hit in hits:
            for key, delta in hit.params.items():
                if key.startswith("_"):
                    continue
                if key in numeric_totals and isinstance(delta, (int, float)):
                    numeric_totals[key] += float(delta) * hit.modifier

            layout = hit.params.get("_layout")
            if isinstance(layout, str):
                # Weight votes by modifier so "very anxious" outvotes a
                # casual mention of "neat". Optional per-keyword
                # ``_layout_weight`` lets explicit style words ("messy",
                # "潜草") outweigh mere persona descriptors ("学生").
                weight = hit.params.get("_layout_weight", 1.0)
                if not isinstance(weight, (int, float)):
                    weight = 1.0
                layout_votes[layout] += max(0.1, hit.modifier) * float(weight)

            style = hit.params.get("_style")
            if isinstance(style, str):
                style_votes[style] += max(0.1, hit.modifier)

            mood = hit.params.get("_mood")
            if isinstance(mood, str):
                mood_tags.append(mood)

        chosen_layout = layout_votes.most_common(1)[0][0] if layout_votes else NATURAL_LAYOUT
        chosen_style = style_votes.most_common(1)[0][0] if style_votes else "default"

        return StyleVector(
            rotation_jitter=base["rotation_jitter"] + numeric_totals["rotation_jitter"],
            scale_jitter=base["scale_jitter"] + numeric_totals["scale_jitter"],
            ink_density=base["ink_density"] + numeric_totals["ink_density"],
            baseline_jitter=base["baseline_jitter"] + numeric_totals["baseline_jitter"],
            char_spacing=base["char_spacing"] + numeric_totals["char_spacing"],
            line_spacing=base["line_spacing"] + numeric_totals["line_spacing"],
            style_name=chosen_style,
            suggested_layout=chosen_layout,
            mood_tags=mood_tags,
        )


def parse_description(text: str) -> StyleVector:
    """Module-level convenience wrapper around :class:`NaturalLanguageStyler`."""
    return NaturalLanguageStyler().parse(text)


__all__ = [
    "NaturalLanguageStyler",
    "parse_description",
    "NEAT_LAYOUT",
    "NATURAL_LAYOUT",
    "CURSIVE_LAYOUT",
]
