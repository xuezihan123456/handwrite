"""Sequence diff utilities for error notebook.

Compute character-level edit diffs between a (wrong) student answer and the
known correct answer.  The diff result is a list of segments, each tagged with
a kind in ``{"equal", "delete", "insert", "replace"}``.  ``delete`` segments
are present in the wrong answer but missing from the correct one, ``insert``
segments are present in the correct answer only, ``replace`` segments differ
in both, and ``equal`` segments are shared.

The algorithm preserves token boundaries when possible -- the wrong and
correct strings are tokenised by whitespace and grouped into LaTeX clusters so
that ``\\frac{1}{2}`` is kept as one block, then a longest-common-subsequence
walk is used to derive the segments.  When tokenisation collapses both inputs
to a single token, the function falls back to ``difflib.SequenceMatcher`` on
characters so that the result stays well-defined for any pair of strings.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import List

__all__ = [
    "DIFF_EQUAL",
    "DIFF_DELETE",
    "DIFF_INSERT",
    "DIFF_REPLACE",
    "DiffSegment",
    "diff_answers",
    "tokenize_answer",
]


DIFF_EQUAL = "equal"
DIFF_DELETE = "delete"
DIFF_INSERT = "insert"
DIFF_REPLACE = "replace"

_VALID_KINDS = {DIFF_EQUAL, DIFF_DELETE, DIFF_INSERT, DIFF_REPLACE}


# Match LaTeX-style command clusters such as ``\frac{1}{2}`` or ``x^{2}``.
_LATEX_PATTERN = re.compile(
    r"\\[A-Za-z]+(?:\{[^{}]*\})*"
    r"|[A-Za-z0-9]+(?:\^\{[^{}]*\}|_\{[^{}]*\}|\^\w|_\w)+"
)


@dataclass(frozen=True)
class DiffSegment:
    """A single diff hunk between wrong and correct answers."""

    kind: str
    wrong: str
    correct: str

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"DiffSegment.kind must be one of {_VALID_KINDS}, got {self.kind!r}"
            )

    @property
    def is_change(self) -> bool:
        """Whether this segment denotes a difference."""
        return self.kind != DIFF_EQUAL

    @property
    def text(self) -> str:
        """Preferred display text (correct if available, else wrong)."""
        return self.correct or self.wrong


def tokenize_answer(text: str) -> List[str]:
    """Split *text* into diff-friendly tokens.

    LaTeX-like clusters and runs of digits or letters are kept together so
    that diffs stay readable for math equations.  Whitespace is preserved as
    its own token to make rejoining lossless.
    """
    if not text:
        return []

    tokens: List[str] = []
    cursor = 0
    length = len(text)

    while cursor < length:
        char = text[cursor]

        if char.isspace():
            run_end = cursor + 1
            while run_end < length and text[run_end].isspace():
                run_end += 1
            tokens.append(text[cursor:run_end])
            cursor = run_end
            continue

        match = _LATEX_PATTERN.match(text, cursor)
        if match and match.end() > cursor:
            tokens.append(match.group())
            cursor = match.end()
            continue

        if char.isdigit():
            run_end = cursor + 1
            while run_end < length and text[run_end].isdigit():
                run_end += 1
            tokens.append(text[cursor:run_end])
            cursor = run_end
            continue

        if char.isalpha() and ord(char) < 128:
            run_end = cursor + 1
            while (
                run_end < length
                and text[run_end].isalpha()
                and ord(text[run_end]) < 128
            ):
                run_end += 1
            tokens.append(text[cursor:run_end])
            cursor = run_end
            continue

        tokens.append(char)
        cursor += 1

    return tokens


def diff_answers(wrong: str, correct: str) -> List[DiffSegment]:
    """Return a list of :class:`DiffSegment` describing wrong -> correct.

    The function never raises for empty inputs and always returns a list
    whose ``"".join(seg.wrong for seg in result) == wrong`` and
    ``"".join(seg.correct for seg in result) == correct`` invariants hold.
    """
    if not isinstance(wrong, str):
        raise TypeError("wrong must be a str")
    if not isinstance(correct, str):
        raise TypeError("correct must be a str")

    if wrong == correct:
        if not wrong:
            return []
        return [DiffSegment(DIFF_EQUAL, wrong, correct)]

    if not wrong:
        return [DiffSegment(DIFF_INSERT, "", correct)]
    if not correct:
        return [DiffSegment(DIFF_DELETE, wrong, "")]

    wrong_tokens = tokenize_answer(wrong)
    correct_tokens = tokenize_answer(correct)

    if len(wrong_tokens) <= 1 and len(correct_tokens) <= 1:
        return _diff_by_chars(wrong, correct)

    return _diff_by_tokens(wrong_tokens, correct_tokens)


def _diff_by_tokens(
    wrong_tokens: List[str],
    correct_tokens: List[str],
) -> List[DiffSegment]:
    matcher = difflib.SequenceMatcher(a=wrong_tokens, b=correct_tokens, autojunk=False)
    segments: List[DiffSegment] = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        wrong_text = "".join(wrong_tokens[i1:i2])
        correct_text = "".join(correct_tokens[j1:j2])
        kind = _opcode_to_kind(opcode)
        if kind == DIFF_EQUAL and not wrong_text and not correct_text:
            continue
        segments.append(DiffSegment(kind, wrong_text, correct_text))
    return _coalesce_segments(segments)


def _diff_by_chars(wrong: str, correct: str) -> List[DiffSegment]:
    matcher = difflib.SequenceMatcher(a=wrong, b=correct, autojunk=False)
    segments: List[DiffSegment] = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        wrong_text = wrong[i1:i2]
        correct_text = correct[j1:j2]
        kind = _opcode_to_kind(opcode)
        if kind == DIFF_EQUAL and not wrong_text and not correct_text:
            continue
        segments.append(DiffSegment(kind, wrong_text, correct_text))
    return _coalesce_segments(segments)


def _opcode_to_kind(opcode: str) -> str:
    if opcode == "equal":
        return DIFF_EQUAL
    if opcode == "delete":
        return DIFF_DELETE
    if opcode == "insert":
        return DIFF_INSERT
    return DIFF_REPLACE


def _coalesce_segments(segments: List[DiffSegment]) -> List[DiffSegment]:
    """Merge adjacent segments of the same kind for cleaner output."""
    if not segments:
        return []

    merged: List[DiffSegment] = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        if segment.kind == previous.kind:
            merged[-1] = DiffSegment(
                kind=previous.kind,
                wrong=previous.wrong + segment.wrong,
                correct=previous.correct + segment.correct,
            )
        else:
            merged.append(segment)
    return merged
