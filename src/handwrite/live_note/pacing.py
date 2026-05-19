"""Pacing strategies for the live-note writer.

Determines how many frames each character should occupy on the canvas
based on its kind (regular character vs punctuation vs whitespace) and
the configured writing speed.
"""

from __future__ import annotations

from typing import Literal

# Punctuation that should slow the pen down (Chinese + ASCII).
_HEAVY_PUNCTUATION = set(
    "\u3002\uff01\uff1f"  # 。！？
    ".!?"
)
_LIGHT_PUNCTUATION = set(
    "\uff0c\uff1b\uff1a\u3001"  # ，；：、
    ",;:"
)
_BREATH_PUNCTUATION = set(
    "\u2014\u2026"  # — …
    "-"
)

PacingStrategy = Literal["linear", "punctuation_pause", "breath_pause"]

# Minimum and maximum frame counts per glyph to keep videos sane.
_MIN_FRAMES_PER_CHAR = 1
_MAX_FRAMES_PER_CHAR = 480

# Standard characters-per-second target for a base 60wpm baseline.
_BASE_WPM = 60


def _frames_per_base_char(base_fps: int, wpm: int) -> float:
    """Return the (possibly fractional) frame count for a baseline glyph."""
    fps = max(1, int(base_fps))
    speed = max(1, int(wpm))
    # Characters per second = wpm / 60. Frames per char = fps / (wpm / 60).
    return fps * _BASE_WPM / (speed * _BASE_WPM)


def _scaled(frames: float, multiplier: float) -> int:
    raw = int(round(frames * multiplier))
    return max(_MIN_FRAMES_PER_CHAR, min(_MAX_FRAMES_PER_CHAR, raw))


def is_heavy_punctuation(char: str) -> bool:
    """Return ``True`` for sentence-ending punctuation that warrants a pause."""
    return char in _HEAVY_PUNCTUATION


def is_light_punctuation(char: str) -> bool:
    """Return ``True`` for clause-level punctuation with a small pause."""
    return char in _LIGHT_PUNCTUATION


def is_breath_punctuation(char: str) -> bool:
    """Return ``True`` for breath-style punctuation (em-dash, ellipsis)."""
    return char in _BREATH_PUNCTUATION


def compute_frame_count_for_char(
    char: str,
    base_fps: int,
    wpm: int,
    strategy: PacingStrategy = "punctuation_pause",
) -> int:
    """Compute the number of frames the writer should spend on ``char``.

    Args:
        char: Single character (may include punctuation, whitespace, or newline).
        base_fps: Frames per second for the output animation.
        wpm: Writing speed expressed in characters-per-minute.
        strategy: Pacing strategy to use.

    Returns:
        Integer number of frames to occupy with this character.
    """
    if len(char) != 1:
        raise ValueError(f"compute_frame_count_for_char expects single character, got {char!r}")

    base_frames = _frames_per_base_char(base_fps, wpm)

    # Whitespace and newlines should always be very quick.
    if char == "\n":
        return _scaled(base_frames, 0.3)
    if char.isspace():
        return _scaled(base_frames, 0.4)

    if strategy == "linear":
        return _scaled(base_frames, 1.0)

    plain = _scaled(base_frames, 1.0)

    if strategy == "punctuation_pause":
        if is_heavy_punctuation(char):
            return max(plain + 2, _scaled(base_frames, 2.4))
        if is_light_punctuation(char):
            return max(plain + 1, _scaled(base_frames, 1.6))
        if is_breath_punctuation(char):
            return max(plain + 1, _scaled(base_frames, 1.8))
        return plain

    if strategy == "breath_pause":
        if is_heavy_punctuation(char):
            return max(plain + 3, _scaled(base_frames, 3.0))
        if is_light_punctuation(char):
            return max(plain + 1, _scaled(base_frames, 1.4))
        if is_breath_punctuation(char):
            return max(plain + 2, _scaled(base_frames, 2.2))
        # Add a tiny breath every regular character.
        return max(plain, _scaled(base_frames, 1.15))

    raise ValueError(f"Unknown pacing strategy: {strategy!r}")


def plan_frame_budget(
    text: str,
    base_fps: int,
    wpm: int,
    strategy: PacingStrategy = "punctuation_pause",
) -> list[int]:
    """Return a list of frame counts aligned with ``text`` character by character."""
    return [
        compute_frame_count_for_char(c, base_fps=base_fps, wpm=wpm, strategy=strategy)
        for c in text
    ]


__all__ = [
    "PacingStrategy",
    "compute_frame_count_for_char",
    "is_heavy_punctuation",
    "is_light_punctuation",
    "is_breath_punctuation",
    "plan_frame_budget",
]
