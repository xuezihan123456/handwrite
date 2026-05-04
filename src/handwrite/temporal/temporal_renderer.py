"""Temporal renderer for age-dependent handwriting composition.

Handles the layout and composition of characters with age-appropriate
positioning, spacing, and baseline effects.
"""

from __future__ import annotations

import random
from typing import Union

import numpy as np
from PIL import Image, ImageDraw

from .age_profiles import AgeGroup, AgeProfile, get_age_profile
from .skill_simulator import SkillSimulator


class TemporalRenderer:
    """Composes character images with age-dependent layout effects.

    Simulates developmental writing characteristics in page composition:
    - Baseline waviness (younger writers)
    - Character spacing variation
    - Line height variation
    - Margin irregularity
    """

    def __init__(
        self,
        age_group: AgeGroup = AgeGroup.ADULT,
        profile: AgeProfile | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the temporal renderer.

        Args:
            age_group: Target age group.
            profile: Optional custom profile overriding age_group.
            seed: Random seed for reproducible output.
        """
        self._age_group = age_group
        self._profile = profile if profile is not None else get_age_profile(age_group)
        self._simulator = SkillSimulator(
            age_group=age_group,
            profile=self._profile,
            seed=seed,
        )
        self._rng = random.Random(seed)

    @property
    def simulator(self) -> SkillSimulator:
        """Return the associated skill simulator."""
        return self._simulator

    def render_text(
        self,
        chars: list[Image.Image],
        text: str,
        page_size: tuple[int, int] = (2480, 3508),
        font_size: int = 80,
        margins: tuple[int, int, int, int] = (200, 200, 200, 200),
    ) -> Image.Image:
        """Render a full page of text with age-dependent layout effects.

        Args:
            chars: List of character images (one per non-whitespace character).
            text: The full text to render.
            page_size: Output page dimensions.
            font_size: Base character size.
            margins: Page margins (top, right, bottom, left).

        Returns:
            A composed page image with temporal effects applied.
        """
        profile = self._profile
        top, right, bottom, left = margins

        # Clamp margins to ensure at least one line/character fits
        page_width, page_height = page_size
        if top + bottom + font_size > page_height:
            max_total = max(0, page_height - font_size)
            total = top + bottom
            if total > 0:
                top = int(top * max_total / total)
                bottom = max_total - top
        if left + right + font_size > page_width:
            max_total = max(0, page_width - font_size)
            total = left + right
            if total > 0:
                left = int(left * max_total / total)
                right = max_total - left

        # Calculate layout parameters
        char_gap = max(4, int(font_size * 0.15))
        line_height = font_size + max(8, int(font_size * 0.4))

        # Create blank page
        page = Image.new("L", page_size, color=255)

        # Build token list
        tokens = _build_tokens(chars, text)

        # Render with age-dependent effects
        y = float(top)
        char_index = 0

        for line in _split_lines(tokens):
            if y + font_size > page_height - bottom:
                break

            # Apply baseline waviness
            baseline_offset = self._generate_baseline_offset(len(line), page_width - left - right)

            x = float(left)
            for symbol, image in line:
                if x + font_size > page_width - right:
                    break

                if image is not None and not symbol.isspace():
                    # Apply age-dependent character transform
                    transformed = self._simulator.apply_to_image(image, font_size)

                    # Calculate position with jitter and baseline effects
                    bx = x + baseline_offset.get(int(x), 0)
                    jittered_x, jittered_y = self._simulator.apply_jitter_to_offset(bx, y)

                    _paste_char(page, transformed, (int(jittered_x), int(jittered_y)), font_size)

                # Advance with spacing variation
                spacing = char_gap + self._rng.gauss(0, char_gap * profile.size_variation * 0.5)
                x += font_size + max(2, int(spacing))

            # Advance line with height variation
            line_variation = self._rng.gauss(0, line_height * (1.0 - profile.line_straightness) * 0.1)
            y += line_height + max(0, int(line_variation))

        return page

    def _generate_baseline_offset(
        self,
        num_chars: int,
        available_width: int,
    ) -> dict[int, float]:
        """Generate per-character baseline offset for wavy writing.

        Younger writers produce more wavy baselines.

        Args:
            num_chars: Number of characters on the line.
            available_width: Width available for the line.

        Returns:
            Mapping from x-position to vertical offset.
        """
        profile = self._profile
        waviness = 1.0 - profile.line_straightness

        if waviness < 0.05:
            return {}

        offsets: dict[int, float] = {}
        # Use a slow sine wave + noise for natural waviness
        amplitude = waviness * 8.0  # max offset in pixels
        frequency = self._rng.uniform(0.002, 0.008)

        for x in range(0, available_width, 10):
            wave = amplitude * np.sin(x * frequency + self._rng.uniform(0, 6.28))
            noise = self._rng.gauss(0, amplitude * 0.3)
            offsets[x] = float(wave + noise)

        return offsets

    def vary_line_spacing(self, base_spacing: int) -> int:
        """Return a varied line spacing value.

        Args:
            base_spacing: The nominal line spacing.

        Returns:
            Spacing with age-appropriate variation.
        """
        variation = self._profile.size_variation * 0.3
        factor = 1.0 + self._rng.gauss(0, variation)
        factor = max(0.7, min(1.3, factor))
        return max(8, int(base_spacing * factor))


def _build_tokens(
    chars: list[Image.Image],
    text: str,
) -> list[tuple[str, Image.Image | None]]:
    """Build token list pairing characters with their images."""
    tokens: list[tuple[str, Image.Image | None]] = []
    char_index = 0

    for symbol in text:
        if symbol == "\n":
            tokens.append((symbol, None))
            continue
        if symbol.isspace():
            tokens.append((symbol, None))
            continue
        if char_index >= len(chars):
            break
        tokens.append((symbol, chars[char_index]))
        char_index += 1

    return tokens


def _split_lines(
    tokens: list[tuple[str, Image.Image | None]],
) -> list[list[tuple[str, Image.Image | None]]]:
    """Split tokens into lines at newline boundaries."""
    lines: list[list[tuple[str, Image.Image | None]]] = [[]]

    for token in tokens:
        symbol, _ = token
        if symbol == "\n":
            lines.append([])
            continue
        lines[-1].append(token)

    return [line for line in lines if line]


def _paste_char(
    page: Image.Image,
    char_image: Image.Image,
    origin: tuple[int, int],
    font_size: int,
) -> None:
    """Paste a character image onto the page at the given origin."""
    # Convert RGBA to grayscale mask
    rgba = char_image.convert("RGBA")
    alpha = rgba.getchannel("A")

    # Resize if needed
    if char_image.size != (font_size, font_size):
        alpha = alpha.resize((font_size, font_size), Image.Resampling.LANCZOS)

    # Ensure we don't paste outside the page
    x, y = origin
    if x < 0 or y < 0:
        return
    if x + font_size > page.size[0] or y + font_size > page.size[1]:
        return

    # Use alpha as mask for pasting dark ink
    page.paste(0, (x, y), alpha)


__all__ = ["TemporalRenderer"]
