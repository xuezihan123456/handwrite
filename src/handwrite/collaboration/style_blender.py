"""Style blender for smooth transitions at paragraph boundaries.

When two adjacent paragraphs are written by different contributors the
blender produces a per-character *style weight* that linearly interpolates
between the source and target style over a configurable number of lines
(default: 3).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlendRegion:
    """Describes a style transition zone between two paragraphs.

    Attributes:
        paragraph_index: The paragraph that begins the transition.
        line_offset: Starting line index *within* that paragraph (0-based).
        length: Number of lines in the blend region.
        source_style_id: Style fading out.
        target_style_id: Style fading in.
    """

    paragraph_index: int
    line_offset: int
    length: int
    source_style_id: int
    target_style_id: int


_DEFAULT_BLEND_LINES = 3


class StyleBlender:
    """Compute per-line blend weights across paragraph boundaries.

    Parameters
    ----------
    blend_lines:
        Number of lines over which to transition between styles.  The
        transition spans the last *blend_lines* of the outgoing paragraph
        and the first *blend_lines* of the incoming paragraph.
    """

    def __init__(self, blend_lines: int = _DEFAULT_BLEND_LINES) -> None:
        if blend_lines < 1:
            raise ValueError("blend_lines must be >= 1")
        self.blend_lines = blend_lines

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_char_weights(
        self,
        paragraphs: list[str],
        style_ids: list[int],
    ) -> list[list[float]]:
        """Return per-character blend weights for every paragraph.

        Each weight is in [0.0, 1.0] where **0.0** means "use the assigned
        contributor's style fully" and **1.0** means "fully use the *next*
        contributor's style".  When a paragraph is surrounded by the same
        style on both sides, all weights are 0.0.

        Parameters
        ----------
        paragraphs:
            The raw text of each paragraph.
        style_ids:
            The assigned ``style_id`` for each paragraph (same length as
            ``paragraphs``).

        Returns
        -------
        list[list[float]]
            Outer list has one entry per paragraph.  Inner list has one
            float per *character* in that paragraph.
        """
        if len(paragraphs) != len(style_ids):
            raise ValueError("paragraphs and style_ids must have the same length")

        paragraph_count = len(paragraphs)
        all_weights: list[list[float]] = []

        for pidx, text in enumerate(paragraphs):
            chars = list(text)
            char_count = len(chars)
            weights = [0.0] * char_count

            # Blend with the *previous* paragraph (tail region).
            if pidx > 0 and style_ids[pidx - 1] != style_ids[pidx]:
                self._apply_incoming_blend(
                    weights, chars, style_ids[pidx - 1], style_ids[pidx], side="start",
                )

            # Blend with the *next* paragraph (head region).
            if pidx < paragraph_count - 1 and style_ids[pidx] != style_ids[pidx + 1]:
                self._apply_outgoing_blend(
                    weights, chars, style_ids[pidx], style_ids[pidx + 1], side="end",
                )

            all_weights.append(weights)

        return all_weights

    def compute_line_weights(
        self,
        paragraphs: list[str],
        style_ids: list[int],
        chars_per_line: int,
    ) -> list[list[float]]:
        """Return per-line blend weights for every paragraph.

        Useful when composing at the line level rather than character level.

        Parameters
        ----------
        paragraphs, style_ids:
            Same semantics as :meth:`compute_char_weights`.
        chars_per_line:
            Approximate characters per line used during layout.

        Returns
        -------
        list[list[float]]
            Outer list: one entry per paragraph.  Inner list: one float per
            *line* in that paragraph.
        """
        if chars_per_line < 1:
            raise ValueError("chars_per_line must be >= 1")

        paragraph_count = len(paragraphs)
        all_weights: list[list[float]] = []

        for pidx, text in enumerate(paragraphs):
            line_count = max(1, (len(text) + chars_per_line - 1) // chars_per_line)
            weights = [0.0] * line_count

            if pidx > 0 and style_ids[pidx - 1] != style_ids[pidx]:
                self._apply_line_blend(weights, side="start")

            if pidx < paragraph_count - 1 and style_ids[pidx] != style_ids[pidx + 1]:
                self._apply_line_blend(weights, side="end")

            all_weights.append(weights)

        return all_weights

    def identify_blend_regions(
        self,
        style_ids: list[int],
        paragraph_line_counts: list[int],
    ) -> list[BlendRegion]:
        """Identify all transition regions in the document.

        Parameters
        ----------
        style_ids:
            Assigned style id per paragraph.
        paragraph_line_counts:
            Number of lines in each paragraph.

        Returns
        -------
        list[BlendRegion]
            One entry per paragraph boundary where styles differ.
        """
        regions: list[BlendRegion] = []
        for pidx in range(len(style_ids) - 1):
            if style_ids[pidx] == style_ids[pidx + 1]:
                continue
            line_count = paragraph_line_counts[pidx]
            blend_start = max(0, line_count - self.blend_lines)
            blend_length = line_count - blend_start
            if blend_length > 0:
                regions.append(
                    BlendRegion(
                        paragraph_index=pidx,
                        line_offset=blend_start,
                        length=blend_length,
                        source_style_id=style_ids[pidx],
                        target_style_id=style_ids[pidx + 1],
                    )
                )
        return regions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_incoming_blend(
        self,
        weights: list[float],
        chars: list[str],
        source_style: int,
        target_style: int,
        *,
        side: str,
    ) -> None:
        """Blend the *start* of a paragraph with the previous style."""
        blend_chars = self._count_blend_chars(chars, side="start")
        for idx in range(blend_chars):
            t = (idx + 1) / (blend_chars + 1)  # 0 -> source, 1 -> target
            # At the start, weight represents how much of *previous* style
            # to mix in, so we store (1 - t) for the incoming side.
            weights[idx] = max(weights[idx], 1.0 - t)

    def _apply_outgoing_blend(
        self,
        weights: list[float],
        chars: list[str],
        source_style: int,
        target_style: int,
        *,
        side: str,
    ) -> None:
        """Blend the *end* of a paragraph with the next style."""
        blend_chars = self._count_blend_chars(chars, side="end")
        total = len(chars)
        for offset in range(blend_chars):
            idx = total - blend_chars + offset
            t = (offset + 1) / (blend_chars + 1)  # 0 -> source, 1 -> target
            weights[idx] = max(weights[idx], t)

    def _apply_line_blend(self, weights: list[float], *, side: str) -> None:
        line_count = len(weights)
        blend_count = min(self.blend_lines, line_count)
        if side == "start":
            for idx in range(blend_count):
                t = (idx + 1) / (blend_count + 1)
                weights[idx] = max(weights[idx], 1.0 - t)
        else:  # end
            for offset in range(blend_count):
                idx = line_count - blend_count + offset
                t = (offset + 1) / (blend_count + 1)
                weights[idx] = max(weights[idx], t)

    def _count_blend_chars(self, chars: list[str], *, side: str) -> int:
        """Count non-whitespace characters in the blend zone."""
        total = len(chars)
        blend_lines = min(self.blend_lines, total)
        # Approximate: distribute blend over N characters from the edge.
        # We use blend_lines * ~20 chars as a heuristic for character-level
        # blending, but cap at half the paragraph to guarantee an unblended
        # region always remains (important for short paragraphs).
        approx = max(blend_lines, blend_lines * 20)
        approx = min(approx, max(1, (total - 1) // 2)) if total > 1 else total
        if side == "start":
            region = chars[:approx]
        else:
            region = chars[-approx:]
        non_space = sum(1 for c in region if not c.isspace())
        return max(1, non_space)
