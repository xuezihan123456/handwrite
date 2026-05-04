"""Collaborative page composer.

Generates handwriting pages where different paragraphs may use different
contributor styles, with smooth style transitions at boundaries.
"""

from __future__ import annotations

from PIL import Image

from handwrite.collaboration.contributor import Contributor
from handwrite.collaboration.segment_assigner import Assignment, assign_segments
from handwrite.collaboration.style_blender import StyleBlender
from handwrite.composer import (
    CURSIVE_LAYOUT,
    NATURAL_LAYOUT,
    NEAT_LAYOUT,
    WHITE_PAPER,
    compose_page,
)


class CollaborativeComposer:
    """Compose a multi-contributor handwritten document.

    Parameters
    ----------
    contributors:
        List of 2--6 contributors.
    blend_lines:
        Number of lines over which to transition between styles at paragraph
        boundaries.  Defaults to 3.
    """

    def __init__(
        self,
        contributors: list[Contributor],
        blend_lines: int = 3,
    ) -> None:
        if not isinstance(contributors, list):
            raise TypeError("contributors must be a list")
        count = len(contributors)
        if count < 2 or count > 6:
            raise ValueError(f"Expected 2--6 contributors, got {count}")
        self.contributors = list(contributors)
        self.blender = StyleBlender(blend_lines=blend_lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose_paragraph(
        self,
        text: str,
        contributor: Contributor,
        char_images: list[Image.Image],
        *,
        page_size: tuple[int, int] = (2480, 3508),
        font_size: int = 80,
        margins: tuple[int, int, int, int] = (200, 200, 200, 200),
        layout: str = NATURAL_LAYOUT,
        paper: str = WHITE_PAPER,
    ) -> Image.Image:
        """Render a single paragraph with a contributor's style.

        This delegates to the standard ``compose_page`` to maintain output
        format compatibility.
        """
        return compose_page(
            char_images,
            text,
            page_size=page_size,
            font_size=font_size,
            margins=margins,
            layout=layout,
            paper=paper,
        )

    def get_assignments(
        self,
        paragraph_count: int,
        manual_mapping: list[int] | None = None,
    ) -> list[Assignment]:
        """Get the paragraph-to-contributor mapping."""
        return assign_segments(paragraph_count, self.contributors, manual_mapping)

    def get_blend_weights(
        self,
        paragraphs: list[str],
        assignments: list[Assignment],
    ) -> list[list[float]]:
        """Compute per-character blend weights for all paragraphs."""
        style_ids = [
            self.contributors[cidx].style_id for _, cidx in assignments
        ]
        return self.blender.compute_char_weights(paragraphs, style_ids)

    def blend_char_images(
        self,
        images_a: list[Image.Image],
        images_b: list[Image.Image],
        weights: list[float],
    ) -> list[Image.Image]:
        """Alpha-blend two sets of character images according to weights.

        For each position *i*, the output is::

            result[i] = images_a[i] * (1 - weights[i]) + images_b[i] * weights[i]

        When the two lists differ in length the shorter one is padded with
        blank (white) images.

        Parameters
        ----------
        images_a:
            Characters rendered in the source style.
        images_b:
            Characters rendered in the target style.
        weights:
            Per-character blend factor in [0, 1].  0 = fully *a*, 1 = fully *b*.

        Returns
        -------
        list[Image.Image]
            Blended character images.
        """
        max_len = max(len(images_a), len(images_b), len(weights))
        result: list[Image.Image] = []

        for idx in range(max_len):
            w = weights[idx] if idx < len(weights) else 0.0
            img_a = self._get_or_blank(images_a, idx)
            img_b = self._get_or_blank(images_b, idx)
            blended = _alpha_blend_images(img_a, img_b, w)
            result.append(blended)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_or_blank(images: list[Image.Image], index: int) -> Image.Image:
        if index < len(images):
            return images[index]
        return Image.new("L", (256, 256), color=255)


def _alpha_blend_images(
    img_a: Image.Image,
    img_b: Image.Image,
    weight: float,
) -> Image.Image:
    """Blend two grayscale images: ``a * (1-w) + b * w``."""
    weight = max(0.0, min(1.0, weight))
    if weight < 1e-6:
        return img_a.copy()
    if weight > 1.0 - 1e-6:
        return img_b.copy()

    size = img_a.size
    if img_b.size != size:
        img_b = img_b.resize(size, Image.Resampling.LANCZOS)

    import numpy as np

    arr_a = np.asarray(img_a.convert("L"), dtype=np.float32)
    arr_b = np.asarray(img_b.convert("L"), dtype=np.float32)
    blended = arr_a * (1.0 - weight) + arr_b * weight
    return Image.fromarray(blended.round().astype(np.uint8), mode="L")
