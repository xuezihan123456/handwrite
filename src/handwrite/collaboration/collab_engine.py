"""Main entry point for collaborative handwriting document generation.

Provides ``generate_collaborative_document()`` which orchestrates the full
pipeline: splitting text into paragraphs, assigning contributors, generating
character images with style blending, and composing the final pages.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from handwrite.collaboration.collaborative_composer import CollaborativeComposer
from handwrite.collaboration.contributor import Contributor
from handwrite.collaboration.segment_assigner import assign_segments
from handwrite.collaboration.style_blender import StyleBlender
from handwrite.composer import (
    NATURAL_LAYOUT,
    WHITE_PAPER,
    compose_page,
)
from handwrite.styles import BUILTIN_STYLES


def generate_collaborative_document(
    text: str,
    contributors: list[Contributor],
    *,
    manual_mapping: list[int] | None = None,
    blend_lines: int = 3,
    page_size: tuple[int, int] = (2480, 3508),
    font_size: int = 80,
    margins: tuple[int, int, int, int] = (200, 200, 200, 200),
    layout: str = NATURAL_LAYOUT,
    paper: str = WHITE_PAPER,
    prototype_pack: str | Path | None = None,
) -> dict[str, object]:
    """Generate a collaborative handwriting document.

    The document text is split into paragraphs (separated by blank lines),
    each paragraph is assigned to a contributor, and the final pages are
    composed with smooth style transitions at boundaries.

    Parameters
    ----------
    text:
        Full document text.  Paragraphs are separated by one or more blank
        lines.
    contributors:
        List of 2--6 :class:`Contributor` instances.
    manual_mapping:
        Optional explicit mapping: ``manual_mapping[i]`` is the contributor
        index for paragraph *i*.  When *None*, round-robin is used.
    blend_lines:
        Number of lines over which to transition between styles.
    page_size, font_size, margins, layout, paper:
        Passed through to the underlying page composer.
    prototype_pack:
        Optional path to a custom prototype pack for character generation.

    Returns
    -------
    dict
        A dictionary compatible with the single-person ``generate_pages``
        output, plus collaboration metadata::

            {
                "pages": list[Image.Image],
                "paragraphs": list[str],
                "assignments": list[tuple[int, int]],
                "contributors": list[dict],
                "blend_lines": int,
                "page_count": int,
            }
    """
    # --- validate inputs ---------------------------------------------------
    if not text.strip():
        return _empty_result(contributors, blend_lines)

    if not isinstance(contributors, list):
        raise TypeError("contributors must be a list of Contributor instances")
    count = len(contributors)
    if count < 2 or count > 6:
        raise ValueError(f"Expected 2--6 contributors, got {count}")

    # --- split into paragraphs ---------------------------------------------
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return _empty_result(contributors, blend_lines)

    # --- assign contributors -----------------------------------------------
    assignments = assign_segments(len(paragraphs), contributors, manual_mapping)

    # --- generate character images per paragraph ---------------------------
    engine = _get_engine(prototype_pack)
    style_blender = StyleBlender(blend_lines=blend_lines)
    style_ids = [contributors[cidx].style_id for _, cidx in assignments]
    char_weights = style_blender.compute_char_weights(paragraphs, style_ids)

    paragraph_images: list[list[Image.Image]] = []
    for pidx, para_text in enumerate(paragraphs):
        cidx = assignments[pidx][1]
        contributor = contributors[cidx]
        weights = char_weights[pidx]
        images = _generate_paragraph_chars(
            para_text, contributor, engine, weights, contributors, assignments, pidx,
        )
        paragraph_images.append(images)

    # --- compose pages -----------------------------------------------------
    full_text = "\n\n".join(paragraphs)
    all_chars: list[Image.Image] = []
    for imgs in paragraph_images:
        all_chars.extend(imgs)

    pages = _compose_all_pages(
        all_chars,
        full_text,
        page_size=page_size,
        font_size=font_size,
        margins=margins,
        layout=layout,
        paper=paper,
    )

    return {
        "pages": pages,
        "paragraphs": paragraphs,
        "assignments": assignments,
        "contributors": [
            {"name": c.name, "style_id": c.style_id, "params": c.params}
            for c in contributors
        ],
        "blend_lines": blend_lines,
        "page_count": len(pages),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs separated by blank lines."""
    paragraphs: list[str] = []
    current: list[str] = []

    for line in text.split("\n"):
        if line.strip() == "":
            if current:
                paragraphs.append("\n".join(current))
                current = []
        else:
            current.append(line)

    if current:
        paragraphs.append("\n".join(current))

    return paragraphs


def _get_engine(prototype_pack: str | Path | None):
    """Lazy-load the StyleEngine to avoid heavy imports at module level."""
    from handwrite.engine.model import StyleEngine

    return StyleEngine(prototype_pack=prototype_pack)


def _generate_paragraph_chars(
    text: str,
    contributor: Contributor,
    engine,
    weights: list[float],
    contributors: list[Contributor],
    assignments: list[tuple[int, int]],
    pidx: int,
) -> list[Image.Image]:
    """Generate character images for a paragraph, applying style blending."""
    chars = [c for c in text if c not in {" ", "\n"}]
    own_style = contributor.style_id

    # Check if blending is needed with adjacent paragraphs.
    needs_blend = any(w > 1e-6 for w in weights)

    if not needs_blend:
        return [engine.generate_char(c, own_style) for c in chars]

    # Determine the neighboring style to blend with.
    neighbor_style = _neighbor_style(pidx, assignments, contributors)
    if neighbor_style is None or neighbor_style == own_style:
        return [engine.generate_char(c, own_style) for c in chars]

    images: list[Image.Image] = []
    for cidx, char in enumerate(chars):
        w = weights[cidx] if cidx < len(weights) else 0.0
        if w < 1e-6:
            images.append(engine.generate_char(char, own_style))
        elif w > 1.0 - 1e-6:
            images.append(engine.generate_char(char, neighbor_style))
        else:
            img_a = engine.generate_char(char, own_style)
            img_b = engine.generate_char(char, neighbor_style)
            images.append(_blend_images(img_a, img_b, w))
    return images


def _neighbor_style(
    pidx: int,
    assignments: list[tuple[int, int]],
    contributors: list[Contributor],
) -> int | None:
    """Return the most relevant neighbor's style_id for blending."""
    # Prefer next paragraph; fall back to previous.
    if pidx < len(assignments) - 1:
        return contributors[assignments[pidx + 1][1]].style_id
    if pidx > 0:
        return contributors[assignments[pidx - 1][1]].style_id
    return None


def _blend_images(
    img_a: Image.Image,
    img_b: Image.Image,
    weight: float,
) -> Image.Image:
    """Blend two grayscale images: ``a*(1-w) + b*w``."""
    import numpy as np

    weight = max(0.0, min(1.0, weight))
    size = img_a.size
    if img_b.size != size:
        img_b = img_b.resize(size, Image.Resampling.LANCZOS)

    arr_a = np.asarray(img_a.convert("L"), dtype=np.float32)
    arr_b = np.asarray(img_b.convert("L"), dtype=np.float32)
    blended = arr_a * (1.0 - weight) + arr_b * weight
    return Image.fromarray(blended.round().astype(np.uint8), mode="L")


def _compose_all_pages(
    chars: list[Image.Image],
    text: str,
    *,
    page_size: tuple[int, int],
    font_size: int,
    margins: tuple[int, int, int, int],
    layout: str,
    paper: str,
) -> list[Image.Image]:
    """Split text into page-sized chunks and compose each page.

    Reuses the same pagination logic as ``handwrite.__init__.generate_pages``
    to maintain output format compatibility.
    """
    from handwrite import _split_text_into_pages

    page_chunks = _split_text_into_pages(
        text, font_size=font_size, layout=layout, paper=paper,
    )

    pages: list[Image.Image] = []
    char_offset = 0
    for chunk in page_chunks:
        chunk_chars = [
            c for c in chunk if c not in {" ", "\n"}
        ]
        needed = len(chunk_chars)
        page_chars = chars[char_offset : char_offset + needed]
        # Pad if we ran out of generated chars.
        while len(page_chars) < needed:
            page_chars.append(Image.new("L", (256, 256), color=255))
        char_offset += needed

        page = compose_page(
            page_chars,
            chunk,
            page_size=page_size,
            font_size=font_size,
            margins=margins,
            layout=layout,
            paper=paper,
        )
        pages.append(page)

    return pages or [Image.new("L", page_size, color=255)]


def _empty_result(
    contributors: list[Contributor], blend_lines: int,
) -> dict[str, object]:
    return {
        "pages": [],
        "paragraphs": [],
        "assignments": [],
        "contributors": [
            {"name": c.name, "style_id": c.style_id, "params": c.params}
            for c in contributors
        ],
        "blend_lines": blend_lines,
        "page_count": 0,
    }
