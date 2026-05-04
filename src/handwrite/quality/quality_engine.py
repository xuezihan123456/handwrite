"""Main quality evaluation engine.

Provides the primary API for evaluating handwriting quality at both
page and character levels.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from handwrite.quality.authenticity_scorer import score_authenticity
from handwrite.quality.improvement_advisor import generate_char_advice, generate_page_advice
from handwrite.quality.naturalness_scorer import score_naturalness
from handwrite.quality.quality_report import (
    CharacterQualityReport,
    DimensionScore,
    PageQualityReport,
)

_INK_THRESHOLD = 128
_MIN_COMPONENT_SIZE = 4  # Minimum ink pixels to consider a character region


def evaluate_page(image: Image.Image) -> PageQualityReport:
    """Evaluate the quality of a full handwriting page.

    Performs holistic page-level analysis and per-character analysis
    where individual characters can be isolated.

    Args:
        image: Grayscale or RGBA handwriting page image.

    Returns:
        PageQualityReport with scores, diagnostics, and improvement tips.
    """
    # Page-level scores
    authenticity = score_authenticity(image)
    naturalness = score_naturalness(image)

    overall = _weighted_overall(authenticity.score, naturalness.score)

    # Collect page-level dimensions
    dimensions = _collect_page_dimensions(authenticity, naturalness)

    # Try to evaluate individual characters
    char_reports = _evaluate_characters(image)

    # Generate improvement tips
    temp_report = PageQualityReport(
        overall_score=overall,
        authenticity_score=authenticity.score,
        naturalness_score=naturalness.score,
        dimensions=dimensions,
        char_reports=tuple(char_reports),
    )
    tips = generate_page_advice(temp_report)

    return PageQualityReport(
        overall_score=overall,
        authenticity_score=authenticity.score,
        naturalness_score=naturalness.score,
        dimensions=dimensions,
        improvement_tips=tips,
        char_reports=tuple(char_reports),
    )


def evaluate_char(image: Image.Image, char: str = "?") -> CharacterQualityReport:
    """Evaluate the quality of a single character image.

    Args:
        image: Grayscale or RGBA character image.
        char: The character string (for labeling in the report).

    Returns:
        CharacterQualityReport with scores and improvement tips.
    """
    authenticity = score_authenticity(image)
    naturalness = score_naturalness(image)

    overall = _weighted_overall(authenticity.score, naturalness.score)

    tips = generate_char_advice(char, authenticity, naturalness, overall)

    return CharacterQualityReport(
        char=char,
        authenticity=authenticity,
        naturalness=naturalness,
        overall_score=overall,
        improvement_tips=tips,
    )


def _weighted_overall(authenticity_score: float, naturalness_score: float) -> float:
    """Compute weighted overall score from authenticity and naturalness."""
    # Authenticity weighted slightly higher (55/45)
    overall = authenticity_score * 0.55 + naturalness_score * 0.45
    return round(min(100.0, max(0.0, overall)), 1)


def _collect_page_dimensions(
    authenticity: DimensionScore,
    naturalness: DimensionScore,
) -> tuple[DimensionScore, ...]:
    """Collect all sub-dimensions from page-level scores."""
    dims: list[DimensionScore] = []

    # Parse details to extract sub-dimension scores
    if authenticity.details:
        dims.extend(_parse_dimension_details(authenticity.details, "authenticity"))
    if naturalness.details:
        dims.extend(_parse_dimension_details(naturalness.details, "naturalness"))

    return tuple(dims)


def _parse_dimension_details(
    details: str, source: str
) -> list[DimensionScore]:
    """Parse dimension detail string into DimensionScore objects."""
    dims: list[DimensionScore] = []
    parts = details.split(",")

    for part in parts:
        part = part.strip()
        if "=" in part:
            name, score_str = part.split("=", 1)
            try:
                score = float(score_str.strip())
                dims.append(
                    DimensionScore(
                        name=name.strip(),
                        score=score,
                        weight=0.5 if source == "authenticity" else 0.5,
                    )
                )
            except ValueError:
                continue

    return dims


def _evaluate_characters(image: Image.Image) -> list[CharacterQualityReport]:
    """Attempt to isolate and evaluate individual characters in the page."""
    binary = _to_binary_array(image)
    if not binary.any():
        return []

    height, width = binary.shape

    # Find character regions via column projection
    col_projection = binary.any(axis=0).astype(np.int32)

    char_regions: list[tuple[int, int]] = []  # (start_x, end_x)
    in_char = False
    start_x = 0

    for x in range(width):
        if col_projection[x]:
            if not in_char:
                start_x = x
                in_char = True
        else:
            if in_char:
                if x - start_x >= 3:  # Minimum width
                    char_regions.append((start_x, x))
                in_char = False

    if in_char and width - start_x >= 3:
        char_regions.append((start_x, width))

    if len(char_regions) < 2:
        return []

    # Evaluate each character region
    char_reports: list[CharacterQualityReport] = []
    for i, (x_start, x_end) in enumerate(char_regions[:20]):  # Limit to 20 chars
        # Find vertical bounds
        region = binary[:, x_start:x_end]
        ink_rows = np.where(region.any(axis=1))[0]
        if len(ink_rows) == 0:
            continue
        y_start = max(0, ink_rows[0] - 2)
        y_end = min(height, ink_rows[-1] + 3)

        # Crop the character
        char_image = image.crop((x_start, y_start, x_end, y_end))

        # Only evaluate if there's enough ink
        char_binary = _to_binary_array(char_image)
        if char_binary.sum() < _MIN_COMPONENT_SIZE:
            continue

        report = evaluate_char(char_image, char=str(i + 1))
        char_reports.append(report)

    return char_reports


def _to_binary_array(image: Image.Image) -> np.ndarray:
    """Convert image to binary array (True where ink is present)."""
    grayscale = image.convert("L")
    arr = np.array(grayscale, dtype=np.float32)
    return arr < _INK_THRESHOLD


__all__ = ["evaluate_page", "evaluate_char"]
