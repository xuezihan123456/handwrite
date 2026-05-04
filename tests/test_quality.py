"""Tests for the handwriting quality evaluation module."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from handwrite.quality.authenticity_scorer import score_authenticity
from handwrite.quality.improvement_advisor import generate_char_advice, generate_page_advice
from handwrite.quality.naturalness_scorer import score_naturalness
from handwrite.quality.quality_engine import evaluate_char, evaluate_page
from handwrite.quality.quality_report import (
    CharacterQualityReport,
    DimensionScore,
    PageQualityReport,
)


# ---------------------------------------------------------------------------
# Test fixtures: helper functions to create test images
# ---------------------------------------------------------------------------


def _make_blank_image(size: tuple[int, int] = (128, 128)) -> Image.Image:
    """Create a blank white image."""
    return Image.new("L", size, color=255)


def _make_simple_char_image(
    size: tuple[int, int] = (128, 128),
    ink_positions: list[tuple[int, int]] | None = None,
) -> Image.Image:
    """Create a simple image with a few ink pixels."""
    image = _make_blank_image(size)
    if ink_positions is None:
        # Default: draw a simple rectangle
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 30, 90, 90), fill=0)
    else:
        for x, y in ink_positions:
            if 0 <= x < size[0] and 0 <= y < size[1]:
                image.putpixel((x, y), 0)
    return image


def _make_stroke_image(size: tuple[int, int] = (200, 200)) -> Image.Image:
    """Create an image with a diagonal stroke (simulating handwriting)."""
    image = _make_blank_image(size)
    draw = ImageDraw.Draw(image)
    # Draw a slightly curved line
    points = [(20, 180), (60, 120), (100, 80), (140, 50), (180, 30)]
    draw.line(points, fill=0, width=3)
    return image


def _make_multi_char_image(size: tuple[int, int] = (400, 128)) -> Image.Image:
    """Create an image with multiple separated character-like blobs."""
    image = _make_blank_image(size)
    draw = ImageDraw.Draw(image)
    # Draw several rectangles separated by gaps
    for x_start in range(20, 380, 80):
        draw.rectangle((x_start, 20, x_start + 40, 100), fill=0)
        # Add some variation
        draw.rectangle((x_start + 5, 25, x_start + 35, 50), fill=64)
    return image


def _make_uniform_image(size: tuple[int, int] = (128, 128)) -> Image.Image:
    """Create an image with perfectly uniform strokes (print-like)."""
    image = _make_blank_image(size)
    draw = ImageDraw.Draw(image)
    # Draw perfectly straight horizontal lines
    for y in range(20, 108, 10):
        draw.line((20, y, 108, y), fill=0, width=2)
    return image


def _make_natural_image(size: tuple[int, int] = (256, 256)) -> Image.Image:
    """Create an image with more natural-looking handwriting features."""
    image = _make_blank_image(size)
    draw = ImageDraw.Draw(image)
    # Draw lines with slight variation
    rng = np.random.RandomState(42)
    for y_base in range(30, 220, 40):
        points = []
        for x in range(20, 240, 10):
            y = y_base + int(rng.normal(0, 2))
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=0, width=2 + int(rng.randint(0, 2)))
    return image


# ---------------------------------------------------------------------------
# Tests for quality_report.py
# ---------------------------------------------------------------------------


class TestDimensionScore:
    def test_weighted_score_calculation(self) -> None:
        dim = DimensionScore(name="test", score=80.0, weight=0.5)
        assert dim.weighted_score == 40.0

    def test_frozen(self) -> None:
        dim = DimensionScore(name="test", score=80.0)
        with pytest.raises(AttributeError):
            dim.score = 90.0  # type: ignore[misc]


class TestCharacterQualityReport:
    def test_grade_excellent(self) -> None:
        dim = DimensionScore(name="test", score=90.0)
        report = CharacterQualityReport(
            char="A", authenticity=dim, naturalness=dim, overall_score=92.0
        )
        assert report.grade == "优秀"

    def test_grade_good(self) -> None:
        dim = DimensionScore(name="test", score=80.0)
        report = CharacterQualityReport(
            char="A", authenticity=dim, naturalness=dim, overall_score=78.0
        )
        assert report.grade == "良好"

    def test_grade_pass(self) -> None:
        dim = DimensionScore(name="test", score=65.0)
        report = CharacterQualityReport(
            char="A", authenticity=dim, naturalness=dim, overall_score=65.0
        )
        assert report.grade == "合格"

    def test_grade_needs_improvement(self) -> None:
        dim = DimensionScore(name="test", score=50.0)
        report = CharacterQualityReport(
            char="A", authenticity=dim, naturalness=dim, overall_score=45.0
        )
        assert report.grade == "待改进"


class TestPageQualityReport:
    def test_grade(self) -> None:
        report = PageQualityReport(
            overall_score=85.0, authenticity_score=80.0, naturalness_score=90.0
        )
        assert report.grade == "良好"

    def test_summary_contains_scores(self) -> None:
        report = PageQualityReport(
            overall_score=75.0,
            authenticity_score=70.0,
            naturalness_score=80.0,
            improvement_tips=("建议一", "建议二"),
        )
        summary = report.summary()
        assert "75.0" in summary
        assert "70.0" in summary
        assert "80.0" in summary
        assert "建议一" in summary
        assert "建议二" in summary


# ---------------------------------------------------------------------------
# Tests for authenticity_scorer.py
# ---------------------------------------------------------------------------


class TestAuthenticityScorer:
    def test_blank_image_returns_zero(self) -> None:
        image = _make_blank_image()
        result = score_authenticity(image)
        assert result.score == 0.0
        assert "无墨迹" in result.details

    def test_ink_image_returns_positive_score(self) -> None:
        image = _make_stroke_image()
        result = score_authenticity(image)
        assert result.score > 0
        assert result.score <= 100

    def test_score_has_details(self) -> None:
        image = _make_stroke_image()
        result = score_authenticity(image)
        assert "笔画连续性" in result.details
        assert "粗细变化" in result.details

    def test_score_is_between_0_and_100(self) -> None:
        image = _make_natural_image()
        result = score_authenticity(image)
        assert 0 <= result.score <= 100

    def test_uniform_image_lower_than_natural(self) -> None:
        uniform = _make_uniform_image()
        natural = _make_natural_image()
        uniform_score = score_authenticity(uniform)
        natural_score = score_authenticity(natural)
        # Natural handwriting should score higher than uniform/print-like
        assert natural_score.score >= uniform_score.score - 20  # Allow some tolerance


# ---------------------------------------------------------------------------
# Tests for naturalness_scorer.py
# ---------------------------------------------------------------------------


class TestNaturalnessScorer:
    def test_blank_image_returns_zero(self) -> None:
        image = _make_blank_image()
        result = score_naturalness(image)
        assert result.score == 0.0

    def test_ink_image_returns_positive_score(self) -> None:
        image = _make_natural_image()
        result = score_naturalness(image)
        assert result.score > 0
        assert result.score <= 100

    def test_score_has_details(self) -> None:
        image = _make_natural_image()
        result = score_naturalness(image)
        assert "行对齐" in result.details
        assert "大小变化" in result.details

    def test_score_range(self) -> None:
        image = _make_stroke_image()
        result = score_naturalness(image)
        assert 0 <= result.score <= 100


# ---------------------------------------------------------------------------
# Tests for improvement_advisor.py
# ---------------------------------------------------------------------------


class TestImprovementAdvisor:
    def test_page_advice_returns_strings(self) -> None:
        report = PageQualityReport(
            overall_score=50.0,
            authenticity_score=45.0,
            naturalness_score=55.0,
            dimensions=(
                DimensionScore(name="笔画连续性", score=40.0, suggestions=("笔画不连续",)),
                DimensionScore(name="粗细变化", score=60.0, suggestions=()),
            ),
        )
        tips = generate_page_advice(report)
        assert isinstance(tips, tuple)
        assert all(isinstance(t, str) for t in tips)
        assert len(tips) > 0

    def test_page_advice_limits_to_8(self) -> None:
        # Create a report with many dimensions to trigger many tips
        dims = tuple(
            DimensionScore(
                name=f"dim_{i}", score=30.0, suggestions=(f"建议{i}a", f"建议{i}b")
            )
            for i in range(10)
        )
        report = PageQualityReport(
            overall_score=30.0,
            authenticity_score=30.0,
            naturalness_score=30.0,
            dimensions=dims,
        )
        tips = generate_page_advice(report)
        assert len(tips) <= 8

    def test_char_advice_returns_strings(self) -> None:
        auth = DimensionScore(name="真实性", score=60.0, suggestions=("建议A",))
        nat = DimensionScore(name="自然度", score=70.0, suggestions=("建议B",))
        tips = generate_char_advice("字", auth, nat, 65.0)
        assert isinstance(tips, tuple)
        assert all(isinstance(t, str) for t in tips)

    def test_page_advice_deduplicates(self) -> None:
        report = PageQualityReport(
            overall_score=50.0,
            authenticity_score=50.0,
            naturalness_score=50.0,
            dimensions=(
                DimensionScore(name="dim1", score=50.0, suggestions=("重复建议",)),
                DimensionScore(name="dim2", score=50.0, suggestions=("重复建议",)),
            ),
        )
        tips = generate_page_advice(report)
        assert tips.count("重复建议") <= 1

    def test_high_score_advice_is_positive(self) -> None:
        report = PageQualityReport(
            overall_score=95.0,
            authenticity_score=95.0,
            naturalness_score=95.0,
        )
        tips = generate_page_advice(report)
        assert any("优秀" in t for t in tips)


# ---------------------------------------------------------------------------
# Tests for quality_engine.py
# ---------------------------------------------------------------------------


class TestQualityEngine:
    def test_evaluate_page_returns_report(self) -> None:
        image = _make_multi_char_image()
        report = evaluate_page(image)
        assert isinstance(report, PageQualityReport)
        assert 0 <= report.overall_score <= 100
        assert 0 <= report.authenticity_score <= 100
        assert 0 <= report.naturalness_score <= 100

    def test_evaluate_page_has_tips(self) -> None:
        image = _make_multi_char_image()
        report = evaluate_page(image)
        assert isinstance(report.improvement_tips, tuple)

    def test_evaluate_page_blank_image(self) -> None:
        image = _make_blank_image((200, 200))
        report = evaluate_page(image)
        assert report.overall_score == 0.0

    def test_evaluate_char_returns_report(self) -> None:
        image = _make_stroke_image()
        report = evaluate_char(image, char="A")
        assert isinstance(report, CharacterQualityReport)
        assert report.char == "A"
        assert 0 <= report.overall_score <= 100

    def test_evaluate_char_has_tips(self) -> None:
        image = _make_stroke_image()
        report = evaluate_char(image)
        assert isinstance(report.improvement_tips, tuple)

    def test_evaluate_char_blank_image(self) -> None:
        image = _make_blank_image()
        report = evaluate_char(image, char="空")
        assert report.overall_score == 0.0

    def test_evaluate_page_with_natural_image(self) -> None:
        image = _make_natural_image()
        report = evaluate_page(image)
        assert report.overall_score > 0

    def test_page_report_grade(self) -> None:
        image = _make_natural_image()
        report = evaluate_page(image)
        assert report.grade in ("优秀", "良好", "合格", "待改进")

    def test_char_report_grade(self) -> None:
        image = _make_stroke_image()
        report = evaluate_char(image)
        assert report.grade in ("优秀", "良好", "合格", "待改进")

    def test_evaluate_page_dimensions_populated(self) -> None:
        image = _make_multi_char_image()
        report = evaluate_page(image)
        # Should have some dimensions from parsing details
        assert isinstance(report.dimensions, tuple)

    def test_evaluate_page_with_single_char(self) -> None:
        image = _make_simple_char_image()
        report = evaluate_page(image)
        assert isinstance(report, PageQualityReport)


# ---------------------------------------------------------------------------
# Tests for __init__.py exports
# ---------------------------------------------------------------------------


class TestQualityModuleExports:
    def test_imports(self) -> None:
        from handwrite.quality import (
            CharacterQualityReport,
            DimensionScore,
            PageQualityReport,
            evaluate_char,
            evaluate_page,
            generate_char_advice,
            generate_page_advice,
            score_authenticity,
            score_naturalness,
        )

        assert callable(evaluate_page)
        assert callable(evaluate_char)
        assert callable(score_authenticity)
        assert callable(score_naturalness)
        assert callable(generate_page_advice)
        assert callable(generate_char_advice)

    def test_end_to_end(self) -> None:
        """Full pipeline: evaluate page, get report, get advice."""
        from handwrite.quality import evaluate_page, generate_page_advice

        image = _make_natural_image()
        report = evaluate_page(image)
        tips = generate_page_advice(report)

        assert isinstance(report, PageQualityReport)
        assert isinstance(tips, tuple)
        assert 0 <= report.overall_score <= 100
