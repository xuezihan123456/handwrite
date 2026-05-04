"""Tests for the handwrite.grading module."""

from __future__ import annotations

import pytest

from handwrite.grading.error_detector import ErrorDetector, ErrorInfo, ErrorType
from handwrite.grading.annotation_renderer import (
    ANNOTATION_RED,
    Annotation,
    AnnotationRenderer,
    AnnotationType,
)
from handwrite.grading.score_calculator import GradeLevel, ScoreCalculator, ScoreResult
from handwrite.grading.feedback_generator import FeedbackGenerator
from handwrite.grading.grading_engine import GradingEngine, GradingResult, grade, annotate


def _has_pil() -> bool:
    """Check if Pillow is available."""
    try:
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        return False


def _has_color_pixels(
    img: "Image.Image", color: tuple[int, int, int], tolerance: int = 10
) -> bool:
    """Check if an image contains pixels of the given color."""
    from PIL import Image  # noqa: F401
    pixels = list(img.getdata())
    for pixel in pixels:
        if isinstance(pixel, tuple) and len(pixel) >= 3:
            r, g, b = pixel[:3]
            if abs(r - color[0]) <= tolerance and abs(g - color[1]) <= tolerance and abs(b - color[2]) <= tolerance:
                return True
    return False


# ======================================================================
# ErrorDetector tests
# ======================================================================


class TestErrorDetector:
    """Tests for ErrorDetector."""

    def test_detect_typos_from_dict(self) -> None:
        detector = ErrorDetector()
        errors = detector.detect("他按装了软件")
        assert len(errors) >= 1
        typo = errors[0]
        assert typo.error_type == ErrorType.TYPO
        assert typo.text == "按装"
        assert typo.suggestion == "安装"
        assert typo.position == 1

    def test_detect_multiple_typos(self) -> None:
        detector = ErrorDetector()
        errors = detector.detect("他的脉博很强，气慨不凡")
        typo_texts = [e.text for e in errors if e.error_type == ErrorType.TYPO]
        assert "脉博" in typo_texts
        assert "气慨" in typo_texts

    def test_detect_punctuation_chinese_period(self) -> None:
        detector = ErrorDetector()
        errors = detector.detect("你好.世界")
        punct_errors = [e for e in errors if e.error_type == ErrorType.PUNCTUATION]
        assert len(punct_errors) >= 1

    def test_detect_repeated_punctuation(self) -> None:
        detector = ErrorDetector()
        errors = detector.detect("你好，，世界")
        punct_errors = [e for e in errors if e.error_type == ErrorType.PUNCTUATION]
        assert len(punct_errors) >= 1

    def test_detect_grammar_about_left_right(self) -> None:
        detector = ErrorDetector()
        errors = detector.detect("大约有100左右人。")
        grammar_errors = [e for e in errors if e.error_type == ErrorType.GRAMMAR]
        assert len(grammar_errors) >= 1

    def test_detect_format_tab(self) -> None:
        detector = ErrorDetector()
        errors = detector.detect("hello\tworld")
        format_errors = [e for e in errors if e.error_type == ErrorType.FORMAT]
        assert len(format_errors) >= 1

    def test_detect_no_errors_clean_text(self) -> None:
        detector = ErrorDetector()
        errors = detector.detect("这是一段完全正确的文本。")
        assert len(errors) == 0

    def test_detect_typos_only(self) -> None:
        detector = ErrorDetector()
        errors = detector.detect_typos("他按装了软件")
        for e in errors:
            assert e.error_type == ErrorType.TYPO

    def test_detect_grammar_only(self) -> None:
        detector = ErrorDetector()
        errors = detector.detect_grammar("大约有100左右人。")
        for e in errors:
            assert e.error_type == ErrorType.GRAMMAR

    def test_extra_typo_dict(self) -> None:
        extra = {"自已": ("自己", "'已'应为'己'")}
        detector = ErrorDetector(extra_typo_dict=extra)
        errors = detector.detect_typos("我自已来")
        assert any(e.text == "自已" for e in errors)

    def test_errors_sorted_by_position(self) -> None:
        detector = ErrorDetector()
        errors = detector.detect("按装软件后，他的脉博正常。")
        positions = [e.position for e in errors]
        assert positions == sorted(positions)

    def test_error_info_end_position(self) -> None:
        info = ErrorInfo(
            error_type=ErrorType.TYPO,
            position=5,
            length=2,
            text="test",
            message="msg",
        )
        assert info.end_position() == 7


# ======================================================================
# ScoreCalculator tests
# ======================================================================


class TestScoreCalculator:
    """Tests for ScoreCalculator."""

    def test_no_errors_perfect_score(self) -> None:
        calc = ScoreCalculator()
        result = calc.calculate(error_count=0)
        assert result.score == 100
        assert result.grade == GradeLevel.EXCELLENT
        assert result.is_passing
        assert result.total_deducted == 0

    def test_five_errors(self) -> None:
        calc = ScoreCalculator()
        result = calc.calculate(error_count=5)
        assert result.score == 75
        assert result.grade == GradeLevel.AVERAGE

    def test_many_errors_floor_at_zero(self) -> None:
        calc = ScoreCalculator()
        result = calc.calculate(error_count=50)
        assert result.score == 0
        assert result.grade == GradeLevel.FAIL
        assert not result.is_passing

    def test_custom_deduction(self) -> None:
        calc = ScoreCalculator(deduction_per_error=10)
        result = calc.calculate(error_count=3)
        assert result.score == 70
        assert result.grade == GradeLevel.AVERAGE

    def test_grade_thresholds(self) -> None:
        calc = ScoreCalculator()
        # 90 -> excellent
        assert calc._score_to_grade(95) == GradeLevel.EXCELLENT
        assert calc._score_to_grade(90) == GradeLevel.EXCELLENT
        # 80-89 -> good
        assert calc._score_to_grade(85) == GradeLevel.GOOD
        assert calc._score_to_grade(80) == GradeLevel.GOOD
        # 70-79 -> average
        assert calc._score_to_grade(75) == GradeLevel.AVERAGE
        # 60-69 -> pass
        assert calc._score_to_grade(65) == GradeLevel.PASS
        assert calc._score_to_grade(60) == GradeLevel.PASS
        # <60 -> fail
        assert calc._score_to_grade(59) == GradeLevel.FAIL
        assert calc._score_to_grade(0) == GradeLevel.FAIL

    def test_error_breakdown(self) -> None:
        calc = ScoreCalculator()
        breakdown = {"typo": 2, "grammar": 1}
        result = calc.calculate(error_count=3, error_breakdown=breakdown)
        assert result.error_breakdown == {"typo": 2, "grammar": 1}

    def test_weighted_calculation(self) -> None:
        calc = ScoreCalculator()
        breakdown = {"typo": 2, "grammar": 1}
        weights = {"typo": 3, "grammar": 10}
        result = calc.calculate_weighted(breakdown, weights)
        # 2*3 + 1*10 = 16 deducted
        assert result.score == 84
        assert result.total_deducted == 16

    def test_summary_format(self) -> None:
        calc = ScoreCalculator()
        result = calc.calculate(error_count=2)
        summary = result.summary()
        assert "85" in summary or "90" in summary  # depends on deduction
        assert "优秀" in summary or "良好" in summary

    def test_percentage_property(self) -> None:
        calc = ScoreCalculator()
        result = calc.calculate(error_count=0)
        assert result.percentage == 100.0


# ======================================================================
# AnnotationRenderer tests
# ======================================================================


class TestAnnotationRenderer:
    """Tests for AnnotationRenderer."""

    def test_from_errors_builds_annotations(self) -> None:
        renderer = AnnotationRenderer()
        errors = [
            ErrorInfo(
                error_type=ErrorType.TYPO,
                position=0,
                length=2,
                text="按装",
                message="test",
                suggestion="安装",
            )
        ]
        char_bboxes = [(10, 20, 30, 40), (40, 20, 30, 40)]
        annotations = renderer.from_errors(errors, char_bboxes)
        assert len(annotations) == 1
        ann = annotations[0]
        assert ann.annotation_type == AnnotationType.STRIKETHROUGH
        assert ann.x == 10
        assert ann.y == 20
        assert ann.width == 60  # 40+30 - 10
        assert ann.height == 40

    def test_from_errors_grammar_uses_wave(self) -> None:
        renderer = AnnotationRenderer()
        errors = [
            ErrorInfo(
                error_type=ErrorType.GRAMMAR,
                position=0,
                length=1,
                text="x",
                message="test",
            )
        ]
        char_bboxes = [(0, 0, 10, 10)]
        annotations = renderer.from_errors(errors, char_bboxes)
        assert annotations[0].annotation_type == AnnotationType.WAVE_UNDERLINE

    def test_from_errors_punctuation_uses_circle(self) -> None:
        renderer = AnnotationRenderer()
        errors = [
            ErrorInfo(
                error_type=ErrorType.PUNCTUATION,
                position=0,
                length=1,
                text="!",
                message="test",
            )
        ]
        char_bboxes = [(5, 5, 10, 10)]
        annotations = renderer.from_errors(errors, char_bboxes)
        assert annotations[0].annotation_type == AnnotationType.CIRCLE

    def test_from_errors_out_of_range_skipped(self) -> None:
        renderer = AnnotationRenderer()
        errors = [
            ErrorInfo(
                error_type=ErrorType.TYPO,
                position=100,
                length=2,
                text="xx",
                message="test",
            )
        ]
        char_bboxes = [(0, 0, 10, 10)] * 5
        annotations = renderer.from_errors(errors, char_bboxes)
        assert len(annotations) == 0

    def test_make_annotation(self) -> None:
        renderer = AnnotationRenderer()
        ann = renderer.make_annotation(
            AnnotationType.CIRCLE, (10, 20, 30, 40), "test"
        )
        assert ann.annotation_type == AnnotationType.CIRCLE
        assert ann.bbox == (10, 20, 30, 40)
        assert ann.message == "test"
        assert ann.color == ANNOTATION_RED

    def test_annotation_properties(self) -> None:
        ann = Annotation(
            annotation_type=AnnotationType.STRIKETHROUGH,
            bbox=(10, 20, 30, 40),
        )
        assert ann.x == 10
        assert ann.y == 20
        assert ann.width == 30
        assert ann.height == 40

    @pytest.mark.skipif(
        not _has_pil(), reason="Pillow not available"
    )
    def test_render_strikethrough(self) -> None:
        from PIL import Image

        renderer = AnnotationRenderer()
        img = Image.new("RGB", (200, 100), "white")
        ann = Annotation(
            annotation_type=AnnotationType.STRIKETHROUGH,
            bbox=(10, 20, 50, 30),
            color=ANNOTATION_RED,
            line_width=2,
        )
        result = renderer.render(img, [ann])
        assert result is img  # same object
        # Verify some red pixels were drawn
        assert _has_color_pixels(img, ANNOTATION_RED)

    @pytest.mark.skipif(
        not _has_pil(), reason="Pillow not available"
    )
    def test_render_circle(self) -> None:
        from PIL import Image

        renderer = AnnotationRenderer()
        img = Image.new("RGB", (200, 100), "white")
        ann = Annotation(
            annotation_type=AnnotationType.CIRCLE,
            bbox=(10, 20, 50, 30),
            color=ANNOTATION_RED,
            line_width=2,
        )
        renderer.render(img, [ann])
        assert _has_color_pixels(img, ANNOTATION_RED)

    @pytest.mark.skipif(
        not _has_pil(), reason="Pillow not available"
    )
    def test_render_wave_underline(self) -> None:
        from PIL import Image

        renderer = AnnotationRenderer()
        img = Image.new("RGB", (200, 100), "white")
        ann = Annotation(
            annotation_type=AnnotationType.WAVE_UNDERLINE,
            bbox=(10, 20, 50, 30),
            color=ANNOTATION_RED,
            line_width=2,
        )
        renderer.render(img, [ann])
        assert _has_color_pixels(img, ANNOTATION_RED)

    @pytest.mark.skipif(
        not _has_pil(), reason="Pillow not available"
    )
    def test_render_to_new(self) -> None:
        renderer = AnnotationRenderer()
        ann = Annotation(
            annotation_type=AnnotationType.STRIKETHROUGH,
            bbox=(10, 10, 50, 30),
            color=ANNOTATION_RED,
            line_width=2,
        )
        img = renderer.render_to_new((200, 100), [ann])
        assert img.size == (200, 100)

    def test_default_annotation_type_mapping(self) -> None:
        renderer = AnnotationRenderer()
        for etype, expected in [
            (ErrorType.TYPO, AnnotationType.STRIKETHROUGH),
            (ErrorType.GRAMMAR, AnnotationType.WAVE_UNDERLINE),
            (ErrorType.PUNCTUATION, AnnotationType.CIRCLE),
            (ErrorType.FORMAT, AnnotationType.MARGIN_NOTE),
        ]:
            errors = [
                ErrorInfo(
                    error_type=etype,
                    position=0,
                    length=1,
                    text="x",
                    message="test",
                )
            ]
            bboxes = [(0, 0, 10, 10)]
            annotations = renderer.from_errors(errors, bboxes)
            assert annotations[0].annotation_type == expected

    def test_custom_annotation_type_override(self) -> None:
        renderer = AnnotationRenderer()
        errors = [
            ErrorInfo(
                error_type=ErrorType.TYPO,
                position=0,
                length=1,
                text="x",
                message="test",
            )
        ]
        bboxes = [(0, 0, 10, 10)]
        annotations = renderer.from_errors(
            errors, bboxes, default_annotation_type=AnnotationType.CIRCLE
        )
        assert annotations[0].annotation_type == AnnotationType.CIRCLE


# ======================================================================
# FeedbackGenerator tests
# ======================================================================


class TestFeedbackGenerator:
    """Tests for FeedbackGenerator."""

    def _make_score(self, score: int = 85, errors: int = 3) -> ScoreResult:
        return ScoreResult(
            score=score,
            error_count=errors,
            error_breakdown={"typo": 2, "grammar": 1},
            grade=ScoreCalculator._score_to_grade(score),
            total_deducted=15,
        )

    def test_markdown_contains_title(self) -> None:
        gen = FeedbackGenerator()
        score = self._make_score()
        md = gen.generate_markdown([], score)
        assert "# 手写批改报告" in md

    def test_markdown_contains_score(self) -> None:
        gen = FeedbackGenerator()
        score = self._make_score(score=85, errors=3)
        md = gen.generate_markdown([], score)
        assert "85" in md
        assert "良好" in md

    def test_markdown_contains_student_name(self) -> None:
        gen = FeedbackGenerator()
        score = self._make_score()
        md = gen.generate_markdown([], score, student_name="张三")
        assert "张三" in md

    def test_markdown_contains_error_details(self) -> None:
        gen = FeedbackGenerator()
        errors = [
            ErrorInfo(
                error_type=ErrorType.TYPO,
                position=0,
                length=2,
                text="按装",
                message="应为安装",
                suggestion="安装",
            )
        ]
        score = self._make_score()
        md = gen.generate_markdown(errors, score)
        assert "按装" in md
        assert "安装" in md

    def test_markdown_contains_suggestions(self) -> None:
        gen = FeedbackGenerator()
        errors = [
            ErrorInfo(
                error_type=ErrorType.TYPO,
                position=0,
                length=2,
                text="按装",
                message="test",
            )
        ]
        score = self._make_score()
        md = gen.generate_markdown(errors, score)
        assert "建议" in md

    def test_markdown_extra_comments(self) -> None:
        gen = FeedbackGenerator()
        score = self._make_score()
        md = gen.generate_markdown([], score, extra_comments="继续努力！")
        assert "继续努力" in md
        assert "教师评语" in md

    def test_plain_text_format(self) -> None:
        gen = FeedbackGenerator()
        score = self._make_score()
        text = gen.generate_plain_text([], score)
        assert "=" in text  # underline decoration
        assert "85" in text
        assert "良好" in text

    def test_plain_text_with_errors(self) -> None:
        gen = FeedbackGenerator()
        errors = [
            ErrorInfo(
                error_type=ErrorType.TYPO,
                position=0,
                length=2,
                text="按装",
                message="应为安装",
                suggestion="安装",
            )
        ]
        score = self._make_score()
        text = gen.generate_plain_text(errors, score)
        assert "typo" in text or "按装" in text

    def test_markdown_error_breakdown_table(self) -> None:
        gen = FeedbackGenerator()
        score = self._make_score()
        md = gen.generate_markdown([], score)
        assert "错别字" in md
        assert "语法错误" in md

    def test_markdown_footer(self) -> None:
        gen = FeedbackGenerator()
        score = self._make_score()
        md = gen.generate_markdown([], score)
        assert "HandWrite" in md


# ======================================================================
# GradingEngine tests
# ======================================================================


class TestGradingEngine:
    """Tests for GradingEngine."""

    def test_grade_text_basic(self) -> None:
        engine = GradingEngine()
        result = engine.grade_text("他按装了软件")
        assert isinstance(result, GradingResult)
        assert result.total_errors >= 1
        assert result.score.score <= 95

    def test_grade_text_clean(self) -> None:
        engine = GradingEngine()
        result = engine.grade_text("今天天气很好。")
        assert result.total_errors == 0
        assert result.score.score == 100

    def test_grade_text_with_student(self) -> None:
        engine = GradingEngine(student_name="李四", assignment_name="作文一")
        result = engine.grade_text("他的按装教程")
        assert "李四" in result.markdown_report
        assert "作文一" in result.markdown_report

    def test_grade_text_extra_comments(self) -> None:
        engine = GradingEngine()
        result = engine.grade_text("按装", extra_comments="注意错别字！")
        assert "注意错别字" in result.markdown_report

    def test_grade_result_summary(self) -> None:
        engine = GradingEngine()
        result = engine.grade_text("按装")
        summary = result.summary()
        assert "得分" in summary

    def test_grade_result_grade_property(self) -> None:
        engine = GradingEngine()
        result = grade("今天天气很好。")
        assert result.grade == "优秀"

    def test_set_student_and_assignment(self) -> None:
        engine = GradingEngine()
        engine.set_student("王五")
        engine.set_assignment("周记")
        result = engine.grade_text("天气不错。")
        assert "王五" in result.markdown_report
        assert "周记" in result.markdown_report

    def test_annotations_built_from_errors(self) -> None:
        engine = GradingEngine()
        result = engine.grade_text("他按装了软件")
        assert len(result.annotations) >= 1
        # Annotations should have placeholder bboxes
        assert result.annotations[0].bbox == (0, 0, 0, 0)

    def test_custom_deduction_per_error(self) -> None:
        engine = GradingEngine(deduction_per_error=10)
        result = engine.grade_text("他按装了软件")
        # 1 error * 10 = 10 deducted
        if result.total_errors == 1:
            assert result.score.score == 90

    @pytest.mark.skipif(not _has_pil(), reason="Pillow not available")
    def test_annotate_image(self) -> None:
        from PIL import Image

        engine = GradingEngine()
        img = Image.new("RGB", (200, 100), "white")
        errors = [
            ErrorInfo(
                error_type=ErrorType.TYPO,
                position=0,
                length=1,
                text="x",
                message="test",
            )
        ]
        char_bboxes = [(10, 10, 50, 30)]
        result = engine.annotate_image(img, errors, char_bboxes)
        assert result is img


# ======================================================================
# Module-level convenience function tests
# ======================================================================


class TestConvenienceFunctions:
    """Tests for module-level grade() and annotate()."""

    def test_grade_function(self) -> None:
        result = grade("他按装了软件")
        assert isinstance(result, GradingResult)
        assert result.total_errors >= 1

    def test_grade_function_with_options(self) -> None:
        result = grade(
            "按装",
            student_name="张三",
            assignment_name="作文",
            extra_comments="加油",
        )
        assert "张三" in result.markdown_report
        assert "作文" in result.markdown_report
        assert "加油" in result.markdown_report

    @pytest.mark.skipif(not _has_pil(), reason="Pillow not available")
    def test_annotate_function(self) -> None:
        from PIL import Image

        img = Image.new("RGB", (200, 100), "white")
        errors = [
            ErrorInfo(
                error_type=ErrorType.TYPO,
                position=0,
                length=1,
                text="x",
                message="test",
            )
        ]
        char_bboxes = [(10, 10, 50, 30)]
        annotated, result = annotate(
            "测试", img, char_bboxes
        )
        assert isinstance(result, GradingResult)
        assert annotated is img


# ======================================================================
# Integration tests
# ======================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_typo(self) -> None:
        result = grade("他的按装过程很顺利。")
        assert result.score.score < 100
        assert any(e.error_type == ErrorType.TYPO for e in result.errors)
        assert "手写批改报告" in result.markdown_report
        assert "错别字" in result.markdown_report

    def test_full_pipeline_multiple_error_types(self) -> None:
        text = "按装软件后，大约有100左右人使用，他的脉博正常。"
        result = grade(text)
        error_types = {e.error_type for e in result.errors}
        assert ErrorType.TYPO in error_types
        # Should have grammar or other types too
        assert len(error_types) >= 1

    def test_full_pipeline_clean_text(self) -> None:
        result = grade("这是一段完美的中文文本，没有任何错误。")
        assert result.score.score == 100
        assert result.grade == "优秀"
        assert result.total_errors == 0

    def test_grade_level_feedback_varies(self) -> None:
        # Excellent
        r1 = grade("完美的文本。")
        assert r1.score.grade == GradeLevel.EXCELLENT

        # Fail -- simulate many errors
        engine = GradingEngine(deduction_per_error=100)
        r2 = engine.grade_text("按装")
        assert r2.score.grade == GradeLevel.FAIL

    def test_markdown_report_structure(self) -> None:
        result = grade("按装软件，大约100左右人。")
        md = result.markdown_report
        # Check structural elements
        assert "# 手写批改报告" in md
        assert "## 评分" in md
        assert "## 错误详情" in md
        assert "## 改进建议" in md
        assert "---" in md

