"""Grading engine -- main entry point for the handwrite grading module.

Provides ``grade()`` and ``annotate()`` as top-level convenience functions,
and ``GradingEngine`` as a configurable class for batch usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from handwrite.grading.annotation_renderer import (
    ANNOTATION_RED,
    Annotation,
    AnnotationRenderer,
    AnnotationType,
)
from handwrite.grading.error_detector import ErrorDetector, ErrorInfo, ErrorType
from handwrite.grading.feedback_generator import FeedbackGenerator
from handwrite.grading.score_calculator import ScoreCalculator, ScoreResult


@dataclass
class GradingResult:
    """Complete result of a grading pass.

    Contains the score, all detected errors, annotations, and a Markdown report.
    """

    score: ScoreResult
    errors: list[ErrorInfo]
    annotations: list[Annotation]
    markdown_report: str
    plain_text_report: str

    @property
    def total_errors(self) -> int:
        return self.score.error_count

    @property
    def grade(self) -> str:
        return self.score.grade.value

    def summary(self) -> str:
        return self.score.summary()


@dataclass
class GradingEngine:
    """Configurable grading engine that orchestrates detection, scoring, and reporting.

    Example::

        engine = GradingEngine()
        result = engine.grade_text("他按装了软件，大约100左右人。")
        print(result.score)         # ScoreResult
        print(result.markdown_report)  # Markdown string
    """

    error_detector: ErrorDetector = field(default_factory=ErrorDetector)
    score_calculator: ScoreCalculator = field(default_factory=ScoreCalculator)
    feedback_generator: FeedbackGenerator = field(default_factory=FeedbackGenerator)
    annotation_renderer: AnnotationRenderer = field(default_factory=AnnotationRenderer)

    # Grading options
    deduction_per_error: int = 5
    student_name: Optional[str] = None
    assignment_name: Optional[str] = None

    def __post_init__(self) -> None:
        self.score_calculator.deduction_per_error = self.deduction_per_error

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def grade_text(
        self,
        text: str,
        *,
        extra_comments: Optional[str] = None,
    ) -> GradingResult:
        """Grade a piece of text: detect errors, calculate score, generate report.

        Args:
            text: The text to grade.
            extra_comments: Optional extra teacher comments for the report.

        Returns:
            A ``GradingResult`` with all grading information.
        """
        # Step 1: Detect errors
        errors = self.error_detector.detect(text)

        # Step 2: Build error breakdown
        breakdown: dict[str, int] = {}
        for err in errors:
            key = err.error_type.value
            breakdown[key] = breakdown.get(key, 0) + 1

        # Step 3: Calculate score
        score = self.score_calculator.calculate(
            error_count=len(errors),
            error_breakdown=breakdown,
        )

        # Step 4: Build annotations (without image bbox data, just the info)
        annotations = self._build_annotations_from_errors(errors)

        # Step 5: Generate reports
        markdown = self.feedback_generator.generate_markdown(
            errors,
            score,
            student_name=self.student_name,
            assignment_name=self.assignment_name,
            extra_comments=extra_comments,
        )
        plain = self.feedback_generator.generate_plain_text(
            errors,
            score,
            student_name=self.student_name,
            assignment_name=self.assignment_name,
        )

        return GradingResult(
            score=score,
            errors=errors,
            annotations=annotations,
            markdown_report=markdown,
            plain_text_report=plain,
        )

    def annotate_image(
        self,
        image: "Image.Image",
        errors: Sequence[ErrorInfo],
        char_bboxes: Sequence[tuple[int, int, int, int]],
    ) -> "Image.Image":
        """Render red annotations onto an image.

        Args:
            image: The PIL Image to annotate.
            errors: Detected errors to annotate.
            char_bboxes: Per-character bounding boxes (x, y, w, h).

        Returns:
            The annotated image (modified in-place and also returned).
        """
        annotations = self.annotation_renderer.from_errors(errors, char_bboxes)
        return self.annotation_renderer.render(image, annotations)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def set_student(self, name: str) -> None:
        """Set the student name for reports."""
        self.student_name = name

    def set_assignment(self, name: str) -> None:
        """Set the assignment name for reports."""
        self.assignment_name = name

    def _build_annotations_from_errors(
        self, errors: Sequence[ErrorInfo]
    ) -> list[Annotation]:
        """Build Annotation objects from errors (without image coordinates)."""
        annotations: list[Annotation] = []
        for err in errors:
            ann_type = {
                ErrorType.TYPO: AnnotationType.STRIKETHROUGH,
                ErrorType.GRAMMAR: AnnotationType.WAVE_UNDERLINE,
                ErrorType.PUNCTUATION: AnnotationType.CIRCLE,
                ErrorType.FORMAT: AnnotationType.MARGIN_NOTE,
            }.get(err.error_type, AnnotationType.STRIKETHROUGH)

            message = err.message
            if err.suggestion:
                message = f"{message} -> {err.suggestion}"

            # Placeholder bbox -- actual coordinates depend on OCR/layout
            annotations.append(
                Annotation(
                    annotation_type=ann_type,
                    bbox=(0, 0, 0, 0),
                    message=message,
                    color=ANNOTATION_RED,
                )
            )
        return annotations


# ======================================================================
# Module-level convenience functions
# ======================================================================


def grade(
    text: str,
    *,
    student_name: Optional[str] = None,
    assignment_name: Optional[str] = None,
    extra_comments: Optional[str] = None,
    deduction_per_error: int = 5,
) -> GradingResult:
    """Grade text and return a full ``GradingResult``.

    This is the simplest way to use the grading module.

    Example::

        from handwrite.grading import grade

        result = grade("他按装了软件")
        print(result.markdown_report)
    """
    engine = GradingEngine(
        student_name=student_name,
        assignment_name=assignment_name,
        deduction_per_error=deduction_per_error,
    )
    return engine.grade_text(text, extra_comments=extra_comments)


def annotate(
    text: str,
    image: "Image.Image",
    char_bboxes: Sequence[tuple[int, int, int, int]],
    *,
    deduction_per_error: int = 5,
) -> tuple["Image.Image", GradingResult]:
    """Grade text and annotate an image in one call.

    Args:
        text: The text to grade.
        image: The PIL Image to annotate.
        char_bboxes: Per-character bounding boxes.
        deduction_per_error: Points deducted per error.

    Returns:
        A tuple of (annotated_image, grading_result).
    """
    engine = GradingEngine(deduction_per_error=deduction_per_error)
    result = engine.grade_text(text)
    annotated = engine.annotate_image(image, result.errors, char_bboxes)
    return annotated, result
