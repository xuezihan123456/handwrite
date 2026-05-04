"""Score calculator -- compute percentage scores and grade levels.

Scoring rule: ``score = max(0, 100 - error_count * 5)``
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GradeLevel(str, Enum):
    """Grade levels mapped from percentage scores."""

    EXCELLENT = "优秀"       # 90-100
    GOOD = "良好"            # 80-89
    AVERAGE = "中等"         # 70-79
    PASS = "及格"            # 60-69
    FAIL = "不及格"          # 0-59


# Score thresholds for each grade level
_GRADE_THRESHOLDS: list[tuple[int, GradeLevel]] = [
    (90, GradeLevel.EXCELLENT),
    (80, GradeLevel.GOOD),
    (70, GradeLevel.AVERAGE),
    (60, GradeLevel.PASS),
    (0, GradeLevel.FAIL),
]


@dataclass(frozen=True)
class ScoreResult:
    """Result of a score calculation."""

    score: int               # 0-100
    error_count: int         # number of errors found
    error_breakdown: dict[str, int]  # counts by error type
    grade: GradeLevel        # letter grade
    total_deducted: int      # total points deducted

    @property
    def is_passing(self) -> bool:
        return self.score >= 60

    @property
    def percentage(self) -> float:
        return float(self.score)

    def summary(self) -> str:
        """Return a short human-readable summary."""
        parts = [
            f"得分: {self.score}/100 ({self.grade.value})",
            f"错误数: {self.error_count}",
        ]
        if self.error_breakdown:
            items = ", ".join(
                f"{k}:{v}" for k, v in self.error_breakdown.items() if v > 0
            )
            if items:
                parts.append(f"错误分布: {items}")
        return " | ".join(parts)


@dataclass
class ScoreCalculator:
    """Calculate scores from error counts.

    Default formula: ``score = max(0, 100 - error_count * deduction_per_error)``

    Example::

        calc = ScoreCalculator()
        result = calc.calculate(error_count=3)
        print(result.score)   # 85
        print(result.grade)   # GradeLevel.GOOD
    """

    deduction_per_error: int = 5
    max_score: int = 100
    min_score: int = 0

    def calculate(
        self,
        error_count: int = 0,
        error_breakdown: Optional[dict[str, int]] = None,
    ) -> ScoreResult:
        """Compute the score for a given number of errors.

        Args:
            error_count: Total number of errors detected.
            error_breakdown: Optional dict mapping error type name to count.

        Returns:
            A ``ScoreResult`` with the computed score, grade, and breakdown.
        """
        if error_breakdown is None:
            error_breakdown = {}

        total_deducted = error_count * self.deduction_per_error
        score = max(self.min_score, self.max_score - total_deducted)
        grade = self._score_to_grade(score)

        return ScoreResult(
            score=score,
            error_count=error_count,
            error_breakdown=dict(error_breakdown),
            grade=grade,
            total_deducted=total_deducted,
        )

    def calculate_weighted(
        self,
        error_breakdown: dict[str, int],
        weights: Optional[dict[str, int]] = None,
    ) -> ScoreResult:
        """Calculate score with different deduction weights per error type.

        Args:
            error_breakdown: Dict mapping error type name to count.
            weights: Dict mapping error type name to deduction per error.
                     Defaults to ``deduction_per_error`` for unspecified types.

        Returns:
            A ``ScoreResult`` with the computed score, grade, and breakdown.
        """
        if weights is None:
            weights = {}

        total_errors = sum(error_breakdown.values())
        total_deducted = 0

        for error_type, count in error_breakdown.items():
            per_error = weights.get(error_type, self.deduction_per_error)
            total_deducted += count * per_error

        score = max(self.min_score, self.max_score - total_deducted)
        grade = self._score_to_grade(score)

        return ScoreResult(
            score=score,
            error_count=total_errors,
            error_breakdown=dict(error_breakdown),
            grade=grade,
            total_deducted=total_deducted,
        )

    @staticmethod
    def _score_to_grade(score: int) -> GradeLevel:
        """Map a numeric score to a grade level."""
        for threshold, level in _GRADE_THRESHOLDS:
            if score >= threshold:
                return level
        return GradeLevel.FAIL
