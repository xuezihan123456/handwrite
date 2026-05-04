"""Quality report data structures for handwriting evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionScore:
    """A single evaluation dimension with score and diagnostic details."""

    name: str
    score: float
    weight: float = 1.0
    details: str = ""
    suggestions: tuple[str, ...] = ()

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass(frozen=True)
class CharacterQualityReport:
    """Quality evaluation for a single character image."""

    char: str
    authenticity: DimensionScore
    naturalness: DimensionScore
    overall_score: float
    improvement_tips: tuple[str, ...] = ()

    @property
    def grade(self) -> str:
        if self.overall_score >= 90:
            return "优秀"
        if self.overall_score >= 75:
            return "良好"
        if self.overall_score >= 60:
            return "合格"
        return "待改进"


@dataclass(frozen=True)
class PageQualityReport:
    """Quality evaluation for a full page image."""

    overall_score: float
    authenticity_score: float
    naturalness_score: float
    dimensions: tuple[DimensionScore, ...] = ()
    improvement_tips: tuple[str, ...] = ()
    char_reports: tuple[CharacterQualityReport, ...] = ()

    @property
    def grade(self) -> str:
        if self.overall_score >= 90:
            return "优秀"
        if self.overall_score >= 75:
            return "良好"
        if self.overall_score >= 60:
            return "合格"
        return "待改进"

    def summary(self) -> str:
        lines = [
            f"手写质量评估报告",
            f"{'=' * 40}",
            f"综合评分: {self.overall_score:.1f}/100 ({self.grade})",
            f"真实性评分: {self.authenticity_score:.1f}/100",
            f"自然度评分: {self.naturalness_score:.1f}/100",
        ]
        if self.improvement_tips:
            lines.append("")
            lines.append("改进建议:")
            for i, tip in enumerate(self.improvement_tips, 1):
                lines.append(f"  {i}. {tip}")
        return "\n".join(lines)


__all__ = [
    "DimensionScore",
    "CharacterQualityReport",
    "PageQualityReport",
]
