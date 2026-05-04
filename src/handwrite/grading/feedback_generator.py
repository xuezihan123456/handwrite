"""Feedback generator -- produce Markdown and plain-text grading reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

from handwrite.grading.error_detector import ErrorInfo, ErrorType
from handwrite.grading.score_calculator import GradeLevel, ScoreResult


# Grade-specific feedback templates
_GRADE_FEEDBACK: dict[GradeLevel, str] = {
    GradeLevel.EXCELLENT: "书写非常出色，继续保持！",
    GradeLevel.GOOD: "书写良好，注意细节，可以更上一层楼。",
    GradeLevel.AVERAGE: "书写中等，建议多加练习，注意常见错误。",
    GradeLevel.PASS: "书写基本及格，但存在较多问题，需要加强练习。",
    GradeLevel.FAIL: "书写不及格，请认真对照批注逐项改正。",
}

# Error-type-specific advice
_ERROR_ADVICE: dict[ErrorType, str] = {
    ErrorType.TYPO: "建议使用词典检查易混淆字词，多做组词练习。",
    ErrorType.GRAMMAR: "建议复习语法知识，注意关联词搭配和句子成分。",
    ErrorType.PUNCTUATION: "建议注意中英文标点的区分，避免重复标点。",
    ErrorType.FORMAT: "建议统一格式规范，注意缩进和换行符的使用。",
}


@dataclass
class FeedbackGenerator:
    """Generate grading feedback in Markdown or plain text.

    Example::

        gen = FeedbackGenerator()
        md = gen.generate_markdown(errors, score_result)
        print(md)
    """

    title: str = "手写批改报告"
    include_timestamp: bool = True
    include_error_details: bool = True
    include_suggestions: bool = True

    def generate_markdown(
        self,
        errors: Sequence[ErrorInfo],
        score: ScoreResult,
        *,
        student_name: Optional[str] = None,
        assignment_name: Optional[str] = None,
        extra_comments: Optional[str] = None,
    ) -> str:
        """Generate a full Markdown grading report.

        Returns:
            A Markdown-formatted string suitable for rendering or saving.
        """
        lines: list[str] = []

        # Title
        lines.append(f"# {self.title}")
        lines.append("")

        # Metadata
        if self.include_timestamp:
            lines.append(f"**批改时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if student_name:
            lines.append(f"**学生姓名**: {student_name}")
        if assignment_name:
            lines.append(f"**作业名称**: {assignment_name}")
        if any([student_name, assignment_name, self.include_timestamp]):
            lines.append("")

        # Score section
        lines.append("## 评分")
        lines.append("")
        lines.append(f"| 项目 | 结果 |")
        lines.append(f"|------|------|")
        lines.append(f"| 得分 | **{score.score}** / 100 |")
        lines.append(f"| 等级 | **{score.grade.value}** |")
        lines.append(f"| 错误总数 | {score.error_count} |")
        lines.append(f"| 扣分 | -{score.total_deducted} |")
        lines.append("")

        # Error breakdown table
        if score.error_breakdown:
            lines.append("### 错误分类统计")
            lines.append("")
            lines.append("| 错误类型 | 数量 |")
            lines.append("|----------|------|")
            type_names = {
                "typo": "错别字",
                "grammar": "语法错误",
                "punctuation": "标点错误",
                "format": "格式问题",
            }
            for etype, count in score.error_breakdown.items():
                display = type_names.get(etype, etype)
                lines.append(f"| {display} | {count} |")
            lines.append("")

        # Error details
        if self.include_error_details and errors:
            lines.append("## 错误详情")
            lines.append("")

            # Group errors by type
            grouped: dict[ErrorType, list[ErrorInfo]] = {}
            for err in errors:
                grouped.setdefault(err.error_type, []).append(err)

            type_titles = {
                ErrorType.TYPO: "错别字",
                ErrorType.GRAMMAR: "语法错误",
                ErrorType.PUNCTUATION: "标点错误",
                ErrorType.FORMAT: "格式问题",
            }

            for etype in ErrorType:
                group = grouped.get(etype, [])
                if not group:
                    continue
                lines.append(f"### {type_titles.get(etype, etype.value)}")
                lines.append("")
                for i, err in enumerate(group, 1):
                    marker = f"~~{err.text}~~" if err.text else ""
                    suggestion = f" -> **{err.suggestion}**" if err.suggestion else ""
                    lines.append(f"{i}. 位置 {err.position}: `{marker}`{suggestion}")
                    lines.append(f"   - {err.message}")
                lines.append("")

        # Suggestions
        if self.include_suggestions and errors:
            lines.append("## 改进建议")
            lines.append("")
            seen_types: set[ErrorType] = set()
            for err in errors:
                if err.error_type not in seen_types:
                    seen_types.add(err.error_type)
                    advice = _ERROR_ADVICE.get(err.error_type, "")
                    if advice:
                        lines.append(f"- {advice}")

            # Grade-level feedback
            grade_msg = _GRADE_FEEDBACK.get(score.grade, "")
            if grade_msg:
                lines.append(f"- {grade_msg}")
            lines.append("")

        # Extra comments
        if extra_comments:
            lines.append("## 教师评语")
            lines.append("")
            lines.append(extra_comments)
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("*本报告由 HandWrite 批改系统自动生成*")

        return "\n".join(lines)

    def generate_plain_text(
        self,
        errors: Sequence[ErrorInfo],
        score: ScoreResult,
        *,
        student_name: Optional[str] = None,
        assignment_name: Optional[str] = None,
    ) -> str:
        """Generate a plain-text grading report (no Markdown syntax)."""
        lines: list[str] = []

        lines.append(self.title)
        lines.append("=" * len(self.title.encode("gbk", errors="replace")) * 2)
        lines.append("")

        if student_name:
            lines.append(f"学生姓名: {student_name}")
        if assignment_name:
            lines.append(f"作业名称: {assignment_name}")
        if self.include_timestamp:
            lines.append(f"批改时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        lines.append(f"得分: {score.score}/100  等级: {score.grade.value}")
        lines.append(f"错误总数: {score.error_count}  扣分: -{score.total_deducted}")
        lines.append("")

        if score.error_breakdown:
            lines.append("错误分布:")
            type_names = {
                "typo": "错别字",
                "grammar": "语法错误",
                "punctuation": "标点错误",
                "format": "格式问题",
            }
            for etype, count in score.error_breakdown.items():
                display = type_names.get(etype, etype)
                lines.append(f"  {display}: {count}")
            lines.append("")

        if self.include_error_details and errors:
            lines.append("错误详情:")
            lines.append("-" * 40)
            for i, err in enumerate(errors, 1):
                suggestion = f" -> {err.suggestion}" if err.suggestion else ""
                lines.append(f"  {i}. [{err.error_type.value}] {err.text}{suggestion}")
                lines.append(f"     {err.message}")
            lines.append("")

        grade_msg = _GRADE_FEEDBACK.get(score.grade, "")
        if grade_msg:
            lines.append(f"总评: {grade_msg}")

        return "\n".join(lines)
