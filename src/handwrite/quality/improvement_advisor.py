"""Improvement advisor for handwriting quality.

Generates actionable, Chinese-language improvement suggestions based on
evaluation scores and diagnostics.
"""

from __future__ import annotations

from handwrite.quality.quality_report import (
    CharacterQualityReport,
    DimensionScore,
    PageQualityReport,
)


def generate_page_advice(report: PageQualityReport) -> tuple[str, ...]:
    """Generate prioritized improvement suggestions for a page.

    Suggestions are sorted by impact (lowest scoring dimensions first)
    and deduplicated.

    Args:
        report: The evaluated page quality report.

    Returns:
        Tuple of Chinese improvement suggestion strings.
    """
    tips: list[str] = []
    seen: set[str] = set()

    # Collect tips from page-level dimensions
    scored_dims: list[tuple[float, DimensionScore]] = []
    for dim in report.dimensions:
        scored_dims.append((dim.score, dim))

    # Sort by score ascending (worst first)
    scored_dims.sort(key=lambda x: x[0])

    for score, dim in scored_dims:
        dim_tips = _dimension_tips(dim)
        for tip in dim_tips:
            if tip not in seen:
                seen.add(tip)
                tips.append(tip)

    # Add tips from character reports
    for char_report in report.char_reports:
        for tip in char_report.improvement_tips:
            if tip not in seen:
                seen.add(tip)
                tips.append(tip)

    # Add overall advice based on overall score
    overall_tips = _overall_tips(report.overall_score)
    for tip in overall_tips:
        if tip not in seen:
            seen.add(tip)
            tips.append(tip)

    # Limit to top 8 suggestions
    return tuple(tips[:8])


def generate_char_advice(
    char: str,
    authenticity: DimensionScore,
    naturalness: DimensionScore,
    overall_score: float,
) -> tuple[str, ...]:
    """Generate improvement suggestions for a single character.

    Args:
        char: The character being evaluated.
        authenticity: Authenticity dimension score.
        naturalness: Naturalness dimension score.
        overall_score: Overall quality score.

    Returns:
        Tuple of Chinese improvement suggestion strings.
    """
    tips: list[str] = []
    seen: set[str] = set()

    # Collect suggestions from both dimensions
    for dim in (authenticity, naturalness):
        for suggestion in dim.suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                tips.append(suggestion)

    # Add character-specific advice
    char_tips = _char_specific_tips(char, overall_score)
    for tip in char_tips:
        if tip not in seen:
            seen.add(tip)
            tips.append(tip)

    return tuple(tips[:5])


def _dimension_tips(dim: DimensionScore) -> list[str]:
    """Generate tips for a specific dimension based on its score."""
    tips: list[str] = []

    # Use existing suggestions from the dimension
    tips.extend(dim.suggestions)

    # Add supplementary advice based on score range
    if dim.score < 50:
        tips.extend(_critical_tips(dim.name))
    elif dim.score < 70:
        tips.extend(_moderate_tips(dim.name))

    return tips


def _critical_tips(dimension_name: str) -> list[str]:
    """Generate critical-level tips for very low scores."""
    critical_map: dict[str, list[str]] = {
        "真实性": [
            "真实性评分极低，建议重新生成并调整渲染参数",
            "当前输出与手写差异较大，建议更换手写风格",
        ],
        "自然度": [
            "自然度评分极低，建议检查生成模型的随机性设置",
            "当前输出过于机械，建议增加扰动参数",
        ],
        "笔画连续性": [
            "笔画严重不连续，建议提高渲染分辨率",
        ],
        "粗细变化": [
            "笔画粗细变化异常，建议调整压力模拟参数",
        ],
        "间距一致性": [
            "字间距严重不均，建议调整字距参数",
        ],
        "打印感检测": [
            "打印感过强，建议大幅增加手写抖动效果",
        ],
        "行对齐": [
            "行对齐严重偏差，建议重新调整行距设置",
        ],
        "大小变化": [
            "字大小严重不均，建议调整字体缩放范围",
        ],
        "倾斜一致性": [
            "字体倾斜角度混乱，建议统一书写角度",
        ],
        "墨迹浓淡": [
            "墨迹浓淡异常，建议检查墨迹模拟参数",
        ],
    }
    return critical_map.get(dimension_name, [])


def _moderate_tips(dimension_name: str) -> list[str]:
    """Generate moderate-level tips for medium scores."""
    moderate_map: dict[str, list[str]] = {
        "真实性": [
            "真实性有提升空间，可微调笔画细节参数",
        ],
        "自然度": [
            "自然度有提升空间，可适当增加随机扰动",
        ],
        "笔画连续性": [
            "笔画连续性尚可，可进一步优化边缘平滑度",
        ],
        "粗细变化": [
            "笔画粗细变化偏大或偏小，建议微调压力曲线",
        ],
        "间距一致性": [
            "字间距可进一步优化，建议微调字距参数",
        ],
        "打印感检测": [
            "仍有轻微打印感，建议适当增加手写特征",
        ],
        "行对齐": [
            "行对齐可进一步优化，建议微调基线波动参数",
        ],
        "大小变化": [
            "字大小变化可更自然，建议微调缩放范围",
        ],
        "倾斜一致性": [
            "倾斜角度可更一致，建议微调旋转参数",
        ],
        "墨迹浓淡": [
            "墨迹浓淡可更自然，建议微调压力模拟",
        ],
    }
    return moderate_map.get(dimension_name, [])


def _overall_tips(overall_score: float) -> list[str]:
    """Generate overall advice based on the total score."""
    if overall_score >= 90:
        return ["整体质量优秀，继续保持当前的书写风格"]
    if overall_score >= 75:
        return ["整体质量良好，可在细节上进一步打磨"]
    if overall_score >= 60:
        return [
            "整体质量合格，建议重点关注评分较低的维度",
            "可尝试调整生成参数后重新生成",
        ]
    return [
        "整体质量需要改进，建议重新调整生成参数",
        "建议逐步优化各维度后再生成完整页面",
        "可先用单字测试找到最佳参数组合",
    ]


def _char_specific_tips(char: str, overall_score: float) -> list[str]:
    """Generate character-specific tips."""
    tips: list[str] = []

    if overall_score < 60:
        tips.append(f"字符「{char}」质量偏低，建议单独优化该字的生成参数")

    return tips


__all__ = ["generate_page_advice", "generate_char_advice"]
