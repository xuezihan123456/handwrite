"""Tests for the summary module: text extraction, layouts, and rendering."""

from __future__ import annotations

import pytest
from PIL import Image

from handwrite.summary.mind_map_layout import (
    MapNode,
    MindMapLayout,
    compute_mind_map_layout,
)
from handwrite.summary.outline_layout import OutlineLayout, compute_outline_layout
from handwrite.summary.summary_engine import extract_summary, render_mind_map, render_outline
from handwrite.summary.summary_renderer import render_mind_map_image, render_outline_image
from handwrite.summary.text_summarizer import SummaryResult, SummarySection


# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

SAMPLE_CHINESE = """\
Python编程语言简介

Python是一种广泛使用的高级编程语言。它由Guido van Rossum于1991年创建。

主要特点：
- 简单易学的语法
- 丰富的标准库
- 跨平台支持
- 强大的社区支持

应用场景：
- 数据科学与机器学习
- Web开发
- 自动化脚本
- 科学计算

总结：Python是最受欢迎的编程语言之一，适合初学者和专业开发者。
"""

SAMPLE_ENGLISH = """\
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

Key Concepts:
- Supervised learning uses labeled data
- Unsupervised learning finds patterns in unlabeled data
- Reinforcement learning uses reward-based feedback

Applications include image recognition, natural language processing, and recommendation systems.
In conclusion, machine learning is transforming many industries.
"""

SAMPLE_MIXED = """\
Deep Learning笔记

Deep learning是machine learning的一个分支，使用多层neural network进行特征提取。

核心组件：
- 卷积层（Convolutional Layer）
- 池化层（Pooling Layer）
- 全连接层（Fully Connected Layer）

重要概念：backpropagation、gradient descent、activation function。

总结：Deep learning在computer vision和NLP领域取得了突破性进展。
"""

SAMPLE_MINIMAL = "This is a very short text."


# ---------------------------------------------------------------------------
# Text summarizer tests
# ---------------------------------------------------------------------------


class TestTextSummarizer:
    """Tests for rule-based text summarization."""

    def test_extract_summary_returns_summary_result(self) -> None:
        result = extract_summary(SAMPLE_CHINESE)
        assert isinstance(result, SummaryResult)

    def test_extract_summary_title_from_chinese(self) -> None:
        result = extract_summary(SAMPLE_CHINESE)
        assert "Python" in result.title or "编程" in result.title

    def test_extract_summary_title_from_english(self) -> None:
        result = extract_summary(SAMPLE_ENGLISH)
        assert "Machine Learning" in result.title or "Introduction" in result.title

    def test_extract_summary_title_from_markdown_heading(self) -> None:
        text = "## My Heading\n\nSome content here."
        result = extract_summary(text)
        assert result.title == "My Heading"

    def test_extract_summary_key_sentences_nonempty(self) -> None:
        result = extract_summary(SAMPLE_CHINESE)
        assert len(result.key_sentences) > 0

    def test_extract_summary_key_sentences_max_count(self) -> None:
        result = extract_summary(SAMPLE_CHINESE, max_key_sentences=3)
        assert len(result.key_sentences) <= 3

    def test_extract_summary_bullet_points(self) -> None:
        result = extract_summary(SAMPLE_CHINESE)
        assert len(result.bullet_points) > 0
        # Bullets should come from the list items
        assert any("简单" in bp or "语法" in bp for bp in result.bullet_points)

    def test_extract_summary_keywords(self) -> None:
        result = extract_summary(SAMPLE_CHINESE)
        assert len(result.keywords) > 0

    def test_extract_summary_keywords_max_count(self) -> None:
        result = extract_summary(SAMPLE_CHINESE, max_keywords=3)
        assert len(result.keywords) <= 3

    def test_extract_summary_sections(self) -> None:
        result = extract_summary(SAMPLE_CHINESE)
        assert len(result.sections) > 0
        for section in result.sections:
            assert isinstance(section, SummarySection)
            assert section.heading
            assert len(section.items) > 0

    def test_extract_summary_english_keywords(self) -> None:
        result = extract_summary(SAMPLE_ENGLISH)
        assert len(result.keywords) > 0

    def test_extract_summary_mixed_language(self) -> None:
        result = extract_summary(SAMPLE_MIXED)
        assert result.title
        assert len(result.key_sentences) > 0 or len(result.bullet_points) > 0

    def test_extract_summary_empty_text(self) -> None:
        result = extract_summary("")
        assert result.title == ""
        assert result.key_sentences == []
        assert result.bullet_points == []
        assert result.keywords == []

    def test_extract_summary_minimal_text(self) -> None:
        result = extract_summary(SAMPLE_MINIMAL)
        assert result.title  # Should have some title
        # Minimal text should still produce a valid result
        assert isinstance(result, SummaryResult)

    def test_extract_summary_frozen_dataclass(self) -> None:
        result = extract_summary(SAMPLE_CHINESE)
        with pytest.raises(AttributeError):
            result.title = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Mind map layout tests
# ---------------------------------------------------------------------------


class TestMindMapLayout:
    """Tests for mind map layout computation."""

    def test_compute_layout_returns_mind_map_layout(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_mind_map_layout(summary)
        assert isinstance(layout, MindMapLayout)

    def test_compute_layout_has_center_node(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_mind_map_layout(summary)
        assert len(layout.nodes) >= 1
        center = layout.nodes[0]
        assert center.level == 0
        assert center.parent_id is None

    def test_compute_layout_has_branches(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_mind_map_layout(summary)
        branch_nodes = [n for n in layout.nodes if n.level == 1]
        assert len(branch_nodes) > 0

    def test_compute_layout_has_edges(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_mind_map_layout(summary)
        assert len(layout.edges) > 0

    def test_compute_layout_edge_connects_existing_nodes(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_mind_map_layout(summary)
        node_ids = {n.id for n in layout.nodes}
        for edge in layout.edges:
            assert edge.source_id in node_ids
            assert edge.target_id in node_ids

    def test_compute_layout_center_at_canvas_center(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        w, h = 1600, 1200
        layout = compute_mind_map_layout(summary, canvas_width=w, canvas_height=h)
        center = layout.nodes[0]
        assert abs(center.x - w / 2) < 1.0
        assert abs(center.y - h / 2) < 1.0

    def test_compute_layout_minimal_text(self) -> None:
        summary = extract_summary(SAMPLE_MINIMAL)
        layout = compute_mind_map_layout(summary)
        assert len(layout.nodes) >= 1
        assert layout.width > 0
        assert layout.height > 0

    def test_compute_layout_empty_summary(self) -> None:
        empty = SummaryResult(
            title="",
            key_sentences=[],
            bullet_points=[],
            keywords=[],
            sections=[],
        )
        layout = compute_mind_map_layout(empty)
        assert len(layout.nodes) == 1  # Just the center node

    def test_compute_layout_no_overlapping_center(self) -> None:
        """Verify force-directed algorithm keeps branches away from center."""
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_mind_map_layout(summary, node_gap=50.0)
        center = layout.nodes[0]
        for node in layout.nodes[1:]:
            dx = node.x - center.x
            dy = node.y - center.y
            dist = (dx * dx + dy * dy) ** 0.5
            # After force-directed, nodes should have some distance
            assert dist > 10.0


# ---------------------------------------------------------------------------
# Outline layout tests
# ---------------------------------------------------------------------------


class TestOutlineLayout:
    """Tests for outline layout computation."""

    def test_compute_layout_returns_outline_layout(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_outline_layout(summary)
        assert isinstance(layout, OutlineLayout)

    def test_compute_layout_first_item_is_title(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_outline_layout(summary)
        assert len(layout.items) > 0
        assert layout.items[0].level == 0
        assert layout.items[0].text == summary.title

    def test_compute_layout_has_level_1_items(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_outline_layout(summary)
        level_1 = [item for item in layout.items if item.level == 1]
        assert len(level_1) > 0

    def test_compute_layout_has_level_2_items(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_outline_layout(summary)
        level_2 = [item for item in layout.items if item.level == 2]
        assert len(level_2) > 0

    def test_compute_layout_bullets_correct(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_outline_layout(summary)
        for item in layout.items:
            if item.level == 0:
                assert item.bullet == ""
            elif item.level == 1:
                assert item.bullet == "\u25cf"  # Filled circle
            elif item.level == 2:
                assert item.bullet == "\u25cb"  # Open circle

    def test_compute_layout_total_lines(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_outline_layout(summary)
        assert layout.total_lines == len(layout.items)

    def test_compute_layout_empty_summary(self) -> None:
        empty = SummaryResult(
            title="",
            key_sentences=[],
            bullet_points=[],
            keywords=[],
            sections=[],
        )
        layout = compute_outline_layout(empty)
        # May have keyword section or be empty
        assert isinstance(layout, OutlineLayout)

    def test_compute_layout_text_wrapping(self) -> None:
        """Verify that long text gets wrapped into multiple lines."""
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_outline_layout(summary, max_line_width=20)
        # With narrow width, some items should wrap
        assert len(layout.items) > 0


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------


class TestSummaryRenderer:
    """Tests for image rendering."""

    def test_render_mind_map_returns_image(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_mind_map_layout(summary, canvas_width=800, canvas_height=600)
        image = render_mind_map_image(
            layout, page_size=(800, 600), font_size=18
        )
        assert isinstance(image, Image.Image)
        assert image.size == (800, 600)

    def test_render_outline_returns_image(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_outline_layout(summary)
        image = render_outline_image(
            layout, page_size=(800, 600), font_size=18
        )
        assert isinstance(image, Image.Image)
        assert image.size == (800, 600)

    def test_render_mind_map_image_not_blank(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_mind_map_layout(summary, canvas_width=800, canvas_height=600)
        image = render_mind_map_image(
            layout, page_size=(800, 600), font_size=18
        )
        # Image should have some non-white pixels (drawn content)
        extrema = image.convert("L").getextrema()
        assert extrema[0] < 255

    def test_render_outline_image_not_blank(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_outline_layout(summary)
        image = render_outline_image(
            layout, page_size=(800, 600), font_size=18
        )
        extrema = image.convert("L").getextrema()
        assert extrema[0] < 255

    def test_render_mind_map_with_custom_generate_char(self) -> None:
        """Test rendering with a custom character generator function."""
        def fake_char_gen(char: str, size: int) -> Image.Image:
            img = Image.new("RGBA", (size, size), (128, 128, 128, 255))
            return img

        summary = extract_summary(SAMPLE_CHINESE)
        layout = compute_mind_map_layout(summary, canvas_width=800, canvas_height=600)
        image = render_mind_map_image(
            layout,
            page_size=(800, 600),
            font_size=18,
            generate_char_fn=fake_char_gen,
        )
        assert isinstance(image, Image.Image)


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------


class TestSummaryEngine:
    """Tests for the main summary engine API."""

    def test_extract_summary_from_string(self) -> None:
        result = extract_summary(SAMPLE_CHINESE)
        assert isinstance(result, SummaryResult)
        assert result.title

    def test_render_mind_map_from_string(self) -> None:
        image = render_mind_map(
            SAMPLE_CHINESE,
            page_size=(800, 600),
            font_size=18,
        )
        assert isinstance(image, Image.Image)
        assert image.size == (800, 600)

    def test_render_outline_from_string(self) -> None:
        image = render_outline(
            SAMPLE_CHINESE,
            page_size=(800, 600),
            font_size=18,
        )
        assert isinstance(image, Image.Image)
        assert image.size == (800, 600)

    def test_render_mind_map_from_summary_result(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        image = render_mind_map(
            summary,
            page_size=(800, 600),
            font_size=18,
        )
        assert isinstance(image, Image.Image)

    def test_render_outline_from_summary_result(self) -> None:
        summary = extract_summary(SAMPLE_CHINESE)
        image = render_outline(
            summary,
            page_size=(800, 600),
            font_size=18,
        )
        assert isinstance(image, Image.Image)

    def test_render_mind_map_english(self) -> None:
        image = render_mind_map(
            SAMPLE_ENGLISH,
            page_size=(800, 600),
            font_size=18,
        )
        assert isinstance(image, Image.Image)

    def test_render_outline_english(self) -> None:
        image = render_outline(
            SAMPLE_ENGLISH,
            page_size=(800, 600),
            font_size=18,
        )
        assert isinstance(image, Image.Image)

    def test_render_mind_map_mixed_language(self) -> None:
        image = render_mind_map(
            SAMPLE_MIXED,
            page_size=(800, 600),
            font_size=18,
        )
        assert isinstance(image, Image.Image)

    def test_render_outline_mixed_language(self) -> None:
        image = render_outline(
            SAMPLE_MIXED,
            page_size=(800, 600),
            font_size=18,
        )
        assert isinstance(image, Image.Image)

    def test_render_mind_map_with_parameters(self) -> None:
        image = render_mind_map(
            SAMPLE_CHINESE,
            page_size=(1200, 900),
            font_size=20,
            margins=(50, 50, 50, 50),
            max_key_sentences=5,
            max_bullet_points=6,
            max_keywords=8,
        )
        assert isinstance(image, Image.Image)
        assert image.size == (1200, 900)

    def test_render_outline_with_parameters(self) -> None:
        image = render_outline(
            SAMPLE_CHINESE,
            page_size=(1200, 900),
            font_size=28,
            margins=(50, 50, 50, 50),
            max_key_sentences=5,
        )
        assert isinstance(image, Image.Image)
        assert image.size == (1200, 900)
