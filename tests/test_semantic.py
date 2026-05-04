"""Tests for the semantic-aware typesetting module."""

from __future__ import annotations

from PIL import Image, ImageDraw

from handwrite.semantic import (
    Decoration,
    INK_BLACK,
    INK_BLUE,
    INK_RED,
    LayoutPlanner,
    SegmentLayout,
    SemanticRole,
    TextAnalyzer,
    TextSegment,
    compose_semantic_page,
    extract_clean_text,
    render_annotations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_char_image(size: int = 256) -> Image.Image:
    """Create a simple synthetic character image (black square on white)."""
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 176, 176), fill=0)
    return image


def _make_chars(text: str, size: int = 256) -> list[Image.Image]:
    """Create one character image per non-whitespace character in *text*."""
    return [_make_char_image(size) for ch in text if not ch.isspace()]


# ===========================================================================
# TextAnalyzer tests
# ===========================================================================


class TestTextAnalyzer:
    """Tests for TextAnalyzer.analyze()."""

    def test_empty_text(self) -> None:
        analyzer = TextAnalyzer()
        assert analyzer.analyze("") == []

    def test_plain_body_text(self) -> None:
        analyzer = TextAnalyzer()
        segments = analyzer.analyze("hello world")
        assert len(segments) == 1
        assert segments[0].role == SemanticRole.BODY
        assert segments[0].text == "hello world"

    def test_heading_detected(self) -> None:
        analyzer = TextAnalyzer()
        segments = analyzer.analyze("# Chapter One")
        roles = [s.role for s in segments]
        assert SemanticRole.TITLE in roles

    def test_heading_level(self) -> None:
        analyzer = TextAnalyzer()
        segments = analyzer.analyze("## Subtitle")
        title_seg = next(s for s in segments if s.role == SemanticRole.TITLE)
        assert title_seg.level == 2

    def test_emphasis_bold_stars(self) -> None:
        analyzer = TextAnalyzer()
        segments = analyzer.analyze("This is **important** text")
        emphasis = [s for s in segments if s.role == SemanticRole.EMPHASIS]
        assert len(emphasis) == 1
        assert "important" in emphasis[0].text

    def test_emphasis_bold_underscores(self) -> None:
        analyzer = TextAnalyzer()
        segments = analyzer.analyze("This is __important__ text")
        emphasis = [s for s in segments if s.role == SemanticRole.EMPHASIS]
        assert len(emphasis) == 1

    def test_inline_formula(self) -> None:
        analyzer = TextAnalyzer()
        segments = analyzer.analyze("The equation $E=mc^2$ is famous")
        formulas = [s for s in segments if s.role == SemanticRole.FORMULA]
        assert len(formulas) == 1
        assert "E=mc^2" in formulas[0].text

    def test_block_formula_dollars(self) -> None:
        analyzer = TextAnalyzer()
        segments = analyzer.analyze("$$x^2 + y^2 = z^2$$")
        formulas = [s for s in segments if s.role == SemanticRole.FORMULA]
        assert len(formulas) == 1
        assert formulas[0].level == 1

    def test_block_formula_brackets(self) -> None:
        analyzer = TextAnalyzer()
        segments = analyzer.analyze(r"\[a^2 + b^2\]")
        formulas = [s for s in segments if s.role == SemanticRole.FORMULA]
        assert len(formulas) == 1

    def test_ordered_list(self) -> None:
        analyzer = TextAnalyzer()
        segments = analyzer.analyze("1. First item\n2. Second item")
        items = [s for s in segments if s.role == SemanticRole.LIST_ITEM]
        assert len(items) == 2

    def test_unordered_list(self) -> None:
        analyzer = TextAnalyzer()
        segments = analyzer.analyze("- Alpha\n- Beta\n- Gamma")
        items = [s for s in segments if s.role == SemanticRole.LIST_ITEM]
        assert len(items) == 3

    def test_mixed_content(self) -> None:
        text = (
            "# Title\n"
            "Some body text with **emphasis** and $x=1$.\n"
            "- item one\n"
            "- item two"
        )
        analyzer = TextAnalyzer()
        segments = analyzer.analyze(text)
        roles = {s.role for s in segments}
        assert SemanticRole.TITLE in roles
        assert SemanticRole.BODY in roles
        assert SemanticRole.EMPHASIS in roles
        assert SemanticRole.FORMULA in roles
        assert SemanticRole.LIST_ITEM in roles


# ===========================================================================
# extract_clean_text tests
# ===========================================================================


class TestExtractCleanText:
    """Tests for extract_clean_text()."""

    def test_heading_stripped(self) -> None:
        seg = TextSegment(text="## Hello", role=SemanticRole.TITLE, level=2)
        assert extract_clean_text(seg) == "Hello"

    def test_emphasis_stripped(self) -> None:
        seg = TextSegment(text="**bold**", role=SemanticRole.EMPHASIS)
        assert extract_clean_text(seg) == "bold"

    def test_formula_dollar_stripped(self) -> None:
        seg = TextSegment(text="$E=mc^2$", role=SemanticRole.FORMULA)
        assert extract_clean_text(seg) == "E=mc^2"

    def test_formula_bracket_stripped(self) -> None:
        seg = TextSegment(text=r"\[x+y\]", role=SemanticRole.FORMULA)
        assert extract_clean_text(seg) == "x+y"

    def test_list_item_stripped(self) -> None:
        seg = TextSegment(text="1. First", role=SemanticRole.LIST_ITEM)
        assert extract_clean_text(seg) == "First"

    def test_body_unchanged(self) -> None:
        seg = TextSegment(text="plain text", role=SemanticRole.BODY)
        assert extract_clean_text(seg) == "plain text"


# ===========================================================================
# LayoutPlanner tests
# ===========================================================================


class TestLayoutPlanner:
    """Tests for LayoutPlanner.plan()."""

    def test_body_default_style(self) -> None:
        planner = LayoutPlanner()
        seg = TextSegment(text="body", role=SemanticRole.BODY)
        plan = planner.plan([seg])
        assert len(plan) == 1
        assert plan[0].font_size_multiplier == 1.0
        assert plan[0].ink_color == INK_BLACK
        assert plan[0].decoration == Decoration.NONE

    def test_title_scaled_and_blue(self) -> None:
        planner = LayoutPlanner()
        seg = TextSegment(text="# Title", role=SemanticRole.TITLE, level=1)
        plan = planner.plan([seg])
        assert len(plan) == 1
        assert plan[0].font_size_multiplier == 1.5
        assert plan[0].ink_color == INK_BLUE
        assert plan[0].decoration == Decoration.UNDERLINE

    def test_emphasis_red_wave(self) -> None:
        planner = LayoutPlanner()
        seg = TextSegment(text="**key**", role=SemanticRole.EMPHASIS)
        plan = planner.plan([seg])
        assert len(plan) == 1
        assert plan[0].ink_color == INK_RED
        assert plan[0].decoration == Decoration.WAVE_UNDERLINE

    def test_formula_grid_paper(self) -> None:
        planner = LayoutPlanner()
        seg = TextSegment(text="$x$", role=SemanticRole.FORMULA)
        plan = planner.plan([seg])
        assert len(plan) == 1
        assert plan[0].use_grid_paper is True
        assert plan[0].decoration == Decoration.HIGHLIGHT

    def test_list_circle_decoration(self) -> None:
        planner = LayoutPlanner()
        seg = TextSegment(text="- item", role=SemanticRole.LIST_ITEM)
        plan = planner.plan([seg])
        assert len(plan) == 1
        assert plan[0].decoration == Decoration.CIRCLE

    def test_custom_title_scale(self) -> None:
        planner = LayoutPlanner(title_scale=2.0)
        seg = TextSegment(text="# Big", role=SemanticRole.TITLE, level=1)
        plan = planner.plan([seg])
        assert plan[0].font_size_multiplier == 2.0

    def test_empty_text_filtered(self) -> None:
        planner = LayoutPlanner()
        seg = TextSegment(text="   ", role=SemanticRole.BODY)
        plan = planner.plan([seg])
        assert len(plan) == 0

    def test_multiple_segments(self) -> None:
        planner = LayoutPlanner()
        segments = [
            TextSegment(text="# T", role=SemanticRole.TITLE, level=1),
            TextSegment(text="body", role=SemanticRole.BODY),
            TextSegment(text="**e**", role=SemanticRole.EMPHASIS),
        ]
        plan = planner.plan(segments)
        assert len(plan) == 3
        assert plan[0].role == SemanticRole.TITLE
        assert plan[1].role == SemanticRole.BODY
        assert plan[2].role == SemanticRole.EMPHASIS


# ===========================================================================
# annotation_renderer tests
# ===========================================================================


class TestAnnotationRenderer:
    """Tests for render_annotations()."""

    def _sample_boxes(self) -> list[tuple[int, int, int, int]]:
        return [(100, 100, 40, 40), (140, 100, 40, 40), (180, 100, 40, 40)]

    def test_no_op_on_none_decoration(self) -> None:
        page = Image.new("L", (400, 400), color=255)
        original = page.copy()
        result = render_annotations(page, self._sample_boxes(), Decoration.NONE)
        assert list(result.getdata()) == list(original.getdata())

    def test_no_op_on_empty_boxes(self) -> None:
        page = Image.new("L", (400, 400), color=255)
        original = page.copy()
        result = render_annotations(page, [], Decoration.UNDERLINE)
        assert list(result.getdata()) == list(original.getdata())

    def test_underline_modifies_page(self) -> None:
        page = Image.new("L", (400, 400), color=255)
        result = render_annotations(page, self._sample_boxes(), Decoration.UNDERLINE)
        # The underline should introduce dark pixels below the boxes
        extrema = result.convert("L").getextrema()
        assert extrema[0] < 255

    def test_wave_underline_modifies_page(self) -> None:
        page = Image.new("L", (400, 400), color=255)
        result = render_annotations(page, self._sample_boxes(), Decoration.WAVE_UNDERLINE)
        extrema = result.convert("L").getextrema()
        assert extrema[0] < 255

    def test_circle_modifies_page(self) -> None:
        page = Image.new("L", (400, 400), color=255)
        result = render_annotations(page, self._sample_boxes(), Decoration.CIRCLE)
        extrema = result.convert("L").getextrema()
        assert extrema[0] < 255

    def test_highlight_returns_image(self) -> None:
        page = Image.new("L", (400, 400), color=255)
        result = render_annotations(page, self._sample_boxes(), Decoration.HIGHLIGHT)
        assert isinstance(result, Image.Image)
        assert result.size == (400, 400)

    def test_underline_respects_color(self) -> None:
        page = Image.new("RGB", (400, 400), color=(255, 255, 255))
        boxes = [(100, 100, 40, 40)]
        result = render_annotations(page, boxes, Decoration.UNDERLINE, color=(255, 0, 0))
        # Check that red pixels appear in the underline region
        found_red = False
        for y in range(142, 148):
            for x in range(100, 141):
                r, g, b = result.getpixel((x, y))
                if r > 200 and g < 50 and b < 50:
                    found_red = True
                    break
            if found_red:
                break
        assert found_red


# ===========================================================================
# compose_semantic_page tests
# ===========================================================================


class TestComposeSemanticPage:
    """Tests for compose_semantic_page()."""

    def test_returns_image(self) -> None:
        chars = _make_chars("hello")
        page = compose_semantic_page(chars, "hello", page_size=(600, 400), font_size=40)
        assert isinstance(page, Image.Image)
        assert page.size == (600, 400)

    def test_title_has_ink(self) -> None:
        text = "# Title"
        chars = _make_chars(text)
        page = compose_semantic_page(
            chars,
            text,
            page_size=(600, 200),
            font_size=40,
            margins=(20, 20, 20, 20),
        )
        # Page should have dark pixels from the title characters
        extrema = page.convert("L").getextrema()
        assert extrema[0] < 128

    def test_mixed_content_renders(self) -> None:
        text = "# T\nbody text\n**bold**"
        chars = _make_chars(text)
        page = compose_semantic_page(chars, text, page_size=(800, 600), font_size=36)
        assert isinstance(page, Image.Image)
        extrema = page.convert("L").getextrema()
        assert extrema[0] < 128

    def test_empty_text_produces_blank_page(self) -> None:
        page = compose_semantic_page([], "", page_size=(400, 300), font_size=40)
        extrema = page.convert("L").getextrema()
        assert extrema == (255, 255)

    def test_custom_title_scale(self) -> None:
        text = "# T"
        chars = _make_chars(text)
        page = compose_semantic_page(
            chars, text, page_size=(600, 200), font_size=40, title_scale=2.0
        )
        assert isinstance(page, Image.Image)

    def test_respects_margins(self) -> None:
        text = "ab"
        chars = _make_chars(text)
        margin = 50
        page = compose_semantic_page(
            chars,
            text,
            page_size=(300, 200),
            font_size=40,
            margins=(margin, margin, margin, margin),
        )
        # The top-left margin area should be white
        pixel = page.convert("L").getpixel((10, 10))
        assert pixel > 200

    def test_formula_highlights(self) -> None:
        text = "$x=1$"
        chars = _make_chars(text)
        page = compose_semantic_page(
            chars,
            text,
            page_size=(600, 200),
            font_size=40,
            margins=(20, 20, 20, 20),
        )
        # Should produce a non-blank page
        extrema = page.convert("L").getextrema()
        assert extrema[0] < 255
