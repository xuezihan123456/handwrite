"""Tests for the collaborative handwriting document module."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from handwrite.collaboration.collab_engine import (
    generate_collaborative_document,
    _split_paragraphs,
)
from handwrite.collaboration.collaborative_composer import (
    CollaborativeComposer,
    _alpha_blend_images,
)
from handwrite.collaboration.contributor import Contributor
from handwrite.collaboration.segment_assigner import (
    assign_segments,
    assign_segments_round_robin,
)
from handwrite.collaboration.style_blender import StyleBlender


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_char_image(size: int = 256, fill: int = 0) -> Image.Image:
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    inset = 60
    draw.rectangle((inset, inset, size - inset, size - inset), fill=fill)
    return image


def _two_contributors() -> list[Contributor]:
    return [
        Contributor(name="Alice", style_id=0),
        Contributor(name="Bob", style_id=2),
    ]


def _three_contributors() -> list[Contributor]:
    return [
        Contributor(name="Alice", style_id=0),
        Contributor(name="Bob", style_id=2),
        Contributor(name="Carol", style_id=4),
    ]


# ---------------------------------------------------------------------------
# Contributor tests
# ---------------------------------------------------------------------------

class TestContributor:
    def test_create_with_defaults(self) -> None:
        c = Contributor(name="Alice", style_id=0)
        assert c.name == "Alice"
        assert c.style_id == 0
        assert c.params is None

    def test_create_with_params(self) -> None:
        c = Contributor(name="Bob", style_id=2, params={"slant": 5})
        assert c.params == {"slant": 5}

    def test_frozen(self) -> None:
        c = Contributor(name="Alice", style_id=0)
        with pytest.raises(AttributeError):
            c.name = "Bob"  # type: ignore[misc]

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Contributor(name="", style_id=0)

    def test_rejects_whitespace_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Contributor(name="   ", style_id=0)

    def test_rejects_non_int_style_id(self) -> None:
        with pytest.raises(TypeError, match="style_id must be int"):
            Contributor(name="Alice", style_id=1.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Segment assigner tests
# ---------------------------------------------------------------------------

class TestSegmentAssigner:
    def test_round_robin_two_contributors(self) -> None:
        contributors = _two_contributors()
        assignments = assign_segments_round_robin(4, contributors)
        assert assignments == [(0, 0), (1, 1), (2, 0), (3, 1)]

    def test_round_robin_three_contributors(self) -> None:
        contributors = _three_contributors()
        assignments = assign_segments_round_robin(5, contributors)
        assert assignments == [(0, 0), (1, 1), (2, 2), (3, 0), (4, 1)]

    def test_manual_mapping(self) -> None:
        contributors = _three_contributors()
        mapping = [2, 0, 1]
        assignments = assign_segments(3, contributors, manual_mapping=mapping)
        assert assignments == [(0, 2), (1, 0), (2, 1)]

    def test_default_is_round_robin(self) -> None:
        contributors = _two_contributors()
        assignments = assign_segments(3, contributors)
        assert assignments == [(0, 0), (1, 1), (2, 0)]

    def test_rejects_wrong_mapping_length(self) -> None:
        contributors = _two_contributors()
        with pytest.raises(ValueError, match="manual_mapping length"):
            assign_segments(3, contributors, manual_mapping=[0, 1])

    def test_rejects_out_of_range_mapping(self) -> None:
        contributors = _two_contributors()
        with pytest.raises(ValueError, match="out of range"):
            assign_segments(2, contributors, manual_mapping=[0, 5])

    def test_rejects_too_few_contributors(self) -> None:
        with pytest.raises(ValueError, match="2--6"):
            assign_segments(3, [Contributor(name="Solo", style_id=0)])

    def test_rejects_too_many_contributors(self) -> None:
        contributors = [Contributor(name=f"P{i}", style_id=i % 5) for i in range(7)]
        with pytest.raises(ValueError, match="2--6"):
            assign_segments(3, contributors)

    def test_rejects_non_int_mapping(self) -> None:
        contributors = _two_contributors()
        with pytest.raises(TypeError, match="must be int"):
            assign_segments(2, contributors, manual_mapping=[0, 1.5])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# StyleBlender tests
# ---------------------------------------------------------------------------

class TestStyleBlender:
    def test_no_blend_when_same_style(self) -> None:
        blender = StyleBlender(blend_lines=3)
        paragraphs = ["hello world", "foo bar"]
        style_ids = [0, 0]
        weights = blender.compute_char_weights(paragraphs, style_ids)
        assert all(w == 0.0 for w in weights[0])
        assert all(w == 0.0 for w in weights[1])

    def test_blend_at_boundary(self) -> None:
        blender = StyleBlender(blend_lines=3)
        paragraphs = ["abcdefghij" * 5, "klmnopqrst" * 5]
        style_ids = [0, 2]
        weights = blender.compute_char_weights(paragraphs, style_ids)

        # First paragraph: end should have non-zero weights.
        assert weights[0][-1] > 0.0
        # First paragraph: start should be 0.0.
        assert weights[0][0] == 0.0

        # Second paragraph: start should have non-zero weights.
        assert weights[1][0] > 0.0
        # Second paragraph: end should be 0.0 (no next paragraph).
        assert weights[1][-1] == 0.0

    def test_blend_weights_are_monotonic(self) -> None:
        blender = StyleBlender(blend_lines=3)
        text = "a" * 100
        paragraphs = [text, text]
        style_ids = [0, 4]
        weights = blender.compute_char_weights(paragraphs, style_ids)

        # End of first paragraph: weights should be non-decreasing.
        end_weights = [w for w in weights[0] if w > 0]
        for i in range(len(end_weights) - 1):
            assert end_weights[i] <= end_weights[i + 1] + 1e-9

    def test_three_paragraphs_middle_blends_both_directions(self) -> None:
        blender = StyleBlender(blend_lines=2)
        paragraphs = ["a" * 50, "b" * 50, "c" * 50]
        style_ids = [0, 2, 4]
        weights = blender.compute_char_weights(paragraphs, style_ids)

        # Middle paragraph should have non-zero weights at both ends.
        mid = weights[1]
        assert mid[0] > 0.0  # blends with previous
        assert mid[-1] > 0.0  # blends with next
        # Middle section should be 0.0.
        center = len(mid) // 2
        assert mid[center] == 0.0

    def test_rejects_mismatched_lengths(self) -> None:
        blender = StyleBlender(blend_lines=3)
        with pytest.raises(ValueError, match="same length"):
            blender.compute_char_weights(["a"], [0, 1])

    def test_rejects_blend_lines_less_than_one(self) -> None:
        with pytest.raises(ValueError, match="blend_lines must be >= 1"):
            StyleBlender(blend_lines=0)

    def test_identify_blend_regions(self) -> None:
        blender = StyleBlender(blend_lines=3)
        style_ids = [0, 2, 2, 4]
        line_counts = [10, 8, 6, 12]
        regions = blender.identify_blend_regions(style_ids, line_counts)

        assert len(regions) == 2  # boundaries at 0-1 and 2-3
        assert regions[0].paragraph_index == 0
        assert regions[0].source_style_id == 0
        assert regions[0].target_style_id == 2
        assert regions[1].paragraph_index == 2
        assert regions[1].source_style_id == 2
        assert regions[1].target_style_id == 4

    def test_compute_line_weights(self) -> None:
        blender = StyleBlender(blend_lines=2)
        paragraphs = ["a" * 40, "b" * 40]
        style_ids = [0, 2]
        line_weights = blender.compute_line_weights(paragraphs, style_ids, chars_per_line=20)

        assert len(line_weights) == 2
        assert len(line_weights[0]) == 2  # 40 chars / 20 per line
        assert len(line_weights[1]) == 2
        # End of first paragraph line weights should be non-zero.
        assert line_weights[0][-1] > 0.0
        # Start of second paragraph line weights should be non-zero.
        assert line_weights[1][0] > 0.0


# ---------------------------------------------------------------------------
# CollaborativeComposer tests
# ---------------------------------------------------------------------------

class TestCollaborativeComposer:
    def test_compose_paragraph_returns_image(self) -> None:
        contributors = _two_contributors()
        composer = CollaborativeComposer(contributors)
        chars = [_make_char_image() for _ in range(4)]
        page = composer.compose_paragraph("ABCD", contributors[0], chars, page_size=(320, 240), font_size=48)
        assert isinstance(page, Image.Image)
        assert page.size == (320, 240)

    def test_get_assignments(self) -> None:
        contributors = _two_contributors()
        composer = CollaborativeComposer(contributors)
        assignments = composer.get_assignments(3)
        assert assignments == [(0, 0), (1, 1), (2, 0)]

    def test_blend_char_images(self) -> None:
        contributors = _two_contributors()
        composer = CollaborativeComposer(contributors)
        imgs_a = [Image.new("L", (64, 64), color=0) for _ in range(4)]
        imgs_b = [Image.new("L", (64, 64), color=255) for _ in range(4)]
        weights = [0.0, 0.3, 0.7, 1.0]
        result = composer.blend_char_images(imgs_a, imgs_b, weights)

        assert len(result) == 4
        # First image should be close to black.
        assert np.asarray(result[0]).mean() < 5
        # Last image should be close to white.
        assert np.asarray(result[3]).mean() > 250

    def test_rejects_single_contributor(self) -> None:
        with pytest.raises(ValueError, match="2--6"):
            CollaborativeComposer([Contributor(name="Solo", style_id=0)])

    def test_rejects_seven_contributors(self) -> None:
        cs = [Contributor(name=f"P{i}", style_id=i % 5) for i in range(7)]
        with pytest.raises(ValueError, match="2--6"):
            CollaborativeComposer(cs)


# ---------------------------------------------------------------------------
# Alpha blend helper tests
# ---------------------------------------------------------------------------

class TestAlphaBlend:
    def test_weight_zero_returns_first(self) -> None:
        a = Image.new("L", (32, 32), color=100)
        b = Image.new("L", (32, 32), color=200)
        result = _alpha_blend_images(a, b, 0.0)
        assert np.asarray(result).mean() == pytest.approx(100, abs=1)

    def test_weight_one_returns_second(self) -> None:
        a = Image.new("L", (32, 32), color=100)
        b = Image.new("L", (32, 32), color=200)
        result = _alpha_blend_images(a, b, 1.0)
        assert np.asarray(result).mean() == pytest.approx(200, abs=1)

    def test_weight_half_is_average(self) -> None:
        a = Image.new("L", (32, 32), color=100)
        b = Image.new("L", (32, 32), color=200)
        result = _alpha_blend_images(a, b, 0.5)
        assert np.asarray(result).mean() == pytest.approx(150, abs=1)


# ---------------------------------------------------------------------------
# Paragraph splitting tests
# ---------------------------------------------------------------------------

class TestSplitParagraphs:
    def test_simple_split(self) -> None:
        text = "Hello world\n\nFoo bar"
        paragraphs = _split_paragraphs(text)
        assert paragraphs == ["Hello world", "Foo bar"]

    def test_multiple_blank_lines(self) -> None:
        text = "AAA\n\n\n\nBBB"
        paragraphs = _split_paragraphs(text)
        assert paragraphs == ["AAA", "BBB"]

    def test_single_paragraph(self) -> None:
        text = "Just one paragraph"
        paragraphs = _split_paragraphs(text)
        assert paragraphs == ["Just one paragraph"]

    def test_empty_text(self) -> None:
        assert _split_paragraphs("") == []

    def test_whitespace_only(self) -> None:
        assert _split_paragraphs("   \n\n  ") == []

    def test_multiline_paragraphs(self) -> None:
        text = "Line 1\nLine 2\n\nLine 3\nLine 4"
        paragraphs = _split_paragraphs(text)
        assert len(paragraphs) == 2
        assert paragraphs[0] == "Line 1\nLine 2"
        assert paragraphs[1] == "Line 3\nLine 4"


# ---------------------------------------------------------------------------
# Full integration test
# ---------------------------------------------------------------------------

class TestGenerateCollaborativeDocument:
    def test_basic_generation(self) -> None:
        contributors = _two_contributors()
        text = "Hello world this is a test paragraph.\n\nSecond paragraph here."
        result = generate_collaborative_document(
            text,
            contributors,
            page_size=(640, 480),
            font_size=40,
            margins=(20, 20, 20, 20),
        )

        assert result["page_count"] >= 1
        assert len(result["pages"]) >= 1
        assert isinstance(result["pages"][0], Image.Image)
        assert result["blend_lines"] == 3
        assert len(result["contributors"]) == 2
        assert result["contributors"][0]["name"] == "Alice"

    def test_empty_text_returns_empty(self) -> None:
        contributors = _two_contributors()
        result = generate_collaborative_document("", contributors)
        assert result["page_count"] == 0
        assert result["pages"] == []

    def test_manual_mapping(self) -> None:
        contributors = _two_contributors()
        text = "Para one.\n\nPara two.\n\nPara three."
        result = generate_collaborative_document(
            text,
            contributors,
            manual_mapping=[1, 0, 1],
            page_size=(640, 480),
            font_size=40,
            margins=(20, 20, 20, 20),
        )
        assignments = result["assignments"]
        assert assignments[0][1] == 1
        assert assignments[1][1] == 0
        assert assignments[2][1] == 1

    def test_output_format_compatibility(self) -> None:
        """Verify the result dict has keys compatible with single-person workflow."""
        contributors = _two_contributors()
        text = "Test paragraph."
        result = generate_collaborative_document(
            text,
            contributors,
            page_size=(640, 480),
            font_size=40,
            margins=(20, 20, 20, 20),
        )
        # Must have 'pages' and 'page_count' like single-person generate_pages.
        assert "pages" in result
        assert "page_count" in result
        assert isinstance(result["pages"], list)

    def test_rejects_single_contributor(self) -> None:
        with pytest.raises(ValueError, match="2--6"):
            generate_collaborative_document("text", [Contributor(name="Solo", style_id=0)])

    def test_blend_lines_configurable(self) -> None:
        contributors = _two_contributors()
        text = "A" * 200 + "\n\n" + "B" * 200
        result = generate_collaborative_document(
            text,
            contributors,
            blend_lines=5,
            page_size=(640, 480),
            font_size=40,
            margins=(20, 20, 20, 20),
        )
        assert result["blend_lines"] == 5
