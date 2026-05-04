"""Tests for the temporal handwriting style module."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from handwrite.temporal.age_profiles import (
    AgeGroup,
    AgeProfile,
    get_age_profile,
    interpolate_profiles,
    list_age_groups,
)
from handwrite.temporal.historical_style import (
    HistoricalInstrument,
    apply_historical_style,
)
from handwrite.temporal.skill_simulator import SkillSimulator
from handwrite.temporal.temporal_renderer import TemporalRenderer


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_test_char(size: int = 256) -> Image.Image:
    """Create a test character image with a simple stroke pattern."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    mid = size // 2
    # Horizontal stroke
    draw.line((40, mid, size - 40, mid), fill=(0, 0, 0, 255), width=6)
    # Vertical stroke
    draw.line((mid, 40, mid, size - 40), fill=(0, 0, 0, 255), width=6)
    return image


def _make_simple_char(size: int = 256) -> Image.Image:
    """Create a simpler test character with one horizontal stroke."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    y = size // 2
    draw.line((30, y, size - 30, y), fill=(0, 0, 0, 255), width=5)
    return image


def _count_ink_pixels(image: Image.Image, threshold: int = 240) -> int:
    """Count ink pixels in a grayscale image."""
    arr = np.array(image.convert("L"))
    return int(np.sum(arr < threshold))


def _has_ink(image: Image.Image, threshold: int = 240) -> bool:
    """Check if image contains any ink."""
    return _count_ink_pixels(image, threshold) > 0


# ---------------------------------------------------------------------------
# Age profile tests
# ---------------------------------------------------------------------------


class TestAgeGroup:
    def test_all_enum_values_exist(self) -> None:
        assert AgeGroup.LOWER_ELEMENTARY.value == "lower_elementary"
        assert AgeGroup.UPPER_ELEMENTARY.value == "upper_elementary"
        assert AgeGroup.MIDDLE_SCHOOL.value == "middle_school"
        assert AgeGroup.HIGH_SCHOOL.value == "high_school"
        assert AgeGroup.ADULT.value == "adult"

    def test_list_age_groups_returns_all(self) -> None:
        groups = list_age_groups()
        assert len(groups) == 5
        assert groups[0] == AgeGroup.LOWER_ELEMENTARY
        assert groups[-1] == AgeGroup.ADULT


class TestAgeProfile:
    def test_get_age_profile_returns_valid_profile(self) -> None:
        profile = get_age_profile(AgeGroup.LOWER_ELEMENTARY)
        assert isinstance(profile, AgeProfile)
        assert profile.jitter_x > 0
        assert profile.jitter_y > 0
        assert 0.0 <= profile.stability <= 1.0

    def test_elementary_has_more_jitter_than_adult(self) -> None:
        elementary = get_age_profile(AgeGroup.LOWER_ELEMENTARY)
        adult = get_age_profile(AgeGroup.ADULT)
        assert elementary.jitter_x > adult.jitter_x
        assert elementary.jitter_y > adult.jitter_y

    def test_elementary_has_lower_stability(self) -> None:
        elementary = get_age_profile(AgeGroup.LOWER_ELEMENTARY)
        adult = get_age_profile(AgeGroup.ADULT)
        assert elementary.stability < adult.stability

    def test_high_school_has_more_connection(self) -> None:
        hs = get_age_profile(AgeGroup.HIGH_SCHOOL)
        elementary = get_age_profile(AgeGroup.LOWER_ELEMENTARY)
        assert hs.stroke_connection > elementary.stroke_connection

    def test_developmental_progression(self) -> None:
        """Verify parameters improve monotonically across age groups."""
        groups = list_age_groups()
        stabilities = [get_age_profile(g).stability for g in groups]
        # Stability should generally increase
        for i in range(len(stabilities) - 1):
            assert stabilities[i] <= stabilities[i + 1] + 0.01

    def test_descriptions_are_chinese(self) -> None:
        for group in list_age_groups():
            profile = get_age_profile(group)
            assert len(profile.description) > 0
            # Should contain Chinese characters
            assert any(ord(c) > 0x4E00 for c in profile.description)


class TestInterpolateProfiles:
    def test_interpolation_at_zero_returns_first(self) -> None:
        a = get_age_profile(AgeGroup.LOWER_ELEMENTARY)
        b = get_age_profile(AgeGroup.ADULT)
        result = interpolate_profiles(a, b, 0.0)
        assert abs(result.jitter_x - a.jitter_x) < 1e-6
        assert abs(result.stability - a.stability) < 1e-6

    def test_interpolation_at_one_returns_second(self) -> None:
        a = get_age_profile(AgeGroup.LOWER_ELEMENTARY)
        b = get_age_profile(AgeGroup.ADULT)
        result = interpolate_profiles(a, b, 1.0)
        assert abs(result.jitter_x - b.jitter_x) < 1e-6
        assert abs(result.stability - b.stability) < 1e-6

    def test_interpolation_at_half_is_midpoint(self) -> None:
        a = get_age_profile(AgeGroup.LOWER_ELEMENTARY)
        b = get_age_profile(AgeGroup.ADULT)
        result = interpolate_profiles(a, b, 0.5)
        expected_jitter = (a.jitter_x + b.jitter_x) / 2
        assert abs(result.jitter_x - expected_jitter) < 1e-6

    def test_interpolation_clamps_t(self) -> None:
        a = get_age_profile(AgeGroup.LOWER_ELEMENTARY)
        b = get_age_profile(AgeGroup.ADULT)
        result = interpolate_profiles(a, b, -0.5)
        assert abs(result.jitter_x - a.jitter_x) < 1e-6
        result2 = interpolate_profiles(a, b, 1.5)
        assert abs(result2.jitter_x - b.jitter_x) < 1e-6


# ---------------------------------------------------------------------------
# Skill simulator tests
# ---------------------------------------------------------------------------


class TestSkillSimulator:
    def test_output_is_rgba(self) -> None:
        sim = SkillSimulator(AgeGroup.LOWER_ELEMENTARY, seed=42)
        char = _make_test_char(64)
        result = sim.apply_to_image(char, 64)
        assert result.mode == "RGBA"

    def test_output_size_matches_target(self) -> None:
        sim = SkillSimulator(AgeGroup.MIDDLE_SCHOOL, seed=42)
        char = _make_test_char(64)
        result = sim.apply_to_image(char, 128)
        assert result.size == (128, 128)

    def test_elementary_produces_different_output_than_adult(self) -> None:
        char = _make_test_char(64)
        sim_child = SkillSimulator(AgeGroup.LOWER_ELEMENTARY, seed=42)
        sim_adult = SkillSimulator(AgeGroup.ADULT, seed=42)

        result_child = sim_child.apply_to_image(char, 64)
        result_adult = sim_adult.apply_to_image(char, 64)

        arr_child = np.array(result_child)
        arr_adult = np.array(result_adult)
        # The images should differ due to different age effects
        assert not np.array_equal(arr_child, arr_adult)

    def test_seed_produces_reproducible_output(self) -> None:
        char = _make_test_char(64)
        sim1 = SkillSimulator(AgeGroup.LOWER_ELEMENTARY, seed=123)
        sim2 = SkillSimulator(AgeGroup.LOWER_ELEMENTARY, seed=123)

        result1 = sim1.apply_to_image(char, 64)
        result2 = sim2.apply_to_image(char, 64)

        assert np.array_equal(np.array(result1), np.array(result2))

    def test_jitter_offset_varies(self) -> None:
        sim = SkillSimulator(AgeGroup.LOWER_ELEMENTARY, seed=42)
        offsets = [sim.apply_jitter_to_offset(100.0, 100.0) for _ in range(20)]
        x_values = [o[0] for o in offsets]
        # Should have variation
        assert max(x_values) - min(x_values) > 0

    def test_vary_size_returns_positive(self) -> None:
        sim = SkillSimulator(AgeGroup.LOWER_ELEMENTARY, seed=42)
        for _ in range(10):
            size = sim.vary_size(100)
            assert size > 0

    def test_preserves_ink(self) -> None:
        """Ensure the character still has visible ink after transformation."""
        sim = SkillSimulator(AgeGroup.MIDDLE_SCHOOL, seed=42)
        char = _make_test_char(64)
        result = sim.apply_to_image(char, 64)
        alpha = np.array(result.getchannel("A"))
        assert np.any(alpha > 10)


# ---------------------------------------------------------------------------
# Historical style tests
# ---------------------------------------------------------------------------


class TestHistoricalStyle:
    def test_brush_pen_output_is_rgba(self) -> None:
        char = _make_test_char(64)
        result = apply_historical_style(char, HistoricalInstrument.BRUSH_PEN)
        assert result.mode == "RGBA"

    def test_fountain_pen_output_is_rgba(self) -> None:
        char = _make_test_char(64)
        result = apply_historical_style(char, HistoricalInstrument.FOUNTAIN_PEN)
        assert result.mode == "RGBA"

    def test_ballpoint_pen_output_is_rgba(self) -> None:
        char = _make_test_char(64)
        result = apply_historical_style(char, HistoricalInstrument.BALLPOINT_PEN)
        assert result.mode == "RGBA"

    def test_reed_pen_output_is_rgba(self) -> None:
        char = _make_test_char(64)
        result = apply_historical_style(char, HistoricalInstrument.REED_PEN)
        assert result.mode == "RGBA"

    def test_custom_ink_color(self) -> None:
        char = _make_test_char(64)
        red_ink = (200, 0, 0)
        result = apply_historical_style(
            char, HistoricalInstrument.FOUNTAIN_PEN, ink_color=red_ink
        )
        # Check that the ink color is applied
        arr = np.array(result)
        ink_mask = arr[:, :, 3] > 100
        if np.any(ink_mask):
            # Red channel should dominate
            avg_r = np.mean(arr[ink_mask, 0])
            avg_b = np.mean(arr[ink_mask, 2])
            assert avg_r > avg_b

    def test_blank_image_handled(self) -> None:
        blank = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        result = apply_historical_style(blank, HistoricalInstrument.BRUSH_PEN)
        assert result.size == (64, 64)

    def test_all_instruments_produce_output(self) -> None:
        char = _make_test_char(64)
        for instrument in HistoricalInstrument:
            result = apply_historical_style(char, instrument)
            assert result.size == char.size

    def test_instruments_produce_visibly_different_output(self) -> None:
        char = _make_test_char(64)
        results = {}
        for instrument in HistoricalInstrument:
            result = apply_historical_style(char, instrument)
            results[instrument] = np.array(result)

        # At least brush pen and ballpoint should differ
        brush = results[HistoricalInstrument.BRUSH_PEN]
        ballpoint = results[HistoricalInstrument.BALLPOINT_PEN]
        assert not np.array_equal(brush, ballpoint)


# ---------------------------------------------------------------------------
# Temporal renderer tests
# ---------------------------------------------------------------------------


class TestTemporalRenderer:
    def test_renders_single_line(self) -> None:
        renderer = TemporalRenderer(AgeGroup.MIDDLE_SCHOOL, seed=42)
        chars = [_make_simple_char(64) for _ in range(3)]
        text = "你好吗"
        page = renderer.render_text(chars, text, page_size=(800, 400), font_size=40)
        assert page.size == (800, 400)
        assert _has_ink(page)

    def test_renders_multiple_lines(self) -> None:
        renderer = TemporalRenderer(AgeGroup.LOWER_ELEMENTARY, seed=42)
        chars = [_make_simple_char(64) for _ in range(20)]
        text = "一二三四五六七八九十" * 2
        page = renderer.render_text(chars, text, page_size=(800, 600), font_size=30)
        assert page.size == (800, 600)
        assert _has_ink(page)

    def test_elementary_output_differs_from_adult(self) -> None:
        chars_child = [_make_simple_char(64) for _ in range(5)]
        chars_adult = [_make_simple_char(64) for _ in range(5)]
        text = "你好世界大"

        r_child = TemporalRenderer(AgeGroup.LOWER_ELEMENTARY, seed=42)
        r_adult = TemporalRenderer(AgeGroup.ADULT, seed=42)

        page_child = r_child.render_text(chars_child, text, page_size=(600, 300), font_size=40)
        page_adult = r_adult.render_text(chars_adult, text, page_size=(600, 300), font_size=40)

        arr_child = np.array(page_child)
        arr_adult = np.array(page_adult)
        assert not np.array_equal(arr_child, arr_adult)

    def test_vary_line_spacing(self) -> None:
        renderer = TemporalRenderer(AgeGroup.LOWER_ELEMENTARY, seed=42)
        spacings = [renderer.vary_line_spacing(80) for _ in range(50)]
        assert all(s > 0 for s in spacings)
        # Should have some variation
        assert max(spacings) - min(spacings) > 0

    def test_simulator_property(self) -> None:
        renderer = TemporalRenderer(AgeGroup.MIDDLE_SCHOOL, seed=42)
        assert isinstance(renderer.simulator, SkillSimulator)

    def test_handles_empty_text(self) -> None:
        renderer = TemporalRenderer(AgeGroup.ADULT, seed=42)
        page = renderer.render_text([], "", page_size=(400, 400), font_size=40)
        assert page.size == (400, 400)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestTemporalIntegration:
    def test_all_age_groups_render(self) -> None:
        """Verify all age groups can render a page without errors."""
        text = "测试文字"
        for group in list_age_groups():
            chars = [_make_simple_char(64) for _ in range(4)]
            renderer = TemporalRenderer(group, seed=42)
            page = renderer.render_text(chars, text, page_size=(400, 200), font_size=30)
            assert page.size == (400, 200)
            assert _has_ink(page)

    def test_all_instruments_render(self) -> None:
        """Verify all historical instruments can be applied."""
        char = _make_test_char(64)
        for instrument in HistoricalInstrument:
            result = apply_historical_style(char, instrument)
            # Should still have ink
            alpha = np.array(result.getchannel("A"))
            assert np.any(alpha > 10)

    def test_skill_simulator_consistency(self) -> None:
        """Verify simulator produces consistent output with same seed."""
        char = _make_test_char(64)
        results = []
        for _ in range(3):
            sim = SkillSimulator(AgeGroup.LOWER_ELEMENTARY, seed=99)
            results.append(sim.apply_to_image(char, 64))

        for i in range(1, len(results)):
            assert np.array_equal(
                np.array(results[0]), np.array(results[i])
            )
