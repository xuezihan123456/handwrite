"""Tests for the ``handwrite.nl_style`` natural-language style picker."""

from __future__ import annotations

import pytest

from handwrite.composer import CURSIVE_LAYOUT, NATURAL_LAYOUT, NEAT_LAYOUT
from handwrite.nl_style import (
    EMOTION_TO_PARAMS,
    INTENSITY_MODIFIERS,
    KEYWORD_TO_STYLE,
    NaturalLanguageStyler,
    StyleVector,
    parse_description,
)


# ---------------------------------------------------------------------------
# Imports / public API
# ---------------------------------------------------------------------------


class TestPublicApi:
    def test_parse_description_callable(self) -> None:
        vector = parse_description("neat")
        assert isinstance(vector, StyleVector)

    def test_class_can_be_instantiated(self) -> None:
        styler = NaturalLanguageStyler()
        assert isinstance(styler.parse("neat"), StyleVector)

    def test_style_vector_dataclass_has_required_fields(self) -> None:
        vector = StyleVector()
        for field_name in (
            "rotation_jitter",
            "scale_jitter",
            "ink_density",
            "baseline_jitter",
            "char_spacing",
            "line_spacing",
            "style_name",
            "suggested_layout",
            "mood_tags",
        ):
            assert hasattr(vector, field_name), f"missing field: {field_name}"

    def test_keyword_dictionaries_meet_size_floor(self) -> None:
        # The spec requires at least 80 keyword entries combined.
        combined = len(KEYWORD_TO_STYLE) + len(EMOTION_TO_PARAMS)
        assert combined >= 80, f"only {combined} keywords defined"
        assert len(INTENSITY_MODIFIERS) >= 5


# ---------------------------------------------------------------------------
# Defaults / empty input
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_empty_input_returns_default_vector(self) -> None:
        vector = parse_description("")
        assert vector == StyleVector()
        assert vector.suggested_layout == NATURAL_LAYOUT
        assert vector.mood_tags == []
        assert vector.style_name == "default"

    def test_whitespace_input_returns_default_vector(self) -> None:
        assert parse_description("   \n  \t").suggested_layout == NATURAL_LAYOUT

    def test_unknown_text_returns_default_vector(self) -> None:
        vector = parse_description("xyzzy plover")
        assert vector.style_name == "default"
        assert vector.suggested_layout == NATURAL_LAYOUT


# ---------------------------------------------------------------------------
# Basic keyword recognition (Chinese + English)
# ---------------------------------------------------------------------------


class TestBasicKeywords:
    def test_chinese_neat_keyword_maps_to_neat_layout(self) -> None:
        vector = parse_description("\u5de5\u6574\u7684\u5b57")  # 工整的字
        assert vector.suggested_layout == NEAT_LAYOUT
        assert vector.style_name in {"neat", "default"}
        # Neat reduces jitter from the baseline.
        assert vector.rotation_jitter < 1.5

    def test_chinese_sloppy_keyword_maps_to_cursive_layout(self) -> None:
        vector = parse_description("\u6f5c\u8349\u7684\u5b57")  # 潜草的字
        assert vector.suggested_layout == CURSIVE_LAYOUT
        assert vector.rotation_jitter > 1.5

    def test_english_neat_keyword(self) -> None:
        vector = parse_description("neat handwriting")
        assert vector.suggested_layout == NEAT_LAYOUT
        assert vector.style_name == "neat"

    def test_english_sloppy_keyword(self) -> None:
        vector = parse_description("sloppy scribble")
        assert vector.suggested_layout == CURSIVE_LAYOUT
        assert vector.rotation_jitter > 3.0


# ---------------------------------------------------------------------------
# Intensity modifiers
# ---------------------------------------------------------------------------


class TestIntensityModifiers:
    def test_very_increases_jitter(self) -> None:
        plain = parse_description("anxious")
        amplified = parse_description("very anxious")
        assert amplified.rotation_jitter > plain.rotation_jitter

    def test_slightly_reduces_jitter(self) -> None:
        plain = parse_description("anxious")
        reduced = parse_description("slightly anxious")
        assert reduced.rotation_jitter < plain.rotation_jitter

    def test_chinese_极_amplifies(self) -> None:
        plain = parse_description("\u7126\u8651")            # 焦虑
        amplified = parse_description("\u6781\u7126\u8651")  # 极焦虑
        assert amplified.rotation_jitter > plain.rotation_jitter

    def test_chinese_稍微_reduces(self) -> None:
        plain = parse_description("\u7126\u8651")                # 焦虑
        reduced = parse_description("\u7a0d\u5fae\u7126\u8651")  # 稍微焦虑
        assert reduced.rotation_jitter < plain.rotation_jitter


# ---------------------------------------------------------------------------
# Emotion mapping
# ---------------------------------------------------------------------------


class TestEmotions:
    def test_anxious_produces_more_jitter_than_calm(self) -> None:
        anxious = parse_description("anxious")
        calm = parse_description("calm")
        assert anxious.rotation_jitter > calm.rotation_jitter
        assert anxious.baseline_jitter > calm.baseline_jitter

    def test_calm_yields_neat_layout(self) -> None:
        vector = parse_description("calm and focused")
        assert vector.suggested_layout == NEAT_LAYOUT
        assert "calm" in vector.mood_tags or "focused" in vector.mood_tags

    def test_chinese_emotion_tags_recorded(self) -> None:
        vector = parse_description("\u7d27\u5f20")  # 紧张
        assert "nervous" in vector.mood_tags

    def test_tired_lowers_ink_density(self) -> None:
        vector = parse_description("tired")
        assert vector.ink_density < 1.0


# ---------------------------------------------------------------------------
# Multi-keyword combinations
# ---------------------------------------------------------------------------


class TestMultiKeyword:
    def test_high_school_rushed_description_chinese(self) -> None:
        vector = parse_description(
            "\u7d27\u5f20\u7126\u8651\u7684\u9ad8\u4e09\u5b66\u751f\u8d76\u65f6\u95f4\u7684\u5b57"
        )
        # Tense + anxious + rushed should produce a cursive, jittery vector.
        assert vector.suggested_layout == CURSIVE_LAYOUT
        assert vector.rotation_jitter > 3.0
        assert vector.baseline_jitter > 0.2
        # At least one mood tag from the emotion list.
        assert any(tag in {"anxious", "nervous"} for tag in vector.mood_tags)

    def test_calm_neat_teacher_english(self) -> None:
        vector = parse_description("calm, neat elementary teacher")
        assert vector.suggested_layout == NEAT_LAYOUT
        assert vector.rotation_jitter < 1.5
        assert "calm" in vector.mood_tags

    def test_child_sloppy_combination(self) -> None:
        vector = parse_description("childlike messy notes")
        assert vector.suggested_layout == CURSIVE_LAYOUT
        assert vector.rotation_jitter > 2.0


# ---------------------------------------------------------------------------
# Composer integration
# ---------------------------------------------------------------------------


class TestComposerIntegration:
    def test_to_composer_kwargs_contains_valid_layout(self) -> None:
        vector = parse_description("neat tidy careful")
        kwargs = vector.to_composer_kwargs()
        assert kwargs["layout"] in {NEAT_LAYOUT, NATURAL_LAYOUT, CURSIVE_LAYOUT}
        assert kwargs["layout"] == NEAT_LAYOUT
        assert isinstance(kwargs["style_params"], dict)
        assert "rotation_jitter" in kwargs["style_params"]

    def test_apply_to_layout_default(self) -> None:
        vector = parse_description("sloppy")
        assert vector.apply_to_layout() == CURSIVE_LAYOUT

    def test_apply_to_layout_with_override(self) -> None:
        vector = parse_description("sloppy")
        assert vector.apply_to_layout(NEAT_LAYOUT) == NEAT_LAYOUT

    def test_apply_to_layout_ignores_invalid_override(self) -> None:
        vector = parse_description("neat")
        assert vector.apply_to_layout("invalid_layout") == NEAT_LAYOUT

    def test_layouts_map_to_composer_constants(self) -> None:
        """The module's layout names must match composer's constants exactly."""
        from handwrite.nl_style import (
            CURSIVE_LAYOUT as nl_cursive,
            NATURAL_LAYOUT as nl_natural,
            NEAT_LAYOUT as nl_neat,
        )

        assert nl_neat == NEAT_LAYOUT
        assert nl_natural == NATURAL_LAYOUT
        assert nl_cursive == CURSIVE_LAYOUT


# ---------------------------------------------------------------------------
# Determinism / clamping
# ---------------------------------------------------------------------------


class TestDeterminismAndClamping:
    @pytest.mark.parametrize(
        "text",
        [
            "calm neat elementary teacher",
            "\u7d27\u5f20\u7126\u8651\u7684\u9ad8\u4e09\u5b66\u751f\u8d76\u65f6\u95f4\u7684\u5b57",
            "very anxious sloppy",
            "",
        ],
    )
    def test_same_input_same_output(self, text: str) -> None:
        a = parse_description(text)
        b = parse_description(text)
        assert a.to_dict() == b.to_dict()

    def test_extreme_input_does_not_overflow(self) -> None:
        # Pile on every messy keyword we can think of.
        vector = parse_description(
            "extremely sloppy very messy super hurried extremely scribble"
        )
        # Values must remain inside the declared StyleVector ranges.
        assert 0.0 <= vector.rotation_jitter <= 15.0
        assert 0.0 <= vector.scale_jitter <= 1.0
        assert 0.0 <= vector.baseline_jitter <= 1.0
        assert 0.5 <= vector.ink_density <= 1.5
        assert 0.5 <= vector.char_spacing <= 2.0
        assert 0.5 <= vector.line_spacing <= 2.0

    def test_mood_tags_are_unique_and_ordered(self) -> None:
        vector = parse_description("anxious anxious nervous calm")
        # 'anxious' should not appear twice even though we wrote it twice.
        assert vector.mood_tags.count("anxious") == 1

    def test_to_dict_round_trips(self) -> None:
        vector = parse_description("neat tidy")
        payload = vector.to_dict()
        assert payload["suggested_layout"] == NEAT_LAYOUT
        assert payload["style_name"] in {"neat", "default"}
