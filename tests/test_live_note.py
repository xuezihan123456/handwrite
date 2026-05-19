"""Tests for the live classroom-note writing module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from handwrite.live_note import (
    LiveNoteEngine,
    NoteAnimationConfig,
    live_note_video,
)
from handwrite.live_note.cursor import (
    draw_pen_cursor,
    overlay_cursor_on_frames,
)
from handwrite.live_note.pacing import (
    compute_frame_count_for_char,
    is_breath_punctuation,
    is_heavy_punctuation,
    is_light_punctuation,
    plan_frame_budget,
)


# Small render config keeps tests fast.
_TINY = NoteAnimationConfig(
    font_size=24,
    fps=4,
    wpm=240,
    page_size=(128, 128),
    margins=(8, 8, 8, 8),
    cursor=False,
)


def _has_change(before: Image.Image, after: Image.Image) -> bool:
    a = np.array(before.convert("L"))
    b = np.array(after.convert("L"))
    return bool(np.any(a != b))


def _darkest_value(image: Image.Image) -> int:
    arr = np.array(image.convert("L"))
    return int(arr.min())


# ---------------------------------------------------------------------------
# Imports + construction
# ---------------------------------------------------------------------------


class TestImportsAndConstruction:
    def test_imports_public_api(self) -> None:
        # Reaching this point means all imports succeeded.
        assert callable(live_note_video)
        assert callable(getattr(LiveNoteEngine, "render"))

    def test_default_config_values(self) -> None:
        cfg = NoteAnimationConfig()
        assert cfg.fps > 0
        assert cfg.wpm > 0
        assert cfg.cursor is True
        assert cfg.font_size > 0

    def test_engine_constructs_with_custom_config(self) -> None:
        engine = LiveNoteEngine(config=_TINY)
        assert engine.config is _TINY


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------


class TestPacing:
    def test_punctuation_classifiers(self) -> None:
        assert is_heavy_punctuation("\u3002")  # 。
        assert is_light_punctuation("\uff0c")  # ，
        assert is_breath_punctuation("\u2026")  # …
        assert not is_heavy_punctuation("a")

    def test_linear_strategy_is_uniform_for_regular_chars(self) -> None:
        a = compute_frame_count_for_char("a", base_fps=24, wpm=120, strategy="linear")
        b = compute_frame_count_for_char("b", base_fps=24, wpm=120, strategy="linear")
        assert a == b
        assert a >= 1

    def test_punctuation_pause_slows_heavy_punct(self) -> None:
        plain = compute_frame_count_for_char(
            "a", base_fps=24, wpm=120, strategy="punctuation_pause"
        )
        heavy = compute_frame_count_for_char(
            "\u3002", base_fps=24, wpm=120, strategy="punctuation_pause"
        )
        assert heavy > plain

    def test_breath_pause_slows_everything_a_bit(self) -> None:
        plain_linear = compute_frame_count_for_char(
            "a", base_fps=24, wpm=120, strategy="linear"
        )
        plain_breath = compute_frame_count_for_char(
            "a", base_fps=24, wpm=120, strategy="breath_pause"
        )
        assert plain_breath >= plain_linear

    def test_plan_frame_budget_matches_text_length(self) -> None:
        text = "Hi\u3002"
        budget = plan_frame_budget(text, base_fps=12, wpm=120)
        assert len(budget) == len(text)
        assert all(value >= 1 for value in budget)

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_frame_count_for_char(
                "a", base_fps=24, wpm=120, strategy="weird"  # type: ignore[arg-type]
            )

    def test_invalid_char_length_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_frame_count_for_char(
                "ab", base_fps=24, wpm=120, strategy="linear"
            )


# ---------------------------------------------------------------------------
# Cursor overlay
# ---------------------------------------------------------------------------


class TestCursorOverlay:
    def test_draw_pen_cursor_changes_pixels(self) -> None:
        frame = Image.new("L", (64, 64), color=255)
        result = draw_pen_cursor(frame, (32, 32), radius=4, halo=False)
        assert result.size == frame.size
        assert _has_change(frame, result)

    def test_draw_pen_cursor_does_not_mutate_input(self) -> None:
        frame = Image.new("L", (32, 32), color=255)
        before = np.array(frame.copy())
        draw_pen_cursor(frame, (16, 16))
        after = np.array(frame)
        assert np.array_equal(before, after)

    def test_overlay_handles_empty_inputs(self) -> None:
        assert overlay_cursor_on_frames([], []) == []
        frame = Image.new("L", (16, 16), color=255)
        out = overlay_cursor_on_frames([frame], [])
        assert len(out) == 1

    def test_overlay_produces_darker_channel(self) -> None:
        frame = Image.new("L", (32, 32), color=255)
        out = overlay_cursor_on_frames([frame], [(16, 16)], radius=4, halo=True)
        assert _darkest_value(out[0]) < 255


# ---------------------------------------------------------------------------
# Engine rendering
# ---------------------------------------------------------------------------


class TestEngineRender:
    def test_short_text_returns_frames(self) -> None:
        engine = LiveNoteEngine(config=_TINY)
        frames = engine.render("\u4f60\u597d")  # 你好
        assert len(frames) >= 2
        assert all(isinstance(f, Image.Image) for f in frames)
        # Frames should match the configured canvas.
        assert frames[0].size == _TINY.page_size

    def test_empty_text_raises(self) -> None:
        engine = LiveNoteEngine(config=_TINY)
        with pytest.raises(ValueError, match="empty"):
            engine.render("")
        with pytest.raises(ValueError, match="empty"):
            engine.render("   ")

    def test_punctuation_extends_frame_count(self) -> None:
        engine = LiveNoteEngine(config=_TINY)
        plain = engine.render("\u4f60\u597d")  # 你好
        punctuated = engine.render("\u4f60\u597d\u3002")  # 你好。
        # Each glyph contributes its budget plus the intro frame so the
        # punctuated version must produce strictly more frames.
        assert len(punctuated) > len(plain)

    def test_multi_line_text_renders(self) -> None:
        engine = LiveNoteEngine(config=_TINY)
        frames = engine.render("AB\nCD")
        assert len(frames) > 2
        assert frames[0].size == _TINY.page_size

    def test_cursor_overlay_changes_frames(self) -> None:
        cursor_cfg = NoteAnimationConfig(
            font_size=24,
            fps=4,
            wpm=240,
            page_size=(128, 128),
            margins=(8, 8, 8, 8),
            cursor=True,
        )
        plain_engine = LiveNoteEngine(config=_TINY)
        cursor_engine = LiveNoteEngine(config=cursor_cfg)
        text = "\u4f60\u597d"
        plain_frames = plain_engine.render(text)
        cursor_frames = cursor_engine.render(text)
        # Both should produce the same number of frames.
        assert len(plain_frames) == len(cursor_frames)
        # A representative middle frame should differ once cursor is drawn.
        mid = len(plain_frames) // 2
        assert _has_change(plain_frames[mid], cursor_frames[mid])

    def test_progressive_ink_increase(self) -> None:
        engine = LiveNoteEngine(config=_TINY)
        frames = engine.render("\u4f60\u597d")
        ink_first = (255 - np.array(frames[0].convert("L"))).sum()
        ink_last = (255 - np.array(frames[-1].convert("L"))).sum()
        assert ink_last >= ink_first


# ---------------------------------------------------------------------------
# Public live_note_video helper
# ---------------------------------------------------------------------------


class TestLiveNoteVideo:
    def test_gif_export_round_trip(self, tmp_path: Path) -> None:
        output = tmp_path / "note.gif"
        info = live_note_video(
            text="\u4f60\u597d",
            output_path=str(output),
            font_size=24,
            fps=4,
            wpm=240,
            cursor=False,
            format="gif",
        )
        assert output.exists()
        assert output.stat().st_size > 0
        assert info["frame_count"] >= 2
        assert info["duration_s"] > 0
        assert info["output_path"] == str(output)

    def test_rejects_empty_text(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            live_note_video(
                text="",
                output_path=str(tmp_path / "x.gif"),
                font_size=24,
                fps=4,
                wpm=240,
                format="gif",
            )
