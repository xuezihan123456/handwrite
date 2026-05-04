"""Tests for the handwriting animation module."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from handwrite.animation.animation_composer import (
    calculate_layout,
    compose_text_animation,
)
from handwrite.animation.animation_engine import (
    export_animation,
    generate_char_animation,
)
from handwrite.animation.frame_renderer import render_animation_frames
from handwrite.animation.stroke_order import extract_strokes
from handwrite.animation.trajectory_generator import (
    evaluate_bezier,
    generate_trajectories,
)
from handwrite.exporter import export_animation as exporter_export_animation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_char_with_horizontal_line(size: int = 256) -> Image.Image:
    """Create a test image with a single horizontal line (like stroke 一)."""
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    y = size // 2
    draw.line((40, y, size - 40, y), fill=0, width=6)
    return image


def _make_char_with_cross(size: int = 256) -> Image.Image:
    """Create a test image with a cross pattern (horizontal + vertical)."""
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    mid = size // 2
    draw.line((40, mid, size - 40, mid), fill=0, width=6)
    draw.line((mid, 40, mid, size - 40), fill=0, width=6)
    return image


def _make_char_with_l_shape(size: int = 256) -> Image.Image:
    """Create a test image with an L-shape (two connected strokes)."""
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    # Vertical stroke
    draw.line((60, 40, 60, size - 40), fill=0, width=6)
    # Horizontal stroke at bottom
    draw.line((60, size - 40, size - 60, size - 40), fill=0, width=6)
    return image


def _make_blank_image(size: int = 256) -> Image.Image:
    """Create a blank white image."""
    return Image.new("L", (size, size), color=255)


def _has_ink(image: Image.Image, threshold: int = 240) -> bool:
    """Check if an image contains any ink pixels."""
    arr = np.array(image.convert("L"))
    return bool(np.any(arr < threshold))


def _count_ink_pixels(image: Image.Image, threshold: int = 240) -> int:
    """Count the number of ink pixels in an image."""
    arr = np.array(image.convert("L"))
    return int(np.sum(arr < threshold))


# ---------------------------------------------------------------------------
# Stroke extraction tests
# ---------------------------------------------------------------------------


class TestExtractStrokes:
    def test_extracts_strokes_from_horizontal_line(self) -> None:
        image = _make_char_with_horizontal_line()
        strokes = extract_strokes(image)

        assert len(strokes) >= 1
        # Each stroke should have at least 2 points
        for stroke in strokes:
            assert len(stroke) >= 2

    def test_extracts_multiple_strokes_from_cross(self) -> None:
        image = _make_char_with_cross()
        strokes = extract_strokes(image)

        # A cross should produce at least 1 stroke (may merge at intersection)
        assert len(strokes) >= 1

    def test_returns_empty_for_blank_image(self) -> None:
        image = _make_blank_image()
        strokes = extract_strokes(image)
        assert strokes == []

    def test_stroke_points_are_xy_tuples(self) -> None:
        image = _make_char_with_horizontal_line()
        strokes = extract_strokes(image)

        for stroke in strokes:
            for point in stroke:
                assert len(point) == 2
                x, y = point
                assert isinstance(x, int)
                assert isinstance(y, int)

    def test_strokes_are_ordered_top_to_bottom(self) -> None:
        image = _make_char_with_l_shape()
        strokes = extract_strokes(image)

        if len(strokes) >= 2:
            # First stroke should start higher (smaller y) than later strokes
            first_start_y = strokes[0][0][1]
            last_start_y = strokes[-1][0][1]
            assert first_start_y <= last_start_y


# ---------------------------------------------------------------------------
# Trajectory generation tests
# ---------------------------------------------------------------------------


class TestGenerateTrajectories:
    def test_generates_trajectories_from_strokes(self) -> None:
        strokes = [[(10, 100), (50, 100), (100, 100), (150, 100), (200, 100)]]
        trajectories = generate_trajectories(strokes, samples_per_stroke=20)

        assert len(trajectories) == 1
        assert len(trajectories[0]) == 20

    def test_trajectory_points_are_float_tuples(self) -> None:
        strokes = [[(10, 50), (100, 50), (200, 50)]]
        trajectories = generate_trajectories(strokes)

        for traj in trajectories:
            for point in traj:
                assert len(point) == 2
                assert isinstance(point[0], float)
                assert isinstance(point[1], float)

    def test_handles_empty_strokes(self) -> None:
        trajectories = generate_trajectories([])
        assert trajectories == []

    def test_handles_single_point_stroke(self) -> None:
        strokes = [[(100, 100)]]
        trajectories = generate_trajectories(strokes)
        # Single-point strokes should be skipped
        assert trajectories == []

    def test_multiple_strokes_produce_multiple_trajectories(self) -> None:
        strokes = [
            [(10, 50), (100, 50), (200, 50)],
            [(10, 150), (100, 150), (200, 150)],
        ]
        trajectories = generate_trajectories(strokes, samples_per_stroke=15)

        assert len(trajectories) == 2
        for traj in trajectories:
            assert len(traj) == 15


class TestEvaluateBezier:
    def test_returns_start_at_t_zero(self) -> None:
        cp = ((0.0, 0.0), (10.0, 20.0), (30.0, 20.0), (40.0, 0.0))
        result = evaluate_bezier(cp, 0.0)
        assert abs(result[0] - 0.0) < 1e-6
        assert abs(result[1] - 0.0) < 1e-6

    def test_returns_end_at_t_one(self) -> None:
        cp = ((0.0, 0.0), (10.0, 20.0), (30.0, 20.0), (40.0, 0.0))
        result = evaluate_bezier(cp, 1.0)
        assert abs(result[0] - 40.0) < 1e-6
        assert abs(result[1] - 0.0) < 1e-6

    def test_midpoint_is_between_start_and_end(self) -> None:
        cp = ((0.0, 0.0), (10.0, 50.0), (30.0, 50.0), (40.0, 0.0))
        result = evaluate_bezier(cp, 0.5)
        assert 0.0 < result[0] < 40.0


# ---------------------------------------------------------------------------
# Frame renderer tests
# ---------------------------------------------------------------------------


class TestRenderAnimationFrames:
    def test_returns_correct_number_of_frames(self) -> None:
        image = _make_char_with_horizontal_line(64)
        strokes = extract_strokes(image)
        trajectories = generate_trajectories(strokes, samples_per_stroke=10)

        frames = render_animation_frames(
            image, trajectories, num_frames=10, canvas_size=(64, 64)
        )
        assert len(frames) == 10

    def test_first_frame_is_mostly_blank(self) -> None:
        image = _make_char_with_horizontal_line(64)
        strokes = extract_strokes(image)
        trajectories = generate_trajectories(strokes, samples_per_stroke=10)

        frames = render_animation_frames(
            image, trajectories, num_frames=10, canvas_size=(64, 64)
        )
        # First frame should have very little ink
        ink_count = _count_ink_pixels(frames[0])
        total_pixels = 64 * 64
        assert ink_count < total_pixels * 0.1

    def test_last_frame_has_most_ink(self) -> None:
        image = _make_char_with_horizontal_line(64)
        strokes = extract_strokes(image)
        trajectories = generate_trajectories(strokes, samples_per_stroke=10)

        frames = render_animation_frames(
            image, trajectories, num_frames=10, canvas_size=(64, 64)
        )
        ink_counts = [_count_ink_pixels(f) for f in frames]
        # Ink should generally increase over time
        assert ink_counts[-1] >= ink_counts[0]

    def test_handles_empty_trajectories(self) -> None:
        image = _make_blank_image(64)
        frames = render_animation_frames(
            image, [], num_frames=5, canvas_size=(64, 64)
        )
        assert len(frames) == 5
        for frame in frames:
            assert not _has_ink(frame)

    def test_frame_size_matches_canvas(self) -> None:
        image = _make_char_with_horizontal_line(64)
        strokes = extract_strokes(image)
        trajectories = generate_trajectories(strokes)

        frames = render_animation_frames(
            image, trajectories, num_frames=5, canvas_size=(128, 128)
        )
        for frame in frames:
            assert frame.size == (128, 128)

    def test_zero_frames_returns_empty(self) -> None:
        image = _make_char_with_horizontal_line(64)
        frames = render_animation_frames(image, [], num_frames=0, canvas_size=(64, 64))
        assert frames == []


# ---------------------------------------------------------------------------
# Animation composer tests
# ---------------------------------------------------------------------------


class TestComposeTextAnimation:
    def test_composes_two_character_animations(self) -> None:
        frame_a = [Image.new("L", (64, 64), color=255) for _ in range(5)]
        frame_b = [Image.new("L", (64, 64), color=255) for _ in range(5)]

        composed = compose_text_animation(
            [frame_a, frame_b],
            char_size=64,
            char_gap=10,
            line_gap=10,
            chars_per_line=2,
            inter_char_delay_frames=0,
        )
        assert len(composed) > 0
        # Canvas should be wide enough for 2 characters
        assert composed[0].size[0] >= 128

    def test_handles_single_character(self) -> None:
        frames = [Image.new("L", (64, 64), color=255) for _ in range(5)]
        composed = compose_text_animation([frames], char_size=64)
        assert len(composed) > 0

    def test_handles_empty_input(self) -> None:
        composed = compose_text_animation([])
        assert len(composed) == 1

    def test_inter_char_delay_extends_total_frames(self) -> None:
        frames = [Image.new("L", (64, 64), color=255) for _ in range(5)]

        short = compose_text_animation(
            [frames, frames],
            char_size=64,
            inter_char_delay_frames=0,
        )
        long = compose_text_animation(
            [frames, frames],
            char_size=64,
            inter_char_delay_frames=10,
        )
        assert len(long) > len(short)


class TestCalculateLayout:
    def test_single_character(self) -> None:
        w, h = calculate_layout(1, char_size=100)
        assert w == 100
        assert h == 100

    def test_multiple_characters_one_line(self) -> None:
        w, h = calculate_layout(3, char_size=100, char_gap=10, chars_per_line=5)
        assert w == 3 * 100 + 2 * 10
        assert h == 100

    def test_wraps_to_multiple_lines(self) -> None:
        w, h = calculate_layout(
            5, char_size=100, char_gap=10, line_gap=20, chars_per_line=3
        )
        # 3 on first line, 2 on second
        assert w == 3 * 100 + 2 * 10
        assert h == 2 * 100 + 1 * 20


# ---------------------------------------------------------------------------
# Animation engine tests
# ---------------------------------------------------------------------------


class TestGenerateCharAnimation:
    def test_returns_frames_list(self) -> None:
        frames = generate_char_animation("一", fps=10, duration=0.5, char_size=64)
        assert isinstance(frames, list)
        assert len(frames) >= 2
        assert all(isinstance(f, Image.Image) for f in frames)

    def test_frame_count_matches_fps_and_duration(self) -> None:
        frames = generate_char_animation("一", fps=20, duration=1.0, char_size=64)
        assert len(frames) == 20

    def test_duration_clamped_to_valid_range(self) -> None:
        frames_short = generate_char_animation("一", fps=10, duration=0.1, char_size=64)
        frames_long = generate_char_animation("一", fps=10, duration=5.0, char_size=64)
        # 0.1 should be clamped to 0.5 -> 5 frames at fps=10
        assert len(frames_short) == 5
        # 5.0 should be clamped to 2.0 -> 20 frames at fps=10
        assert len(frames_long) == 20

    def test_rejects_multi_char_input(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="single character"):
            generate_char_animation("你好", fps=10, duration=0.5)


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExportAnimation:
    def test_export_gif_creates_file(self, tmp_path: Path) -> None:
        frames = [
            Image.new("L", (64, 64), color=c) for c in [255, 200, 150, 100, 50]
        ]
        output = tmp_path / "test.gif"

        result = export_animation(frames, str(output), format="gif", fps=10)

        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0

    def test_export_mp4_creates_file(self, tmp_path: Path) -> None:
        frames = [
            Image.new("L", (64, 64), color=c) for c in [255, 200, 150, 100, 50]
        ]
        output = tmp_path / "test.mp4"

        result = export_animation(frames, str(output), format="mp4", fps=10)

        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0

    def test_export_rejects_empty_frames(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(ValueError, match="at least one frame"):
            export_animation([], str(tmp_path / "empty.gif"), format="gif")

    def test_export_rejects_unsupported_format(self, tmp_path: Path) -> None:
        import pytest

        frames = [Image.new("L", (64, 64), color=255)]
        with pytest.raises(ValueError, match="Unsupported format"):
            export_animation(frames, str(tmp_path / "test.webm"), format="webm")

    def test_export_creates_parent_directories(self, tmp_path: Path) -> None:
        frames = [Image.new("L", (64, 64), color=255)]
        output = tmp_path / "subdir" / "deep" / "anim.gif"

        export_animation(frames, str(output), format="gif", fps=10)

        assert output.exists()


class TestExporterExportAnimation:
    def test_exporter_animation_gif(self, tmp_path: Path) -> None:
        frames = [
            Image.new("L", (64, 64), color=c) for c in [255, 200, 150, 100, 50]
        ]
        output = tmp_path / "via_exporter.gif"

        result = exporter_export_animation(frames, str(output), format="gif", fps=10)

        assert result == output
        assert output.exists()

    def test_exporter_animation_mp4(self, tmp_path: Path) -> None:
        frames = [
            Image.new("L", (64, 64), color=c) for c in [255, 200, 150, 100, 50]
        ]
        output = tmp_path / "via_exporter.mp4"

        result = exporter_export_animation(frames, str(output), format="mp4", fps=10)

        assert result == output
        assert output.exists()


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline_horizontal_stroke(self, tmp_path: Path) -> None:
        """Test the full pipeline: image -> strokes -> trajectories -> frames -> export."""
        image = _make_char_with_horizontal_line(64)

        # Extract strokes
        strokes = extract_strokes(image)
        assert len(strokes) >= 1

        # Generate trajectories
        trajectories = generate_trajectories(strokes, samples_per_stroke=15)
        assert len(trajectories) >= 1

        # Render frames
        frames = render_animation_frames(
            image, trajectories, num_frames=8, canvas_size=(64, 64)
        )
        assert len(frames) == 8

        # Export GIF
        gif_path = tmp_path / "integration.gif"
        export_animation(frames, str(gif_path), format="gif", fps=10)
        assert gif_path.exists()

        # Export MP4
        mp4_path = tmp_path / "integration.mp4"
        export_animation(frames, str(mp4_path), format="mp4", fps=10)
        assert mp4_path.exists()

    def test_full_pipeline_cross_pattern(self, tmp_path: Path) -> None:
        """Test with a more complex character pattern."""
        image = _make_char_with_cross(64)

        strokes = extract_strokes(image)
        trajectories = generate_trajectories(strokes, samples_per_stroke=10)
        frames = render_animation_frames(
            image, trajectories, num_frames=6, canvas_size=(64, 64)
        )

        assert len(frames) == 6

        # Verify progressive ink increase
        ink_counts = [_count_ink_pixels(f) for f in frames]
        # Last frame should have more ink than first
        assert ink_counts[-1] >= ink_counts[0]

    def test_animation_frames_show_progressive_reveal(self) -> None:
        """Verify that intermediate frames show partial character."""
        image = _make_char_with_horizontal_line(64)
        strokes = extract_strokes(image)
        trajectories = generate_trajectories(strokes, samples_per_stroke=20)

        frames = render_animation_frames(
            image, trajectories, num_frames=20, canvas_size=(64, 64)
        )

        # Check that ink increases monotonically (approximately)
        ink_counts = [_count_ink_pixels(f) for f in frames]
        # At least the trend should be upward
        first_quarter_avg = sum(ink_counts[:5]) / 5
        last_quarter_avg = sum(ink_counts[-5:]) / 5
        assert last_quarter_avg > first_quarter_avg
