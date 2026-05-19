"""Tests for the error-notebook module (innovation #05).

The tests intentionally use a small ``font_size`` and short strings so that
each page renders quickly while still exercising the full pipeline:
diff -> page composition -> red-pen annotation overlays -> PDF export.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from handwrite.error_notebook import (
    DIFF_DELETE,
    DIFF_EQUAL,
    DIFF_INSERT,
    DIFF_REPLACE,
    DiffSegment,
    ErrorEntry,
    ErrorNotebookBuilder,
    build_error_notebook,
    diff_answers,
    red_correction,
    strike_through,
    tokenize_answer,
    underline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_image(
    size: tuple[int, int] = (200, 80),
    color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _pixel_difference_count(a: Image.Image, b: Image.Image) -> int:
    """Count pixels that differ between two RGB images of equal size."""
    assert a.size == b.size, "images must be the same size"
    a_data = list(a.convert("RGB").getdata())
    b_data = list(b.convert("RGB").getdata())
    return sum(1 for pa, pb in zip(a_data, b_data) if pa != pb)


def _sample_entry(prefix: str = "Q") -> ErrorEntry:
    return ErrorEntry(
        question=f"{prefix} 2 + 3 = ?",
        wrong="6",
        correct="5",
        reflection="",
    )


# ---------------------------------------------------------------------------
# Imports / public API surface
# ---------------------------------------------------------------------------


class TestImports:
    def test_package_exposes_public_api(self) -> None:
        from handwrite import error_notebook

        assert hasattr(error_notebook, "ErrorEntry")
        assert hasattr(error_notebook, "ErrorNotebookBuilder")
        assert hasattr(error_notebook, "build_error_notebook")
        assert hasattr(error_notebook, "DiffSegment")
        assert hasattr(error_notebook, "diff_answers")

    def test_annotation_primitives_exposed(self) -> None:
        from handwrite import error_notebook

        assert callable(error_notebook.strike_through)
        assert callable(error_notebook.underline)
        assert callable(error_notebook.red_correction)


# ---------------------------------------------------------------------------
# Re-exported diff API
# ---------------------------------------------------------------------------


class TestDiffReexports:
    def test_diff_answers_smoke(self) -> None:
        segments = diff_answers("abc", "abd")
        assert isinstance(segments, list)
        assert all(isinstance(seg, DiffSegment) for seg in segments)
        # End-to-end invariant: joining ``wrong`` parts reproduces input.
        joined_wrong = "".join(seg.wrong for seg in segments)
        joined_correct = "".join(seg.correct for seg in segments)
        assert joined_wrong == "abc"
        assert joined_correct == "abd"

    def test_diff_marks_changes(self) -> None:
        segments = diff_answers("cat", "bat")
        kinds = {seg.kind for seg in segments}
        # At least one of the three change kinds should be present.
        assert kinds & {DIFF_DELETE, DIFF_INSERT, DIFF_REPLACE}

    def test_tokenize_keeps_latex_clusters(self) -> None:
        tokens = tokenize_answer(r"\frac{1}{2} + x")
        assert any(tok.startswith("\\frac") for tok in tokens)


# ---------------------------------------------------------------------------
# ErrorEntry data class
# ---------------------------------------------------------------------------


class TestErrorEntry:
    def test_basic_construction(self) -> None:
        entry = ErrorEntry(question="Q", wrong="W", correct="C")
        assert entry.question == "Q"
        assert entry.wrong == "W"
        assert entry.correct == "C"
        assert entry.reflection == ""
        assert entry.scan_image is None

    def test_rejects_non_string_fields(self) -> None:
        with pytest.raises(TypeError):
            ErrorEntry(question=123, wrong="W", correct="C")  # type: ignore[arg-type]

    def test_rejects_completely_empty_entry(self) -> None:
        with pytest.raises(ValueError):
            ErrorEntry(question="", wrong="", correct="")

    def test_accepts_scan_image(self) -> None:
        scan = _solid_image((40, 40), color=(220, 220, 220))
        entry = ErrorEntry(question="Q", wrong="W", correct="C", scan_image=scan)
        assert entry.scan_image is scan


# ---------------------------------------------------------------------------
# Annotation primitives
# ---------------------------------------------------------------------------


class TestAnnotations:
    def test_strike_through_changes_pixels_in_bbox(self) -> None:
        base = _solid_image((200, 80), color=(255, 255, 255))
        result = strike_through(base, bbox=(20, 20, 180, 60))
        assert isinstance(result, Image.Image)
        assert result.size == base.size
        assert result.mode == "RGB"
        # The result must differ from the input — strike-through draws ink.
        diff = _pixel_difference_count(base, result)
        assert diff > 0
        # And the diff must concentrate inside the bbox.
        cropped_base = base.crop((20, 20, 180, 60))
        cropped_result = result.crop((20, 20, 180, 60))
        assert _pixel_difference_count(cropped_base, cropped_result) > 0

    def test_underline_adds_ink_below_bbox(self) -> None:
        base = _solid_image((200, 80), color=(255, 255, 255))
        result = underline(base, bbox=(20, 20, 180, 50))
        assert isinstance(result, Image.Image)
        assert result.size == base.size
        # Difference should appear below the bbox (around y >= 50).
        sliver_base = base.crop((20, 50, 180, 70))
        sliver_result = result.crop((20, 50, 180, 70))
        assert _pixel_difference_count(sliver_base, sliver_result) > 0

    def test_red_correction_draws_text(self) -> None:
        base = _solid_image((220, 80), color=(255, 255, 255))
        result = red_correction(base, position=(10, 10), text="X")
        assert isinstance(result, Image.Image)
        assert _pixel_difference_count(base, result) > 0

    def test_strike_through_invalid_bbox_raises(self) -> None:
        base = _solid_image((40, 40))
        with pytest.raises(ValueError):
            strike_through(base, bbox=(0, 0, 0))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Builder accumulation
# ---------------------------------------------------------------------------


class TestBuilderAccumulation:
    def test_add_entry_chainable(self) -> None:
        builder = ErrorNotebookBuilder()
        result = builder.add_entry(question="Q", wrong="2", correct="3")
        assert result is builder
        assert len(builder.entries) == 1

    def test_empty_builder_renders_raises(self) -> None:
        builder = ErrorNotebookBuilder()
        with pytest.raises(ValueError):
            builder.render(font_size=40)

    def test_multi_entry_returns_one_page_per_entry(self) -> None:
        builder = ErrorNotebookBuilder()
        builder.add_entry(question="Q1: 1+1", wrong="3", correct="2")
        builder.add_entry(question="Q2: 2+2", wrong="5", correct="4")
        builder.add_entry(question="Q3: 3+3", wrong="7", correct="6")
        pages = builder.render(font_size=40)
        assert len(pages) == 3
        for page in pages:
            assert isinstance(page, Image.Image)
            assert page.mode == "RGB"

    def test_extend_accepts_dicts(self) -> None:
        builder = ErrorNotebookBuilder()
        builder.extend(
            [
                {"question": "Q1", "wrong": "1", "correct": "2"},
                ErrorEntry(question="Q2", wrong="3", correct="4"),
            ]
        )
        assert len(builder.entries) == 2

    def test_extend_corrupt_dict_raises_clearly(self) -> None:
        builder = ErrorNotebookBuilder()
        with pytest.raises(ValueError, match="question"):
            builder.extend([{"wrong": "1", "correct": "2"}])


# ---------------------------------------------------------------------------
# Builder rendering
# ---------------------------------------------------------------------------


class TestBuilderRender:
    def test_render_returns_pil_images(self) -> None:
        builder = ErrorNotebookBuilder()
        builder.add_entry(question="Q", wrong="3", correct="2")
        pages = builder.render(font_size=40)
        assert pages and isinstance(pages[0], Image.Image)
        assert pages[0].size == (2480, 3508)

    def test_render_with_latex_question(self) -> None:
        """LaTeX in the question must be preserved as text without crashing."""
        builder = ErrorNotebookBuilder()
        builder.add_entry(
            question=r"求 \frac{1}{2} + \frac{1}{3} = ?",
            wrong=r"\frac{2}{5}",
            correct=r"\frac{5}{6}",
        )
        pages = builder.render(font_size=40)
        assert len(pages) == 1
        assert pages[0].mode == "RGB"

    def test_strike_through_applied_when_diff_present(self) -> None:
        """Pages with a real diff should contain red ink from the overlay."""
        builder = ErrorNotebookBuilder()
        builder.add_entry(question="Q", wrong="abc", correct="xyz")
        pages = builder.render(font_size=40)
        # Scan a portion of the page that should contain the wrong-answer
        # block — check for red-ish pixels left behind by strike_through.
        page = pages[0].convert("RGB")
        red_pixels = 0
        for r, g, b in page.getdata():
            if r > 150 and g < 120 and b < 120:
                red_pixels += 1
        assert red_pixels > 0


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------


class TestPdfExport:
    def test_export_pdf_roundtrip(self, tmp_path: Path) -> None:
        builder = ErrorNotebookBuilder()
        builder.add_entry(question="Q1", wrong="3", correct="2")
        out = tmp_path / "errors.pdf"
        info = builder.export_pdf(out, font_size=40)
        assert info["pdf_path"] == Path(out)
        assert info["page_count"] == 1
        assert info["entry_count"] == 1
        assert Path(out).exists()
        # File must be a PDF — header should start with "%PDF".
        with open(out, "rb") as fh:
            head = fh.read(4)
        assert head == b"%PDF"

    def test_build_error_notebook_wrapper(self, tmp_path: Path) -> None:
        out = tmp_path / "wrapper.pdf"
        info = build_error_notebook(
            [
                {"question": "Q1", "wrong": "1", "correct": "2"},
                ErrorEntry(question="Q2", wrong="3", correct="4"),
            ],
            out,
            font_size=40,
        )
        assert info["pdf_path"] == Path(out)
        assert info["entry_count"] == 2
        assert info["page_count"] == 2
        assert Path(out).exists()

    def test_build_error_notebook_requires_entries(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.pdf"
        with pytest.raises(ValueError):
            build_error_notebook([], out, font_size=40)

    def test_build_error_notebook_rejects_corrupt_dict(self, tmp_path: Path) -> None:
        out = tmp_path / "corrupt.pdf"
        with pytest.raises(ValueError):
            build_error_notebook(
                [{"question": "Q", "wrong": "1"}],  # missing 'correct'
                out,
                font_size=40,
            )
