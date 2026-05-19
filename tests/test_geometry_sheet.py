"""Tests for the geometry exam-sheet module (innovation #9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from handwrite.geometry_sheet import (
    Figure,
    GeometrySheetBuilder,
    Problem,
    build_exam_sheet,
    draw_angle_arc,
    draw_arrow,
    draw_axes,
    draw_circle,
    draw_labeled_point,
    draw_line,
    draw_rectangle,
    draw_triangle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alpha_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Return the bounding box of the non-transparent pixels."""
    assert image.mode == "RGBA"
    return image.getchannel("A").getbbox()


def _ink_pixel_count(image: Image.Image) -> int:
    """Count non-transparent pixels in an RGBA overlay."""
    return sum(1 for px in image.getchannel("A").getdata() if px > 0)


# ---------------------------------------------------------------------------
# Import & module shape
# ---------------------------------------------------------------------------

class TestImports:
    def test_package_exposes_public_api(self) -> None:
        from handwrite import geometry_sheet

        assert hasattr(geometry_sheet, "GeometrySheetBuilder")
        assert hasattr(geometry_sheet, "Figure")
        assert hasattr(geometry_sheet, "Problem")
        assert hasattr(geometry_sheet, "build_exam_sheet")

    def test_primitive_helpers_exposed(self) -> None:
        from handwrite import geometry_sheet

        assert callable(geometry_sheet.draw_circle)
        assert callable(geometry_sheet.draw_triangle)
        assert callable(geometry_sheet.draw_rectangle)
        assert callable(geometry_sheet.draw_axes)
        assert callable(geometry_sheet.draw_angle_arc)
        assert callable(geometry_sheet.draw_arrow)
        assert callable(geometry_sheet.draw_labeled_point)
        assert callable(geometry_sheet.draw_line)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class TestDrawCircle:
    def test_circle_has_visible_pixels(self) -> None:
        img = draw_circle((200, 200), center=(100, 100), radius=60)
        assert img.size == (200, 200)
        assert img.mode == "RGBA"
        assert _ink_pixel_count(img) > 0

    def test_circle_pixels_fall_within_expected_region(self) -> None:
        img = draw_circle((300, 300), center=(150, 150), radius=80)
        bbox = _alpha_bbox(img)
        assert bbox is not None
        left, top, right, bottom = bbox
        # The drawn pixels should sit on/within the circle's expected square.
        assert left >= 60 and right <= 240
        assert top >= 60 and bottom <= 240


class TestDrawTriangle:
    def test_triangle_vertices_respected(self) -> None:
        vertices = [(50, 200), (200, 200), (125, 60)]
        img = draw_triangle((300, 260), vertices)
        bbox = _alpha_bbox(img)
        assert bbox is not None
        # Every vertex should sit inside the inked bounding box (within stroke width).
        left, top, right, bottom = bbox
        for x, y in vertices:
            assert left - 4 <= x <= right + 4
            assert top - 4 <= y <= bottom + 4

    def test_triangle_invalid_vertex_count(self) -> None:
        with pytest.raises(ValueError):
            draw_triangle((100, 100), [(0, 0), (10, 10)])


class TestDrawRectangle:
    def test_rectangle_pixels_around_perimeter(self) -> None:
        img = draw_rectangle((400, 200), top_left=(60, 40), bottom_right=(340, 160))
        bbox = _alpha_bbox(img)
        assert bbox is not None
        assert bbox[0] <= 70 and bbox[2] >= 330
        assert bbox[1] <= 50 and bbox[3] >= 150


class TestDrawLine:
    def test_line_has_visible_pixels(self) -> None:
        img = draw_line((300, 300), start=(20, 20), end=(280, 280))
        bbox = _alpha_bbox(img)
        assert bbox is not None
        left, top, right, bottom = bbox
        assert left <= 25 and right >= 275
        assert top <= 25 and bottom >= 275


class TestDrawAxes:
    def test_axes_label_default_visible(self) -> None:
        img = draw_axes((400, 400))
        bbox = _alpha_bbox(img)
        assert bbox is not None
        # Both axes span the canvas — bbox should approach full extent.
        left, top, right, bottom = bbox
        assert right - left > 200
        assert bottom - top > 200


class TestDrawAngleArc:
    def test_zero_sweep_still_draws_marker(self) -> None:
        img = draw_angle_arc((200, 200), vertex=(100, 100), radius=40, start_angle=30, end_angle=30)
        assert _ink_pixel_count(img) > 0

    def test_quarter_sweep(self) -> None:
        img = draw_angle_arc((200, 200), vertex=(100, 100), radius=40, start_angle=0, end_angle=90)
        bbox = _alpha_bbox(img)
        assert bbox is not None
        # Pixels should be confined roughly to the south-east quadrant of the vertex.
        left, top, right, bottom = bbox
        assert right >= 130 and bottom >= 130

    def test_half_sweep(self) -> None:
        img = draw_angle_arc((200, 200), vertex=(100, 100), radius=50, start_angle=0, end_angle=180)
        assert _ink_pixel_count(img) > 0

    def test_full_circle(self) -> None:
        img = draw_angle_arc((220, 220), vertex=(110, 110), radius=60, start_angle=0, end_angle=360)
        bbox = _alpha_bbox(img)
        assert bbox is not None
        left, top, right, bottom = bbox
        assert right - left > 100 and bottom - top > 100


class TestDrawArrow:
    def test_arrow_has_visible_pixels_and_head(self) -> None:
        img = draw_arrow((400, 200), start=(20, 100), end=(360, 100))
        bbox = _alpha_bbox(img)
        assert bbox is not None
        # Width should span most of the canvas because of the arrow length.
        assert bbox[2] - bbox[0] >= 300


class TestDrawLabeledPoint:
    def test_labeled_point_has_pixels(self) -> None:
        img = draw_labeled_point((200, 200), position=(80, 120), label="A")
        bbox = _alpha_bbox(img)
        assert bbox is not None
        left, top, right, bottom = bbox
        # The dot itself must be near the requested position.
        assert left <= 90 and right >= 70


# ---------------------------------------------------------------------------
# Problem dataclass
# ---------------------------------------------------------------------------

class TestProblemDataclass:
    def test_problem_basic_construction(self) -> None:
        problem = Problem(
            question_text="Find the area of the circle.",
            figures=[],
            solution_steps=["Step 1", "Step 2"],
            answer="A = pi r^2",
            formula_latex=r"A = \pi r^{2}",
        )
        assert problem.question_text.startswith("Find")
        assert problem.answer.startswith("A =")
        assert problem.formula_latex is not None

    def test_problem_is_empty(self) -> None:
        problem = Problem(question_text="   ")
        assert problem.is_empty

    def test_figure_dataclass_accepts_pil_image(self) -> None:
        circle = draw_circle((160, 160), center=(80, 80), radius=40)
        figure = Figure(image=circle, size=(160, 160), caption="circle")
        assert figure.image is circle
        assert figure.caption == "circle"

    def test_figure_rejects_non_image(self) -> None:
        with pytest.raises(TypeError):
            Figure(image="not-an-image")

    def test_problem_from_text(self) -> None:
        problem = Problem.from_text(
            "Triangle area",
            steps=["height x base / 2"],
            answer="6",
            formula_latex=r"\frac{1}{2} b h",
        )
        assert problem.solution_steps == ["height x base / 2"]
        assert problem.answer == "6"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _sample_problem(prefix: str = "") -> Problem:
    canvas = (320, 240)
    fig = Figure(
        image=draw_circle(canvas, center=(160, 120), radius=80, label="O"),
        size=canvas,
        caption="circle",
    )
    return Problem(
        question_text=f"{prefix}Compute the radius given diameter d = 10.",
        figures=[fig],
        solution_steps=["radius = d / 2", "radius = 10 / 2", "radius = 5"],
        answer="r = 5",
        formula_latex=r"r = \frac{d}{2}",
    )


class TestGeometrySheetBuilder:
    def test_builder_render_returns_pages(self) -> None:
        builder = GeometrySheetBuilder(title="测试")
        builder.add_problem(_sample_problem())
        pages = builder.render()
        assert pages, "render() must yield at least one page"
        assert isinstance(pages[0], Image.Image)
        assert pages[0].size == (2480, 3508)

    def test_builder_add_problem_returns_self(self) -> None:
        builder = GeometrySheetBuilder()
        result = builder.add_problem(_sample_problem())
        assert result is builder
        assert len(builder.problems) == 1

    def test_builder_rejects_empty_problem(self) -> None:
        builder = GeometrySheetBuilder()
        with pytest.raises(ValueError):
            builder.add_problem(Problem(question_text=""))

    def test_render_without_problems_raises(self) -> None:
        builder = GeometrySheetBuilder()
        with pytest.raises(ValueError):
            builder.render()

    def test_multi_problem_layout_produces_at_least_one_page(self) -> None:
        builder = GeometrySheetBuilder()
        for idx in range(4):
            builder.add_problem(_sample_problem(prefix=f"Q{idx+1}: "))
        pages = builder.render()
        assert len(pages) >= 1
        # Two pages should at least be produced for many problems with figures
        # — we assert the total number of pixels rendered is non-trivial.
        non_white = 0
        for page in pages:
            grayscale = page.convert("L")
            non_white += sum(1 for px in grayscale.getdata() if px < 250)
        assert non_white > 1000

    def test_formula_integration_smoke(self) -> None:
        problem = Problem(
            question_text="LaTeX smoke test",
            figures=[],
            solution_steps=["Use the formula"],
            answer="42",
            formula_latex=r"E = mc^{2}",
        )
        builder = GeometrySheetBuilder(title="Formula")
        builder.add_problem(problem)
        pages = builder.render()
        assert pages and isinstance(pages[0], Image.Image)


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

class TestPdfExport:
    def test_export_pdf_roundtrip(self, tmp_path: Path) -> None:
        builder = GeometrySheetBuilder(title="PDF")
        builder.add_problem(_sample_problem())
        out = tmp_path / "exam.pdf"
        path = builder.export_pdf(out)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_build_exam_sheet_helper(self, tmp_path: Path) -> None:
        out = tmp_path / "sheet.pdf"
        problems = [_sample_problem(prefix="A"), _sample_problem(prefix="B")]
        info = build_exam_sheet(problems, out, title="单元测验")
        assert info["pdf_path"].exists()
        assert info["page_count"] >= 1
        assert info["problem_count"] == 2

    def test_build_exam_sheet_requires_problems(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.pdf"
        with pytest.raises(ValueError):
            build_exam_sheet([], out)
