"""Exam-sheet builder that composes problems onto A4 pages."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from handwrite.composer import WHITE_PAPER, create_paper
from handwrite.exporter import export_pages_pdf
from handwrite.geometry_sheet.problem import Figure, Problem

# ---------------------------------------------------------------------------
# Page geometry — A4 @ 300dpi
# ---------------------------------------------------------------------------

A4_SIZE: tuple[int, int] = (2480, 3508)
_DEFAULT_MARGINS = (220, 220, 220, 220)  # top, right, bottom, left
_PROBLEM_GAP = 80
_FIGURE_PADDING = 24
_QUESTION_FONT_SIZE = 44
_BODY_FONT_SIZE = 38
_HEADER_FONT_SIZE = 64
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    cached = _FONT_CACHE.get(size)
    if cached is not None:
        return cached
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _text_height(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> int:
    """Return a rough rendered-height estimate for the given text."""
    try:
        bbox = font.getbbox(text or "Ag")
        return max(1, int(bbox[3] - bbox[1]))
    except Exception:
        return 24


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class GeometrySheetBuilder:
    """Compose exam-sheet pages combining geometry, formulas, and handwriting."""

    def __init__(
        self,
        *,
        page_size: tuple[int, int] = A4_SIZE,
        margins: tuple[int, int, int, int] = _DEFAULT_MARGINS,
        paper: str = WHITE_PAPER,
        style: str = "工整楷书",
        title: str = "数学几何练习",
    ) -> None:
        self._page_size = (int(page_size[0]), int(page_size[1]))
        self._margins = tuple(int(m) for m in margins)  # type: ignore[assignment]
        if len(self._margins) != 4:
            raise ValueError("margins must have four elements")
        self._paper = paper
        self._style = style
        self._title = title
        self._problems: list[Problem] = []

    # -- accumulation -------------------------------------------------------

    def add_problem(self, problem: Problem) -> "GeometrySheetBuilder":
        if not isinstance(problem, Problem):
            raise TypeError("problem must be a Problem instance")
        if problem.is_empty:
            raise ValueError("cannot add an empty problem to the exam sheet")
        self._problems.append(problem)
        return self

    def extend(self, problems: Iterable[Problem]) -> "GeometrySheetBuilder":
        for p in problems:
            self.add_problem(p)
        return self

    @property
    def problems(self) -> tuple[Problem, ...]:
        return tuple(self._problems)

    # -- rendering ----------------------------------------------------------

    def render(self) -> list[Image.Image]:
        if not self._problems:
            raise ValueError("no problems registered; add at least one before rendering")
        blocks = [self._render_problem_block(index, problem)
                  for index, problem in enumerate(self._problems, start=1)]
        return list(self._paginate(blocks))

    def export_pdf(self, output_path: str | Path, dpi: int = 300) -> Path:
        pages = self.render()
        return export_pages_pdf(pages, output_path, dpi=dpi)

    # -- internals ----------------------------------------------------------

    def _new_page(self) -> Image.Image:
        return create_paper(self._page_size, self._paper).convert("RGB")

    def _content_width(self) -> int:
        top, right, bottom, left = self._margins
        return self._page_size[0] - left - right

    def _content_top(self) -> int:
        return self._margins[0]

    def _content_bottom(self) -> int:
        return self._page_size[1] - self._margins[2]

    def _content_left(self) -> int:
        return self._margins[3]

    def _paginate(self, blocks: Sequence[Image.Image]) -> list[Image.Image]:
        pages: list[Image.Image] = []
        page = self._new_page()
        self._draw_page_header(page, len(pages) + 1)

        cursor_y = self._content_top() + _HEADER_FONT_SIZE + 32
        x = self._content_left()
        max_y = self._content_bottom()

        for block in blocks:
            block_h = block.size[1]
            if cursor_y + block_h > max_y and cursor_y > self._content_top() + _HEADER_FONT_SIZE + 32:
                pages.append(page)
                page = self._new_page()
                self._draw_page_header(page, len(pages) + 1)
                cursor_y = self._content_top() + _HEADER_FONT_SIZE + 32

            page.paste(block, (x, cursor_y), block if block.mode == "RGBA" else None)
            cursor_y += block_h + _PROBLEM_GAP

        pages.append(page)
        return pages

    def _draw_page_header(self, page: Image.Image, page_number: int) -> None:
        draw = ImageDraw.Draw(page)
        font = _load_font(_HEADER_FONT_SIZE)
        header = f"{self._title}  -  Page {page_number}"
        draw.text((self._content_left(), self._content_top()), header, fill=(20, 20, 20), font=font)
        # Underline.
        underline_y = self._content_top() + _HEADER_FONT_SIZE + 8
        draw.line(
            [(self._content_left(), underline_y),
             (self._content_left() + self._content_width(), underline_y)],
            fill=(60, 60, 60),
            width=2,
        )

    # -- problem block ------------------------------------------------------

    def _render_problem_block(self, index: int, problem: Problem) -> Image.Image:
        content_w = self._content_width()
        question_font = _load_font(_QUESTION_FONT_SIZE)
        body_font = _load_font(_BODY_FONT_SIZE)

        # Wrap question text.
        question_lines = self._wrap_text(
            f"{index}. {problem.question_text}",
            font=question_font,
            max_width=content_w,
        )
        question_line_h = _text_height(question_font, "Ag") + 12

        figure_block = self._compose_figure_strip(problem.figures, content_w)
        formula_image = self._render_formula(problem.formula_latex)
        step_lines: list[str] = []
        for raw_step in problem.solution_steps:
            for line in self._wrap_text(raw_step, font=body_font, max_width=content_w - 40):
                step_lines.append(line)
        body_line_h = _text_height(body_font, "Ag") + 10
        answer_line_h = body_line_h

        # Compute block height.
        block_h = len(question_lines) * question_line_h + 8
        if figure_block is not None:
            block_h += figure_block.size[1] + 16
        if formula_image is not None:
            block_h += formula_image.size[1] + 12
        if step_lines:
            block_h += body_line_h + 8  # "解答:" header
            block_h += len(step_lines) * body_line_h
        if problem.answer.strip():
            block_h += answer_line_h + 8

        block_h = max(64, block_h)
        block = Image.new("RGBA", (content_w, block_h), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(block)
        y = 0
        for line in question_lines:
            draw.text((0, y), line, fill=(20, 20, 20, 255), font=question_font)
            y += question_line_h
        y += 8

        if figure_block is not None:
            fx = (content_w - figure_block.size[0]) // 2
            block.paste(figure_block, (max(0, fx), y), figure_block)
            y += figure_block.size[1] + 16

        if formula_image is not None:
            fx = (content_w - formula_image.size[0]) // 2
            paste = formula_image.convert("RGBA")
            block.paste(paste, (max(0, fx), y), paste)
            y += formula_image.size[1] + 12

        if step_lines:
            draw.text((0, y), "解答:", fill=(40, 40, 40, 255), font=body_font)
            y += body_line_h + 4
            for line in step_lines:
                draw.text((40, y), line, fill=(40, 40, 40, 255), font=body_font)
                y += body_line_h

        if problem.answer.strip():
            y += 4
            draw.text((0, y), f"答: {problem.answer.strip()}", fill=(10, 10, 60, 255), font=body_font)

        return block

    # -- figure strip --------------------------------------------------------

    def _compose_figure_strip(
        self,
        figures: Sequence[Figure],
        max_width: int,
    ) -> Image.Image | None:
        if not figures:
            return None

        caption_font = _load_font(22)
        caption_h = _text_height(caption_font, "Ag") + 6

        # Lay figures out side-by-side, wrapping when they exceed ``max_width``.
        rows: list[list[Figure]] = [[]]
        widths: list[int] = [0]
        for fig in figures:
            fig_w = fig.image.size[0]
            new_width = widths[-1] + fig_w + (_FIGURE_PADDING if rows[-1] else 0)
            if rows[-1] and new_width > max_width:
                rows.append([fig])
                widths.append(fig_w)
            else:
                rows[-1].append(fig)
                widths[-1] = new_width

        row_heights = []
        for row in rows:
            row_h = 0
            for fig in row:
                row_h = max(row_h, fig.image.size[1] + (caption_h if fig.caption else 0))
            row_heights.append(row_h)

        total_w = max(max_width, max(widths) if widths else max_width)
        total_h = sum(row_heights) + _FIGURE_PADDING * (len(rows) - 1)
        if total_h <= 0:
            return None

        strip = Image.new("RGBA", (total_w, total_h), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(strip)

        y = 0
        for row, row_h in zip(rows, row_heights):
            row_width = sum(f.image.size[0] for f in row) + _FIGURE_PADDING * (len(row) - 1)
            x = max(0, (total_w - row_width) // 2)
            for fig in row:
                figure_img = fig.image.convert("RGBA")
                strip.paste(figure_img, (x, y), figure_img)
                if fig.caption:
                    draw.text(
                        (x, y + figure_img.size[1] + 2),
                        fig.caption,
                        fill=(40, 40, 40, 255),
                        font=caption_font,
                    )
                x += figure_img.size[0] + _FIGURE_PADDING
            y += row_h + _FIGURE_PADDING

        return strip

    # -- formula -------------------------------------------------------------

    def _render_formula(self, latex: str | None) -> Image.Image | None:
        if not latex or not latex.strip():
            return None
        try:
            from handwrite.formula import FormulaConfig, render_latex_formula
        except Exception:
            return None
        try:
            return render_latex_formula(latex, FormulaConfig(font_size=42, seed=42))
        except Exception:
            return None

    # -- text wrapping -------------------------------------------------------

    def _wrap_text(
        self,
        text: str,
        *,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
    ) -> list[str]:
        if not text:
            return [""]
        result: list[str] = []
        current = ""
        for paragraph in text.splitlines() or [text]:
            current = ""
            for char in paragraph:
                trial = current + char
                w = self._measure_width(font, trial)
                if w > max_width and current:
                    result.append(current)
                    current = char
                else:
                    current = trial
            if current:
                result.append(current)
                current = ""
        return result or [""]

    def _measure_width(
        self,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        text: str,
    ) -> int:
        try:
            bbox = font.getbbox(text)
            return int(bbox[2] - bbox[0])
        except Exception:
            return int(len(text) * 18)


# ---------------------------------------------------------------------------
# Module helper
# ---------------------------------------------------------------------------

def build_exam_sheet(
    problems: Sequence[Problem],
    output_path: str | Path,
    *,
    style: str = "工整楷书",
    paper: str = WHITE_PAPER,
    title: str = "数学几何练习",
) -> dict[str, object]:
    """Build an exam sheet and write it as a multi-page PDF.

    Returns a dict with ``pdf_path``, ``page_count``, and ``problem_count``.
    """
    if not problems:
        raise ValueError("problems must contain at least one Problem")

    builder = GeometrySheetBuilder(style=style, paper=paper, title=title)
    for problem in problems:
        builder.add_problem(problem)

    pages = builder.render()
    pdf_path = export_pages_pdf(pages, output_path)
    return {
        "pdf_path": Path(pdf_path),
        "page_count": len(pages),
        "problem_count": len(builder.problems),
    }
