"""ErrorNotebookBuilder — compose error-notebook pages from wrong/correct pairs.

The builder takes a list of :class:`ErrorEntry` items and lays each one out as a
single page that shows the original question, the wrong answer (with red
strike-through overlaying the differing tokens), the correct answer, and a
"反思" (reflection) slot for the student to fill in.  The rendered pages are
``PIL.Image`` objects in RGB mode, ready to be exported to PDF via
:func:`handwrite.exporter.export_pages_pdf`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageFont

from handwrite.composer import NATURAL_LAYOUT, RULED_PAPER, WHITE_PAPER, create_paper
from handwrite.exporter import export_pages_pdf
from handwrite.error_notebook.annotation import (
    red_correction,
    strike_through,
    underline,
)
from handwrite.error_notebook.diff import (
    DIFF_DELETE,
    DIFF_REPLACE,
    DiffSegment,
    diff_answers,
)

__all__ = ["ErrorEntry", "ErrorNotebookBuilder"]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class ErrorEntry:
    """A single mistake captured in the notebook.

    Attributes:
        question: The original question text (LaTeX strings are preserved).
        wrong: The student's wrong answer.
        correct: The intended correct answer.
        scan_image: Optional PIL.Image of the original scan to embed.
        reflection: Free-form reflection text (often empty so the student
            can hand-write their own thoughts later).
    """

    question: str
    wrong: str
    correct: str
    scan_image: Image.Image | None = None
    reflection: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.question, str):
            raise TypeError("question must be a str")
        if not isinstance(self.wrong, str):
            raise TypeError("wrong must be a str")
        if not isinstance(self.correct, str):
            raise TypeError("correct must be a str")
        if not isinstance(self.reflection, str):
            raise TypeError("reflection must be a str")
        if self.scan_image is not None and not isinstance(self.scan_image, Image.Image):
            raise TypeError("scan_image must be a PIL.Image.Image or None")
        if not self.question.strip() and not self.wrong.strip() and not self.correct.strip():
            raise ValueError("ErrorEntry must contain at least a question, wrong, or correct field")


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_A4_SIZE: tuple[int, int] = (2480, 3508)
_DEFAULT_MARGINS = (220, 220, 220, 220)
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
_BLOCK_GAP = 36
_LABEL_GAP = 12
_RED_INK = (200, 30, 30)


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


def _text_height(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str = "Ag") -> int:
    try:
        bbox = font.getbbox(text or "Ag")
        return max(1, int(bbox[3] - bbox[1]))
    except Exception:
        return 24


def _text_width(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> int:
    if not text:
        return 0
    try:
        bbox = font.getbbox(text)
        return max(0, int(bbox[2] - bbox[0]))
    except Exception:
        return int(len(text) * 12)


def _wrap_text(
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> List[str]:
    if not text:
        return [""]
    lines: List[str] = []
    for paragraph in text.splitlines() or [text]:
        current = ""
        for char in paragraph:
            trial = current + char
            if _text_width(font, trial) > max_width and current:
                lines.append(current)
                current = char
            else:
                current = trial
        if current:
            lines.append(current)
    return lines or [""]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ErrorNotebookBuilder:
    """Collect :class:`ErrorEntry` items and render them as paginated images."""

    def __init__(
        self,
        *,
        page_size: tuple[int, int] = _A4_SIZE,
        margins: tuple[int, int, int, int] = _DEFAULT_MARGINS,
        title: str = "错题本",
    ) -> None:
        if len(margins) != 4:
            raise ValueError("margins must contain four integers")
        self._page_size = (int(page_size[0]), int(page_size[1]))
        self._margins = tuple(int(m) for m in margins)  # type: ignore[assignment]
        self._title = title
        self._entries: List[ErrorEntry] = []

    # -- accumulation -------------------------------------------------------

    def add_entry(
        self,
        question: str,
        wrong: str,
        correct: str,
        scan_image: Image.Image | None = None,
        reflection: str = "",
    ) -> "ErrorNotebookBuilder":
        entry = ErrorEntry(
            question=question,
            wrong=wrong,
            correct=correct,
            scan_image=scan_image,
            reflection=reflection,
        )
        self._entries.append(entry)
        return self

    def extend(self, entries: Iterable[ErrorEntry | dict]) -> "ErrorNotebookBuilder":
        for item in entries:
            if isinstance(item, ErrorEntry):
                self._entries.append(item)
            elif isinstance(item, dict):
                self._entries.append(_coerce_dict_entry(item))
            else:
                raise TypeError("entries must be ErrorEntry or dict instances")
        return self

    @property
    def entries(self) -> tuple[ErrorEntry, ...]:
        return tuple(self._entries)

    # -- rendering ----------------------------------------------------------

    def render(
        self,
        style: str = "工整楷书",
        paper: str = RULED_PAPER,
        layout: str = NATURAL_LAYOUT,
        font_size: int = 64,
    ) -> List[Image.Image]:
        if not self._entries:
            raise ValueError("no entries registered; add at least one before rendering")
        pages: List[Image.Image] = []
        for index, entry in enumerate(self._entries, start=1):
            pages.append(
                self._render_entry_page(
                    index=index,
                    entry=entry,
                    style=style,
                    paper=paper,
                    layout=layout,
                    font_size=font_size,
                )
            )
        return pages

    def export_pdf(
        self,
        output_path: str | Path,
        style: str = "工整楷书",
        paper: str = RULED_PAPER,
        layout: str = NATURAL_LAYOUT,
        font_size: int = 64,
        dpi: int = 300,
    ) -> dict[str, object]:
        pages = self.render(style=style, paper=paper, layout=layout, font_size=font_size)
        pdf_path = export_pages_pdf(pages, output_path, dpi=dpi)
        return {
            "pdf_path": Path(pdf_path),
            "page_count": len(pages),
            "entry_count": len(self._entries),
        }

    # -- single-entry page --------------------------------------------------

    def _render_entry_page(
        self,
        *,
        index: int,
        entry: ErrorEntry,
        style: str,
        paper: str,
        layout: str,
        font_size: int,
    ) -> Image.Image:
        # Background paper.
        page = create_paper(self._page_size, paper, line_spacing=max(40, font_size)).convert("RGB")
        draw = ImageDraw.Draw(page)

        top_margin, right_margin, bottom_margin, left_margin = self._margins
        content_width = self._page_size[0] - left_margin - right_margin

        # Header.
        header_font = _load_font(max(48, font_size))
        label_font = _load_font(max(32, int(font_size * 0.55)))
        body_font = _load_font(max(28, int(font_size * 0.5)))

        header_text = f"{self._title} · 第{index}题"
        draw.text(
            (left_margin, top_margin),
            header_text,
            fill=(20, 20, 20),
            font=header_font,
        )
        cursor_y = top_margin + _text_height(header_font, header_text) + 24
        draw.line(
            [
                (left_margin, cursor_y),
                (self._page_size[0] - right_margin, cursor_y),
            ],
            fill=(80, 80, 80),
            width=2,
        )
        cursor_y += 18

        # Question section.
        cursor_y = self._draw_labeled_block(
            page,
            draw,
            label="题目:",
            text=entry.question or "(空题面)",
            label_font=label_font,
            body_font=body_font,
            content_width=content_width,
            origin=(left_margin, cursor_y),
            body_color=(20, 20, 20),
        )
        cursor_y += _BLOCK_GAP

        # Wrong answer with strike-through annotation over differing tokens.
        cursor_y, wrong_block_bbox = self._draw_block_with_bbox(
            page,
            draw,
            label="错答:",
            text=entry.wrong or "(空)",
            label_font=label_font,
            body_font=body_font,
            content_width=content_width,
            origin=(left_margin, cursor_y),
            body_color=(60, 60, 60),
        )

        # Apply strike-through to portion of the wrong-answer block matching
        # the diff segments flagged as DELETE or REPLACE.
        if wrong_block_bbox is not None and entry.wrong:
            page = self._annotate_wrong_block(
                page,
                bbox=wrong_block_bbox,
                wrong=entry.wrong,
                correct=entry.correct,
            )
            draw = ImageDraw.Draw(page)

        cursor_y += _BLOCK_GAP

        # Correct answer.
        cursor_y = self._draw_labeled_block(
            page,
            draw,
            label="正解:",
            text=entry.correct or "(空)",
            label_font=label_font,
            body_font=body_font,
            content_width=content_width,
            origin=(left_margin, cursor_y),
            body_color=(20, 80, 30),
        )
        cursor_y += _BLOCK_GAP

        # Reflection slot — always show the label even when empty so students
        # have a place to hand-write their thoughts.
        cursor_y = self._draw_labeled_block(
            page,
            draw,
            label="反思:",
            text=entry.reflection or "",
            label_font=label_font,
            body_font=body_font,
            content_width=content_width,
            origin=(left_margin, cursor_y),
            body_color=(40, 40, 40),
        )

        # Embed the scan image, if provided, in the remaining space below.
        if entry.scan_image is not None:
            cursor_y = self._embed_scan(page, entry.scan_image, cursor_y, content_width)

        return page

    # -- helpers ------------------------------------------------------------

    def _draw_labeled_block(
        self,
        page: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        label: str,
        text: str,
        label_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        body_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        content_width: int,
        origin: tuple[int, int],
        body_color: tuple[int, int, int],
    ) -> int:
        cursor_y, _ = self._draw_block_with_bbox(
            page,
            draw,
            label=label,
            text=text,
            label_font=label_font,
            body_font=body_font,
            content_width=content_width,
            origin=origin,
            body_color=body_color,
        )
        return cursor_y

    def _draw_block_with_bbox(
        self,
        page: Image.Image,
        draw: ImageDraw.ImageDraw,
        *,
        label: str,
        text: str,
        label_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        body_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        content_width: int,
        origin: tuple[int, int],
        body_color: tuple[int, int, int],
    ) -> tuple[int, tuple[int, int, int, int] | None]:
        x, y = origin
        page_bottom = self._page_size[1] - self._margins[2]

        # Label line.
        draw.text((x, y), label, fill=(40, 40, 40), font=label_font)
        y += _text_height(label_font, label) + _LABEL_GAP

        # Body wrap.
        body_lines = _wrap_text(text, font=body_font, max_width=content_width)
        line_h = _text_height(body_font, "Ag") + 8

        block_top = y
        first_line_y: int | None = None
        last_line_bottom = y
        block_left = x
        block_right = x

        for line in body_lines:
            if y + line_h > page_bottom:
                break
            if first_line_y is None and line.strip():
                first_line_y = y
            draw.text((x, y), line, fill=body_color, font=body_font)
            last_line_bottom = y + line_h
            block_right = max(block_right, x + _text_width(body_font, line))
            y = last_line_bottom

        # Even if the body was empty, advance a little so the next block has air.
        if not body_lines or all(not line for line in body_lines):
            y = block_top + line_h

        bbox: tuple[int, int, int, int] | None = None
        if first_line_y is not None and block_right > block_left:
            bbox = (block_left, first_line_y, block_right, last_line_bottom)

        return y, bbox

    def _annotate_wrong_block(
        self,
        page: Image.Image,
        *,
        bbox: tuple[int, int, int, int],
        wrong: str,
        correct: str,
    ) -> Image.Image:
        """Strike through the section of the wrong-answer block.

        We use the diff API to decide whether any token is wrong; if so, we
        cross the entire wrong-answer block.  This keeps the visual focus
        clear without re-implementing a full character-level mapping.
        """
        segments: Sequence[DiffSegment] = diff_answers(wrong, correct)
        has_difference = any(
            seg.kind in {DIFF_DELETE, DIFF_REPLACE} for seg in segments
        )
        if not has_difference:
            return page
        return strike_through(page, bbox, color=(*_RED_INK, 210))

    def _embed_scan(
        self,
        page: Image.Image,
        scan_image: Image.Image,
        cursor_y: int,
        content_width: int,
    ) -> int:
        top_margin, right_margin, bottom_margin, left_margin = self._margins
        page_bottom = self._page_size[1] - bottom_margin
        available_h = page_bottom - cursor_y
        if available_h <= 64 or content_width <= 64:
            return cursor_y

        scan = scan_image.convert("RGBA")
        target_w = min(content_width, scan.width)
        ratio = target_w / max(1, scan.width)
        target_h = min(available_h, int(scan.height * ratio))
        if target_h <= 0:
            return cursor_y
        scan_resized = scan.resize((target_w, target_h))
        page.paste(scan_resized, (left_margin, cursor_y), scan_resized)
        return cursor_y + target_h


# ---------------------------------------------------------------------------
# Helpers used by the dict-entry path
# ---------------------------------------------------------------------------


def _coerce_dict_entry(item: dict) -> ErrorEntry:
    try:
        return ErrorEntry(
            question=str(item["question"]),
            wrong=str(item["wrong"]),
            correct=str(item["correct"]),
            scan_image=item.get("scan_image"),
            reflection=str(item.get("reflection", "")),
        )
    except KeyError as missing:
        raise ValueError(
            f"entry dict missing required key: {missing.args[0]!r}"
        ) from missing
