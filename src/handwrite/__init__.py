"""Public API for the HandWrite package."""

from PIL import Image

from handwrite.composer import (
    CURSIVE_LAYOUT,
    GRID_PAPER,
    MI_PAPER,
    NATURAL_LAYOUT,
    NEAT_LAYOUT,
    RULED_PAPER,
    WHITE_PAPER,
    compose_page,
)
from handwrite.engine.model import StyleEngine
from handwrite.styles import BUILTIN_STYLES, list_style_names

_ENGINE: StyleEngine | None = None
_DEFAULT_STYLE = list_style_names()[0]
_DEFAULT_PAGE_SIZE = (2480, 3508)
_DEFAULT_MARGINS = (200, 200, 200, 200)
_GUIDED_PAPERS = {RULED_PAPER, GRID_PAPER, MI_PAPER}
_GRID_PAPERS = {GRID_PAPER, MI_PAPER}
_LEADING_PUNCTUATION = set(
    "\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001,.!?;:)]}"
    "\u3011\uff09\u300b\u300d\u300f\u2019\u201d"
)


def _get_engine() -> StyleEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = StyleEngine()
    return _ENGINE


def list_styles() -> list[str]:
    """Return the built-in handwriting styles."""
    return list_style_names()


def char(text: str, style: str = _DEFAULT_STYLE) -> Image.Image:
    """Generate a single handwriting character image."""
    style_id = BUILTIN_STYLES[style]
    return _get_engine().generate_char(text, style_id)


def generate(
    text: str,
    style: str = _DEFAULT_STYLE,
    paper: str = WHITE_PAPER,
    layout: str = NATURAL_LAYOUT,
    font_size: int = 80,
) -> Image.Image:
    """Generate a single handwriting page image."""
    images = [char(c, style=style) for c in text if c not in {" ", "\n"}]
    return compose_page(images, text, font_size=font_size, layout=layout, paper=paper)


def generate_pages(text: str, **kwargs) -> list[Image.Image]:
    """Generate one or more handwriting page images for longer text."""
    font_size = int(kwargs.get("font_size", 80))
    layout = kwargs.get("layout", NATURAL_LAYOUT)
    paper = kwargs.get("paper", WHITE_PAPER)
    page_chunks = _split_text_into_pages(
        text,
        font_size=font_size,
        layout=layout,
        paper=paper,
    )
    return [generate(chunk, **kwargs) for chunk in page_chunks]


def _split_text_into_pages(
    text: str,
    *,
    font_size: int,
    layout: str,
    paper: str,
) -> list[str]:
    if font_size <= 0:
        raise ValueError("font_size must be positive")

    max_columns, max_rows = _page_grid(font_size=font_size, layout=layout, paper=paper)
    if max_columns < 1 or max_rows < 1:
        raise ValueError(
            f"font_size={font_size} does not fit within page_size={_DEFAULT_PAGE_SIZE} "
            f"and margins={_DEFAULT_MARGINS}"
        )

    lines = _wrap_text_to_lines(text, max_columns=max_columns)

    return [
        _lines_to_text(lines[index : index + max_rows])
        for index in range(0, len(lines), max_rows)
    ] or [""]


def _page_grid(*, font_size: int, layout: str, paper: str) -> tuple[int, int]:
    top, right, bottom, left = _DEFAULT_MARGINS
    page_width, page_height = _DEFAULT_PAGE_SIZE
    char_gap, line_gap = _estimated_spacing(font_size, layout)
    line_height = font_size + line_gap
    column_step = line_height if paper in _GRID_PAPERS else font_size + char_gap
    first_column_x = _aligned_origin(left, column_step) if paper in _GRID_PAPERS else left
    first_row_y = _aligned_origin(top, line_height) if paper in _GUIDED_PAPERS else top

    max_columns = _count_slots(
        total_extent=page_width,
        start=first_column_x,
        end_margin=right,
        item_extent=font_size,
        step=column_step,
    )
    max_rows = _count_slots(
        total_extent=page_height,
        start=first_row_y,
        end_margin=bottom,
        item_extent=font_size,
        step=line_height,
    )
    return max_columns, max_rows


def _estimated_spacing(font_size: int, layout: str) -> tuple[int, int]:
    if layout == NEAT_LAYOUT:
        return max(4, font_size // 12), max(8, font_size // 4)
    if layout == CURSIVE_LAYOUT:
        return max(8, font_size // 8), max(14, font_size // 3)
    return max(6, font_size // 10), max(12, font_size // 3)


def _aligned_origin(margin: int, step: int) -> int:
    return ((margin + step - 1) // step) * step


def _count_slots(
    *,
    total_extent: int,
    start: int,
    end_margin: int,
    item_extent: int,
    step: int,
) -> int:
    available_extent = total_extent - end_margin - start
    if available_extent < item_extent:
        return 0
    return 1 + max(0, (available_extent - item_extent) // step)


def _wrap_text_to_lines(text: str, *, max_columns: int) -> list[str]:
    lines: list[list[str]] = [[]]

    for symbol in text:
        current_line = lines[-1]

        if symbol == "\n":
            _trim_trailing_spaces(current_line)
            lines.append([])
            continue

        if symbol.isspace() and not current_line:
            continue

        if len(current_line) >= max_columns:
            _trim_trailing_spaces(current_line)

            if len(current_line) < max_columns:
                current_line = lines[-1]
            else:
                if symbol.isspace():
                    lines.append([])
                    continue

                if _is_leading_punctuation(symbol) and max_columns > 1:
                    moved = _pop_reflow_suffix(current_line, max_columns)
                    if moved is not None:
                        _trim_trailing_spaces(current_line)
                        lines.append(moved + [symbol])
                        continue

                lines.append([])
                current_line = lines[-1]

        if symbol.isspace() and not current_line:
            continue

        current_line.append(symbol)

    for line in lines:
        _trim_trailing_spaces(line)

    return ["".join(line) for line in lines]


def _is_leading_punctuation(symbol: str) -> bool:
    return symbol in _LEADING_PUNCTUATION


def _pop_reflow_suffix(line: list[str], max_columns: int) -> list[str] | None:
    if not line:
        return None

    split_index = len(line) - 1
    while split_index >= 0 and _is_leading_punctuation(line[split_index]):
        split_index -= 1

    if split_index < 0 or line[split_index].isspace():
        return None

    moved = line[split_index:]
    if len(moved) + 1 > max_columns:
        return None

    del line[split_index:]
    return moved


def _trim_trailing_spaces(line: list[str]) -> None:
    while line and line[-1].isspace():
        line.pop()


def _lines_to_text(lines: list[str]) -> str:
    return "\n".join(lines)
