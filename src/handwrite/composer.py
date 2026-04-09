"""Page composition utilities."""

from __future__ import annotations

from PIL import Image, ImageChops, ImageDraw, ImageOps

WHITE_PAPER = "\u767d\u7eb8"
RULED_PAPER = "\u6a2a\u7ebf\u7eb8"
GRID_PAPER = "\u65b9\u683c\u7eb8"
MI_PAPER = "\u7c73\u5b57\u683c"

NATURAL_LAYOUT = "\u81ea\u7136"
NEAT_LAYOUT = "\u5de5\u6574"
CURSIVE_LAYOUT = "\u6f47\u8349"

_SUPPORTED_PAPERS = {WHITE_PAPER, RULED_PAPER, GRID_PAPER, MI_PAPER}
_LINE_COLOR = 214
_DIAGONAL_COLOR = 228
_LEADING_PUNCTUATION = set(
    "\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001,.!?;:)]}"
    "\u3011\uff09\u300b\u300d\u300f\u2019\u201d"
)
_GRID_PAPERS = {GRID_PAPER, MI_PAPER}


def create_paper(
    size: tuple[int, int],
    paper_type: str,
    line_spacing: int = 80,
) -> Image.Image:
    """Create a simple grayscale paper background."""
    if paper_type not in _SUPPORTED_PAPERS:
        raise ValueError(f"Unsupported paper type: {paper_type}")

    width, height = size
    spacing = max(8, int(line_spacing))
    paper = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(paper)

    if paper_type in {RULED_PAPER, GRID_PAPER, MI_PAPER}:
        for y in range(0, height, spacing):
            draw.line((0, y, width, y), fill=_LINE_COLOR, width=1)

    if paper_type in {GRID_PAPER, MI_PAPER}:
        for x in range(0, width, spacing):
            draw.line((x, 0, x, height), fill=_LINE_COLOR, width=1)

    if paper_type == MI_PAPER:
        for top in range(0, height, spacing):
            bottom = min(top + spacing, height - 1)
            for left in range(0, width, spacing):
                right = min(left + spacing, width - 1)
                draw.line((left, top, right, bottom), fill=_DIAGONAL_COLOR, width=1)
                draw.line((right, top, left, bottom), fill=_DIAGONAL_COLOR, width=1)

    return paper


def compose_page(
    chars: list[Image.Image],
    text: str,
    page_size: tuple[int, int] = (2480, 3508),
    font_size: int = 80,
    margins: tuple[int, int, int, int] = (200, 200, 200, 200),
    layout: str = NATURAL_LAYOUT,
    paper: str = WHITE_PAPER,
) -> Image.Image:
    """Compose character images onto a paper background in reading order."""
    top, right, bottom, left = margins
    char_gap, line_gap = _layout_spacing(font_size, layout)
    line_height = font_size + line_gap
    paper_spacing = line_height if paper != WHITE_PAPER else font_size
    page = create_paper(page_size, paper, line_spacing=paper_spacing)
    page_width, page_height = page.size
    column_step = _column_step(font_size, char_gap, line_height, paper)
    first_column_x = _first_column_x(left, column_step, paper)
    max_columns = _max_columns(page_width, first_column_x, right, font_size, column_step)
    lines = _layout_lines(_build_tokens(chars, text), max_columns)
    y = _first_row_y(top, line_height, paper)

    for line in lines:
        if y + font_size > page_height - bottom:
            break

        x = first_column_x
        for symbol, image in line:
            if x + font_size > page_width - right:
                break

            if image is not None and not symbol.isspace():
                _paste_char(page, image, (x, y), font_size)

            x += column_step

        y += line_height

    return page


def _layout_spacing(font_size: int, layout: str) -> tuple[int, int]:
    if layout == NEAT_LAYOUT:
        return max(4, font_size // 12), max(8, font_size // 4)
    if layout == CURSIVE_LAYOUT:
        return max(8, font_size // 8), max(14, font_size // 3)
    return max(6, font_size // 10), max(12, font_size // 3)


def _build_tokens(
    chars: list[Image.Image],
    text: str,
) -> list[tuple[str, Image.Image | None]]:
    tokens: list[tuple[str, Image.Image | None]] = []
    char_index = 0

    for symbol in text:
        if symbol == "\n":
            tokens.append((symbol, None))
            continue

        if symbol.isspace():
            tokens.append((symbol, None))
            continue

        if char_index >= len(chars):
            break

        tokens.append((symbol, chars[char_index]))
        char_index += 1

    return tokens


def _layout_lines(
    tokens: list[tuple[str, Image.Image | None]],
    max_columns: int,
) -> list[list[tuple[str, Image.Image | None]]]:
    lines: list[list[tuple[str, Image.Image | None]]] = [[]]

    for token in tokens:
        symbol, _ = token
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
                        lines.append(moved + [token])
                        continue

                lines.append([])
                current_line = lines[-1]

        if symbol.isspace() and not current_line:
            continue

        current_line.append(token)

    for line in lines:
        _trim_trailing_spaces(line)

    return lines


def _max_columns(
    page_width: int,
    first_column_x: int,
    right: int,
    font_size: int,
    column_step: int,
) -> int:
    available_width = page_width - right - first_column_x
    if available_width < font_size:
        return 1
    return 1 + max(0, (available_width - font_size) // column_step)


def _first_row_y(top: int, line_height: int, paper: str) -> int:
    if paper == WHITE_PAPER:
        return top
    return ((top + line_height - 1) // line_height) * line_height


def _column_step(font_size: int, char_gap: int, line_height: int, paper: str) -> int:
    if paper in _GRID_PAPERS:
        return line_height
    return font_size + char_gap


def _first_column_x(left: int, column_step: int, paper: str) -> int:
    if paper not in _GRID_PAPERS:
        return left
    return ((left + column_step - 1) // column_step) * column_step


def _is_leading_punctuation(symbol: str) -> bool:
    return symbol in _LEADING_PUNCTUATION


def _pop_reflow_suffix(
    line: list[tuple[str, Image.Image | None]],
    max_columns: int,
) -> list[tuple[str, Image.Image | None]] | None:
    if not line:
        return None

    split_index = len(line) - 1
    while split_index >= 0 and _is_leading_punctuation(line[split_index][0]):
        split_index -= 1

    if split_index < 0 or line[split_index][0].isspace():
        return None

    moved = line[split_index:]
    if len(moved) + 1 > max_columns:
        return None

    del line[split_index:]
    return moved


def _trim_trailing_spaces(line: list[tuple[str, Image.Image | None]]) -> None:
    while line and line[-1][0].isspace():
        line.pop()


def _paste_char(
    page: Image.Image,
    char_image: Image.Image,
    origin: tuple[int, int],
    font_size: int,
) -> None:
    glyph_mask = _prepare_char_mask(char_image, font_size)
    if glyph_mask.getbbox() is None:
        return
    page.paste(0, (origin[0], origin[1]), glyph_mask)


def _prepare_char_mask(char_image: Image.Image, font_size: int) -> Image.Image:
    rgba = char_image.convert("RGBA")
    grayscale = ImageOps.grayscale(rgba)
    alpha = rgba.getchannel("A")
    glyph = ImageChops.multiply(ImageOps.invert(grayscale), alpha)
    bbox = glyph.getbbox()
    if bbox is None:
        return Image.new("L", (font_size, font_size), color=0)

    cropped = glyph.crop(bbox)
    resized = ImageOps.contain(
        cropped,
        (font_size, font_size),
        Image.Resampling.LANCZOS,
    )
    mask = Image.new("L", (font_size, font_size), color=0)
    x_offset = (font_size - resized.width) // 2
    y_offset = (font_size - resized.height) // 2
    mask.paste(resized, (x_offset, y_offset))
    return mask
