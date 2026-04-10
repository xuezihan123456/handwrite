from PIL import Image, ImageChops, ImageDraw

from handwrite.composer import compose_page, create_paper


WHITE_PAPER = "\u767d\u7eb8"
RULED_PAPER = "\u6a2a\u7ebf\u7eb8"
GRID_PAPER = "\u65b9\u683c\u7eb8"
MI_PAPER = "\u7c73\u5b57\u683c"
NEAT_LAYOUT = "\u5de5\u6574"


def _make_char_image(size: int = 256) -> Image.Image:
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 176, 176), fill=0)
    return image


def _make_filled_char_image(size: int = 256) -> Image.Image:
    return Image.new("L", (size, size), color=0)


def _make_punctuation_image(size: int = 256) -> Image.Image:
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    draw.ellipse((120, 148, 172, 208), fill=0)
    return image


def _rows_with_ink(image: Image.Image, threshold: int = 240) -> list[int]:
    grayscale = image.convert("L")
    width, height = grayscale.size
    rows: list[int] = []
    for y in range(height):
        for x in range(width):
            if grayscale.getpixel((x, y)) < threshold:
                rows.append(y)
                break
    return rows


def _columns_with_ink(image: Image.Image, threshold: int = 240) -> list[int]:
    grayscale = image.convert("L")
    width, height = grayscale.size
    columns: list[int] = []
    for x in range(width):
        for y in range(height):
            if grayscale.getpixel((x, y)) < threshold:
                columns.append(x)
                break
    return columns


def _region_has_ink(
    image: Image.Image,
    box: tuple[int, int, int, int],
    threshold: int = 32,
) -> bool:
    crop = image.convert("L").crop(box)
    width, height = crop.size
    for y in range(height):
        for x in range(width):
            if crop.getpixel((x, y)) < threshold:
                return True
    return False


def _cell_box(x: int, y: int, font_size: int, inset: int = 4) -> tuple[int, int, int, int]:
    return (
        x + inset,
        y + inset,
        x + font_size - inset,
        y + font_size - inset,
    )


def test_create_paper_supports_current_paper_types() -> None:
    size = (160, 200)
    papers = {
        WHITE_PAPER: create_paper(size, WHITE_PAPER, line_spacing=40),
        RULED_PAPER: create_paper(size, RULED_PAPER, line_spacing=40),
        GRID_PAPER: create_paper(size, GRID_PAPER, line_spacing=40),
        MI_PAPER: create_paper(size, MI_PAPER, line_spacing=40),
    }

    for paper in papers.values():
        assert isinstance(paper, Image.Image)
        assert paper.size == size

    for paper_type in (RULED_PAPER, GRID_PAPER, MI_PAPER):
        extrema = papers[paper_type].convert("L").getextrema()
        assert extrema[0] < extrema[1]

    distinct_pairs = (
        (WHITE_PAPER, RULED_PAPER),
        (WHITE_PAPER, GRID_PAPER),
        (WHITE_PAPER, MI_PAPER),
        (RULED_PAPER, GRID_PAPER),
        (RULED_PAPER, MI_PAPER),
        (GRID_PAPER, MI_PAPER),
    )
    for left, right in distinct_pairs:
        diff = ImageChops.difference(papers[left].convert("L"), papers[right].convert("L"))
        assert diff.getbbox() is not None


def test_compose_page_returns_requested_size() -> None:
    page = compose_page(
        [_make_char_image(), _make_char_image()],
        "AB",
        page_size=(320, 240),
        font_size=48,
        margins=(20, 20, 20, 20),
        paper=WHITE_PAPER,
    )

    assert isinstance(page, Image.Image)
    assert page.size == (320, 240)


def test_compose_page_draws_characters_onto_the_page() -> None:
    page_size = (200, 200)
    page = compose_page(
        [_make_char_image()],
        "A",
        page_size=page_size,
        font_size=48,
        margins=(20, 20, 20, 20),
        paper=WHITE_PAPER,
    )

    background = create_paper(page_size, WHITE_PAPER, line_spacing=48)
    diff = ImageChops.difference(page.convert("L"), background.convert("L"))

    assert diff.getbbox() is not None


def test_compose_page_respects_explicit_newlines() -> None:
    top_margin = 20
    font_size = 40
    page = compose_page(
        [_make_char_image(), _make_char_image()],
        "A\nB",
        page_size=(240, 240),
        font_size=font_size,
        margins=(top_margin, 20, 20, 20),
        paper=WHITE_PAPER,
    )

    rows_with_ink = _rows_with_ink(page)

    assert rows_with_ink
    assert min(rows_with_ink) < top_margin + font_size
    assert max(rows_with_ink) > top_margin + font_size + 5


def test_compose_page_aligns_rows_to_guided_paper_spacing() -> None:
    top_margin = 23
    font_size = 40
    line_height = font_size + max(8, font_size // 4)
    page = compose_page(
        [_make_filled_char_image(), _make_filled_char_image()],
        "A\nB",
        page_size=(240, 240),
        font_size=font_size,
        margins=(top_margin, 20, 20, 20),
        layout=NEAT_LAYOUT,
        paper=RULED_PAPER,
    )

    rows_with_ink = _rows_with_ink(page, threshold=32)
    first_row_top = min(rows_with_ink)
    second_row_top = min(row for row in rows_with_ink if row > first_row_top + font_size)

    assert first_row_top >= top_margin
    assert first_row_top % line_height == 0
    assert second_row_top == first_row_top + line_height


def test_compose_page_aligns_columns_to_grid_paper_spacing() -> None:
    left_margin = 23
    font_size = 40
    line_height = font_size + max(8, font_size // 4)
    page = compose_page(
        [_make_filled_char_image(), _make_filled_char_image()],
        "AB",
        page_size=(240, 180),
        font_size=font_size,
        margins=(20, 20, 20, left_margin),
        layout=NEAT_LAYOUT,
        paper=GRID_PAPER,
    )

    columns_with_ink = _columns_with_ink(page, threshold=32)
    first_column_left = min(columns_with_ink)
    second_column_left = min(column for column in columns_with_ink if column > first_column_left + font_size)

    assert first_column_left >= left_margin
    assert first_column_left % line_height == 0
    assert second_column_left == first_column_left + line_height


def test_compose_page_keeps_wrapped_punctuation_off_line_start() -> None:
    font_size = 40
    line_height = font_size + max(8, font_size // 4)
    char_gap = max(4, font_size // 12)
    left_margin = 20
    top_margin = 20
    second_cell_x = left_margin + font_size + char_gap
    second_line_y = top_margin + line_height
    page = compose_page(
        [_make_filled_char_image(), _make_filled_char_image(), _make_punctuation_image()],
        "AB\uff0c",
        page_size=(140, 180),
        font_size=font_size,
        margins=(top_margin, 20, 20, left_margin),
        layout=NEAT_LAYOUT,
        paper=WHITE_PAPER,
    )

    first_line_second_cell = _cell_box(second_cell_x, top_margin, font_size)
    second_line_first_cell = _cell_box(left_margin, second_line_y, font_size)
    second_line_second_cell = _cell_box(second_cell_x, second_line_y, font_size)

    assert not _region_has_ink(page, first_line_second_cell)
    assert _region_has_ink(page, second_line_first_cell)
    assert _region_has_ink(page, second_line_second_cell)


def test_compose_page_keeps_multi_punctuation_wrap_off_line_start() -> None:
    font_size = 40
    line_height = font_size + max(8, font_size // 4)
    char_gap = max(4, font_size // 12)
    left_margin = 20
    top_margin = 20
    second_cell_x = left_margin + font_size + char_gap
    third_cell_x = second_cell_x + font_size + char_gap
    second_line_y = top_margin + line_height
    page = compose_page(
        [
            _make_filled_char_image(),
            _make_filled_char_image(),
            _make_punctuation_image(),
            _make_punctuation_image(),
        ],
        "AB\uff0c\u3002",
        page_size=(168, 180),
        font_size=font_size,
        margins=(top_margin, 20, 20, left_margin),
        layout=NEAT_LAYOUT,
        paper=WHITE_PAPER,
    )

    first_line_second_cell = _cell_box(second_cell_x, top_margin, font_size)
    second_line_first_cell = _cell_box(left_margin, second_line_y, font_size)
    second_line_second_cell = _cell_box(second_cell_x, second_line_y, font_size)
    second_line_third_cell = _cell_box(third_cell_x, second_line_y, font_size)

    assert not _region_has_ink(page, first_line_second_cell)
    assert _region_has_ink(page, second_line_first_cell)
    assert _region_has_ink(page, second_line_second_cell)
    assert _region_has_ink(page, second_line_third_cell)
