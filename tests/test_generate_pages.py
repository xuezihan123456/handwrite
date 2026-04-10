from PIL import Image, ImageChops
import pytest

import handwrite
from handwrite.composer import WHITE_PAPER, _build_tokens, _layout_lines, create_paper


def _page_has_visible_content(page: Image.Image) -> bool:
    background = create_paper(page.size, WHITE_PAPER, line_spacing=80)
    diff = ImageChops.difference(page.convert("L"), background.convert("L"))
    return diff.getbbox() is not None


def _renderer_expected_page_chunks(text: str, *, font_size: int) -> list[str]:
    max_columns, max_rows = handwrite._page_grid(
        font_size=font_size,
        layout=handwrite.NATURAL_LAYOUT,
        paper=WHITE_PAPER,
    )
    chars = [Image.new("L", (1, 1), color=255) for symbol in text if symbol not in {" ", "\n"}]
    lines = [
        "".join(symbol for symbol, _ in line)
        for line in _layout_lines(_build_tokens(chars, text), max_columns)
    ]
    return [
        handwrite._lines_to_text(lines[index : index + max_rows])
        for index in range(0, len(lines), max_rows)
    ] or [""]


def test_generate_pages_returns_one_page_for_short_text(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    stub_page = Image.new("L", (32, 32), color=255)

    def fake_generate(text: str, **kwargs) -> Image.Image:
        calls.append((text, kwargs))
        return stub_page

    monkeypatch.setattr(handwrite, "generate", fake_generate)

    pages = handwrite.generate_pages("short note")

    assert pages == [stub_page]
    assert calls == [("short note", {})]


def test_generate_pages_splits_longer_text_into_multiple_pages(monkeypatch) -> None:
    calls: list[str] = []

    def fake_generate(text: str, **kwargs) -> Image.Image:
        calls.append(text)
        return Image.new("L", (32, 32), color=255)

    monkeypatch.setattr(handwrite, "generate", fake_generate)

    pages = handwrite.generate_pages("abcdefghijklmnopqrstuvwxyz1234", font_size=400)

    assert len(pages) > 1
    assert len(calls) == len(pages)


def test_generate_pages_forwards_kwargs_to_generate(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_generate(
        text: str,
        *,
        style: str = "default-style",
        paper: str = "plain-paper",
        layout: str = "natural-layout",
        font_size: int = 80,
    ) -> Image.Image:
        calls.append(
            {
                "text": text,
                "style": style,
                "paper": paper,
                "layout": layout,
                "font_size": font_size,
            }
        )
        return Image.new("L", (32, 32), color=255)

    monkeypatch.setattr(handwrite, "generate", fake_generate)

    handwrite.generate_pages(
        "abcdefghijklmnopqrstuvwxyz1234",
        style="demo-style",
        paper="demo-paper",
        layout="demo-layout",
        font_size=400,
    )

    assert calls
    assert all(call["style"] == "demo-style" for call in calls)
    assert all(call["paper"] == "demo-paper" for call in calls)
    assert all(call["layout"] == "demo-layout" for call in calls)
    assert all(call["font_size"] == 400 for call in calls)


def test_generate_pages_preserves_non_whitespace_content_order(monkeypatch) -> None:
    calls: list[str] = []

    def fake_generate(text: str, **kwargs) -> Image.Image:
        calls.append(text)
        return Image.new("L", (32, 32), color=255)

    monkeypatch.setattr(handwrite, "generate", fake_generate)

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234"

    handwrite.generate_pages(text, font_size=400)

    reconstructed = "".join(character for chunk in calls for character in chunk if not character.isspace())
    assert reconstructed == text


def test_generate_pages_renders_visible_content_without_monkeypatch() -> None:
    pages = handwrite.generate_pages("AB")

    assert len(pages) == 1
    assert isinstance(pages[0], Image.Image)
    assert pages[0].size == (2480, 3508)
    assert _page_has_visible_content(pages[0])


def test_generate_pages_matches_renderer_reflow_across_page_boundary() -> None:
    font_size = 400
    text = "ABCD," * 5

    expected_chunks = _renderer_expected_page_chunks(text, font_size=font_size)
    actual_pages = handwrite.generate_pages(text, font_size=font_size)
    expected_pages = [handwrite.generate(chunk, font_size=font_size) for chunk in expected_chunks]

    assert expected_chunks == [
        "ABC\nD,AB\nCD,A\nBCD,\nABC\nD,AB",
        "CD,",
    ]
    assert len(actual_pages) == len(expected_pages) == 2

    for actual, expected in zip(actual_pages, expected_pages):
        diff = ImageChops.difference(actual.convert("L"), expected.convert("L"))
        assert diff.getbbox() is None


def test_generate_pages_raises_for_impossible_page_geometry() -> None:
    with pytest.raises(ValueError, match="font_size"):
        handwrite.generate_pages("AB", font_size=4000)
