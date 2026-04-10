from PIL import Image
import pytest

import handwrite
import handwrite.exporter as exporter_module


def test_list_styles_exposes_builtin_names() -> None:
    styles = handwrite.list_styles()
    assert styles == [
        "工整楷书",
        "圆润可爱",
        "行书流畅",
        "偏瘦紧凑",
        "随意潦草",
    ]


def test_char_returns_256_square_image() -> None:
    image = handwrite.char("你")
    assert isinstance(image, Image.Image)
    assert image.size == (256, 256)


def test_generate_returns_a4_page_image() -> None:
    page = handwrite.generate("你好")
    assert isinstance(page, Image.Image)
    assert page.size == (2480, 3508)


def test_generate_uses_human_facing_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}
    stub_page = Image.new("L", (32, 32), color=255)

    def fake_char(text: str, style: str = "工整楷书") -> Image.Image:
        captured.setdefault("styles", []).append(style)
        return Image.new("L", (256, 256), color=255)

    def fake_compose_page(images, text, font_size, layout, paper) -> Image.Image:
        captured["image_count"] = len(images)
        captured["text"] = text
        captured["font_size"] = font_size
        captured["layout"] = layout
        captured["paper"] = paper
        return stub_page

    monkeypatch.setattr(handwrite, "char", fake_char)
    monkeypatch.setattr(handwrite, "compose_page", fake_compose_page)

    page = handwrite.generate("你好")

    assert page is stub_page
    assert captured == {
        "styles": ["工整楷书", "工整楷书"],
        "image_count": 2,
        "text": "你好",
        "font_size": 80,
        "layout": "自然",
        "paper": "白纸",
    }


def test_export_dispatches_pdf_to_exporter(monkeypatch, tmp_path) -> None:
    page = Image.new("L", (2480, 3508), color=255)
    output_path = tmp_path / "page.pdf"
    captured: dict[str, object] = {}

    def fake_export_pdf(image, destination, dpi=300):
        captured["image"] = image
        captured["destination"] = destination
        captured["dpi"] = dpi
        return output_path

    monkeypatch.setattr(exporter_module, "export_pdf", fake_export_pdf, raising=False)

    written_path = handwrite.export(page, output_path, format="pdf", dpi=144)

    assert written_path == output_path
    assert captured == {
        "image": page,
        "destination": output_path,
        "dpi": 144,
    }


def test_export_pages_dispatches_pdf_to_exporter(monkeypatch, tmp_path) -> None:
    pages = [
        Image.new("L", (2480, 3508), color=255),
        Image.new("L", (2480, 3508), color=240),
    ]
    output_path = tmp_path / "pages.pdf"
    captured: dict[str, object] = {}

    def fake_export_pages_pdf(images, destination, dpi=300):
        captured["images"] = images
        captured["destination"] = destination
        captured["dpi"] = dpi
        return output_path

    monkeypatch.setattr(
        exporter_module,
        "export_pages_pdf",
        fake_export_pages_pdf,
        raising=False,
    )

    written_path = handwrite.export_pages(pages, output_path, format="pdf", dpi=144)

    assert written_path == output_path
    assert captured == {
        "images": pages,
        "destination": output_path,
        "dpi": 144,
    }


def test_export_pages_dispatches_png_to_exporter(monkeypatch, tmp_path) -> None:
    pages = [
        Image.new("L", (2480, 3508), color=255),
        Image.new("L", (2480, 3508), color=240),
    ]
    output_dir = tmp_path / "pages"
    written_paths = [output_dir / "page_001.png", output_dir / "page_002.png"]
    captured: dict[str, object] = {}

    def fake_export_pages_png(images, destination, prefix="page", dpi=300):
        captured["images"] = images
        captured["destination"] = destination
        captured["prefix"] = prefix
        captured["dpi"] = dpi
        return written_paths

    monkeypatch.setattr(
        exporter_module,
        "export_pages_png",
        fake_export_pages_png,
        raising=False,
    )

    result = handwrite.export_pages(
        pages,
        output_dir,
        format="png",
        prefix="sheet",
        dpi=200,
    )

    assert result == written_paths
    assert captured == {
        "images": pages,
        "destination": output_dir,
        "prefix": "sheet",
        "dpi": 200,
    }


def test_export_pages_rejects_invalid_format(tmp_path) -> None:
    pages = [Image.new("L", (2480, 3508), color=255)]

    with pytest.raises(ValueError, match="format must be 'png' or 'pdf'"):
        handwrite.export_pages(pages, tmp_path / "pages", format="jpg")
