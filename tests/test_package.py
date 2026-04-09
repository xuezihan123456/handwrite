from PIL import Image

import handwrite


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
