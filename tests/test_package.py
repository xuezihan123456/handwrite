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

    def fake_char(
        text: str,
        style: str = "行书流畅",
        prototype_pack: str | None = None,
    ) -> Image.Image:
        captured.setdefault("styles", []).append(style)
        captured.setdefault("prototype_packs", []).append(prototype_pack)
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
        "styles": ["行书流畅", "行书流畅"],
        "prototype_packs": [None, None],
        "image_count": 2,
        "text": "你好",
        "font_size": 80,
        "layout": "自然",
        "paper": "白纸",
    }


def test_inspect_text_delegates_to_engine_and_returns_structured_report(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}
    manifest_path = tmp_path / "custom-note" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    engine_report = {
        "style": "行书流畅",
        "total_characters": 4,
        "unique_characters": ["课", "堂", "笔", "记"],
        "prototype_covered_characters": ["课", "记"],
        "model_supported_characters": [],
        "fallback_characters": ["堂", "笔"],
        "suggestions": [
            {
                "char": "堂",
                "reason": "starter prototype pack does not cover this character yet",
                "suggestion": "可保留原文，或改成更常见、已覆盖的表达后再生成。",
            }
        ],
        "summary": "2 / 4 characters are in the prototype-backed path.",
        "report_markdown": "## 课堂笔记预检\n- 2 个字符会走较低真实感路径",
    }

    class StubEngine:
        def inspect_text(self, text: str, style_id: int):
            captured["text"] = text
            captured["style_id"] = style_id
            return engine_report

    monkeypatch.setattr(
        handwrite,
        "_get_engine",
        lambda prototype_pack=None: (
            captured.setdefault("prototype_pack", prototype_pack),
            StubEngine(),
        )[1],
    )

    report = handwrite.inspect_text(
        "课堂笔记",
        style="行书流畅",
        prototype_pack=manifest_path.parent,
    )

    for key, value in engine_report.items():
        assert report[key] == value
    assert report["prototype_pack_name"] == "custom-note"
    assert report["prototype_source"]["kind"] == "custom"
    assert report["prototype_source"]["name"] == "custom-note"
    assert report["prototype_source"]["manifest_path"] == str(manifest_path.resolve())
    assert report["prototype_source"]["label"].startswith("Local prototype pack:")
    assert report["prototype_source_kind"] == "custom"
    assert captured == {
        "prototype_pack": manifest_path.parent,
        "text": "课堂笔记",
        "style_id": 2,
    }


def test_generate_routes_custom_prototype_pack_to_char(monkeypatch) -> None:
    calls: list[tuple[str, str, str | None]] = []
    stub_page = Image.new("L", (32, 32), color=255)

    def fake_char(
        text: str,
        style: str = "行书流畅",
        prototype_pack: str | None = None,
    ) -> Image.Image:
        calls.append((text, style, prototype_pack))
        return Image.new("L", (256, 256), color=255)

    monkeypatch.setattr(handwrite, "char", fake_char)
    monkeypatch.setattr(handwrite, "compose_page", lambda *args, **kwargs: stub_page)

    page = handwrite.generate(
        "你好",
        style="行书流畅",
        prototype_pack="packs/custom-note",
    )

    assert page is stub_page
    assert calls == [
        ("你", "行书流畅", "packs/custom-note"),
        ("好", "行书流畅", "packs/custom-note"),
    ]


def test_inspect_text_marks_prototype_pack_disabled_for_non_note_styles(monkeypatch) -> None:
    class StubEngine:
        def inspect_text(self, text: str, style_id: int):
            return {
                "style": "工整楷书",
                "total_characters": 2,
                "unique_characters": ["你", "好"],
                "prototype_covered_characters": [],
                "model_supported_characters": [],
                "fallback_characters": ["你", "好"],
                "prototype_pack_name": None,
                "prototype_source": None,
                "prototype_source_kind": "disabled",
                "summary": "0 / 2 characters are in a higher-realism path.",
            }

    monkeypatch.setattr(handwrite, "_get_engine", lambda prototype_pack=None: StubEngine())

    report = handwrite.inspect_text("你好", style="工整楷书")

    assert report["prototype_source"] == {
        "active": False,
        "kind": "disabled",
        "name": None,
        "manifest_path": None,
        "root": None,
        "label": "Prototype pack disabled for the selected style",
    }


def test_build_note_session_returns_report_pages_and_status(monkeypatch) -> None:
    first_page = Image.new("L", (2480, 3508), color=255)
    second_page = Image.new("L", (2480, 3508), color=240)
    report = {
        "style": "行书流畅",
        "report_markdown": "## 课堂笔记预检\n- 当前原型字库: Built-in starter pack: default_note",
        "prototype_source": {
            "active": True,
            "kind": "builtin",
            "name": "default_note",
            "manifest_path": None,
            "root": None,
            "label": "Built-in starter pack: default_note",
        },
        "prototype_pack_name": "default_note",
        "prototype_source_kind": "builtin",
        "fallback_characters": ["龘"],
    }

    monkeypatch.setattr(handwrite, "inspect_text", lambda *args, **kwargs: report)
    monkeypatch.setattr(handwrite, "generate_pages", lambda *args, **kwargs: [first_page, second_page])

    session = handwrite.build_note_session(
        "课堂笔记",
        style="行书流畅",
        paper="横线纸",
        layout="自然",
        font_size=88,
    )

    assert session["report"] is report
    assert session["report_markdown"] == report["report_markdown"]
    assert session["pages"] == [first_page, second_page]
    assert session["page_count"] == 2
    assert session["prototype_source"] == report["prototype_source"]
    assert session["status_text"] == "已生成 2 页课堂笔记，当前原型字库：Built-in starter pack: default_note"


def test_build_note_session_marks_empty_generation() -> None:
    session = handwrite.build_note_session("   ")

    assert session["pages"] == []
    assert session["page_count"] == 0
    assert session["status_text"] == "还没有可生成的课堂笔记内容。"


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
