from pathlib import Path
import importlib
import inspect
import sys
import types

import demo.app as demo_app
import handwrite


def test_generate_handwriting_returns_none_for_blank_input() -> None:
    assert demo_app.generate_handwriting("   ", "neat", "plain", "natural", 80) is None


def test_generate_handwriting_defaults_follow_public_api(monkeypatch) -> None:
    def fake_generate(
        text: str,
        style: str = "rounded",
        paper: str = "grid-paper",
        layout: str = "neat-layout",
        font_size: int = 90,
    ):
        return text, style, paper, layout, font_size

    with monkeypatch.context() as patch:
        patch.setattr(handwrite, "generate", fake_generate)
        reloaded_demo_app = importlib.reload(demo_app)

        demo_defaults = inspect.signature(
            reloaded_demo_app.generate_handwriting
        ).parameters
        public_defaults = inspect.signature(fake_generate).parameters

        for name in ("style", "paper", "layout", "font_size"):
            assert demo_defaults[name].default == public_defaults[name].default

    importlib.reload(demo_app)


def test_generate_demo_document_artifacts_returns_preview_and_multi_page_exports(
    monkeypatch, tmp_path
) -> None:
    first_page = object()
    second_page = object()
    generate_pages_calls: list[dict[str, object]] = []

    def fake_generate_pages(text: str, **kwargs):
        generate_pages_calls.append({"text": text, **kwargs})
        return [first_page, second_page]

    def fake_export_pages_png(pages, output_dir, prefix="page", dpi=300):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        written_paths = []
        for index, _ in enumerate(pages, start=1):
            path = directory / f"{prefix}_{index:03d}.png"
            path.write_text(f"png-{index}", encoding="utf-8")
            written_paths.append(path)
        return written_paths

    def fake_export_pages_pdf(pages, output_path, dpi=300):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-demo")
        return path

    monkeypatch.setattr(handwrite, "generate_pages", fake_generate_pages)
    monkeypatch.setattr(demo_app, "export_pages_png", fake_export_pages_png)
    monkeypatch.setattr(demo_app, "export_pages_pdf", fake_export_pages_pdf)

    actual_preview, png_paths, pdf_path = demo_app.generate_demo_document_artifacts(
        "long sample text",
        style="demo-style",
        paper="demo-paper",
        layout="demo-layout",
        font_size=96,
        preview_page_index=1,
        output_dir=tmp_path / "artifacts",
    )

    assert actual_preview is second_page
    assert png_paths == [
        str(tmp_path / "artifacts" / "handwriting_001.png"),
        str(tmp_path / "artifacts" / "handwriting_002.png"),
    ]
    assert Path(pdf_path) == tmp_path / "artifacts" / "handwriting.pdf"
    assert Path(pdf_path).exists()
    assert generate_pages_calls == [
        {
            "text": "long sample text",
            "style": "demo-style",
            "paper": "demo-paper",
            "layout": "demo-layout",
            "font_size": 96,
        }
    ]


def test_generate_demo_document_artifacts_prunes_old_temp_dirs(
    monkeypatch, tmp_path
) -> None:
    stale_dir = tmp_path / "stale"
    fresh_dirs = [tmp_path / f"keep_{index:02d}" for index in range(20)]
    stale_dir.mkdir()
    for directory in fresh_dirs:
        directory.mkdir()

    monkeypatch.setattr(demo_app, "_DEMO_OUTPUT_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(handwrite, "generate_pages", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(
        demo_app,
        "export_pages_png",
        lambda pages, output_dir, prefix="page", dpi=300: [
            Path(output_dir) / f"{prefix}_001.png"
        ],
    )
    monkeypatch.setattr(
        demo_app,
        "export_pages_pdf",
        lambda pages, output_path, dpi=300: Path(output_path),
    )

    _, png_paths, pdf_path = demo_app.generate_demo_document_artifacts("sample text")

    assert png_paths is not None and pdf_path is not None
    assert stale_dir.exists() is False


def test_generate_demo_document_artifacts_returns_empty_outputs_for_blank_input(
    tmp_path,
) -> None:
    output_dir = tmp_path / "artifacts"

    assert demo_app.generate_demo_document_artifacts("   ", output_dir=output_dir) == (
        None,
        None,
        None,
    )
    assert not output_dir.exists()


def test_generate_demo_document_artifacts_uses_generate_pages_not_single_page_path(
    monkeypatch, tmp_path
) -> None:
    preview_page = object()
    generate_pages_calls: list[str] = []

    def fake_generate_pages(text: str, **kwargs):
        generate_pages_calls.append(text)
        return [preview_page]

    monkeypatch.setattr(handwrite, "generate_pages", fake_generate_pages)
    monkeypatch.setattr(
        demo_app,
        "generate_handwriting",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("single-page helper should not be used")
        ),
    )
    monkeypatch.setattr(
        demo_app,
        "export_pages_png",
        lambda pages, output_dir, prefix="page", dpi=300: [
            Path(output_dir) / f"{prefix}_001.png"
        ],
    )
    monkeypatch.setattr(
        demo_app,
        "export_pages_pdf",
        lambda pages, output_path, dpi=300: Path(output_path),
    )

    preview, png_paths, pdf_path = demo_app.generate_demo_document_artifacts(
        "demo text",
        output_dir=tmp_path / "artifacts",
    )

    assert preview is preview_page
    assert png_paths == [str(tmp_path / "artifacts" / "handwriting_001.png")]
    assert pdf_path == str(tmp_path / "artifacts" / "handwriting.pdf")
    assert generate_pages_calls == ["demo text"]


def test_main_launches_built_demo(monkeypatch) -> None:
    launched: dict[str, object] = {}

    class FakeDemo:
        def launch(self, **kwargs):
            launched["kwargs"] = kwargs

    monkeypatch.setattr(demo_app, "build_demo", lambda: FakeDemo())

    demo_app.main()

    assert launched["kwargs"] == {"server_name": "0.0.0.0", "server_port": 7860}


def test_build_demo_returns_blocks_and_wires_multi_page_outputs(monkeypatch) -> None:
    active_blocks: list["FakeBlocks"] = []

    class FakeComponent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.label = kwargs.get("label")
            self.value = kwargs.get("value")
            self.choices = list(kwargs.get("choices", []))
            if active_blocks:
                active_blocks[-1].components.append(self)

    class FakeTextbox(FakeComponent):
        pass

    class FakeDropdown(FakeComponent):
        pass

    class FakeSlider(FakeComponent):
        pass

    class FakeImage(FakeComponent):
        pass

    class FakeFile(FakeComponent):
        pass

    class FakeButton(FakeComponent):
        def click(self, *, fn, inputs, outputs):
            active_blocks[-1].click_events.append(
                {"fn": fn, "inputs": inputs, "outputs": outputs}
            )

    class FakeMarkdown(FakeComponent):
        pass

    class FakeContainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeRow(FakeContainer):
        pass

    class FakeColumn(FakeContainer):
        pass

    class FakeBlocks(FakeContainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.components = []
            self.click_events = []

        def __enter__(self):
            active_blocks.append(self)
            return self

        def __exit__(self, exc_type, exc, tb):
            active_blocks.pop()
            return False

    fake_gradio = types.SimpleNamespace(
        Blocks=FakeBlocks,
        Row=FakeRow,
        Column=FakeColumn,
        Markdown=FakeMarkdown,
        Textbox=FakeTextbox,
        Dropdown=FakeDropdown,
        Slider=FakeSlider,
        Image=FakeImage,
        File=FakeFile,
        Button=FakeButton,
    )

    def fake_generate(
        text: str,
        style: str = "formal-style",
        paper: str = "essay-paper",
        layout: str = "free-layout",
        font_size: int = 80,
    ):
        return text, style, paper, layout, font_size

    with monkeypatch.context() as patch:
        patch.setattr(handwrite, "generate", fake_generate)
        patch.setattr(handwrite, "list_styles", lambda: ["formal-style"])
        patch.setitem(sys.modules, "gradio", fake_gradio)

        reloaded_demo_app = importlib.reload(demo_app)
        demo = reloaded_demo_app.build_demo()

        click_event = demo.click_events[0]
        dropdown_by_label = {
            component.label: component
            for component in demo.components
            if isinstance(component, FakeDropdown)
        }
        file_outputs = [
            component for component in click_event["outputs"] if isinstance(component, FakeFile)
        ]
        png_output = next(
            component for component in file_outputs if component.label == "PNG Pages"
        )

        assert isinstance(demo, FakeBlocks)
        assert dropdown_by_label["Paper Type"].value == "essay-paper"
        assert "essay-paper" in dropdown_by_label["Paper Type"].choices
        assert dropdown_by_label["Layout Style"].value == "free-layout"
        assert "free-layout" in dropdown_by_label["Layout Style"].choices
        assert click_event["fn"] is reloaded_demo_app.generate_demo_document_artifacts
        assert any(
            isinstance(component, FakeImage) for component in click_event["outputs"]
        )
        assert len(file_outputs) == 2
        assert png_output.kwargs["file_count"] == "multiple"
        assert any(
            isinstance(component, FakeButton)
            and "multi-page" in component.value.lower()
            for component in demo.components
        )
        assert demo.kwargs.get("css")

    importlib.reload(demo_app)


def test_export_handwriting_dispatches_pdf_via_public_api(monkeypatch, tmp_path) -> None:
    page = object()
    output_path = tmp_path / "page.pdf"
    captured: dict[str, object] = {}

    monkeypatch.setattr(demo_app, "generate_handwriting", lambda *args, **kwargs: page)

    def fake_export(image, destination, format="png", **kwargs):
        captured["image"] = image
        captured["destination"] = destination
        captured["format"] = format
        captured["kwargs"] = kwargs
        return output_path

    monkeypatch.setattr(handwrite, "export", fake_export, raising=False)

    written_path = demo_app.export_handwriting(
        "demo text",
        output_path,
        format="pdf",
        font_size=96,
    )

    assert written_path == output_path
    assert captured == {
        "image": page,
        "destination": output_path,
        "format": "pdf",
        "kwargs": {},
    }
