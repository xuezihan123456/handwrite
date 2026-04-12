from pathlib import Path
import importlib
import inspect
import shutil
import sys
from contextlib import contextmanager
import types
from uuid import uuid4

import demo.app as demo_app
import handwrite


@contextmanager
def _project_temp_dir():
    root = Path(r"C:/Users/ASUS/.codex/memories/handwrite-testdirs")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_generate_handwriting_returns_none_for_blank_input() -> None:
    assert demo_app.generate_handwriting("   ", "neat", "plain", "natural", 80) is None


def test_load_note_preset_returns_classroom_template() -> None:
    text = demo_app.load_note_preset("牛顿定律复习")

    assert "牛顿第二定律" in text
    assert "例题" in text


def test_inspect_demo_text_returns_report_markdown(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_inspect_text(text, style="行书流畅", prototype_pack=None):
        captured["text"] = text
        captured["style"] = style
        captured["prototype_pack"] = prototype_pack
        return {
            "report_markdown": (
                "## 课堂笔记预检\n"
                "- 当前原型字库: Local prototype pack: packs/custom-note\n"
                "- 2 个字符会走较低真实感路径"
            )
        }

    monkeypatch.setattr(
        handwrite,
        "inspect_text",
        fake_inspect_text,
        raising=False,
    )

    report = demo_app.inspect_demo_text(
        "课堂笔记",
        style="行书流畅",
        prototype_pack="packs/custom-note",
    )

    assert "Local prototype pack: packs/custom-note" in report
    assert captured == {
        "text": "课堂笔记",
        "style": "行书流畅",
        "prototype_pack": "packs/custom-note",
    }


def test_inspect_demo_text_reports_invalid_custom_pack(monkeypatch) -> None:
    monkeypatch.setattr(
        handwrite,
        "inspect_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing manifest")),
        raising=False,
    )

    report = demo_app.inspect_demo_text(
        "课堂笔记",
        style="行书流畅",
        prototype_pack="packs/missing",
    )

    assert "自定义原型字库不可用" in report
    assert "missing manifest" in report


def test_generate_demo_document_session_reports_invalid_custom_pack(monkeypatch) -> None:
    monkeypatch.setattr(
        demo_app,
        "_generate_demo_document_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing manifest")),
    )
    monkeypatch.setattr(demo_app, "_slider_update", lambda **kwargs: kwargs)

    result = demo_app.generate_demo_document_session(
        "课堂笔记",
        prototype_pack="packs/missing",
    )

    assert result == (
        None,
        None,
        None,
        None,
        None,
        demo_app._page_status_text(0, 0),
        {"minimum": 1, "maximum": 1, "value": 1, "visible": False},
        "⚠️ 自定义原型字库不可用：missing manifest",
    )


def test_generate_handwriting_defaults_follow_public_api(monkeypatch) -> None:
    def fake_generate(
        text: str,
        style: str = "行书流畅",
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
    monkeypatch,
) -> None:
    with _project_temp_dir() as tmp_path:
        first_page = object()
        second_page = object()
        build_session_calls: list[dict[str, object]] = []

        def fake_build_note_session(text: str, **kwargs):
            build_session_calls.append({"text": text, **kwargs})
            return {
                "pages": [first_page, second_page],
                "status_text": "已生成 2 页课堂笔记，当前原型字库：Local prototype pack: packs/custom-note",
                "report_markdown": "## 课堂笔记预检\n- 当前原型字库: Local prototype pack: packs/custom-note",
                "report": {
                    "prototype_source": {
                        "label": "Local prototype pack: packs/custom-note",
                    }
                },
            }

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

        monkeypatch.setattr(handwrite, "build_note_session", fake_build_note_session)
        monkeypatch.setattr(demo_app, "export_pages_png", fake_export_pages_png)
        monkeypatch.setattr(demo_app, "export_pages_pdf", fake_export_pages_pdf)

        actual_preview, png_paths, pdf_path = demo_app.generate_demo_document_artifacts(
            "long sample text",
            style="demo-style",
            paper="demo-paper",
            layout="demo-layout",
            font_size=96,
            prototype_pack="packs/custom-note",
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
        assert build_session_calls == [
            {
                "text": "long sample text",
                "style": "demo-style",
                "paper": "demo-paper",
                "layout": "demo-layout",
                "font_size": 96,
                "prototype_pack": "packs/custom-note",
            }
        ]


def test_generate_demo_document_artifacts_prunes_old_temp_dirs(
    monkeypatch,
) -> None:
    with _project_temp_dir() as tmp_path:
        stale_dir = tmp_path / "stale"
        fresh_dirs = [tmp_path / f"keep_{index:02d}" for index in range(20)]
        stale_dir.mkdir()
        for directory in fresh_dirs:
            directory.mkdir()

        monkeypatch.setattr(demo_app, "_DEMO_OUTPUT_ROOT", tmp_path, raising=False)
        monkeypatch.setattr(
            handwrite,
            "build_note_session",
            lambda *args, **kwargs: {
                "pages": [object()],
                "status_text": "已生成 1 页课堂笔记，当前原型字库：Built-in starter pack: default_note",
                "report_markdown": "## 课堂笔记预检\n- 当前原型字库: Built-in starter pack: default_note",
                "report": {
                    "prototype_source": {
                        "label": "Built-in starter pack: default_note",
                    }
                },
            },
            raising=False,
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

        _, png_paths, pdf_path = demo_app.generate_demo_document_artifacts("sample text")

        assert png_paths is not None and pdf_path is not None
        assert stale_dir.exists() is False


def test_generate_demo_document_artifacts_returns_empty_outputs_for_blank_input(
) -> None:
    with _project_temp_dir() as tmp_path:
        output_dir = tmp_path / "artifacts"

        assert demo_app.generate_demo_document_artifacts("   ", output_dir=output_dir) == (
            None,
            None,
            None,
        )
        assert not output_dir.exists()


def test_generate_demo_document_artifacts_uses_generate_pages_not_single_page_path(
    monkeypatch,
) -> None:
    with _project_temp_dir() as tmp_path:
        preview_page = object()
        build_session_calls: list[str] = []

        def fake_build_note_session(text: str, **kwargs):
            build_session_calls.append(text)
            return {
                "pages": [preview_page],
                "status_text": "已生成 1 页课堂笔记，当前原型字库：Built-in starter pack: default_note",
                "report_markdown": "## 课堂笔记预检\n- 当前原型字库: Built-in starter pack: default_note",
                "report": {
                    "prototype_source": {
                        "label": "Built-in starter pack: default_note",
                    }
                },
            }

        monkeypatch.setattr(handwrite, "build_note_session", fake_build_note_session)
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
        assert build_session_calls == ["demo text"]


def test_generate_demo_document_session_returns_state_and_slider_update(
    monkeypatch,
) -> None:
    with _project_temp_dir() as tmp_path:
        first_page = object()
        second_page = object()
        report_path = str(tmp_path / "artifacts" / "handwriting_report.md")

        monkeypatch.setattr(
            demo_app,
            "_generate_demo_document_bundle",
            lambda *args, **kwargs: (
                {
                    "pages": [first_page, second_page],
                    "status_text": "已生成 2 页课堂笔记，当前原型字库：Built-in starter pack: default_note",
                },
                [
                    str(tmp_path / "artifacts" / "handwriting_001.png"),
                    str(tmp_path / "artifacts" / "handwriting_002.png"),
                ],
                str(tmp_path / "artifacts" / "handwriting.pdf"),
                report_path,
            ),
        )
        monkeypatch.setattr(demo_app, "_slider_update", lambda **kwargs: kwargs)

        (
            preview,
            png_paths,
            pdf_path,
            session_report_path,
            preview_state,
            page_status,
            slider_update,
            generation_status,
        ) = (
            demo_app.generate_demo_document_session(
                "long sample text",
                preview_page_index=1,
                output_dir=tmp_path / "artifacts",
            )
        )

        assert preview is second_page
        assert png_paths == [
            str(tmp_path / "artifacts" / "handwriting_001.png"),
            str(tmp_path / "artifacts" / "handwriting_002.png"),
        ]
        assert pdf_path == str(tmp_path / "artifacts" / "handwriting.pdf")
        assert session_report_path == report_path
        assert preview_state == {
            "pages": [first_page, second_page],
            "png_paths": png_paths,
            "pdf_path": pdf_path,
            "report_path": report_path,
        }
        assert page_status == "Page 2 / 2"
        assert slider_update == {"minimum": 1, "maximum": 2, "value": 2, "visible": True}
        assert "当前原型字库" in generation_status


def test_generate_demo_document_bundle_writes_session_report(monkeypatch) -> None:
    with _project_temp_dir() as tmp_path:
        page = object()
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            handwrite,
            "build_note_session",
            lambda *args, **kwargs: {
                "pages": [page],
                "status_text": "已生成 1 页课堂笔记，当前原型字库：Built-in starter pack: default_note",
                "report_markdown": "## 课堂笔记预检\n- 当前原型字库: Built-in starter pack: default_note",
                "report": {"prototype_source": {"label": "Built-in starter pack: default_note"}},
            },
            raising=False,
        )
        monkeypatch.setattr(
            demo_app,
            "export_pages_png",
            lambda pages, output_dir, prefix="page", dpi=300: [Path(output_dir) / f"{prefix}_001.png"],
        )
        monkeypatch.setattr(
            demo_app,
            "export_pages_pdf",
            lambda pages, output_path, dpi=300: Path(output_path),
        )

        session, png_paths, pdf_path, report_path = demo_app._generate_demo_document_bundle(
            "课堂笔记",
            output_dir=tmp_path / "artifacts",
        )

        assert session["pages"] == [page]
        assert png_paths == [str(tmp_path / "artifacts" / "handwriting_001.png")]
        assert pdf_path == str(tmp_path / "artifacts" / "handwriting.pdf")
        assert Path(report_path).exists()
        assert "HandWrite Note Session" in Path(report_path).read_text(encoding="utf-8")


def test_change_preview_page_reads_cached_state() -> None:
    first_page = object()
    second_page = object()

    preview, page_status = demo_app.change_preview_page(
        2,
        {
            "pages": [first_page, second_page],
            "png_paths": ["one", "two"],
            "pdf_path": "document.pdf",
        },
    )

    assert preview is second_page
    assert page_status == "Page 2 / 2"


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

    class FakeState(FakeComponent):
        pass

    class FakeButton(FakeComponent):
        def click(self, *, fn, inputs, outputs):
            active_blocks[-1].click_events.append(
                {"fn": fn, "inputs": inputs, "outputs": outputs}
            )

    class FakeSlider(FakeComponent):
        def change(self, *, fn, inputs, outputs):
            active_blocks[-1].change_events.append(
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
            self.change_events = []

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
        State=FakeState,
        Button=FakeButton,
        update=lambda **kwargs: kwargs,
    )

    def fake_generate(
        text: str,
        style: str = "行书流畅",
        paper: str = "essay-paper",
        layout: str = "free-layout",
        font_size: int = 80,
        prototype_pack: str | None = None,
    ):
        return text, style, paper, layout, font_size

    with monkeypatch.context() as patch:
        patch.setattr(handwrite, "generate", fake_generate)
        patch.setattr(
            handwrite,
            "inspect_text",
            lambda text, style="行书流畅", prototype_pack=None: {"report_markdown": "ok"},
            raising=False,
        )
        patch.setattr(handwrite, "list_styles", lambda: ["行书流畅"])
        patch.setitem(sys.modules, "gradio", fake_gradio)

        reloaded_demo_app = importlib.reload(demo_app)
        demo = reloaded_demo_app.build_demo()

        inspect_click_event = next(
            event for event in demo.click_events if event["fn"] is reloaded_demo_app.inspect_demo_text
        )
        preset_click_event = next(
            event for event in demo.click_events if event["fn"] is reloaded_demo_app.load_note_preset
        )
        generate_click_event = next(
            event
            for event in demo.click_events
            if event["fn"] is reloaded_demo_app.generate_demo_document_session
        )
        change_event = demo.change_events[0]
        dropdown_by_label = {
            component.label: component
            for component in demo.components
            if isinstance(component, FakeDropdown)
        }
        slider_by_label = {
            component.label: component
            for component in demo.components
            if isinstance(component, FakeSlider)
        }
        textbox_by_label = {
            component.label: component
            for component in demo.components
            if isinstance(component, FakeTextbox)
        }
        markdown_by_value = {
            component.value: component
            for component in demo.components
            if isinstance(component, FakeMarkdown)
        }
        file_outputs = [
            component
            for component in generate_click_event["outputs"]
            if isinstance(component, FakeFile)
        ]
        png_output = next(
            component for component in file_outputs if component.label == "PNG Pages"
        )
        report_output = next(
            component for component in file_outputs if component.label == "Session Report"
        )

        assert isinstance(demo, FakeBlocks)
        assert dropdown_by_label["Paper Type"].value == "essay-paper"
        assert "essay-paper" in dropdown_by_label["Paper Type"].choices
        assert dropdown_by_label["Layout Style"].value == "free-layout"
        assert "free-layout" in dropdown_by_label["Layout Style"].choices
        assert preset_click_event["outputs"] == [textbox_by_label["Text"]]
        assert inspect_click_event["inputs"][-1] is textbox_by_label["Local Prototype Pack (optional)"]
        assert generate_click_event["inputs"][-1] is textbox_by_label["Local Prototype Pack (optional)"]
        assert any(
            isinstance(component, FakeImage) for component in generate_click_event["outputs"]
        )
        assert len(file_outputs) == 3
        assert png_output.kwargs["file_count"] == "multiple"
        assert report_output.label == "Session Report"
        assert slider_by_label["Preview Page"].kwargs["minimum"] == 1
        assert change_event["fn"] is reloaded_demo_app.change_preview_page
        assert "课堂笔记模板" in dropdown_by_label
        assert any(
            isinstance(component, FakeButton)
            and "inspect" in component.value.lower()
            for component in demo.components
        )
        assert any(
            isinstance(component, FakeButton)
            and "preset" in component.value.lower()
            for component in demo.components
        )
        assert any(
            isinstance(component, FakeButton)
            and "multi-page" in component.value.lower()
            for component in demo.components
        )
        assert any(
            value and "Precheck Report" in str(value)
            for value in markdown_by_value
        )
        assert demo.kwargs.get("css")

    importlib.reload(demo_app)


def test_export_handwriting_dispatches_pdf_via_public_api(monkeypatch) -> None:
    with _project_temp_dir() as tmp_path:
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
