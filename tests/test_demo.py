import importlib
import inspect
import sys
import types

import demo.app as demo_app
import handwrite


def test_generate_handwriting_returns_none_for_blank_input() -> None:
    assert demo_app.generate_handwriting("   ", "工整楷书", "白纸", "自然", 80) is None


def test_generate_handwriting_defaults_follow_public_api(monkeypatch) -> None:
    def fake_generate(
        text: str,
        style: str = "圆润可爱",
        paper: str = "方格纸",
        layout: str = "工整",
        font_size: int = 90,
    ):
        return text, style, paper, layout, font_size

    with monkeypatch.context() as patch:
        patch.setattr(handwrite, "generate", fake_generate)
        reloaded_demo_app = importlib.reload(demo_app)

        demo_defaults = inspect.signature(reloaded_demo_app.generate_handwriting).parameters
        public_defaults = inspect.signature(fake_generate).parameters

        for name in ("style", "paper", "layout", "font_size"):
            assert demo_defaults[name].default == public_defaults[name].default

    importlib.reload(demo_app)


def test_main_launches_built_demo(monkeypatch) -> None:
    launched: dict[str, object] = {}

    class FakeDemo:
        def launch(self, **kwargs):
            launched["kwargs"] = kwargs

    monkeypatch.setattr(demo_app, "build_demo", lambda: FakeDemo())

    demo_app.main()

    assert launched["kwargs"] == {"server_name": "0.0.0.0", "server_port": 7860}


def test_build_demo_includes_reflected_defaults_in_choices(monkeypatch) -> None:
    class FakeTextbox:
        def __init__(self, **kwargs):
            self.label = kwargs["label"]

    class FakeDropdown:
        def __init__(self, *, choices, value, label):
            self.choices = list(choices)
            self.value = value
            self.label = label

    class FakeSlider:
        def __init__(self, **kwargs):
            self.label = kwargs["label"]

    class FakeImage:
        def __init__(self, **kwargs):
            self.label = kwargs["label"]

    class FakeInterface:
        def __init__(self, *, fn, inputs, outputs, title, description):
            self.fn = fn
            self.inputs = inputs
            self.outputs = outputs
            self.title = title
            self.description = description

    fake_gradio = types.SimpleNamespace(
        Textbox=FakeTextbox,
        Dropdown=FakeDropdown,
        Slider=FakeSlider,
        Image=FakeImage,
        Interface=FakeInterface,
    )

    def fake_generate(
        text: str,
        style: str = "工整楷书",
        paper: str = "作文本",
        layout: str = "自由布局",
        font_size: int = 80,
    ):
        return text, style, paper, layout, font_size

    with monkeypatch.context() as patch:
        patch.setattr(handwrite, "generate", fake_generate)
        patch.setattr(handwrite, "list_styles", lambda: ["工整楷书"])
        patch.setitem(sys.modules, "gradio", fake_gradio)

        reloaded_demo_app = importlib.reload(demo_app)
        demo = reloaded_demo_app.build_demo()

        paper_input = next(component for component in demo.inputs if component.label == "纸张类型")
        layout_input = next(component for component in demo.inputs if component.label == "排版风格")

        assert paper_input.value == "作文本"
        assert "作文本" in paper_input.choices
        assert layout_input.value == "自由布局"
        assert "自由布局" in layout_input.choices

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
        "浣犲ソ",
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
