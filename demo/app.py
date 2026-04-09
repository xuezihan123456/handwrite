"""Gradio demo entrypoint."""

from inspect import signature

import handwrite


_PAPER_CHOICES = ["白纸", "横线纸", "方格纸", "米字格"]
_LAYOUT_CHOICES = ["工整", "自然", "潦草"]


def _public_generate_defaults() -> dict[str, object]:
    parameters = signature(handwrite.generate).parameters
    return {
        name: parameters[name].default
        for name in ("style", "paper", "layout", "font_size")
    }


_DEFAULTS = _public_generate_defaults()


def _with_default_choice(default: str, choices: list[str]) -> list[str]:
    """Ensure the active default is a valid UI choice."""
    if default in choices:
        return choices
    return [default, *choices]


def generate_handwriting(
    text: str,
    style: str = _DEFAULTS["style"],
    paper: str = _DEFAULTS["paper"],
    layout: str = _DEFAULTS["layout"],
    font_size: int = _DEFAULTS["font_size"],
):
    """Generate a handwriting page for the demo."""
    if not text.strip():
        return None

    return handwrite.generate(
        text=text,
        style=style,
        paper=paper,
        layout=layout,
        font_size=int(font_size),
    )


def build_demo():
    """Build the interactive Gradio demo."""
    import gradio as gr

    return gr.Interface(
        fn=generate_handwriting,
        inputs=[
            gr.Textbox(label="输入文字", lines=5, placeholder="在这里输入要转换的文字..."),
            gr.Dropdown(
                choices=_with_default_choice(_DEFAULTS["style"], handwrite.list_styles()),
                value=_DEFAULTS["style"],
                label="手写风格",
            ),
            gr.Dropdown(
                choices=_with_default_choice(_DEFAULTS["paper"], _PAPER_CHOICES),
                value=_DEFAULTS["paper"],
                label="纸张类型",
            ),
            gr.Dropdown(
                choices=_with_default_choice(_DEFAULTS["layout"], _LAYOUT_CHOICES),
                value=_DEFAULTS["layout"],
                label="排版风格",
            ),
            gr.Slider(
                minimum=40,
                maximum=120,
                value=_DEFAULTS["font_size"],
                step=10,
                label="字号",
            ),
        ],
        outputs=gr.Image(label="生成结果", type="pil"),
        title="HandWrite - AI 中文手写体生成器",
        description="输入中文文字，选择手写风格，生成手写文档图片。",
    )


def main() -> None:
    """Launch the demo app."""
    build_demo().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
