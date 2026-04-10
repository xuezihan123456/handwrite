"""Gradio demo entrypoint."""

import shutil
from inspect import signature
from pathlib import Path
from tempfile import gettempdir
from typing import Optional, Union
from uuid import uuid4

import handwrite


PathLike = Union[str, Path]
_PAPER_CHOICES = ["白纸", "横线纸", "方格纸", "米字格"]
_LAYOUT_CHOICES = ["工整", "自然", "潦草"]
_DEMO_OUTPUT_ROOT = Path(gettempdir()) / "handwrite-demo"
_PNG_FILENAME = "handwriting.png"
_PDF_FILENAME = "handwriting.pdf"
_MAX_DEMO_EXPORT_DIRS = 20
_CUSTOM_CSS = """
#demo-shell {
    max-width: 1180px;
    margin: 0 auto;
    padding: 24px 16px 40px;
}

.demo-hero {
    margin-bottom: 18px;
    padding: 24px 28px;
    border: 1px solid #d8d4cb;
    border-radius: 24px;
    background: linear-gradient(135deg, #fbf6ec 0%, #f1e5cf 100%);
}

.demo-panel {
    border: 1px solid #e1ddd4;
    border-radius: 20px;
    background: #fffdf8;
    box-shadow: 0 18px 40px rgba(91, 71, 43, 0.08);
}

.demo-panel > .gr-block {
    padding: 20px;
}

.demo-hint {
    color: #5b5143;
    font-size: 0.95rem;
}
"""


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


def _resolve_output_dir(output_dir: Optional[PathLike] = None) -> Path:
    if output_dir is not None:
        resolved_dir = Path(output_dir)
        resolved_dir.mkdir(parents=True, exist_ok=True)
        return resolved_dir

    _prune_demo_output_dirs()
    _DEMO_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    request_dir = _DEMO_OUTPUT_ROOT / uuid4().hex
    request_dir.mkdir(parents=True, exist_ok=False)
    return request_dir


def _prune_demo_output_dirs(max_keep: int = _MAX_DEMO_EXPORT_DIRS) -> None:
    if not _DEMO_OUTPUT_ROOT.exists():
        return

    existing_dirs = sorted(
        (path for path in _DEMO_OUTPUT_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale_dir in existing_dirs[max_keep:]:
        shutil.rmtree(stale_dir, ignore_errors=True)


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


def generate_demo_artifacts(
    text: str,
    style: str = _DEFAULTS["style"],
    paper: str = _DEFAULTS["paper"],
    layout: str = _DEFAULTS["layout"],
    font_size: int = _DEFAULTS["font_size"],
    output_dir: Optional[PathLike] = None,
):
    """Generate preview and downloadable export artifacts for the demo."""
    preview = generate_handwriting(
        text=text,
        style=style,
        paper=paper,
        layout=layout,
        font_size=font_size,
    )
    if preview is None:
        return None, None, None

    artifact_dir = _resolve_output_dir(output_dir)
    try:
        png_path = handwrite.export(preview, artifact_dir / _PNG_FILENAME, format="png")
        pdf_path = handwrite.export(preview, artifact_dir / _PDF_FILENAME, format="pdf")
    except Exception:
        if output_dir is None:
            shutil.rmtree(artifact_dir, ignore_errors=True)
        raise
    return preview, str(png_path), str(pdf_path)


def export_handwriting(
    text: str,
    output_path,
    format: str = "png",
    style: str = _DEFAULTS["style"],
    paper: str = _DEFAULTS["paper"],
    layout: str = _DEFAULTS["layout"],
    font_size: int = _DEFAULTS["font_size"],
    **export_kwargs,
):
    """Generate and export a handwriting page for the demo."""
    page = generate_handwriting(
        text=text,
        style=style,
        paper=paper,
        layout=layout,
        font_size=font_size,
    )
    if page is None:
        return None
    return handwrite.export(page, output_path, format=format, **export_kwargs)


def build_demo():
    """Build the interactive Gradio demo."""
    import gradio as gr

    with gr.Blocks(title="HandWrite Demo", css=_CUSTOM_CSS) as demo:
        gr.Markdown(
            value=(
                "<div class='demo-hero'>"
                "<h1>HandWrite 中文手写页生成器</h1>"
                "<p>输入文字后即可预览结果，并直接下载 PNG 或 PDF 文件。</p>"
                "</div>"
            )
        )

        with gr.Row(elem_id="demo-shell"):
            with gr.Column(scale=4, elem_classes=["demo-panel"]):
                gr.Markdown(
                    value="### 生成设置\n<p class='demo-hint'>调整纸张、布局和字号后生成下载文件。</p>"
                )
                text_input = gr.Textbox(
                    label="输入文字",
                    lines=8,
                    placeholder="在这里输入要转换的中文内容……",
                )
                style_input = gr.Dropdown(
                    choices=_with_default_choice(_DEFAULTS["style"], handwrite.list_styles()),
                    value=_DEFAULTS["style"],
                    label="手写风格",
                )
                paper_input = gr.Dropdown(
                    choices=_with_default_choice(_DEFAULTS["paper"], _PAPER_CHOICES),
                    value=_DEFAULTS["paper"],
                    label="纸张类型",
                )
                layout_input = gr.Dropdown(
                    choices=_with_default_choice(_DEFAULTS["layout"], _LAYOUT_CHOICES),
                    value=_DEFAULTS["layout"],
                    label="排版风格",
                )
                font_size_input = gr.Slider(
                    minimum=40,
                    maximum=120,
                    value=_DEFAULTS["font_size"],
                    step=10,
                    label="字号",
                )
                generate_button = gr.Button(value="生成预览并导出", variant="primary")

            with gr.Column(scale=5, elem_classes=["demo-panel"]):
                gr.Markdown(
                    value="### 结果输出\n<p class='demo-hint'>生成后可直接下载导出的图片和 PDF。</p>"
                )
                preview_output = gr.Image(label="预览图", type="pil")
                png_output = gr.File(label="PNG 下载")
                pdf_output = gr.File(label="PDF 下载")

        generate_button.click(
            fn=generate_demo_artifacts,
            inputs=[
                text_input,
                style_input,
                paper_input,
                layout_input,
                font_size_input,
            ],
            outputs=[preview_output, png_output, pdf_output],
        )

    return demo


def main() -> None:
    """Launch the demo app."""
    build_demo().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
