"""Gradio demo entrypoint."""

import shutil
from inspect import signature
from pathlib import Path
from tempfile import gettempdir
from typing import Optional, Union
from uuid import uuid4

import handwrite
from handwrite.exporter import export_pages_pdf, export_pages_png


PathLike = Union[str, Path]
_PAPER_CHOICES = ["白纸", "横线纸", "方格纸", "米字格"]
_LAYOUT_CHOICES = ["工整", "自然", "潦草"]
_DEMO_OUTPUT_ROOT = Path(gettempdir()) / "handwrite-demo"
_PNG_PREFIX = "handwriting"
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


def _resolve_preview_page_index(preview_page_index: int, page_count: int) -> int:
    if page_count < 1:
        raise ValueError("page_count must be positive")
    normalized_index = int(preview_page_index)
    return max(0, min(normalized_index, page_count - 1))


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


def generate_demo_document_artifacts(
    text: str,
    style: str = _DEFAULTS["style"],
    paper: str = _DEFAULTS["paper"],
    layout: str = _DEFAULTS["layout"],
    font_size: int = _DEFAULTS["font_size"],
    preview_page_index: int = 0,
    output_dir: Optional[PathLike] = None,
):
    """Generate preview and multi-page exports for the demo."""
    if not text.strip():
        return None, None, None

    pages = handwrite.generate_pages(
        text=text,
        style=style,
        paper=paper,
        layout=layout,
        font_size=int(font_size),
    )
    if not pages:
        return None, None, None

    preview = pages[_resolve_preview_page_index(preview_page_index, len(pages))]
    artifact_dir = _resolve_output_dir(output_dir)
    try:
        png_paths = export_pages_png(pages, artifact_dir, prefix=_PNG_PREFIX)
        pdf_path = export_pages_pdf(pages, artifact_dir / _PDF_FILENAME)
    except Exception:
        if output_dir is None:
            shutil.rmtree(artifact_dir, ignore_errors=True)
        raise

    return preview, [str(path) for path in png_paths], str(pdf_path)


def generate_demo_artifacts(
    text: str,
    style: str = _DEFAULTS["style"],
    paper: str = _DEFAULTS["paper"],
    layout: str = _DEFAULTS["layout"],
    font_size: int = _DEFAULTS["font_size"],
    output_dir: Optional[PathLike] = None,
):
    """Backward-compatible alias for demo artifact generation."""
    return generate_demo_document_artifacts(
        text=text,
        style=style,
        paper=paper,
        layout=layout,
        font_size=font_size,
        output_dir=output_dir,
    )


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
                "<h1>HandWrite Chinese Handwriting Demo</h1>"
                "<p>Paste a short note or a long passage. The preview shows one page, "
                "while downloads include the full multi-page PNG set and PDF document.</p>"
                "</div>"
            )
        )

        with gr.Row(elem_id="demo-shell"):
            with gr.Column(scale=4, elem_classes=["demo-panel"]):
                gr.Markdown(
                    value=(
                        "### Generation Settings\n"
                        "<p class='demo-hint'>Long text is split automatically across "
                        "multiple pages for download.</p>"
                    )
                )
                text_input = gr.Textbox(
                    label="Text",
                    lines=10,
                    placeholder=(
                        "Paste Chinese text here. Long passages will be exported as "
                        "multi-page PNG and PDF files."
                    ),
                )
                style_input = gr.Dropdown(
                    choices=_with_default_choice(_DEFAULTS["style"], handwrite.list_styles()),
                    value=_DEFAULTS["style"],
                    label="Handwriting Style",
                )
                paper_input = gr.Dropdown(
                    choices=_with_default_choice(_DEFAULTS["paper"], _PAPER_CHOICES),
                    value=_DEFAULTS["paper"],
                    label="Paper Type",
                )
                layout_input = gr.Dropdown(
                    choices=_with_default_choice(_DEFAULTS["layout"], _LAYOUT_CHOICES),
                    value=_DEFAULTS["layout"],
                    label="Layout Style",
                )
                font_size_input = gr.Slider(
                    minimum=40,
                    maximum=120,
                    value=_DEFAULTS["font_size"],
                    step=10,
                    label="Font Size",
                )
                generate_button = gr.Button(
                    value="Generate Preview + Multi-Page Files",
                    variant="primary",
                )

            with gr.Column(scale=5, elem_classes=["demo-panel"]):
                gr.Markdown(
                    value=(
                        "### Output\n"
                        "<p class='demo-hint'>Preview shows the selected page image. "
                        "Downloads include every rendered PNG page plus a combined PDF.</p>"
                    )
                )
                preview_output = gr.Image(label="Preview Page", type="pil")
                png_output = gr.File(label="PNG Pages", file_count="multiple")
                pdf_output = gr.File(label="PDF Document")

        generate_button.click(
            fn=generate_demo_document_artifacts,
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
