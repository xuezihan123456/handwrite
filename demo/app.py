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
_PAPER_DISPLAY = ["📄 白纸", "📝 横线纸", "🔲 方格纸", "⊞ 米字格"]
_PAPER_DISPLAY_TO_API = dict(zip(_PAPER_DISPLAY, _PAPER_CHOICES))
_LAYOUT_CHOICES = ["工整", "自然", "潦草"]
_NOTE_PRESETS = {
    "牛顿定律复习": (
        "今天上课主要复习牛顿第二定律：F = ma。\n"
        "老师强调先判断受力，再列出已知量，最后代入公式求解。\n"
        "例题一是水平推箱子，例题二是斜面受力分析。"
    ),
    "古诗文背诵": (
        "语文课整理：\n"
        "1. 先通读全文，标出停顿和重点字词。\n"
        "2. 第二遍背诵时抓住意象和情感变化。\n"
        "3. 默写前再检查易错字和标点。"
    ),
    "英语课堂摘记": (
        "English notes:\n"
        "Topic: passive voice in scientific writing.\n"
        "Key idea: focus on the process, not the actor.\n"
        "Examples: The sample was heated. The result was recorded."
    ),
}
_DEMO_OUTPUT_ROOT = Path(gettempdir()) / "handwrite-demo"
_PNG_PREFIX = "handwriting"
_PDF_FILENAME = "handwriting.pdf"
_REPORT_FILENAME = "handwriting_report.md"
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


def load_note_preset(preset_name: str) -> str:
    """Load a built-in classroom-note template into the text box."""
    return _NOTE_PRESETS.get(preset_name, "")


def _optional_prototype_kwargs(prototype_pack: Optional[PathLike] = None) -> dict[str, object]:
    if prototype_pack is None:
        return {}
    if isinstance(prototype_pack, Path):
        return {"prototype_pack": prototype_pack}
    normalized = str(prototype_pack).strip()
    if not normalized:
        return {}
    return {"prototype_pack": normalized}


def _format_precheck_for_user(report: dict[str, object] | None) -> str:
    """把技术性预检字典转为中文用户友好提示。"""
    try:
        if not isinstance(report, dict):
            return "ℹ️ 预检完成，可以开始生成"
        unique_characters = report.get("unique_characters") or []
        total_unique = len(unique_characters) if isinstance(unique_characters, list) else 0
        prototype_covered = report.get("prototype_covered_characters") or []
        model_supported = report.get("model_supported_characters") or []
        high_realism = (
            len(prototype_covered) if isinstance(prototype_covered, list) else 0
        ) + (
            len(model_supported) if isinstance(model_supported, list) else 0
        )
        if total_unique == 0:
            return "ℹ️ 预检完成，可以开始生成"
        coverage_pct = round(high_realism / total_unique * 100)
        if coverage_pct >= 80:
            return f"✅ 手写覆盖率 {coverage_pct}%，生成效果自然流畅"
        if coverage_pct >= 40:
            return f"⚠️ 手写覆盖率 {coverage_pct}%，部分字符使用标准字体替代"
        return f"ℹ️ 当前主要使用标准字体模式（覆盖率 {coverage_pct}%）"
    except Exception:
        return "ℹ️ 预检完成，可以开始生成"


def _friendly_error(e: Exception) -> str:
    """把技术性异常转为用户友好中文提示。"""
    msg = str(e).lower()
    if "prototype_pack" in msg or "manifest" in msg:
        return "原型库路径无效，请检查路径是否正确"
    if "too long" in msg or "capacity" in msg:
        return "文字太长，请减少文字数量后重试"
    if "too small" in msg:
        return "图片尺寸太小"
    return f"生成遇到问题，请重试（{type(e).__name__}）"


def _generation_status_from_report(report: dict[str, object] | None) -> str:
    if not isinstance(report, dict):
        return "生成完成。"
    prototype_source = report.get("prototype_source")
    if isinstance(prototype_source, dict):
        label = prototype_source.get("label")
        if isinstance(label, str) and label.strip():
            return f"生成完成，当前原型字库：{label}"
    return "生成完成。"


def _session_report_markdown(session: dict[str, object]) -> str:
    status_text = str(session.get("status_text", "生成完成。"))
    report_markdown = str(session.get("report_markdown", "")).strip()
    if report_markdown:
        return f"# HandWrite Note Session\n\n{status_text}\n\n{report_markdown}\n"
    return f"# HandWrite Note Session\n\n{status_text}\n"


def _write_session_report(output_dir: Path, session: dict[str, object]) -> str:
    report_path = output_dir / _REPORT_FILENAME
    report_path.write_text(_session_report_markdown(session), encoding="utf-8")
    return str(report_path)


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


def _page_status_text(page_number: int, page_count: int) -> str:
    if page_count < 1:
        return "Page 0 / 0"
    return f"Page {page_number} / {page_count}"


def _slider_update(**kwargs):
    import gradio as gr

    update = getattr(gr, "update", None)
    if callable(update):
        return update(**kwargs)
    return kwargs


def generate_handwriting(
    text: str,
    style: str = _DEFAULTS["style"],
    paper: str = _DEFAULTS["paper"],
    layout: str = _DEFAULTS["layout"],
    font_size: int = _DEFAULTS["font_size"],
    prototype_pack: Optional[PathLike] = None,
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
        **_optional_prototype_kwargs(prototype_pack),
    )


def inspect_demo_text(
    text: str,
    style: str = _DEFAULTS["style"],
    prototype_pack: Optional[PathLike] = None,
):
    """Inspect realism coverage for classroom-note text before generation."""
    if not text.strip():
        return (
            "## 课堂笔记预检\n"
            "- 还没有检测到正文内容。先贴入课堂笔记，再检查真实感覆盖。"
        )

    try:
        report = handwrite.inspect_text(
            text=text,
            style=style,
            **_optional_prototype_kwargs(prototype_pack),
        )
    except (FileNotFoundError, OSError) as error:
        return (
            "## 课堂笔记预检\n"
            f"- 自定义原型字库不可用: `{error}`\n"
            "- 请确认你填的是 prototype pack 目录或 manifest.json 路径。"
        )
    user_summary = _format_precheck_for_user(report)
    report_markdown = report.get("report_markdown")
    if isinstance(report_markdown, str) and report_markdown.strip():
        return f"{user_summary}\n\n{report_markdown}"
    return f"{user_summary}\n\n## 课堂笔记预检\n- 当前没有可展示的预检结果。"


def _generate_demo_document_bundle(
    text: str,
    style: str = _DEFAULTS["style"],
    paper: str = _DEFAULTS["paper"],
    layout: str = _DEFAULTS["layout"],
    font_size: int = _DEFAULTS["font_size"],
    prototype_pack: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
):
    if not text.strip():
        return None, None, None, None

    session = handwrite.build_note_session(
        text=text,
        style=style,
        paper=paper,
        layout=layout,
        font_size=int(font_size),
        **_optional_prototype_kwargs(prototype_pack),
    )
    pages = session["pages"]
    if not pages:
        return session, None, None, None

    artifact_dir = _resolve_output_dir(output_dir)
    try:
        png_paths = export_pages_png(pages, artifact_dir, prefix=_PNG_PREFIX)
        pdf_path = export_pages_pdf(pages, artifact_dir / _PDF_FILENAME)
        report_path = _write_session_report(artifact_dir, session)
    except Exception:
        if output_dir is None:
            shutil.rmtree(artifact_dir, ignore_errors=True)
        raise

    return session, [str(path) for path in png_paths], str(pdf_path), report_path


def generate_demo_document_artifacts(
    text: str,
    style: str = _DEFAULTS["style"],
    paper: str = _DEFAULTS["paper"],
    layout: str = _DEFAULTS["layout"],
    font_size: int = _DEFAULTS["font_size"],
    prototype_pack: Optional[PathLike] = None,
    preview_page_index: int = 0,
    output_dir: Optional[PathLike] = None,
):
    """Generate preview and multi-page exports for the demo."""
    session, png_paths, pdf_path, _report_path = _generate_demo_document_bundle(
        text=text,
        style=style,
        paper=paper,
        layout=layout,
        font_size=font_size,
        prototype_pack=prototype_pack,
        output_dir=output_dir,
    )
    if not session or not session.get("pages"):
        return None, None, None

    pages = session["pages"]
    preview = pages[_resolve_preview_page_index(preview_page_index, len(pages))]
    return preview, png_paths, pdf_path


def generate_demo_document_session(
    text: str,
    style: str = _DEFAULTS["style"],
    paper: str = _DEFAULTS["paper"],
    layout: str = _DEFAULTS["layout"],
    font_size: int = _DEFAULTS["font_size"],
    prototype_pack: Optional[PathLike] = None,
    preview_page_index: int = 0,
    output_dir: Optional[PathLike] = None,
):
    """Generate demo outputs plus cached preview state for page browsing."""
    try:
        session, png_paths, pdf_path, report_path = _generate_demo_document_bundle(
            text=text,
            style=style,
            paper=paper,
            layout=layout,
            font_size=font_size,
            prototype_pack=prototype_pack,
            output_dir=output_dir,
        )
    except Exception as error:
        return (
            None,
            None,
            None,
            None,
            None,
            _page_status_text(0, 0),
            _slider_update(minimum=1, maximum=1, value=1, visible=False),
            f"⚠️ {_friendly_error(error)}",
        )
    if not session or not session.get("pages"):
        return (
            None,
            None,
            None,
            None,
            None,
            _page_status_text(0, 0),
            _slider_update(minimum=1, maximum=1, value=1, visible=False),
            "还没有生成内容。",
        )

    pages = session["pages"]
    page_index = _resolve_preview_page_index(preview_page_index, len(pages))
    page_number = page_index + 1
    preview_state = {
        "pages": pages,
        "png_paths": png_paths,
        "pdf_path": pdf_path,
        "report_path": report_path,
    }
    return (
        pages[page_index],
        png_paths,
        pdf_path,
        report_path,
        preview_state,
        _page_status_text(page_number, len(pages)),
        _slider_update(
            minimum=1,
            maximum=len(pages),
            value=page_number,
            visible=len(pages) > 1,
        ),
        str(session.get("status_text", _generation_status_from_report(session.get("report")))),
    )


def change_preview_page(preview_page_number: int, preview_state):
    """Switch the preview image using cached pages from the last generation."""
    if not preview_state or not preview_state.get("pages"):
        return None, _page_status_text(0, 0)

    pages = preview_state["pages"]
    page_index = _resolve_preview_page_index(int(preview_page_number) - 1, len(pages))
    return pages[page_index], _page_status_text(page_index + 1, len(pages))


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
                "<h1>HandWrite Classroom Note Studio</h1>"
                "<p>Inspect realism first, then generate a flowing classroom note. "
                "Preview shows one page, while downloads include the full multi-page "
                "PNG set and PDF document.</p>"
                "</div>"
            )
        )

        with gr.Row(elem_id="demo-shell"):
            with gr.Column(scale=4, elem_classes=["demo-panel"]):
                gr.Markdown(
                    value=(
                        "### Classroom Note Input\n"
                        "<p class='demo-hint'>Start with a realism precheck. Long text "
                        "is split automatically across multiple pages for download.</p>"
                    )
                )
                text_input = gr.Textbox(
                    label="Text",
                    lines=10,
                    placeholder=(
                        "Paste classroom note text here. Inspect realism first, then "
                        "export the generated note as multi-page PNG and PDF files."
                    ),
                )
                preset_input = gr.Dropdown(
                    choices=list(_NOTE_PRESETS.keys()),
                    value=list(_NOTE_PRESETS.keys())[0],
                    label="课堂笔记模板",
                )
                preset_button = gr.Button(
                    value="Load Preset Note",
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
                prototype_pack_input = gr.Textbox(
                    label="Local Prototype Pack (optional)",
                    lines=1,
                    placeholder="留空使用内置包 | 扩充包示例：data/prototypes/font_note",
                )
                inspect_button = gr.Button(
                    value="Inspect Note Realism",
                )
                generate_button = gr.Button(
                    value="Generate Preview + Multi-Page Files",
                    variant="primary",
                )

            with gr.Column(scale=5, elem_classes=["demo-panel"]):
                gr.Markdown(
                    value=(
                        "### Precheck Report\n"
                        "<p class='demo-hint'>Use this report to spot lower-realism "
                        "characters before generating the full note.</p>"
                    )
                )
                precheck_report_output = gr.Markdown(
                    value=(
                        "## 课堂笔记预检\n"
                        "- 先点击 Inspect Note Realism，查看哪些字符会走较低真实感路径。"
                    )
                )
                gr.Markdown(
                    value=(
                        "### Output\n"
                        "<p class='demo-hint'>Preview shows the selected page image. "
                        "Downloads include every rendered PNG page plus a combined PDF.</p>"
                    )
                )
                preview_state = gr.State(value=None)
                preview_output = gr.Image(label="Preview Page", type="pil")
                page_status_output = gr.Markdown(value=_page_status_text(0, 0))
                generation_status_output = gr.Markdown(value="还没有开始生成。")
                preview_page_input = gr.Slider(
                    minimum=1,
                    maximum=1,
                    value=1,
                    step=1,
                    label="Preview Page",
                    visible=False,
                )
                png_output = gr.File(label="PNG Pages", file_count="multiple")
                pdf_output = gr.File(label="PDF Document")
                report_output = gr.File(label="Session Report")

        preset_button.click(
            fn=load_note_preset,
            inputs=[preset_input],
            outputs=[text_input],
        )
        inspect_button.click(
            fn=inspect_demo_text,
            inputs=[text_input, style_input, prototype_pack_input],
            outputs=[precheck_report_output],
        )
        generate_button.click(
            fn=generate_demo_document_session,
            inputs=[
                text_input,
                style_input,
                paper_input,
                layout_input,
                font_size_input,
                prototype_pack_input,
            ],
            outputs=[
                preview_output,
                png_output,
                pdf_output,
                report_output,
                preview_state,
                page_status_output,
                preview_page_input,
                generation_status_output,
            ],
        )
        preview_page_input.change(
            fn=change_preview_page,
            inputs=[preview_page_input, preview_state],
            outputs=[preview_output, page_status_output],
        )

        gr.Examples(
            examples=[
                [
                    "语文课笔记：\n今天学了《岳阳楼记》，重点句式：先天下之忧而忧，后天下之乐而乐。\n课后要求背诵全文，注意断句和情感变化。",
                    "行书流畅",
                    "横线纸",
                    "自然",
                    80,
                    "data/prototypes/font_note",
                ],
                [
                    "数学笔记：\n二次函数 y = ax² + bx + c\n顶点坐标：(-b/2a, (4ac-b²)/4a)\n判别式 Δ = b²-4ac，Δ>0两个实根，Δ=0两相等实根，Δ<0无实根。",
                    "工整楷书",
                    "方格纸",
                    "工整",
                    72,
                    "data/prototypes/font_note",
                ],
                [
                    "English Notes:\nGrammar focus: The Present Perfect Tense.\nForm: have/has + past participle.\nUsage: actions that happened at an unspecified time before now.\nExample: I have visited Beijing twice.",
                    "工整楷书",
                    "白纸",
                    "工整",
                    68,
                    "data/prototypes/font_note",
                ],
            ],
            inputs=[
                text_input,
                style_input,
                paper_input,
                layout_input,
                font_size_input,
                prototype_pack_input,
            ],
            label="内置示例（点击填入）",
        )

    return demo


def main() -> None:
    """Launch the demo app."""
    build_demo().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
