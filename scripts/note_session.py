"""CLI entrypoint for generating classroom-note sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import handwrite

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
_PNG_PREFIX = "note"
_PDF_FILENAME = "note.pdf"
_REPORT_FILENAME = "note_session_report.md"


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a classroom-note session with exports and a report."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", type=str, help="Inline note text.")
    source_group.add_argument(
        "--text_file",
        type=Path,
        help="UTF-8 text file containing note text.",
    )
    source_group.add_argument(
        "--preset",
        type=str,
        choices=sorted(_NOTE_PRESETS),
        help="Built-in note preset.",
    )
    parser.add_argument("--style", type=str, default="行书流畅")
    parser.add_argument("--paper", type=str, default="横线纸")
    parser.add_argument("--layout", type=str, default="自然")
    parser.add_argument("--font_size", type=int, default=80)
    parser.add_argument("--prototype_pack", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def _resolve_text(args: argparse.Namespace) -> str:
    if isinstance(args.text, str):
        return args.text
    if isinstance(args.text_file, Path):
        return args.text_file.read_text(encoding="utf-8")
    if isinstance(args.preset, str):
        return _NOTE_PRESETS[args.preset]
    raise ValueError("one of --text, --text_file, or --preset is required")


def _write_session_report(output_dir: Path, session: dict[str, Any]) -> Path:
    report_path = output_dir / _REPORT_FILENAME
    status_text = str(session.get("status_text", "生成完成。"))
    report_markdown = str(session.get("report_markdown", "")).strip()
    body = f"# HandWrite Note Session\n\n{status_text}\n"
    if report_markdown:
        body += f"\n{report_markdown}\n"
    report_path.write_text(body, encoding="utf-8")
    return report_path


def build_note_session_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    text = _resolve_text(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    session = handwrite.build_note_session(
        text,
        style=args.style,
        paper=args.paper,
        layout=args.layout,
        font_size=int(args.font_size),
        prototype_pack=args.prototype_pack,
    )

    pages = session["pages"]
    png_paths = (
        handwrite.export_pages(pages, args.output_dir, format="png", prefix=_PNG_PREFIX)
        if pages
        else []
    )
    pdf_path = (
        handwrite.export_pages(pages, args.output_dir / _PDF_FILENAME, format="pdf")
        if pages
        else None
    )
    report_path = _write_session_report(args.output_dir, session)

    return {
        "text_source": (
            "inline"
            if args.text is not None
            else "file"
            if args.text_file is not None
            else "preset"
        ),
        "preset": args.preset,
        "output_dir": str(args.output_dir.resolve()),
        "page_count": int(session["page_count"]),
        "png_paths": [str(path) for path in png_paths],
        "pdf_path": None if pdf_path is None else str(pdf_path),
        "report_path": str(report_path),
        "status_text": str(session["status_text"]),
        "prototype_source": session.get("prototype_source"),
        "prototype_pack_name": session.get("prototype_pack_name"),
        "prototype_source_kind": session.get("prototype_source_kind"),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    summary = build_note_session_artifacts(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
