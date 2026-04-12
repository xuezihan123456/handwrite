from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from PIL import Image
import pytest


def _load_note_session_module():
    sys.modules.pop("scripts.note_session", None)
    return importlib.import_module("scripts.note_session")


def test_argument_parser_accepts_text_or_text_file(tmp_path: Path) -> None:
    note_session = _load_note_session_module()
    parser = note_session._build_argument_parser()

    args = parser.parse_args(
        [
            "--text",
            "\u8bfe\u5802\u7b14\u8bb0",
            "--output_dir",
            str(tmp_path / "output"),
        ]
    )
    assert args.text == "\u8bfe\u5802\u7b14\u8bb0"
    assert args.text_file is None
    assert args.output_dir == tmp_path / "output"
    assert args.prototype_pack is None

    text_file = tmp_path / "note.txt"
    text_file.write_text("\u6765\u81ea\u6587\u4ef6\u7684\u8bfe\u5802\u7b14\u8bb0", encoding="utf-8")
    args = parser.parse_args(
        [
            "--text_file",
            str(text_file),
            "--output_dir",
            str(tmp_path / "artifacts"),
            "--prototype_pack",
            str(tmp_path / "packs" / "custom"),
        ]
    )
    assert args.text is None
    assert args.text_file == text_file
    assert args.output_dir == tmp_path / "artifacts"
    assert args.prototype_pack == tmp_path / "packs" / "custom"


def test_main_builds_note_session_artifacts_from_inline_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    note_session = _load_note_session_module()
    page_one = Image.new("L", (2480, 3508), color=255)
    page_two = Image.new("L", (2480, 3508), color=240)
    output_dir = tmp_path / "session-output"
    calls: dict[str, object] = {}

    def fake_build_note_session(
        text: str,
        style: str,
        paper: str,
        layout: str,
        font_size: int,
        prototype_pack: Path | None = None,
    ) -> dict[str, object]:
        calls["build_note_session"] = {
            "text": text,
            "style": style,
            "paper": paper,
            "layout": layout,
            "font_size": font_size,
            "prototype_pack": prototype_pack,
        }
        return {
            "pages": [page_one, page_two],
            "page_count": 2,
            "status_text": "\u5df2\u751f\u6210 2 \u9875\u8bfe\u5802\u7b14\u8bb0\uff0c\u5f53\u524d\u539f\u578b\u5b57\u5e93\uff1aBuilt-in starter pack: default_note",
            "report_markdown": "## \u8bfe\u5802\u7b14\u8bb0\u9884\u68c0\n- \u5f53\u524d\u539f\u578b\u5b57\u5e93: Built-in starter pack: default_note",
            "prototype_source": {
                "label": "Built-in starter pack: default_note",
            },
            "prototype_pack_name": "default_note",
            "prototype_source_kind": "builtin",
        }

    def fake_export_pages(pages, output_path, format="png", **kwargs):
        calls.setdefault("export_pages", []).append(
            {
                "pages": pages,
                "output_path": output_path,
                "format": format,
                "kwargs": kwargs,
            }
        )
        if format == "png":
            return [
                output_dir / "note_001.png",
                output_dir / "note_002.png",
            ]
        return output_dir / "note.pdf"

    monkeypatch.setattr(note_session.handwrite, "build_note_session", fake_build_note_session)
    monkeypatch.setattr(note_session.handwrite, "export_pages", fake_export_pages)

    exit_code = note_session.main(
        [
            "--text",
            "\u8bfe\u5802\u7b14\u8bb0",
            "--output_dir",
            str(output_dir),
            "--style",
            "\u884c\u4e66\u6d41\u7545",
            "--paper",
            "\u6a2a\u7ebf\u7eb8",
            "--layout",
            "\u81ea\u7136",
            "--font_size",
            "88",
            "--prototype_pack",
            str(tmp_path / "packs" / "custom-pack"),
        ]
    )

    assert exit_code == 0
    assert calls["build_note_session"] == {
        "text": "\u8bfe\u5802\u7b14\u8bb0",
        "style": "\u884c\u4e66\u6d41\u7545",
        "paper": "\u6a2a\u7ebf\u7eb8",
        "layout": "\u81ea\u7136",
        "font_size": 88,
        "prototype_pack": tmp_path / "packs" / "custom-pack",
    }
    assert calls["export_pages"] == [
        {
            "pages": [page_one, page_two],
            "output_path": output_dir,
            "format": "png",
            "kwargs": {"prefix": "note"},
        },
        {
            "pages": [page_one, page_two],
            "output_path": output_dir / "note.pdf",
            "format": "pdf",
            "kwargs": {},
        },
    ]
    report_path = output_dir / "note_session_report.md"
    assert report_path.exists()
    assert "HandWrite Note Session" in report_path.read_text(encoding="utf-8")
    payload = json.loads(capsys.readouterr().out)
    assert payload["page_count"] == 2
    assert payload["report_path"] == str(report_path)
    assert payload["pdf_path"] == str(output_dir / "note.pdf")
    assert payload["png_paths"] == [
        str(output_dir / "note_001.png"),
        str(output_dir / "note_002.png"),
    ]


def test_main_reads_text_file_and_skips_exports_when_no_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    note_session = _load_note_session_module()
    text_file = tmp_path / "note.txt"
    text_file.write_text("\u6765\u81ea\u6587\u4ef6\u7684\u8bfe\u5802\u7b14\u8bb0", encoding="utf-8")
    output_dir = tmp_path / "session-output"
    calls: dict[str, object] = {"export_calls": 0}

    def fake_build_note_session(
        text: str,
        style: str,
        paper: str,
        layout: str,
        font_size: int,
        prototype_pack: Path | None = None,
    ) -> dict[str, object]:
        calls["text"] = text
        return {
            "pages": [],
            "page_count": 0,
            "status_text": "\u8fd8\u6ca1\u6709\u53ef\u751f\u6210\u7684\u8bfe\u5802\u7b14\u8bb0\u5185\u5bb9\u3002",
            "report_markdown": "## \u8bfe\u5802\u7b14\u8bb0\u9884\u68c0\n- \u8fd8\u6ca1\u6709\u53ef\u751f\u6210\u7684\u8bfe\u5802\u7b14\u8bb0\u5185\u5bb9\u3002",
            "prototype_source": {
                "label": "Built-in starter pack: default_note",
            },
            "prototype_pack_name": "default_note",
            "prototype_source_kind": "builtin",
        }

    def fake_export_pages(*args, **kwargs):
        calls["export_calls"] += 1
        raise AssertionError("export should not run when there are no pages")

    monkeypatch.setattr(note_session.handwrite, "build_note_session", fake_build_note_session)
    monkeypatch.setattr(note_session.handwrite, "export_pages", fake_export_pages)

    exit_code = note_session.main(
        [
            "--text_file",
            str(text_file),
            "--output_dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert calls["text"] == "\u6765\u81ea\u6587\u4ef6\u7684\u8bfe\u5802\u7b14\u8bb0"
    assert calls["export_calls"] == 0
    report_path = output_dir / "note_session_report.md"
    assert report_path.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["page_count"] == 0
    assert payload["report_path"] == str(report_path)
    assert payload["png_paths"] == []
    assert payload["pdf_path"] is None


def test_main_supports_note_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    note_session = _load_note_session_module()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        note_session.handwrite,
        "build_note_session",
        lambda text, **kwargs: (
            calls.__setitem__("text", text),
            {
                "pages": [],
                "page_count": 0,
                "status_text": "\u8fd8\u6ca1\u6709\u53ef\u751f\u6210\u7684\u8bfe\u5802\u7b14\u8bb0\u5185\u5bb9\u3002",
                "report_markdown": "## \u8bfe\u5802\u7b14\u8bb0\u9884\u68c0\n- \u7a7a\u5185\u5bb9",
                "prototype_source": {"label": "Built-in starter pack: default_note"},
                "prototype_pack_name": "default_note",
                "prototype_source_kind": "builtin",
            },
        )[1],
    )
    monkeypatch.setattr(note_session.handwrite, "export_pages", lambda *args, **kwargs: [])

    exit_code = note_session.main(
        [
            "--preset",
            "\u725b\u987f\u5b9a\u5f8b\u590d\u4e60",
            "--output_dir",
            str(tmp_path / "session"),
        ]
    )

    assert exit_code == 0
    assert "\u725b\u987f\u7b2c\u4e8c\u5b9a\u5f8b" in calls["text"]
