import json
import math
from pathlib import Path
import struct

import numpy as np
import pytest

import scripts.preprocess as preprocess_module
from handwrite.data.charsets import get_charset


def _require_cjk_font() -> str:
    candidates = [
        Path("C:/Windows/Fonts/NotoSerifSC-VF.ttf"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    pytest.skip("No CJK-capable font found in common system font locations.")


def _build_sample(char: str, width: int, height: int, pixels: bytes) -> bytes:
    label = char.encode("gbk")
    sample_size = 10 + len(pixels)
    return b"".join(
        [
            struct.pack("<I", sample_size),
            label,
            struct.pack("<H", width),
            struct.pack("<H", height),
            pixels,
        ]
    )


def _make_bitmap(height: int = 64, width: int = 64, *, top: int = 16, left: int = 20) -> np.ndarray:
    bitmap = np.full((height, width), 255, dtype=np.uint8)
    bitmap[top : top + 24, left : left + 20] = 0
    return bitmap


def test_main_resolves_default_charset_and_calls_preprocess_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    font_path = tmp_path / "font.ttf"
    raw_dir.mkdir()

    calls: dict[str, object] = {}

    def fake_get_charset(level: str) -> list[str]:
        calls["charset_level"] = level
        return ["\u4f60", "\u597d"]

    def fake_preprocess_dataset(
        raw_dir_arg: str | Path,
        output_dir_arg: str | Path,
        font_path_arg: str | Path,
        *,
        charset: list[str],
        min_writer_coverage: float,
        canvas_size: int = 256,
        content_size: int = 200,
    ) -> dict[str, object]:
        calls["preprocess_args"] = {
            "raw_dir": raw_dir_arg,
            "output_dir": output_dir_arg,
            "font_path": font_path_arg,
            "charset": charset,
            "min_writer_coverage": min_writer_coverage,
            "canvas_size": canvas_size,
            "content_size": content_size,
        }
        return {"stats": {"total_pairs": 0}}

    monkeypatch.setattr(preprocess_module, "get_charset", fake_get_charset)
    monkeypatch.setattr(preprocess_module, "preprocess_dataset", fake_preprocess_dataset)

    exit_code = preprocess_module.main(
        [
            "--raw_dir",
            str(raw_dir),
            "--output_dir",
            str(output_dir),
            "--font_path",
            str(font_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert calls == {
        "charset_level": "500",
        "preprocess_args": {
            "raw_dir": raw_dir,
            "output_dir": output_dir,
            "font_path": font_path,
            "charset": ["\u4f60", "\u597d"],
            "min_writer_coverage": 0.9,
            "canvas_size": 256,
            "content_size": 200,
        },
    }
    assert payload == {
        "metadata_path": str((output_dir / "metadata.json").resolve()),
        "stats": {"total_pairs": 0},
    }
    assert captured.err == ""


def test_main_processes_synthetic_gnt_directory_with_real_defaults(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    charset = get_charset("500")
    gnt_supported_charset = [char for char in charset if len(char.encode("gbk")) == 2]
    required_chars = gnt_supported_charset[: math.ceil(len(gnt_supported_charset) * 0.9)]
    writer_samples = b"".join(
        _build_sample(char, 64, 64, _make_bitmap(top=16 + (index % 4), left=20 + (index % 5)).tobytes())
        for index, char in enumerate(required_chars)
    )
    (raw_dir / "001.gnt").write_bytes(writer_samples)

    exit_code = preprocess_module.main(
        [
            "--raw_dir",
            str(raw_dir),
            "--output_dir",
            str(output_dir),
            "--font_path",
            _require_cjk_font(),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert (output_dir / "metadata.json").exists()
    assert payload == {
        "metadata_path": str((output_dir / "metadata.json").resolve()),
        "stats": metadata["stats"],
    }
    assert payload["stats"] == {
        "total_pairs": len(required_chars),
        "num_writers": 1,
        "num_chars": len(gnt_supported_charset),
    }
    assert metadata["writers"] == ["001"]
    assert metadata["charset"] == gnt_supported_charset
    assert metadata["pairs"]
    first_pair = metadata["pairs"][0]
    assert first_pair["standard"].startswith("writer_001/")
    assert first_pair["handwrite"].startswith("writer_001/")
    assert (output_dir / first_pair["standard"]).exists()
    assert (output_dir / first_pair["handwrite"]).exists()
    assert captured.err == ""
