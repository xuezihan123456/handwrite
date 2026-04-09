import json
from pathlib import Path
import struct

import numpy as np
import pytest
from PIL import Image

from scripts.preprocess import build_processed_dataset, preprocess_dataset


def _require_test_font() -> str:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/NotoSerifSC-VF.ttf"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    pytest.skip("No usable test font found in common system font locations.")


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


def _make_bitmap(height: int = 64, width: int = 64, *, top: int = 16, left: int = 20) -> np.ndarray:
    bitmap = np.full((height, width), 255, dtype=np.uint8)
    bitmap[top : top + 24, left : left + 20] = 0
    return bitmap


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


def test_build_processed_dataset_writes_pairs_and_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    parsed_data = {
        "001": [
            ("A", _make_bitmap()),
            ("A", _make_bitmap(left=22)),
            ("B", _make_bitmap(top=18, left=18)),
            ("C", np.full((64, 64), 255, dtype=np.uint8)),
        ],
        "002": [
            ("A", _make_bitmap()),
            ("B", np.full((64, 64), 255, dtype=np.uint8)),
        ],
    }

    metadata = build_processed_dataset(
        parsed_data,
        output_dir=output_dir,
        font_path=_require_test_font(),
        charset=["A", "B"],
        min_writer_coverage=1.0,
    )

    metadata_path = output_dir / "metadata.json"
    writer_dir = output_dir / "writer_001"

    assert metadata_path.exists()
    assert writer_dir.exists()
    assert not (output_dir / "writer_002").exists()
    assert metadata["writers"] == ["001"]
    assert metadata["charset"] == ["A", "B"]
    assert metadata["stats"] == {
        "total_pairs": 2,
        "num_writers": 1,
        "num_chars": 2,
    }

    pair_map = {pair["unicode_hex"]: pair for pair in metadata["pairs"]}
    assert set(pair_map) == {"0041", "0042"}
    assert pair_map["0041"] == {
        "writer_id": "001",
        "char": "A",
        "unicode_hex": "0041",
        "standard": "writer_001/0041_standard.png",
        "handwrite": "writer_001/0041_handwrite.png",
    }
    assert pair_map["0042"] == {
        "writer_id": "001",
        "char": "B",
        "unicode_hex": "0042",
        "standard": "writer_001/0042_standard.png",
        "handwrite": "writer_001/0042_handwrite.png",
    }

    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata == metadata

    with Image.open(writer_dir / "0041_handwrite.png") as handwrite_image:
        assert handwrite_image.mode == "L"
        assert handwrite_image.size == (256, 256)

    with Image.open(writer_dir / "0041_standard.png") as standard_image:
        assert standard_image.mode == "L"
        assert standard_image.size == (256, 256)


def test_preprocess_dataset_parses_gnt_directory_and_writes_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    first_writer = (
        _build_sample("\u4f60", 64, 64, _make_bitmap().tobytes())
        + _build_sample("\u597d", 64, 64, _make_bitmap(top=18, left=18).tobytes())
    )
    second_writer = _build_sample("\u4f60", 64, 64, _make_bitmap().tobytes())
    (raw_dir / "001.gnt").write_bytes(first_writer)
    (raw_dir / "002.gnt").write_bytes(second_writer)

    metadata = preprocess_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        font_path=_require_cjk_font(),
        charset=["\u4f60", "\u597d"],
        min_writer_coverage=1.0,
    )

    metadata_path = output_dir / "metadata.json"

    assert metadata_path.exists()
    assert metadata["writers"] == ["001"]
    assert metadata["charset"] == ["\u4f60", "\u597d"]
    assert metadata["stats"] == {
        "total_pairs": 2,
        "num_writers": 1,
        "num_chars": 2,
    }
    assert (output_dir / "writer_001" / "4F60_handwrite.png").exists()
    assert (output_dir / "writer_001" / "597D_standard.png").exists()
    assert not (output_dir / "writer_002").exists()

    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata == metadata
