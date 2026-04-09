from pathlib import Path

import numpy as np
import pytest

import scripts.preprocess as preprocess_module
from scripts.preprocess import (
    _relative_metadata_path,
    compute_writer_coverage,
    is_valid_bitmap,
    normalize_bitmap,
    preprocess_dataset,
)


def test_is_valid_bitmap_rejects_extreme_sizes_and_blank_samples() -> None:
    too_small = np.full((16, 16), 255, dtype=np.uint8)
    too_large = np.full((400, 400), 255, dtype=np.uint8)
    blank = np.full((64, 64), 255, dtype=np.uint8)
    full_black = np.zeros((64, 64), dtype=np.uint8)
    valid = np.full((64, 64), 255, dtype=np.uint8)
    valid[16:48, 24:40] = 0

    assert not is_valid_bitmap(too_small)
    assert not is_valid_bitmap(too_large)
    assert not is_valid_bitmap(blank)
    assert not is_valid_bitmap(full_black)
    assert is_valid_bitmap(valid)


def test_normalize_bitmap_centers_foreground_on_256_canvas() -> None:
    bitmap = np.full((80, 60), 255, dtype=np.uint8)
    bitmap[20:70, 10:30] = 0

    normalized = normalize_bitmap(bitmap)

    assert normalized.shape == (256, 256)
    bbox = np.argwhere(normalized < 255)
    assert bbox.size > 0
    top_left = bbox.min(axis=0)
    bottom_right = bbox.max(axis=0)
    center_y = (top_left[0] + bottom_right[0]) / 2
    center_x = (top_left[1] + bottom_right[1]) / 2
    assert abs(center_x - 128) < 24
    assert abs(center_y - 128) < 24


def test_compute_writer_coverage_counts_unique_charset_hits() -> None:
    sample = np.full((32, 32), 255, dtype=np.uint8)
    parsed = {
        "001": [("你", sample), ("你", sample), ("好", sample), ("学", sample)],
        "002": [("你", sample), ("生", sample)],
    }

    coverage = compute_writer_coverage(parsed, {"你", "好", "生"})

    assert coverage == {"001": 2, "002": 2}


def test_preprocess_dataset_parses_raw_dir_and_delegates_to_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    font_path = tmp_path / "font.ttf"
    raw_dir.mkdir()

    parsed_data = {"001": [("你", np.full((64, 64), 127, dtype=np.uint8))]}
    expected_metadata = {"stats": {"total_pairs": 1}}
    calls: dict[str, object] = {}

    def fake_parse_gnt_directory(dir_path: str) -> dict[str, list[tuple[str, np.ndarray]]]:
        calls["raw_dir"] = dir_path
        return parsed_data

    def fake_build_processed_dataset(
        parsed_data_arg: dict[str, list[tuple[str, np.ndarray]]],
        output_dir_arg: str | Path,
        font_path_arg: str | Path,
        **kwargs: object,
    ) -> dict[str, object]:
        calls["parsed_data"] = parsed_data_arg
        calls["output_dir"] = output_dir_arg
        calls["font_path"] = font_path_arg
        calls["kwargs"] = kwargs
        return expected_metadata

    monkeypatch.setattr(preprocess_module, "parse_gnt_directory", fake_parse_gnt_directory)
    monkeypatch.setattr(preprocess_module, "build_processed_dataset", fake_build_processed_dataset)

    metadata = preprocess_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        font_path=font_path,
        charset=["你", "好"],
        min_writer_coverage=0.75,
        canvas_size=128,
        content_size=96,
    )

    assert metadata == expected_metadata
    assert calls == {
        "raw_dir": str(raw_dir),
        "parsed_data": parsed_data,
        "output_dir": output_dir,
        "font_path": font_path,
        "kwargs": {
            "charset": ["你", "好"],
            "min_writer_coverage": 0.75,
            "canvas_size": 128,
            "content_size": 96,
        },
    }


def test_relative_metadata_path_is_dataset_root_relative_for_relative_and_absolute_output_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "processed"
    asset_path = output_dir / "writer_001" / "0041_standard.png"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"")

    absolute_relative_path = _relative_metadata_path(asset_path, output_dir=output_dir)

    monkeypatch.chdir(tmp_path)

    relative_relative_path = _relative_metadata_path(asset_path, output_dir=Path("processed"))

    assert absolute_relative_path == "writer_001/0041_standard.png"
    assert relative_relative_path == absolute_relative_path
