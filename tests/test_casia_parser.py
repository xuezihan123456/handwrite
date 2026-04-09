from pathlib import Path
import struct

import numpy as np

from handwrite.data.casia_parser import parse_gnt_directory, parse_gnt_file


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


def test_parse_gnt_file_reads_samples(tmp_path: Path) -> None:
    gnt_path = tmp_path / "001.gnt"
    gnt_path.write_bytes(
        _build_sample("你", 2, 2, bytes([0, 255, 255, 0]))
        + _build_sample("好", 2, 2, bytes([255, 0, 0, 255]))
    )

    samples = parse_gnt_file(str(gnt_path))

    assert len(samples) == 2
    assert samples[0][0] == "你"
    assert samples[1][0] == "好"
    assert isinstance(samples[0][1], np.ndarray)
    assert samples[0][1].shape == (2, 2)


def test_parse_gnt_directory_groups_by_writer_id(tmp_path: Path) -> None:
    first = tmp_path / "001.gnt"
    second = tmp_path / "002.gnt"
    first.write_bytes(_build_sample("你", 1, 1, bytes([255])))
    second.write_bytes(_build_sample("好", 1, 1, bytes([0])))

    parsed = parse_gnt_directory(str(tmp_path))

    assert set(parsed) == {"001", "002"}
    assert parsed["001"][0][0] == "你"
    assert parsed["002"][0][0] == "好"
