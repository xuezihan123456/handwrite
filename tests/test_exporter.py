from pathlib import Path

from PIL import Image

from handwrite.exporter import export_pages_png, export_png


def test_export_png_writes_a_readable_png_file(tmp_path: Path) -> None:
    output_path = tmp_path / "page.png"
    page = Image.new("L", (2480, 3508), color=255)

    export_png(page, str(output_path), dpi=300)

    assert output_path.exists()

    with Image.open(output_path) as reopened:
        assert reopened.size == (2480, 3508)
        assert reopened.format == "PNG"


def test_export_png_returns_the_written_path(tmp_path: Path) -> None:
    output_path = tmp_path / "exports" / "custom-page.png"
    page = Image.new("L", (1200, 1600), color=255)

    written_path = export_png(page, output_path, dpi=144)

    assert written_path == output_path
    assert written_path.exists()


def test_export_pages_png_writes_numbered_files_in_order(tmp_path: Path) -> None:
    pages = [
        Image.new("L", (800, 1200), color=255),
        Image.new("L", (810, 1210), color=240),
        Image.new("L", (820, 1220), color=225),
    ]

    written_paths = export_pages_png(pages, tmp_path / "batch", prefix="sheet", dpi=200)

    expected_paths = [
        tmp_path / "batch" / "sheet_001.png",
        tmp_path / "batch" / "sheet_002.png",
        tmp_path / "batch" / "sheet_003.png",
    ]
    assert written_paths == expected_paths

    for expected_path, expected_size in zip(expected_paths, [(800, 1200), (810, 1210), (820, 1220)]):
        assert expected_path.exists()
        with Image.open(expected_path) as reopened:
            assert reopened.size == expected_size
            assert reopened.format == "PNG"


def test_export_pages_png_sanitizes_path_like_prefix(tmp_path: Path) -> None:
    output_dir = tmp_path / "batch"
    pages = [Image.new("L", (800, 1200), color=255)]

    written_paths = export_pages_png(pages, output_dir, prefix="../escape", dpi=200)

    expected_path = output_dir / "escape_001.png"
    escaped_path = tmp_path / "escape_001.png"

    assert written_paths == [expected_path]
    assert expected_path.exists()
    assert escaped_path.exists() is False
