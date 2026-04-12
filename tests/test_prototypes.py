from pathlib import Path
import json
from importlib.resources import files

from PIL import Image

from handwrite.prototypes import (
    PrototypeLibrary,
    load_builtin_prototype_library,
    load_prototype_library,
    resolve_prototype_manifest_path,
)
from scripts.build_prototype_library import build_prototype_library


def _write_sample_glyph(path: Path, *, color: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (256, 256), color=255)
    image.paste(color, (96, 96, 160, 160))
    image.save(path)


def test_prototype_library_loads_manifest_and_resolves_glyph(tmp_path: Path) -> None:
    glyph_path = tmp_path / "glyphs" / "U4F60.png"
    _write_sample_glyph(glyph_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "test-pack",
                "glyphs": [
                    {"char": "你", "file": "glyphs/U4F60.png", "writer_id": "starter"}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    library = PrototypeLibrary.from_manifest(manifest_path)

    assert library.name == "test-pack"
    assert library.has_char("你") is True
    assert library.has_char("好") is False
    assert library.get_glyph_path("你") == glyph_path.resolve()
    assert library.get_glyph_image("你").size == (256, 256)
    assert library.coverage_summary("你好你")["covered_chars"] == ["你"]
    assert library.coverage_summary("你好")["missing_chars"] == ["好"]


def test_builtin_prototype_library_contains_note_starter_characters() -> None:
    library = load_builtin_prototype_library()

    assert library.name == "default_note"
    assert library.has_char("你") is True
    assert library.has_char("学") is True
    assert library.get_glyph_image("你").size == (256, 256)


def test_builtin_prototype_package_resources_are_present() -> None:
    handwrite_root = Path(str(files("handwrite")))
    manifest_path = handwrite_root / "assets" / "prototypes" / "default_note" / "manifest.json"
    glyph_path = handwrite_root / "assets" / "prototypes" / "default_note" / "glyphs" / "U4F60.png"

    assert manifest_path.is_file()
    assert glyph_path.is_file()


def test_load_prototype_library_accepts_pack_directory(tmp_path: Path) -> None:
    glyph_path = tmp_path / "glyphs" / "U4F60.png"
    _write_sample_glyph(glyph_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "local-pack",
                "glyphs": [
                    {"char": "你", "file": "glyphs/U4F60.png", "writer_id": "001"}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert resolve_prototype_manifest_path(tmp_path) == manifest_path.resolve()
    library = load_prototype_library(tmp_path)

    assert library.name == "local-pack"
    assert library.source_kind == "custom"
    assert library.manifest_path == manifest_path.resolve()
    assert library.has_char("你") is True


def test_load_prototype_library_accepts_pack_directory_or_manifest_path(tmp_path: Path) -> None:
    glyph_path = tmp_path / "custom-pack" / "glyphs" / "U9F98.png"
    _write_sample_glyph(glyph_path, color=32)
    manifest_path = tmp_path / "custom-pack" / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "custom-pack",
                "glyphs": [
                    {"char": "龘", "file": "glyphs/U9F98.png", "writer_id": "custom"}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    directory_library = load_prototype_library(tmp_path / "custom-pack")
    manifest_library = load_prototype_library(manifest_path)

    assert directory_library.name == "custom-pack"
    assert directory_library.prototype_source == str(manifest_path.resolve())
    assert directory_library.source_kind == "custom"
    assert directory_library.has_char("龘") is True
    assert manifest_library.name == "custom-pack"
    assert manifest_library.prototype_source == str(manifest_path.resolve())
    assert manifest_library.source_kind == "custom"
    assert manifest_library.get_glyph_image("龘").size == (256, 256)


def test_build_prototype_library_copies_one_sample_per_character(tmp_path: Path) -> None:
    source_root = tmp_path / "processed"
    source_root.mkdir()
    image_one = source_root / "writer_001" / "4F60_handwrite.png"
    image_two = source_root / "writer_001" / "5B66_handwrite.png"
    _write_sample_glyph(image_one, color=32)
    _write_sample_glyph(image_two, color=64)

    metadata_path = source_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {"char": "你", "writer_id": "001", "handwrite": "writer_001/4F60_handwrite.png"},
                    {"char": "学", "writer_id": "001", "handwrite": "writer_001/5B66_handwrite.png"},
                    {"char": "你", "writer_id": "002", "handwrite": "writer_001/4F60_handwrite.png"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "prototype-pack"
    summary = build_prototype_library(
        metadata_path=metadata_path,
        output_dir=output_dir,
        pack_name="prototype-pack",
    )

    assert summary["pack_name"] == "prototype-pack"
    assert summary["glyph_count"] == 2
    assert summary["chars"] == ["你", "学"]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "prototype-pack"
    assert manifest["glyphs"] == [
        {"char": "你", "file": "glyphs/U4F60.png", "writer_id": "001"},
        {"char": "学", "file": "glyphs/U5B66.png", "writer_id": "001"},
    ]
    assert (output_dir / "glyphs" / "U4F60.png").exists()
    assert (output_dir / "glyphs" / "U5B66.png").exists()
