"""Prototype glyph library helpers for handwritten starter packs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class PrototypeGlyph:
    """Metadata for a single packaged or local prototype glyph."""

    char: str
    path: Path
    writer_id: str | None = None


class PrototypeLibrary:
    """A small handwritten glyph library used before low-realism fallback."""

    def __init__(
        self,
        *,
        name: str,
        root: Path,
        manifest_path: Path,
        source_kind: str,
        glyphs: dict[str, PrototypeGlyph],
    ) -> None:
        self.name = name
        self.root = root
        self.manifest_path = manifest_path
        self.source_kind = source_kind
        self._glyphs = glyphs

    @property
    def prototype_source(self) -> str:
        """Return the concrete manifest path backing this prototype pack."""
        return str(self.manifest_path)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        source_kind: str = "custom",
    ) -> "PrototypeLibrary":
        manifest_file = Path(manifest_path).resolve()
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        glyphs: dict[str, PrototypeGlyph] = {}
        for entry in payload.get("glyphs", []):
            glyph_path = (manifest_file.parent / entry["file"]).resolve()
            glyphs[entry["char"]] = PrototypeGlyph(
                char=entry["char"],
                path=glyph_path,
                writer_id=entry.get("writer_id"),
            )
        return cls(
            name=payload.get("name", manifest_file.parent.name),
            root=manifest_file.parent.resolve(),
            manifest_path=manifest_file,
            source_kind=source_kind,
            glyphs=glyphs,
        )

    def has_char(self, char: str) -> bool:
        return char in self._glyphs

    def get_glyph_path(self, char: str) -> Path | None:
        glyph = self._glyphs.get(char)
        return None if glyph is None else glyph.path

    def get_glyph_image(self, char: str) -> Image.Image:
        glyph_path = self.get_glyph_path(char)
        if glyph_path is None:
            raise KeyError(f"Prototype glyph not found for {char!r}")
        return Image.open(glyph_path).convert("L")

    def coverage_summary(self, text: str) -> dict[str, Any]:
        ordered_chars: list[str] = []
        seen_chars: set[str] = set()
        for char in text:
            if char.isspace() or char in seen_chars:
                continue
            seen_chars.add(char)
            ordered_chars.append(char)

        covered_chars = sorted(char for char in ordered_chars if self.has_char(char))
        missing_chars = sorted(char for char in ordered_chars if not self.has_char(char))
        total_chars = sum(1 for char in text if not char.isspace())
        return {
            "name": self.name,
            "source_kind": self.source_kind,
            "manifest_path": str(self.manifest_path),
            "prototype_source": self.prototype_source,
            "total_chars": total_chars,
            "unique_chars": len(ordered_chars),
            "covered_chars": covered_chars,
            "missing_chars": missing_chars,
        }


def resolve_prototype_manifest_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        manifest_path = candidate / "manifest.json"
        if manifest_path.is_file():
            return manifest_path
        raise FileNotFoundError(f"Prototype pack directory is missing manifest.json: {candidate}")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Prototype pack path does not exist: {candidate}")


def load_builtin_prototype_library(pack_name: str = "default_note") -> PrototypeLibrary:
    manifest_path = (
        Path(str(files("handwrite")))
        / "assets"
        / "prototypes"
        / pack_name
        / "manifest.json"
    )
    return PrototypeLibrary.from_manifest(manifest_path, source_kind="builtin")


def load_prototype_library(
    prototype_pack: str | Path | None = None,
    *,
    pack_name: str = "default_note",
) -> PrototypeLibrary:
    if prototype_pack is None:
        return load_builtin_prototype_library(pack_name=pack_name)
    return PrototypeLibrary.from_manifest(
        resolve_prototype_manifest_path(prototype_pack),
        source_kind="custom",
    )


__all__ = [
    "PrototypeGlyph",
    "PrototypeLibrary",
    "load_builtin_prototype_library",
    "load_prototype_library",
    "resolve_prototype_manifest_path",
]
