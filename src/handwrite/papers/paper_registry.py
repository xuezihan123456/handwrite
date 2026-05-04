"""Paper registry: manages all available paper template definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from handwrite.papers.builtin_papers import BUILTIN_PAPER_DEFS, builtin_paper_names

_DATA_PAPERS_DIR = Path(__file__).resolve().parents[2] / ".." / "data" / "papers"


def _data_papers_dir() -> Path:
    """Resolve the data/papers directory relative to the project root."""
    candidates = [
        Path(__file__).resolve().parents[2] / ".." / "data" / "papers",
        Path(__file__).resolve().parents[3] / "data" / "papers",
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    return _DATA_PAPERS_DIR.resolve()


class PaperRegistry:
    """Central registry for paper template definitions.

    Supports two sources:
    - Built-in definitions (from builtin_papers module)
    - JSON file definitions (from data/papers/ directory or custom paths)
    """

    def __init__(self) -> None:
        self._definitions: dict[str, dict[str, Any]] = {}
        self._load_builtins()
        self._load_json_papers()

    def _load_builtins(self) -> None:
        for name, definition in BUILTIN_PAPER_DEFS.items():
            self._definitions[name] = definition

    def _load_json_papers(self) -> None:
        papers_dir = _data_papers_dir()
        if not papers_dir.is_dir():
            return
        for json_path in sorted(papers_dir.glob("*.json")):
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
                name = raw.get("name", json_path.stem)
                raw["_source"] = str(json_path)
                self._definitions[name] = raw
            except (json.JSONDecodeError, OSError):
                continue

    def register(self, definition: dict[str, Any]) -> None:
        """Register a paper definition dict directly."""
        name = definition.get("name")
        if not name:
            raise ValueError("Paper definition must have a 'name' field")
        self._definitions[name] = definition

    def load_json(self, path: str | Path) -> str:
        """Load a paper definition from a JSON file and register it.

        Returns the paper name.
        """
        json_path = Path(path)
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        name = raw.get("name", json_path.stem)
        raw["_source"] = str(json_path)
        self._definitions[name] = raw
        return name

    def get(self, name: str) -> dict[str, Any] | None:
        """Get a paper definition by name, or None if not found."""
        return self._definitions.get(name)

    def list_names(self) -> list[str]:
        """Return all registered paper names."""
        return list(self._definitions.keys())

    def list_builtin_names(self) -> list[str]:
        """Return built-in paper names only."""
        return builtin_paper_names()

    def __len__(self) -> int:
        return len(self._definitions)

    def __contains__(self, name: str) -> bool:
        return name in self._definitions


# Module-level singleton for convenience
_DEFAULT_REGISTRY: PaperRegistry | None = None


def _get_default_registry() -> PaperRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = PaperRegistry()
    return _DEFAULT_REGISTRY


def list_papers() -> list[str]:
    """Return all available paper names from the default registry."""
    return _get_default_registry().list_names()


def get_paper(name: str) -> dict[str, Any] | None:
    """Get a paper definition by name from the default registry."""
    return _get_default_registry().get(name)
