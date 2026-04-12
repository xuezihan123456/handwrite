"""Built-in and selected handwriting style definitions."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Union

BUILTIN_STYLES = {
    "工整楷书": 0,
    "圆润可爱": 1,
    "行书流畅": 2,
    "偏瘦紧凑": 3,
    "随意潦草": 4,
}
DEFAULT_STYLE_NAME = "行书流畅"


@dataclass(frozen=True)
class StyleDefinition:
    """A named handwriting style mapped to a numeric style id."""

    id: int
    name: str
    writer_id: Optional[str] = None


def list_style_names() -> list[str]:
    """Return built-in style names in stable order."""
    return list(BUILTIN_STYLES.keys())


def default_style_name() -> str:
    """Return the preferred default style for user-facing note generation."""
    return DEFAULT_STYLE_NAME


def load_selected_styles(path: Union[str, Path]) -> list[StyleDefinition]:
    """Load selected style definitions from a UTF-8 JSON file."""
    styles_path = Path(path)
    payload = json.loads(styles_path.read_text(encoding="utf-8"))

    return [
        StyleDefinition(
            id=style["id"],
            name=style["name"],
            writer_id=style["writer_id"],
        )
        for style in payload["styles"]
    ]


__all__ = [
    "BUILTIN_STYLES",
    "DEFAULT_STYLE_NAME",
    "StyleDefinition",
    "default_style_name",
    "list_style_names",
    "load_selected_styles",
]
