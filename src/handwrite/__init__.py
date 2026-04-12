"""Public API for the HandWrite package."""

from pathlib import Path

from PIL import Image

from . import exporter as _exporter
from handwrite.composer import (
    CURSIVE_LAYOUT,
    GRID_PAPER,
    MI_PAPER,
    NATURAL_LAYOUT,
    NEAT_LAYOUT,
    RULED_PAPER,
    WHITE_PAPER,
    compose_page,
)
from handwrite.engine.model import StyleEngine
from handwrite.prototypes import resolve_prototype_manifest_path
from handwrite.styles import BUILTIN_STYLES, default_style_name, list_style_names

_ENGINE: StyleEngine | None = None
_ENGINE_CACHE: dict[tuple[str | None], StyleEngine] = {}
_DEFAULT_STYLE = default_style_name()
_DEFAULT_PAGE_SIZE = (2480, 3508)
_DEFAULT_MARGINS = (200, 200, 200, 200)
_GUIDED_PAPERS = {RULED_PAPER, GRID_PAPER, MI_PAPER}
_GRID_PAPERS = {GRID_PAPER, MI_PAPER}
_LEADING_PUNCTUATION = set(
    "\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001,.!?;:)]}"
    "\u3011\uff09\u300b\u300d\u300f\u2019\u201d"
)


def _engine_cache_key(prototype_pack: str | Path | None = None) -> tuple[str | None]:
    if prototype_pack is None or str(prototype_pack).strip() == "":
        return (None,)
    resolved = resolve_prototype_manifest_path(prototype_pack)
    return (str(resolved),)


def _get_engine(prototype_pack: str | Path | None = None) -> StyleEngine:
    global _ENGINE
    if prototype_pack is None and _ENGINE is not None:
        return _ENGINE

    cache_key = _engine_cache_key(prototype_pack)
    engine = _ENGINE_CACHE.get(cache_key)
    if engine is None:
        engine = StyleEngine(prototype_pack=None if cache_key[0] is None else cache_key[0])
        _ENGINE_CACHE[cache_key] = engine
        if cache_key[0] is None:
            _ENGINE = engine
    return engine


def list_styles() -> list[str]:
    """Return the built-in handwriting styles."""
    return list_style_names()


def inspect_text(
    text: str,
    style: str = _DEFAULT_STYLE,
    prototype_pack: str | Path | None = None,
) -> dict[str, object]:
    """Inspect text coverage and realism before generating note pages."""
    style_id = BUILTIN_STYLES[style]
    engine = _get_engine(prototype_pack=prototype_pack)
    inspect_fn = getattr(engine, "inspect_text", None)
    if callable(inspect_fn):
        report = inspect_fn(text, style_id)
        return _normalize_inspection_report(text, style, report, prototype_pack=prototype_pack)
    return _normalize_inspection_report(text, style, None, prototype_pack=prototype_pack)


def char(
    text: str,
    style: str = _DEFAULT_STYLE,
    prototype_pack: str | Path | None = None,
) -> Image.Image:
    """Generate a single handwriting character image."""
    style_id = BUILTIN_STYLES[style]
    return _get_engine(prototype_pack=prototype_pack).generate_char(text, style_id)


def _prototype_kwargs(prototype_pack: str | Path | None = None) -> dict[str, object]:
    if prototype_pack is None:
        return {}
    return {"prototype_pack": prototype_pack}


def generate(
    text: str,
    style: str = _DEFAULT_STYLE,
    paper: str = WHITE_PAPER,
    layout: str = NATURAL_LAYOUT,
    font_size: int = 80,
    prototype_pack: str | Path | None = None,
) -> Image.Image:
    """Generate a single handwriting page image."""
    images = [
        char(c, style=style, **_prototype_kwargs(prototype_pack))
        for c in text
        if c not in {" ", "\n"}
    ]
    return compose_page(images, text, font_size=font_size, layout=layout, paper=paper)


def build_note_session(
    text: str,
    style: str = _DEFAULT_STYLE,
    paper: str = WHITE_PAPER,
    layout: str = NATURAL_LAYOUT,
    font_size: int = 80,
    prototype_pack: str | Path | None = None,
) -> dict[str, object]:
    """Build a complete classroom-note generation session payload."""
    if not text.strip():
        return {
            "text": text,
            "style": style,
            "paper": paper,
            "layout": layout,
            "font_size": int(font_size),
            "report": None,
            "report_markdown": "## 课堂笔记预检\n- 还没有可生成的课堂笔记内容。",
            "pages": [],
            "page_count": 0,
            "prototype_source": _prototype_source_from_argument(prototype_pack),
            "prototype_pack_name": _prototype_pack_name_from_argument(prototype_pack),
            "prototype_source_kind": (
                "builtin" if prototype_pack is None or str(prototype_pack).strip() == "" else "custom"
            ),
            "status_text": "还没有可生成的课堂笔记内容。",
        }

    report = inspect_text(text, style=style, prototype_pack=prototype_pack)
    pages = generate_pages(
        text,
        style=style,
        paper=paper,
        layout=layout,
        font_size=font_size,
        prototype_pack=prototype_pack,
    )
    page_count = len(pages)
    status_text = _build_note_session_status(page_count, report)
    return {
        "text": text,
        "style": style,
        "paper": paper,
        "layout": layout,
        "font_size": int(font_size),
        "report": report,
        "report_markdown": report.get("report_markdown", ""),
        "pages": pages,
        "page_count": page_count,
        "prototype_source": report.get("prototype_source"),
        "prototype_pack_name": report.get("prototype_pack_name"),
        "prototype_source_kind": report.get("prototype_source_kind"),
        "status_text": status_text,
    }


def export(page: Image.Image, output_path, format: str = "png", **kwargs):
    """Export a generated page as PNG or PDF."""
    normalized_format = format.lower()
    if normalized_format == "png":
        return _exporter.export_png(page, output_path, **kwargs)
    if normalized_format == "pdf":
        return _exporter.export_pdf(page, output_path, **kwargs)
    raise ValueError("format must be 'png' or 'pdf'")


def _normalize_inspection_report(
    text: str,
    style: str,
    report: dict[str, object] | None,
    *,
    prototype_pack: str | Path | None = None,
) -> dict[str, object]:
    normalized = dict(report or {})
    unique_characters = normalized.get("unique_characters")
    if not isinstance(unique_characters, list):
        unique_characters = _unique_non_whitespace_characters(text)

    prototype_covered_characters = _coerce_character_list(
        normalized.get("prototype_covered_characters")
    )
    model_supported_characters = _coerce_character_list(
        normalized.get("model_supported_characters")
    )
    fallback_characters = _coerce_character_list(normalized.get("fallback_characters"))
    if not prototype_covered_characters and not model_supported_characters and not fallback_characters:
        fallback_characters = list(unique_characters)

    suggestions = normalized.get("suggestions")
    if not isinstance(suggestions, list):
        suggestions = _build_inspection_suggestions(fallback_characters)

    summary = normalized.get("summary")
    if not isinstance(summary, str):
        high_realism_count = len(prototype_covered_characters) + len(model_supported_characters)
        summary = (
            f"{high_realism_count} / {len(unique_characters)} unique characters are in a "
            "higher-realism path."
        )

    prototype_source = _normalize_prototype_source(
        normalized,
        prototype_pack=prototype_pack,
    )
    prototype_pack_name = prototype_source.get("name")
    prototype_source_kind = prototype_source.get("kind")

    normalized.update(
        {
            "style": normalized.get("style", style),
            "total_characters": int(
                normalized.get("total_characters", len(_non_whitespace_characters(text)))
            ),
            "unique_characters": unique_characters,
            "prototype_covered_characters": prototype_covered_characters,
            "model_supported_characters": model_supported_characters,
            "fallback_characters": fallback_characters,
            "suggestions": suggestions,
            "summary": summary,
            "prototype_source": prototype_source,
            "prototype_pack_name": prototype_pack_name,
            "prototype_source_kind": prototype_source_kind,
        }
    )
    existing_report_markdown = normalized.get("report_markdown")
    if not isinstance(existing_report_markdown, str) or not existing_report_markdown.strip():
        normalized["report_markdown"] = _inspection_report_markdown(normalized)
    return normalized


def _inspection_report_markdown(report: dict[str, object]) -> str:
    fallback_characters = _coerce_character_list(report.get("fallback_characters"))
    prototype_covered_characters = _coerce_character_list(
        report.get("prototype_covered_characters")
    )
    model_supported_characters = _coerce_character_list(
        report.get("model_supported_characters")
    )
    suggestions = report.get("suggestions")
    lines = [
        "## 课堂笔记预检",
        f"- 风格: {report.get('style', _DEFAULT_STYLE)}",
        f"- 当前原型字库: {_prototype_source_label(report)}",
        f"- 非空白字符数: {report.get('total_characters', 0)}",
        f"- 原型字库覆盖: {len(prototype_covered_characters)}",
        f"- 模型直出覆盖: {len(model_supported_characters)}",
        f"- 较低真实感字符: {len(fallback_characters)}",
        f"- 摘要: {report.get('summary', '')}",
    ]
    if fallback_characters:
        lines.append(
            "- 较低真实感字符: " + "、".join(fallback_characters[:12])
            + (" ..." if len(fallback_characters) > 12 else "")
        )
    if isinstance(suggestions, list) and suggestions:
        lines.append("- 建议:")
        for item in suggestions[:5]:
            if not isinstance(item, dict):
                continue
            char = item.get("char", "?")
            suggestion = item.get("suggestion", "可尝试换成更常见表达后再生成。")
            lines.append(f"  - `{char}`: {suggestion}")
    return "\n".join(lines)


def _build_note_session_status(page_count: int, report: dict[str, object]) -> str:
    prototype_source = report.get("prototype_source")
    if isinstance(prototype_source, dict):
        label = prototype_source.get("label")
        if isinstance(label, str) and label.strip():
            return f"已生成 {page_count} 页课堂笔记，当前原型字库：{label}"
    return f"已生成 {page_count} 页课堂笔记。"


def _prototype_source_label(report: dict[str, object]) -> str:
    prototype_source = report.get("prototype_source")
    prototype_source_kind = report.get("prototype_source_kind")
    prototype_pack_name = report.get("prototype_pack_name", "default_note")
    if isinstance(prototype_source, dict):
        label = prototype_source.get("label")
        if isinstance(label, str) and label.strip():
            return label
        prototype_source_kind = prototype_source.get("kind", prototype_source_kind)
        prototype_pack_name = prototype_source.get("name", prototype_pack_name)
    if prototype_source_kind == "disabled":
        return "Prototype pack disabled for the selected style"
    if isinstance(prototype_source, str):
        if prototype_source.startswith("builtin:"):
            return f"Built-in starter pack: {prototype_pack_name}"
        if prototype_source.strip():
            return f"Local prototype pack: {prototype_pack_name} ({prototype_source})"
    return "Built-in starter pack: default_note"


def _build_inspection_suggestions(characters: list[str]) -> list[dict[str, str]]:
    suggestions: list[dict[str, str]] = []
    replacement_hints = {
        "“": "改成常见中文引号或普通双引号后再试。",
        "”": "改成常见中文引号或普通双引号后再试。",
        "‘": "改成常见中文引号或普通单引号后再试。",
        "’": "改成常见中文引号或普通单引号后再试。",
        "—": "改成 `-`、`--` 或更常见中文标点后再试。",
        "①": "改成 `1` 或 `一` 后再试。",
        "②": "改成 `2` 或 `二` 后再试。",
        "③": "改成 `3` 或 `三` 后再试。",
    }
    for char in characters[:12]:
        suggestions.append(
            {
                "char": char,
                "reason": "current prototype or high-realism path may not cover this character",
                "suggestion": replacement_hints.get(
                    char, "可保留原文，或改成更常见、已覆盖的表达后再生成。"
                ),
            }
        )
    return suggestions


def _coerce_character_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _non_whitespace_characters(text: str) -> list[str]:
    return [char for char in text if not char.isspace()]


def _unique_non_whitespace_characters(text: str) -> list[str]:
    return list(dict.fromkeys(_non_whitespace_characters(text)))


def _prototype_source_from_argument(
    prototype_pack: str | Path | None,
) -> str:
    if prototype_pack is None or str(prototype_pack).strip() == "":
        return "builtin:default_note"
    manifest_path = resolve_prototype_manifest_path(prototype_pack)
    return str(manifest_path)


def _prototype_pack_name_from_argument(prototype_pack: str | Path | None) -> str:
    if prototype_pack is None or str(prototype_pack).strip() == "":
        return "default_note"
    manifest_path = resolve_prototype_manifest_path(prototype_pack)
    return manifest_path.parent.name


def _normalize_prototype_source(
    report: dict[str, object],
    *,
    prototype_pack: str | Path | None = None,
) -> dict[str, object]:
    prototype_source = report.get("prototype_source")
    prototype_pack_name = report.get("prototype_pack_name")
    prototype_source_kind = report.get("prototype_source_kind")
    if isinstance(prototype_source, dict):
        return prototype_source
    if prototype_source_kind == "disabled":
        return {
            "active": False,
            "kind": "disabled",
            "name": None,
            "manifest_path": None,
            "root": None,
            "label": "Prototype pack disabled for the selected style",
        }
    if isinstance(prototype_source, str) and prototype_source.strip():
        source_kind = (
            prototype_source_kind
            if isinstance(prototype_source_kind, str) and prototype_source_kind.strip()
            else "custom"
        )
        manifest_path = Path(prototype_source).resolve()
        pack_name = (
            prototype_pack_name
            if isinstance(prototype_pack_name, str) and prototype_pack_name.strip()
            else manifest_path.parent.name
        )
        label = (
            f"Built-in starter pack: {pack_name}"
            if source_kind == "builtin"
            else f"Local prototype pack: {pack_name} ({manifest_path})"
        )
        return {
            "active": True,
            "kind": source_kind,
            "name": pack_name,
            "manifest_path": str(manifest_path),
            "root": str(manifest_path.parent),
            "label": label,
        }

    default_source = _prototype_source_from_argument(prototype_pack)
    default_name = _prototype_pack_name_from_argument(prototype_pack)
    if default_source.startswith("builtin:"):
        return {
            "active": True,
            "kind": "builtin",
            "name": default_name,
            "manifest_path": None,
            "root": None,
            "label": f"Built-in starter pack: {default_name}",
        }

    manifest_path = Path(default_source).resolve()
    return {
        "active": True,
        "kind": "custom",
        "name": default_name,
        "manifest_path": str(manifest_path),
        "root": str(manifest_path.parent),
        "label": f"Local prototype pack: {default_name} ({manifest_path})",
    }


def export_pages(pages, output_path, format: str = "png", **kwargs):
    """Export generated pages as PNG files or a multi-page PDF."""
    normalized_format = format.lower()
    if normalized_format == "png":
        return _exporter.export_pages_png(pages, output_path, **kwargs)
    if normalized_format == "pdf":
        return _exporter.export_pages_pdf(pages, output_path, **kwargs)
    raise ValueError("format must be 'png' or 'pdf'")


def generate_pages(text: str, **kwargs) -> list[Image.Image]:
    """Generate one or more handwriting page images for longer text."""
    font_size = int(kwargs.get("font_size", 80))
    layout = kwargs.get("layout", NATURAL_LAYOUT)
    paper = kwargs.get("paper", WHITE_PAPER)
    page_chunks = _split_text_into_pages(
        text,
        font_size=font_size,
        layout=layout,
        paper=paper,
    )
    return [generate(chunk, **kwargs) for chunk in page_chunks]


def _split_text_into_pages(
    text: str,
    *,
    font_size: int,
    layout: str,
    paper: str,
) -> list[str]:
    if font_size <= 0:
        raise ValueError("font_size must be positive")

    max_columns, max_rows = _page_grid(font_size=font_size, layout=layout, paper=paper)
    if max_columns < 1 or max_rows < 1:
        raise ValueError(
            f"font_size={font_size} does not fit within page_size={_DEFAULT_PAGE_SIZE} "
            f"and margins={_DEFAULT_MARGINS}"
        )

    lines = _wrap_text_to_lines(text, max_columns=max_columns)

    return [
        _lines_to_text(lines[index : index + max_rows])
        for index in range(0, len(lines), max_rows)
    ] or [""]


def _page_grid(*, font_size: int, layout: str, paper: str) -> tuple[int, int]:
    top, right, bottom, left = _DEFAULT_MARGINS
    page_width, page_height = _DEFAULT_PAGE_SIZE
    char_gap, line_gap = _estimated_spacing(font_size, layout)
    line_height = font_size + line_gap
    column_step = line_height if paper in _GRID_PAPERS else font_size + char_gap
    first_column_x = _aligned_origin(left, column_step) if paper in _GRID_PAPERS else left
    first_row_y = _aligned_origin(top, line_height) if paper in _GUIDED_PAPERS else top

    max_columns = _count_slots(
        total_extent=page_width,
        start=first_column_x,
        end_margin=right,
        item_extent=font_size,
        step=column_step,
    )
    max_rows = _count_slots(
        total_extent=page_height,
        start=first_row_y,
        end_margin=bottom,
        item_extent=font_size,
        step=line_height,
    )
    return max_columns, max_rows


def _estimated_spacing(font_size: int, layout: str) -> tuple[int, int]:
    if layout == NEAT_LAYOUT:
        return max(4, font_size // 12), max(8, font_size // 4)
    if layout == CURSIVE_LAYOUT:
        return max(8, font_size // 8), max(14, font_size // 3)
    return max(6, font_size // 10), max(12, font_size // 3)


def _aligned_origin(margin: int, step: int) -> int:
    return ((margin + step - 1) // step) * step


def _count_slots(
    *,
    total_extent: int,
    start: int,
    end_margin: int,
    item_extent: int,
    step: int,
) -> int:
    available_extent = total_extent - end_margin - start
    if available_extent < item_extent:
        return 0
    return 1 + max(0, (available_extent - item_extent) // step)


def _wrap_text_to_lines(text: str, *, max_columns: int) -> list[str]:
    lines: list[list[str]] = [[]]

    for symbol in text:
        current_line = lines[-1]

        if symbol == "\n":
            _trim_trailing_spaces(current_line)
            lines.append([])
            continue

        if symbol.isspace() and not current_line:
            continue

        if len(current_line) >= max_columns:
            _trim_trailing_spaces(current_line)

            if len(current_line) < max_columns:
                current_line = lines[-1]
            else:
                if symbol.isspace():
                    lines.append([])
                    continue

                if _is_leading_punctuation(symbol) and max_columns > 1:
                    moved = _pop_reflow_suffix(current_line, max_columns)
                    if moved is not None:
                        _trim_trailing_spaces(current_line)
                        lines.append(moved + [symbol])
                        continue

                lines.append([])
                current_line = lines[-1]

        if symbol.isspace() and not current_line:
            continue

        current_line.append(symbol)

    for line in lines:
        _trim_trailing_spaces(line)

    return ["".join(line) for line in lines]


def _is_leading_punctuation(symbol: str) -> bool:
    return symbol in _LEADING_PUNCTUATION


def _pop_reflow_suffix(line: list[str], max_columns: int) -> list[str] | None:
    if not line:
        return None

    split_index = len(line) - 1
    while split_index >= 0 and _is_leading_punctuation(line[split_index]):
        split_index -= 1

    if split_index < 0 or line[split_index].isspace():
        return None

    moved = line[split_index:]
    if len(moved) + 1 > max_columns:
        return None

    del line[split_index:]
    return moved


def _trim_trailing_spaces(line: list[str]) -> None:
    while line and line[-1].isspace():
        line.pop()


def _lines_to_text(lines: list[str]) -> str:
    return "\n".join(lines)
