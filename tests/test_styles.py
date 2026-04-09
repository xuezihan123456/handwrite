import json
from pathlib import Path

import handwrite.styles as styles


def test_builtin_styles_remain_available_for_name_lookup() -> None:
    assert styles.BUILTIN_STYLES == {
        "工整楷书": 0,
        "圆润可爱": 1,
        "行书流畅": 2,
        "偏瘦紧凑": 3,
        "随意潦草": 4,
    }


def test_list_style_names_returns_builtin_names_in_stable_order() -> None:
    assert styles.list_style_names() == [
        "工整楷书",
        "圆润可爱",
        "行书流畅",
        "偏瘦紧凑",
        "随意潦草",
    ]


def test_load_selected_styles_reads_utf8_json_from_path(tmp_path: Path) -> None:
    styles_path = tmp_path / "selected_styles.json"
    styles_path.write_text(
        json.dumps(
            {
                "styles": [
                    {"id": 0, "name": "工整楷书", "writer_id": "057"},
                    {"id": 1, "name": "圆润可爱", "writer_id": "123"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    style_definition = getattr(styles, "StyleDefinition", None)
    load_selected_styles = getattr(styles, "load_selected_styles", None)

    assert style_definition is not None
    assert callable(load_selected_styles)
    assert load_selected_styles(styles_path) == [
        style_definition(id=0, name="工整楷书", writer_id="057"),
        style_definition(id=1, name="圆润可爱", writer_id="123"),
    ]
