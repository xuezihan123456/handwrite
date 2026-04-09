from pathlib import Path

from PIL import Image
import pytest

from handwrite.data.charsets import get_charset
from handwrite.data.font_renderer import render_standard_char


def _find_chinese_font() -> str:
    candidates = [
        Path("C:/Windows/Fonts/NotoSerifSC-VF.ttf"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("No Chinese-capable font found in expected Windows font paths.")


def test_get_charset_500_has_expected_size_and_uniqueness() -> None:
    charset = get_charset("500")
    assert len(charset) == 577
    assert len(set(charset)) == len(charset)
    assert "的" in charset
    assert "你" in charset
    assert "，" in charset
    assert "0" in charset
    assert "A" in charset


def test_get_charset_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        get_charset("999")


def test_render_standard_char_returns_centered_grayscale_image() -> None:
    image = render_standard_char("你", _find_chinese_font())
    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    assert image.size == (256, 256)
    bbox = image.point(lambda value: 255 - value).getbbox()
    assert bbox is not None

    left, top, right, bottom = bbox
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    assert abs(center_x - 128) < 24
    assert abs(center_y - 128) < 24
