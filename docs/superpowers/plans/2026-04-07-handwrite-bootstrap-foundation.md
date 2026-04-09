# HandWrite Bootstrap Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the initial local Python package, test baseline, and minimal public API scaffold so later data/model/composer work has a stable project structure to land on.

**Architecture:** Start with a docs-first repository and convert it into an installable Python package under `src/handwrite/`. Keep the initial implementation deliberately thin: package metadata, style registry, public API shell, and smoke-tested module boundaries. Defer model training and data processing behavior, but reserve exact files and interfaces required by the implementation guide.

**Tech Stack:** Python 3.9+, setuptools via `pyproject.toml`, pytest, Pillow, NumPy, Gradio placeholders.

---

### Task 1: Create the repository scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/handwrite/__init__.py`
- Create: `src/handwrite/styles.py`
- Create: `src/handwrite/composer.py`
- Create: `src/handwrite/exporter.py`
- Create: `src/handwrite/engine/__init__.py`
- Create: `src/handwrite/engine/model.py`
- Create: `src/handwrite/engine/generator.py`
- Create: `src/handwrite/engine/discriminator.py`
- Create: `src/handwrite/engine/train.py`
- Create: `src/handwrite/data/__init__.py`
- Create: `src/handwrite/data/casia_parser.py`
- Create: `src/handwrite/data/dataset.py`
- Create: `src/handwrite/data/font_renderer.py`
- Create: `src/handwrite/data/charsets.py`
- Create: `demo/app.py`
- Create: `scripts/download_data.py`
- Create: `scripts/preprocess.py`
- Create: `scripts/train.py`
- Create: `scripts/evaluate.py`
- Create: `weights/.gitkeep`
- Test: `tests/test_package.py`

- [ ] **Step 1: Write the failing scaffold test**

```python
from handwrite import list_styles


def test_list_styles_exposes_builtin_names() -> None:
    styles = list_styles()
    assert styles == [
        "工整楷书",
        "圆润可爱",
        "行书流畅",
        "偏瘦紧凑",
        "随意潦草",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'handwrite'`

- [ ] **Step 3: Write the minimal package scaffold**

```python
# src/handwrite/styles.py
BUILTIN_STYLES = {
    "工整楷书": 0,
    "圆润可爱": 1,
    "行书流畅": 2,
    "偏瘦紧凑": 3,
    "随意潦草": 4,
}


def list_style_names() -> list[str]:
    return list(BUILTIN_STYLES.keys())
```

```python
# src/handwrite/__init__.py
from handwrite.styles import BUILTIN_STYLES, list_style_names


def list_styles() -> list[str]:
    return list_style_names()
```

- [ ] **Step 4: Add packaging files and empty module placeholders**

```toml
[project]
name = "handwrite"
version = "0.1.0"
description = "AI Chinese handwriting generator"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
    "Pillow>=10.0",
    "numpy>=1.24",
    "opencv-python>=4.8",
    "gradio>=4.0",
    "tqdm>=4.65",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "matplotlib>=3.7"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

```gitignore
weights/*.pt
weights/*.pth
data/raw/
data/processed/
__pycache__/
*.pyc
.venv/
.pytest_cache/
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_package.py -v`
Expected: PASS

- [ ] **Step 6: Verify install/import behavior**

Run: `python -m pip install -e ".[dev]"`
Expected: editable install succeeds

Run: `python -c "import handwrite; print(handwrite.list_styles())"`
Expected: prints the five built-in style names

---

### Task 2: Add the initial public API shell

**Files:**
- Modify: `src/handwrite/__init__.py`
- Modify: `src/handwrite/engine/model.py`
- Modify: `src/handwrite/composer.py`
- Modify: `src/handwrite/exporter.py`
- Test: `tests/test_package.py`

- [ ] **Step 1: Write the failing API tests**

```python
from PIL import Image

import handwrite


def test_char_returns_pil_image() -> None:
    image = handwrite.char("你")
    assert isinstance(image, Image.Image)
    assert image.size == (256, 256)


def test_generate_returns_page_image() -> None:
    page = handwrite.generate("你好")
    assert isinstance(page, Image.Image)
    assert page.size == (2480, 3508)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_package.py -v`
Expected: FAIL with `AttributeError` for `char` and `generate`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/handwrite/engine/model.py
from PIL import Image, ImageDraw


class StyleEngine:
    def generate_char(self, char: str, style_id: int) -> Image.Image:
        image = Image.new("L", (256, 256), color=255)
        draw = ImageDraw.Draw(image)
        draw.text((96, 96), char, fill=0)
        return image
```

```python
# src/handwrite/composer.py
from PIL import Image


def compose_page(
    chars,
    text: str,
    page_size: tuple[int, int] = (2480, 3508),
    font_size: int = 80,
    margins: tuple[int, int, int, int] = (200, 200, 200, 200),
    layout: str = "自然",
    paper: str = "白纸",
) -> Image.Image:
    return Image.new("L", page_size, color=255)
```

```python
# src/handwrite/__init__.py
from handwrite.composer import compose_page
from handwrite.engine.model import StyleEngine
from handwrite.styles import BUILTIN_STYLES, list_style_names

_ENGINE = None


def _get_engine() -> StyleEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = StyleEngine()
    return _ENGINE


def list_styles() -> list[str]:
    return list_style_names()


def char(text: str, style: str = "工整楷书"):
    return _get_engine().generate_char(text, BUILTIN_STYLES[style])


def generate(
    text: str,
    style: str = "工整楷书",
    paper: str = "白纸",
    layout: str = "自然",
    font_size: int = 80,
):
    images = [char(c, style=style) for c in text if c not in {" ", "\n"}]
    return compose_page(images, text, font_size=font_size, layout=layout, paper=paper)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_package.py -v`
Expected: PASS

---

### Task 3: Reserve the Phase 2 data interfaces

**Files:**
- Modify: `src/handwrite/data/casia_parser.py`
- Modify: `src/handwrite/data/charsets.py`
- Modify: `src/handwrite/data/font_renderer.py`
- Test: `tests/test_data_interfaces.py`

- [ ] **Step 1: Write failing interface tests**

```python
from PIL import Image

from handwrite.data.charsets import get_charset
from handwrite.data.font_renderer import render_standard_char


def test_get_charset_500_contains_core_characters() -> None:
    charset = get_charset("500")
    assert "的" in charset
    assert "你" in charset
    assert "A" in charset


def test_render_standard_char_returns_grayscale_image() -> None:
    image = render_standard_char("你", "missing-font.otf")
    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    assert image.size == (256, 256)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_interfaces.py -v`
Expected: FAIL because the modules/functions do not exist

- [ ] **Step 3: Write the minimal implementation**

```python
# src/handwrite/data/charsets.py
COMMON_500 = list("的一是在不了有人这中大为上个国我以要他")
PUNCTUATION = ["，", "。", "！", "？", "、", "；", "：", "“", "”", "（", "）", "《", "》"]
DIGITS = [str(i) for i in range(10)]
LETTERS = [chr(code) for code in range(ord("a"), ord("z") + 1)] + [
    chr(code) for code in range(ord("A"), ord("Z") + 1)
]


def get_charset(level: str = "500") -> list[str]:
    if level != "500":
        raise ValueError(f"Unsupported charset level: {level}")
    return list(dict.fromkeys(COMMON_500 + PUNCTUATION + DIGITS + LETTERS))
```

```python
# src/handwrite/data/font_renderer.py
from PIL import Image, ImageDraw


def render_standard_char(
    char: str,
    font_path: str,
    image_size: int = 256,
    char_size: int = 200,
) -> Image.Image:
    image = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((image_size // 3, image_size // 3), char, fill=0)
    return image
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data_interfaces.py -v`
Expected: PASS

---

### Task 4: Add the demo entry-point smoke test

**Files:**
- Modify: `demo/app.py`
- Test: `tests/test_demo.py`

- [ ] **Step 1: Write the failing demo smoke test**

```python
from demo.app import generate_handwriting


def test_generate_handwriting_returns_none_for_blank_input() -> None:
    assert generate_handwriting("   ", "工整楷书", "白纸", "自然", 80) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo.py -v`
Expected: FAIL because `demo.app` does not exist

- [ ] **Step 3: Write the minimal implementation**

```python
import handwrite


def generate_handwriting(text, style, paper, layout, font_size):
    if not text.strip():
        return None
    return handwrite.generate(text, style=style, paper=paper, layout=layout, font_size=int(font_size))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_demo.py -v`
Expected: PASS

---

### Task 5: Verify the bootstrap milestone

**Files:**
- Verify only

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_package.py tests/test_data_interfaces.py tests/test_demo.py -v`
Expected: all tests PASS

- [ ] **Step 2: Run import smoke verification**

Run: `python -c "import handwrite; print(handwrite.list_styles()); print(handwrite.char('你').size); print(handwrite.generate('你好').size)"`
Expected: five style names, `(256, 256)`, `(2480, 3508)`

- [ ] **Step 3: Record the next slice**

Next slice after this milestone:
1. Implement real `charsets.py` coverage for the 500-character MVP set.
2. Implement real `font_renderer.py` centering and font loading.
3. Implement `.gnt` parsing and preprocessing pipeline.
4. Implement real page composition behavior and exporter.
