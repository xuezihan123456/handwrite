# HandWrite English Guide

## Project Overview

`HandWrite` is a Python project for Chinese handwriting generation.

Its long-term goal is not simple static font replacement. The intended pipeline is:

- parse the CASIA-HWDB handwriting dataset
- build paired standard-glyph / handwritten-glyph training data
- train a zi2zi-style conditional generation model
- generate single characters, single pages, and multi-page handwritten documents
- expose the workflow through a Python API, CLI scripts, and a Gradio demo

The repository has already moved beyond pure design docs and now contains a working engineering foundation.

## Current Status

Implemented today:

- installable Python package structure
- style registry and `selected_styles.json` loading
- `.gnt` parsing
- preprocessing pipeline with `metadata.json`
- `HandwriteDataset`
- standard glyph rendering
- `Generator` / `Discriminator` model definitions
- `StyleEngine` inference wrapper
- page composition
- PNG export and multi-page PNG export
- training core and training CLI
- evaluation helpers
- multi-page public API via `generate_pages()`
- Gradio demo entrypoint

Not fully done yet:

- no real pretrained weights are bundled in this repository
- PDF export is not implemented yet
- TTF export is not implemented yet
- user-provided custom style training is not implemented yet

## Who This Repo Is For

- developers who want to train a Chinese handwriting generator locally
- researchers or competition participants who want a codebase, not just a concept
- users who prefer local execution instead of a hosted closed service

## Environment Requirements

- Python 3.9+
- Windows is the primary development environment at the moment
- expected libraries:
  - PyTorch
  - torchvision
  - Pillow
  - NumPy
  - OpenCV
  - Gradio
  - pytest

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Python API

### 1. List styles

```python
import handwrite

print(handwrite.list_styles())
```

Current built-in styles:

- 工整楷书
- 圆润可爱
- 行书流畅
- 偏瘦紧凑
- 随意潦草

### 2. Generate one character

```python
import handwrite

img = handwrite.char("你", style="工整楷书")
img.save("single.png")
```

### 3. Generate one page

```python
import handwrite

page = handwrite.generate(
    "今天天气真不错，适合出去走走。",
    style="工整楷书",
    paper="白纸",
    layout="自然",
    font_size=80,
)
page.save("page.png")
```

### 4. Generate multiple pages

```python
import handwrite

pages = handwrite.generate_pages(
    "Long text content" * 200,
    style="工整楷书",
    paper="方格纸",
    layout="自然",
    font_size=80,
)

for index, page in enumerate(pages, start=1):
    page.save(f"page_{index}.png")
```

## Command Line Tools

### 1. Preprocess dataset

```bash
python scripts/preprocess.py \
  --raw_dir data/raw/HWDB1.0trn_gnt \
  --output_dir data/processed \
  --font_path data/fonts/NotoSerifSC-Regular.otf
```

This step:

- reads `.gnt` files
- filters low-quality samples
- normalizes handwriting bitmaps
- renders standard glyph images
- writes `metadata.json`

### 2. Train the model

```bash
python scripts/train.py \
  --data_dir data/processed \
  --styles_file data/processed/selected_styles.json \
  --output_dir weights \
  --batch_size 8 \
  --epochs 30
```

Training outputs include:

- `weights/checkpoint_epoch_{N}.pt`
- `weights/samples/...png`
- `weights/train_log.csv`

### 3. Run evaluation

```bash
python scripts/evaluate.py --output-dir evaluation
```

Current outputs include:

- `evaluation/style_comparison.png`
- `evaluation/complex_chars.png`

### 4. Launch the demo

```bash
python demo/app.py
```

Default URL:

- `http://localhost:7860`

## Export Utilities

### Export a single page PNG

```python
from handwrite.exporter import export_png

export_png(page, "output/page.png", dpi=300)
```

### Export multiple pages as PNG

```python
from handwrite.exporter import export_pages_png

export_pages_png(pages, "output/pages", prefix="page", dpi=300)
```

## StyleEngine Notes

`StyleEngine` currently supports:

- graceful fallback when no weights are available
- explicit `weights_path`
- automatic discovery of common checkpoint/state-dict files under `weights/`
- real `Generator` inference when valid weights are found
- fallback rendering when weights are missing or invalid

That means:

- the repository is structurally ready for real inference
- but without trained weights, current output is still MVP-level behavior rather than final production-quality handwriting

## Repository Layout

```text
handwrite/
├── demo/
├── docs/
├── scripts/
├── src/
│   └── handwrite/
│       ├── data/
│       ├── engine/
│       ├── composer.py
│       ├── exporter.py
│       └── styles.py
├── tests/
└── weights/
```

Important directories:

- `src/handwrite/data/`: parsing, charset, font rendering, dataset
- `src/handwrite/engine/`: generator, discriminator, StyleEngine, training core
- `scripts/`: preprocess / train / evaluate entrypoints
- `demo/`: Gradio app
- `tests/`: unit and regression tests
- `weights/`: model weights directory, without committed real large checkpoints

## Current Limitations

### 1. No bundled trained weights

The repository does not ship with trained `.pt` / `.pth` files, so:

- the API works
- the inference flow works
- but default output is still MVP behavior, not final handwriting quality

### 2. Evaluation uses a lightweight Frechet-style metric

The current `fid_score` in `scripts/evaluate.py` is a lightweight engineering approximation suitable for MVP verification. It is not the same thing as a full `pytorch-fid` workflow.

### 3. PDF export is still missing

The repo currently provides:

- single-page PNG export
- multi-page PNG export

### 4. Large artifacts should stay out of git

Do not commit:

- datasets
- `.gnt`
- large model weights
- large generated evaluation images

## Recommended Usage Order

If you are opening this repository for the first time, the practical order is:

1. `pip install -e ".[dev]"`
2. `pytest`
3. prepare `CASIA-HWDB` and a usable font file
4. run `scripts/preprocess.py`
5. prepare `selected_styles.json`
6. run `scripts/train.py`
7. run `scripts/evaluate.py`
8. inspect results through `demo/app.py`

## Summary

This is not yet a “finished pretrained handwriting product repository”.

It is already:

- structured
- testable
- extendable
- runnable

and it provides a strong base for continuing toward a full Chinese handwriting generation system.
