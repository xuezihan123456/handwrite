# HandWrite English Guide

## Project Overview

`HandWrite` now makes the most sense as a **classroom-note generator**, not just a generic handwriting demo.

The repository already includes data preprocessing, training scaffolding, inference wrappers, page composition, PNG/PDF export, a Gradio demo, and the first note-realism product loop.

It is still not a finished pretrained handwriting product. No real pretrained weights are bundled here. When valid weights are missing, runtime should prefer a prototype-backed note path first, then fall back to more handwritten-looking font/distortion logic. That keeps the product loop usable, but it is not evidence of trained-model quality.

## Current Capabilities

- Installable Python package in `src/handwrite`
- CASIA `.gnt` parsing and dataset preprocessing
- Five built-in style names exposed through `handwrite.list_styles()`
- Public APIs for single-character, single-page, and multi-page generation
- `handwrite.inspect_text(...)` for note-realism precheck
- `StyleEngine` inference wrapper with automatic weight discovery and graceful fallback
- Starter prototype pack for the default classroom-note flow
- Local prototype-library builder path for larger private packs
- PNG export, single-page PDF export, and multi-page PDF export
- Gradio demo with precheck reporting plus multi-page preview/downloads
- Training and evaluation scripts

## Environment and Installation

- Python 3.9+
- Windows is the primary development environment at the moment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Python API

### List built-in styles

```python
import handwrite

print(handwrite.list_styles())
```

The built-in style order stays stable, but the classroom-note product loop should use an explicit flowing-note default instead of relying on style-list insertion order.

### Inspect note text before generation

```python
import handwrite

report = handwrite.inspect_text(
    "Today we reviewed Newton's second law and worked through two examples.",
    style="行书流畅",
)

print(report["summary"])
for advisory in report["suggestions"]:
    print(advisory)
```

The report is meant to tell the user:

- which characters are prototype-covered
- which characters can use higher-quality model/runtime paths
- which characters are likely to fall back to lower-realism rendering
- whether the active source is the built-in starter pack or a local custom pack
- which advisory replacements or rewrites the user may want to consider

It does **not** rewrite the original text automatically.

### Generate one character

```python
import handwrite

img = handwrite.char("你", style="行书流畅")
img.save("single.png")
```

### Generate one note page

```python
import handwrite

page = handwrite.generate(
    "Today we reviewed Newton's second law and worked through two examples.",
    style="行书流畅",
    paper="横线纸",
    layout="自然",
    font_size=80,
)
page.save("note_page.png")
```

### Use a larger local prototype pack

```python
import handwrite

prototype_pack = "data/prototypes/my_note_pack"

report = handwrite.inspect_text(
    "Today we reviewed Newton's second law and worked through two examples.",
    style="行书流畅",
    prototype_pack=prototype_pack,
)
page = handwrite.generate(
    "Today we reviewed Newton's second law and worked through two examples.",
    style="行书流畅",
    paper="横线纸",
    layout="自然",
    font_size=80,
    prototype_pack=prototype_pack,
)
print(report["prototype_source"]["label"])
```

That is the concrete product loop now:
1. build a larger local pack with `scripts/build_prototype_library.py`
2. point runtime or the demo at that pack through `prototype_pack`
3. inspect coverage first, then generate the note

### Generate multiple note pages

```python
import handwrite

pages = handwrite.generate_pages(
    "Classroom notes " * 120,
    style="行书流畅",
    paper="横线纸",
    layout="自然",
    font_size=80,
)

for index, page in enumerate(pages, start=1):
    page.save(f"note_{index:03d}.png")
```

### Export PNG / PDF

```python
import handwrite

page = handwrite.generate("Export example.", style="行书流畅")
pages = handwrite.generate_pages("Long text content " * 80, style="行书流畅")

handwrite.export(page, "output/page.png", format="png", dpi=300)
handwrite.export(page, "output/page.pdf", format="pdf", dpi=300)
handwrite.export_pages(pages, "output/pages", format="png", prefix="note", dpi=300)
handwrite.export_pages(pages, "output/note.pdf", format="pdf", dpi=300)
```

## CLI and Script Usage

### 1. Local data preparation helper

```bash
python scripts/download_data.py --scan_dir downloads --raw_dir data/raw
```

Use it to normalize manually downloaded CASIA/HWDB split archives or extracted directories into `data/raw/<split>`.

### 2. Preprocessing

```bash
python scripts/preprocess.py --raw_dir data/raw/HWDB1.0trn_gnt --output_dir data/processed --font_path data/fonts/NotoSerifSC-Regular.otf --charset_level 500 --min_writer_coverage 0.9
```

This produces normalized paired assets plus the `metadata.json` needed by training and prototype-library building.

### 3. Build a larger prototype library

```bash
python scripts/build_prototype_library.py --metadata data/processed/metadata.json --output_dir data/prototypes/default_note
```

This is the scale-up path:

- the repository ships only a small starter pack
- if you want broader 2000+ note coverage, build a larger private prototype pack locally from processed handwriting data
- the demo now ships with classroom-note presets so you can load a sample note before adapting it
- if `prototype_pack` points to a missing directory or manifest, the demo reports a clear product-level error instead of failing silently

### 4. Style selection file

`scripts/train.py` still expects a separate `selected_styles.json`.

### 5. Note session CLI

```bash
python scripts/note_session.py --preset 牛顿定律复习 --output_dir output/session
```

This is the most product-shaped local entrypoint right now:

- supports `--text`
- supports `--text_file`
- supports `--preset`
- supports `--prototype_pack`
- exports PNG pages, a PDF, and a markdown session report in one run

### 6. Training

```bash
python scripts/train.py --data_dir data/processed --styles_file data/processed/selected_styles.json --output_dir weights --batch_size 8 --epochs 30
```

### 7. Evaluation

```bash
python scripts/evaluate.py --output-dir evaluation
```

### 8. Demo

```bash
python demo/app.py
```

Default URL:

- `http://localhost:7860`

The demo is now meant to support the classroom-note product loop:

- paste note text
- inspect realism before generation
- generate multi-page note pages
- preview page by page
- download PNG/PDF artifacts

## Practical Limits and Honest Notes

### 1. The starter pack is not the full note asset body

The repository should include a small distributable starter prototype pack, not a full 2000+ built-in handwritten asset set.

### 2. Larger realistic coverage still depends on local expansion

If you need broader coverage or more stable realism, build a bigger local prototype library and pass it into `prototype_pack` in the API or demo, or move on to real generator weights.

### 3. The product does not rewrite text automatically

Precheck is advisory. It reports risk and suggestions, but the user keeps control of the text.

### 4. Evaluation is still lightweight

The repo includes a Frechet-style lightweight metric, but not a full research-grade benchmark pipeline.

## Suggested Order of Use

1. `pip install -e ".[dev]"`
2. `pytest`
3. use `handwrite.inspect_text(...)` or the demo precheck before generation
4. try the starter-pack classroom-note flow out of the box
5. prepare `.gnt` data and a usable Chinese font
6. run `scripts/download_data.py` and `scripts/preprocess.py` when you want broader local data
7. run `scripts/build_prototype_library.py` to build a larger private prototype pack
8. only then move on to real style selection, training, and evaluation if you need higher-quality generation
