# HandWrite

AI Chinese handwriting generator built with Python, PyTorch, and Pillow.

这是一个基于 Python、PyTorch 和 Pillow 的中文手写体生成项目。

## Readme

- 中文文档: [README.zh-CN.md](./README.zh-CN.md)
- English documentation: [README.en.md](./README.en.md)

## Current Status

- Package/API scaffold: available
- Data preprocessing pipeline: available
- Generator/Discriminator definitions: available
- Training loop and training CLI: available
- Evaluation script: available
- Demo entrypoint: available
- Real pretrained weights: not included in this repository
- PDF export: not implemented yet

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
python demo/app.py
```

## Python API

```python
import handwrite

img = handwrite.char("你", style="工整楷书")
page = handwrite.generate("今天天气真不错。", style="工整楷书")
pages = handwrite.generate_pages("长文本内容" * 200, style="工整楷书")
print(handwrite.list_styles())
```

## Scripts

```bash
python scripts/preprocess.py --raw_dir data/raw/HWDB1.0trn_gnt --output_dir data/processed --font_path data/fonts/NotoSerifSC-Regular.otf
python scripts/train.py --data_dir data/processed --styles_file data/processed/selected_styles.json --output_dir weights
python scripts/evaluate.py --output-dir evaluation
```

## Notes

- If no trained generator weights are found, the current inference wrapper degrades gracefully to a fallback rendering path.
- Evaluation includes a lightweight Frechet-style score for MVP use; it is not a full `pytorch-fid` integration.
- Large datasets and model weights should stay out of git.
