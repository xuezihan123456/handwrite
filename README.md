# HandWrite

> AI-powered Chinese handwriting generator with 15 innovation modules for classroom notes, personalization, and more.

> AI驱动的中文手写生成器，包含15个创新模块，支持课堂笔记、个性化定制等场景。

[English](./README.en.md) | [中文](./README.zh-CN.md)

## Highlights

- **15 Innovation Modules** covering personalization, dynamics, animation, semantic layout, formula rendering, style mixing, paper templates, OCR-based style extraction, collaborative writing, quality assurance, text summarization, digitization, grading, temporal evolution, and AR integration
- **Classroom-note product loop** with precheck, generation, export, and demo
- **623+ tests** passing across all modules
- Python 3.9+ | PyTorch | Pillow | OpenCV | Gradio

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
pytest
python demo/app.py
```

## Core API

```python
import handwrite

# Generate a single note page
page = handwrite.generate(
    "今天上课主要讲了牛顿第二定律和两个例题。",
    style="行书流畅", paper="横线纸", layout="自然", font_size=80,
)
handwrite.export(page, "output/note.png", format="png", dpi=300)

# Generate multi-page notes
pages = handwrite.generate_pages("..." * 80, style="行书流畅")
handwrite.export_pages(pages, "output/note.pdf", format="pdf", dpi=300)
```

## 15 Innovation Modules

| Module | Description |
|--------|-------------|
| **Personalization** | Analyze handwriting samples, extract style vectors, synthesize personalized glyph packs |
| **Dynamics** | Simulate stroke pressure, ink flow, and writing speed for realistic pen dynamics |
| **Animation** | Stroke-order animation with Bezier trajectories, GIF/MP4 export |
| **Semantic** | Text analysis, intelligent layout planning, semantic annotation rendering |
| **Formula** | LaTeX and chemical formula parsing, layout, and rendering |
| **Style Mixing** | Blend multiple handwriting styles, style transfer, and interpolation |
| **Papers** | Paper template registry with 6+ built-in layouts (cornell, grid, staff, etc.) |
| **OCR Style** | Extract handwriting style from scanned images, generate prototype fonts |
| **Collaboration** | Multi-writer collaborative writing with style blending |
| **Quality** | Authenticity and naturalness scoring with improvement suggestions |
| **Summary** | Text summarization with mind-map and outline layout generation |
| **Digitization** | OCR recognition with style-preserving round-trip editing |
| **Grading** | Error detection, annotation, scoring, and feedback for handwriting |
| **Temporal** | Simulate handwriting evolution across age and skill levels |
| **AR** | Paper detection, perspective transform, and texture blending for AR overlay |

See [README.en.md](./README.en.md) for detailed English documentation and [README.zh-CN.md](./README.zh-CN.md) for full Chinese documentation.

## License

MIT
