# HandWrite

> AI-powered Chinese handwriting generator with **25 innovation modules** (15 Phase-1 + 10 Phase-2) for classroom notes, personalization, animation, geometry sheets, watermarking and more.

> AI驱动的中文手写生成器，包含 **25 个创新模块**（Phase-1 的 15 个 + Phase-2 的 10 个），支持课堂笔记、个性化定制、动画直播、几何考卷、防伪水印等场景。

[English](./README.en.md) | [中文](./README.zh-CN.md)

## Highlights

- **25 Innovation Modules** covering personalization, dynamics, animation, semantic layout, formula rendering, style mixing, paper templates, OCR-based style extraction, collaborative writing, quality assurance, text summarization, digitization, grading, temporal evolution, AR integration, photo-to-style, live-note animation, diffusion, natural-language style, error notebook, service, timeline, watermark, geometry sheet, and 3D pen.
- **Classroom-note product loop** with precheck, generation, export, and demo.
- **600+ tests** passing across all modules.
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

## 15 Phase-1 Innovation Modules

| Module | Description |
|--------|-------------|
| **Personalization** | Analyze handwriting samples, extract style vectors, synthesize personalized glyph packs |
| **Dynamics** | Simulate stroke pressure, ink flow, and writing speed for realistic pen dynamics |
| **Animation** | Stroke-order animation with Bezier trajectories, GIF/MP4 export |
| **Semantic** | Text analysis, intelligent layout planning, semantic annotation rendering |
| **Formula** | LaTeX and chemical formula parsing, layout, and rendering |
| **Style Mixing** | Blend multiple handwriting styles, style transfer, and interpolation |
| **Papers** | Paper template registry with 6+ built-in layouts |
| **OCR Style** | Extract handwriting style from scanned images, generate prototype fonts |
| **Collaboration** | Multi-writer collaborative writing with style blending |
| **Quality** | Authenticity and naturalness scoring with improvement suggestions |
| **Summary** | Text summarization with mind-map and outline layout generation |
| **Digitization** | OCR recognition with style-preserving round-trip editing |
| **Grading** | Error detection, annotation, scoring, and feedback for handwriting |
| **Temporal** | Simulate handwriting evolution across age and skill levels |
| **AR** | Paper detection, perspective transform, and texture blending for AR overlay |

## 10 Phase-2 Innovation Modules

| # | Module | Description |
|---|--------|-------------|
| ① | **photo_style** | 笔迹照相机 — one-shot photo → personal prototype pack via OCR + segmentation + style extraction |
| ② | **live_note** | 课堂笔记直播动画 — character-by-character live writing video (GIF/MP4) with punctuation pacing and pen-tip cursor |
| ③ | **diffusion** | Diffusion 取代 zi2zi — DDPM + ControlNet-style scaffold (UNet + scheduler + training), graceful fallback when weights absent |
| ④ | **nl_style** | 自然语言控笔 — bilingual natural-language description (e.g. "紧张焦虑的高三学生") → concrete `StyleVector` |
| ⑤ | **error_notebook** | 错题本智能复刻 — diff wrong/correct answers, strike-through annotation, multi-page error notebook PDF |
| ⑥ | **service** | HandWrite-as-Service — FastAPI multi-tenant API with `BillingMeter`, Dockerfile, docker-compose |
| ⑦ | **timeline** | 笔迹时光机 — fit per-feature evolution curves from multi-age samples, generate glyph at any age |
| ⑧ | **watermark** | 防伪手写 — LSB + DCT steganographic watermark embed/extract + `StegMarkAdapter` plug-in interface |
| ⑨ | **geometry_sheet** | 几何 + 公式 + 手写考卷 — geometric figures + handwritten solution steps + LaTeX formula |
| ⑩ | **pen_3d** | 3D 笔尖动力学 — extend dynamics to (x,y,t,pressure,tilt,rotation), Wacom WILL-lite JSON export, tilt-aware replay |

```python
import handwrite

# ① Photo to personal style pack
result = handwrite.photo_to_style("my_handwriting.jpg", output_dir="./packs")

# ② Live-note animation video
handwrite.live_note_video("今天上课讲了牛顿第二定律。", "lecture.gif", fps=24)

# ④ Natural-language style
style = handwrite.parse_description("calm and neat elementary teacher")

# ⑤ Error notebook PDF
handwrite.build_error_notebook(entries=[...], output_path="errors.pdf")

# ⑦ Handwriting time machine
model = handwrite.fit_timeline([(7, "child.png"), (15, "teen.png"), (25, "adult.png")])
glyph = handwrite.generate_at_age(model, "学", age=12)

# ⑧ Embed/extract steganographic watermark
watermarked = handwrite.embed_watermark(image, payload=b"author=alice", method="dct")
payload = handwrite.extract_watermark(watermarked, method="dct")

# ⑩ 3D pen simulation
sim = handwrite.Pen3DSimulator(seed=42)
stroke_3d = sim.simulate([(30, 100), (60, 110), (90, 100)])
will_json = handwrite.export_will_json([stroke_3d])
```

See [README.en.md](./README.en.md) for detailed English documentation and [README.zh-CN.md](./README.zh-CN.md) for full Chinese documentation.

## License

MIT
