# HandWrite 中文说明

## 项目介绍

HandWrite 是一个 **AI驱动的中文手写生成器**，包含 **25 个创新模块**（Phase-1 的 15 个 + Phase-2 的 10 个），覆盖从笔迹个性化到 AR 增强、再到 3D 笔尖动力学的完整技术栈。

仓库覆盖：数据预处理、训练骨架、推理封装、页面排版、PNG/PDF 导出、Gradio demo，以及面向课堂笔记场景的真实感预检。

> 当前仓库不包含真实预训练权重。没有有效权重时，运行时会走 prototype-backed 回退路径，保持产品链路可用。

## 环境与安装

- Python 3.9+
- PyTorch、Pillow、OpenCV、Gradio

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
python demo/app.py
```

## 核心 API

### 查看内置风格

```python
import handwrite
print(handwrite.list_styles())
```

### 预检课堂笔记文本

```python
report = handwrite.inspect_text(
    "今天上课主要讲了牛顿第二定律和两个例题。",
    style="行书流畅",
)
```

### 生成单页/多页课堂笔记

```python
page = handwrite.generate(text, style="行书流畅", paper="横线纸", layout="自然", font_size=80)
pages = handwrite.generate_pages(long_text, style="行书流畅")
handwrite.export(page, "output/page.png", format="png", dpi=300)
handwrite.export_pages(pages, "output/note.pdf", format="pdf", dpi=300)
```

## Phase-2 新增 10 个创新模块

> 这一组模块在 Phase-1 的 15 模块基础上扩展，覆盖 photo→风格、动画直播、Diffusion 训练脚手架、自然语言控笔、错题本、云服务、笔迹时光机、隐写水印、几何考卷、3D 笔尖动力学。所有模块均自带 ≥10 个单元测试。

| 编号 | 模块 | 一句话定位 |
|------|------|-----------|
| ① | `photo_style` | 一张照片复刻笔迹：OCR + 分割 + 风格抽取 → 个人 prototype pack |
| ② | `live_note` | 长文逐字"直播书写"动画，GIF/MP4 导出，含标点停顿与笔尖光标 |
| ③ | `diffusion` | DDPM + ControlNet 风格脚手架，替代 zi2zi 架构，无权重时优雅降级 |
| ④ | `nl_style` | 中英文自然语言描述 → 具体 `StyleVector`，无 LLM 依赖 |
| ⑤ | `error_notebook` | 错题智能 diff、红笔批注、多页错题本 PDF |
| ⑥ | `service` | HandWrite-as-Service，FastAPI 多租户 API + `BillingMeter` + Docker |
| ⑦ | `timeline` | 笔迹时光机：多年龄样本拟合演化曲线，任意年龄重建字迹 |
| ⑧ | `watermark` | LSB + DCT 隐写水印 + `StegMarkAdapter` 适配器接口 |
| ⑨ | `geometry_sheet` | 几何图形 + 手写解题步骤 + LaTeX 公式 → 数学考卷 PDF |
| ⑩ | `pen_3d` | 2D 笔迹 → 3D 动力学（压力/倾角/旋转），Wacom WILL-lite JSON 导出 |

```python
import handwrite

# ① 一张照片复刻笔迹
result = handwrite.photo_to_style("my_handwriting.jpg", output_dir="./packs")

# ② 课堂笔记直播动画
handwrite.live_note_video("今天上课讲了牛顿第二定律。", "lecture.gif", fps=24)

# ④ 自然语言控笔
style = handwrite.parse_description("紧张焦虑的高三学生赶时间的字")

# ⑤ 错题本一键生成
handwrite.build_error_notebook(entries=[...], output_path="errors.pdf")

# ⑦ 笔迹时光机
model = handwrite.fit_timeline([(7, "child.png"), (15, "teen.png"), (25, "adult.png")])
glyph = handwrite.generate_at_age(model, "学", age=12)

# ⑧ 隐写水印嵌入与提取
watermarked = handwrite.embed_watermark(image, payload=b"author=alice", method="dct")
payload = handwrite.extract_watermark(watermarked, method="dct")

# ⑩ 3D 笔尖动力学
sim = handwrite.Pen3DSimulator(seed=42)
stroke_3d = sim.simulate([(30, 100), (60, 110), (90, 100)])
will_json = handwrite.export_will_json([stroke_3d])
```

每个模块的详细设计文档位于 `docs/innovations/01..10-*.md`。

## 15 个 Phase-1 创新模块

详见模块表格，覆盖：个性化、动力学、动画、语义排版、公式渲染、风格混合、纸张模板、OCR 风格提取、协作书写、质量评估、文本摘要、数字化、作业批改、时间演化、AR 增强等能力。完整的 API 示例可在 `tests/` 中查看。

## CLI 脚本

```bash
python scripts/download_data.py --scan_dir downloads --raw_dir data/raw
python scripts/preprocess.py --raw_dir data/raw/HWDB1.0trn_gnt --output_dir data/processed
python scripts/build_prototype_library.py --metadata data/processed/metadata.json --output_dir data/prototypes/default_note
python scripts/note_session.py --preset 牛顿定律复习 --output_dir output/session
python scripts/train.py --data_dir data/processed --styles_file data/processed/selected_styles.json --output_dir weights
python scripts/evaluate.py --output-dir evaluation
python demo/app.py
```

## 项目结构

```
src/handwrite/
├── animation/          # 动画
├── ar/                 # AR
├── collaboration/      # 协作
├── data/               # 数据
├── diffusion/          # Phase-2 ③ Diffusion 训练脚手架
├── digitization/       # 数字化
├── dynamics/           # 动力学
├── engine/             # 核心引擎
├── error_notebook/     # Phase-2 ⑤ 错题本
├── formula/            # 公式
├── geometry_sheet/     # Phase-2 ⑨ 几何考卷
├── grading/            # 批改
├── live_note/          # Phase-2 ② 直播动画
├── nl_style/           # Phase-2 ④ 自然语言控笔
├── ocr_style/          # OCR 风格
├── papers/             # 纸张
├── pen_3d/             # Phase-2 ⑩ 3D 笔尖动力学
├── personalization/    # 个性化
├── photo_style/        # Phase-2 ① 照片→风格
├── quality/            # 质量
├── semantic/           # 语义
├── service/            # Phase-2 ⑥ HandWrite-as-Service
├── style_mixing/       # 风格混合
├── summary/            # 摘要
├── temporal/           # 时间演化
├── timeline/           # Phase-2 ⑦ 笔迹时光机
├── watermark/          # Phase-2 ⑧ 防伪水印
├── composer.py         # 页面合成器
├── exporter.py         # 导出器（PNG/PDF）
├── prototypes.py       # 原型管理
└── styles.py           # 风格管理
```

## License

MIT
