# HandWrite 中文说明

## 项目介绍

HandWrite 是一个 **AI驱动的中文手写生成器**，包含 **15个创新模块**，覆盖从笔迹个性化到AR增强的完整技术栈。它既可以作为课堂笔记生成工具使用，也是一个可扩展的手写合成研究平台。

仓库覆盖：数据预处理、训练骨架、推理封装、页面排版、PNG/PDF 导出、Gradio demo，以及面向课堂笔记场景的真实感预检。同时集成了15个创新模块，涵盖笔迹动力学、动画生成、语义排版、公式渲染、风格混合、协作书写、质量评估、手写识别、作业批改、时间演化和AR增强等能力。

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
import handwrite

report = handwrite.inspect_text(
    "今天上课主要讲了牛顿第二定律和两个例题。",
    style="行书流畅",
)
print(report["summary"])
for item in report["suggestions"]:
    print(item)
```

### 生成单页课堂笔记

```python
import handwrite

page = handwrite.generate(
    "今天上课主要讲了牛顿第二定律和两个例题。",
    style="行书流畅", paper="横线纸", layout="自然", font_size=80,
)
page.save("note_page.png")
```

### 生成多页课堂笔记

```python
import handwrite

pages = handwrite.generate_pages(
    "课堂笔记内容" * 120,
    style="行书流畅", paper="横线纸", layout="自然", font_size=80,
)
for index, page in enumerate(pages, start=1):
    page.save(f"note_{index:03d}.png")
```

### 导出 PNG / PDF

```python
import handwrite

handwrite.export(page, "output/page.png", format="png", dpi=300)
handwrite.export(page, "output/page.pdf", format="pdf", dpi=300)
handwrite.export_pages(pages, "output/note.pdf", format="pdf", dpi=300)
```

### 使用本地 prototype pack

```python
import handwrite

prototype_pack = "data/prototypes/my_note_pack"

report = handwrite.inspect_text(
    "今天上课主要讲了牛顿第二定律和两个例题。",
    style="行书流畅", prototype_pack=prototype_pack,
)
page = handwrite.generate(
    "今天上课主要讲了牛顿第二定律和两个例题。",
    style="行书流畅", paper="横线纸", layout="自然", font_size=80,
    prototype_pack=prototype_pack,
)
```

## 15个创新模块

### 1. 个性化模块 (Personalization)

从手写样本图片中提取笔迹特征（笔画宽度、倾斜角度、连笔率、墨迹分布），生成风格向量，并合成个性化字形包。

```python
from handwrite.personalization import SampleAnalyzer, StyleExtractor, GlyphSynthesizer

analyzer = SampleAnalyzer()
features = analyzer.analyze("sample.png")

extractor = StyleExtractor()
style = extractor.extract(features)

synth = GlyphSynthesizer(style)
synth.synthesize_pack(output_dir, charset="你好世界")
```

### 2. 笔迹动力学模块 (Dynamics)

模拟书写过程中的笔压变化、墨水流淌效果和书写速度，生成更真实的笔迹动态效果。

```python
from handwrite.dynamics import DynamicsEngine

engine = DynamicsEngine()
result = engine.simulate(stroke_data)
# 包含 pressure_map, ink_flow, speed_profile
```

### 3. 动画模块 (Animation)

基于笔画顺序检测和贝塞尔轨迹生成，支持逐笔动画渲染，可导出GIF和MP4格式。

```python
from handwrite.animation import StrokeOrderDetector, TrajectoryGenerator, AnimationEngine

detector = StrokeOrderDetector()
order = detector.detect(char_image)

generator = TrajectoryGenerator()
trajectory = generator.generate(order)

engine = AnimationEngine()
frames = engine.render(trajectory)
engine.export_gif(frames, "writing.gif")
```

### 4. 语义排版模块 (Semantic)

分析文本语义结构，智能规划页面布局，支持标题、段落、列表等语义标注渲染。

```python
from handwrite.semantic import TextAnalyzer, LayoutPlanner, SemanticComposer

analyzer = TextAnalyzer()
structure = analyzer.analyze("课堂笔记内容...")

planner = LayoutPlanner()
layout = planner.plan(structure, page_size=(2480, 3508))

composer = SemanticComposer()
page = composer.compose(layout)
```

### 5. 公式渲染模块 (Formula)

支持LaTeX数学公式和化学方程式的解析、排版和渲染，包括矩阵、分数、上下标等复杂结构。

```python
from handwrite.formula import LatexParser, ChemistryParser, FormulaRenderer

parser = LatexParser()
formula = parser.parse(r"\frac{-b \pm \sqrt{b^2-4ac}}{2a}")

chem = ChemistryParser()
equation = chem.parse("2H2 + O2 -> 2H2O")

renderer = FormulaRenderer()
img = renderer.render(formula)
```

### 6. 风格混合模块 (Style Mixing)

支持多种手写风格的混合、迁移和插值，可以创建独特的个人风格。

```python
from handwrite.style_mixing import StyleMixer, StyleTransfer

mixer = StyleMixer()
blended = mixer.mix(style_a, style_b, ratio=0.6)

transfer = StyleTransfer()
transferred = transfer.transfer(source_style, target_content)
```

### 7. 纸张模板模块 (Papers)

内置6种纸张模板（康奈尔笔记、英语练习纸、错题本、作文稿纸、五线谱、思维导图），支持自定义纸张。

```python
from handwrite.papers import PaperRegistry, PaperRenderer

registry = PaperRegistry()
paper = registry.get("cornell")

renderer = PaperRenderer()
img = renderer.render(paper)
```

内置模板：
- `cornell` - 康奈尔笔记（线索栏+笔记栏+总结栏）
- `english_practice` - 英语练习纸（四线三格）
- `error_notebook` - 错题本（题目+错误解法+正确解法+反思）
- `essay_grid` - 作文稿纸（方格纸）
- `staff` - 五线谱
- `mind_map` - 思维导图

### 8. OCR风格提取模块 (OCR Style)

从扫描的手写图片中提取笔迹风格，生成可用于个性化生成的原型字体。

```python
from handwrite.ocr_style import ImagePreprocessor, CharacterSegmenter, StyleExtractor

preprocessor = ImagePreprocessor()
clean = preprocessor.preprocess(scanned_image)

segmenter = CharacterSegmenter()
chars = segmenter.segment(clean)

extractor = StyleExtractor()
style = extractor.extract(chars)
```

### 9. 协作书写模块 (Collaboration)

支持多人协作书写，自动分配书写区域，混合不同书写者的风格。

```python
from handwrite.collaboration import CollaborativeComposer, StyleBlender

composer = CollaborativeComposer()
segments = composer.assign_segments(contributors, content)

blender = StyleBlender()
result = blender.blend(segments)
```

### 10. 质量评估模块 (Quality)

从真实感和自然度两个维度评估生成的手写质量，提供改进建议。

```python
from handwrite.quality import QualityEngine

engine = QualityEngine()
report = engine.evaluate(generated_image)
# report 包含 authenticity_score, naturalness_score, suggestions
```

### 11. 文本摘要模块 (Summary)

自动提取文本关键信息，生成思维导图和大纲布局。

```python
from handwrite.summary import TextSummarizer, MindMapLayout

summarizer = TextSummarizer()
summary = summarizer.summarize(long_text)

layout = MindMapLayout()
mind_map = layout.generate(summary)
```

### 12. 数字化模块 (Digitization)

手写文字OCR识别，支持风格保持的往返编辑——识别后修改文字，再以原风格重新生成。

```python
from handwrite.digitization import HandwritingRecognizer, RoundTripEngine

recognizer = HandwritingRecognizer()
text = recognizer.recognize(handwritten_image)

engine = RoundTripEngine()
edited_image = engine.edit(handwritten_image, original_text=text, new_text="修改后的文字")
```

### 13. 作业批改模块 (Grading)

自动检测手写作业中的错误，生成批注和评分反馈。

```python
from handwrite.grading import GradingEngine

engine = GradingEngine()
result = engine.grade(submission_image, reference_text="正确答案")
# result 包含 score, errors, annotations, feedback
```

### 14. 时间演化模块 (Temporal)

模拟不同年龄段和技能水平的手写变化，支持历史风格重现。

```python
from handwrite.temporal import TemporalEngine

engine = TemporalEngine()
aged = engine.simulate_age(glyph, target_age=10)
skilled = engine.simulate_skill(glyph, skill_level=0.8)
```

### 15. AR增强模块 (AR)

纸张检测、透视变换和纹理融合，实现手写内容的AR增强效果。

```python
from handwrite.ar import AREngine

engine = AREngine()
detected = engine.detect_paper(camera_frame)
transformed = engine.transform(detected)
blended = engine.blend(transformed, digital_content)
```

## CLI 脚本

```bash
# 数据准备
python scripts/download_data.py --scan_dir downloads --raw_dir data/raw

# 预处理
python scripts/preprocess.py --raw_dir data/raw/HWDB1.0trn_gnt --output_dir data/processed

# 构建 prototype library
python scripts/build_prototype_library.py --metadata data/processed/metadata.json --output_dir data/prototypes/default_note

# 课堂笔记 session
python scripts/note_session.py --preset 牛顿定律复习 --output_dir output/session

# 训练
python scripts/train.py --data_dir data/processed --styles_file data/processed/selected_styles.json --output_dir weights

# 评估
python scripts/evaluate.py --output-dir evaluation

# Demo
python demo/app.py
```

## 项目结构

```
src/handwrite/
├── animation/          # 动画模块：笔画顺序、轨迹生成、帧渲染
├── ar/                 # AR模块：纸张检测、透视变换、纹理融合
├── collaboration/      # 协作模块：贡献者管理、区域分配、风格混合
├── data/               # 数据模块：CASIA解析、字符集、数据集
├── digitization/       # 数字化模块：OCR识别、风格保持、往返编辑
├── dynamics/           # 动力学模块：笔压、墨水、速度模拟
├── engine/             # 核心引擎：生成器、判别器、训练
├── formula/            # 公式模块：LaTeX/化学解析、排版、渲染
├── grading/            # 批改模块：错误检测、批注、评分、反馈
├── ocr_style/          # OCR风格模块：预处理、分割、风格提取
├── papers/             # 纸张模块：模板注册、内置纸张、渲染
├── personalization/    # 个性化模块：样本分析、风格提取、字形合成
├── quality/            # 质量模块：真实感、自然度评分、改进建议
├── semantic/           # 语义模块：文本分析、布局规划、标注渲染
├── style_mixing/       # 风格混合模块：混合、迁移、插值
├── summary/            # 摘要模块：文本摘要、思维导图、大纲
├── temporal/           # 时间模块：年龄档案、历史风格、技能模拟
├── composer.py         # 页面合成器
├── exporter.py         # 导出器（PNG/PDF）
├── prototypes.py       # 原型管理
└── styles.py           # 风格管理
```

## 当前限制

1. **无预训练权重** - 仓库不包含真实模型权重，使用 prototype-backed 回退路径
2. **Starter pack 覆盖有限** - 内置原型包较小，大覆盖需本地构建
3. **不自动改原文** - 预检只做报告和建议，不会重写用户内容
4. **评估偏轻量** - 包含基础评估指标，非完整 research-grade benchmark

## 推荐使用顺序

1. `pip install -e ".[dev]"` → 安装依赖
2. `pytest` → 运行测试（623+ tests）
3. `handwrite.inspect_text(...)` → 预检文本覆盖
4. `python demo/app.py` → 体验完整产品链路
5. 准备 `.gnt` 数据和中文字体
6. `scripts/build_prototype_library.py` → 构建更大的 prototype pack
7. 如需更高质量，运行训练/评估脚本
