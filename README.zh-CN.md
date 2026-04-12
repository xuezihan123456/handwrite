# HandWrite 中文说明

## 项目介绍

`HandWrite` 现在更适合作为一个 **课堂笔记生成器** 来理解，而不只是通用手写 demo。

仓库已经覆盖：数据预处理、训练骨架、推理封装、页面排版、PNG/PDF 导出、Gradio demo，以及一套面向课堂笔记场景的真实感预检思路。

它依然不是“自带高质量预训练模型、开箱即用的最终成品”。仓库不包含真实预训练权重。没有有效权重时，运行时会优先走 prototype-backed note fallback，再退到更接近手写观感的字体/形变路径，所以它更适合做产品链路验证、demo、和本地迭代，而不是直接宣称真实模型效果。

## 当前功能

- 可安装的 Python 包，入口在 `src/handwrite`
- CASIA `.gnt` 解析和数据预处理
- 5 个内置风格名，通过 `handwrite.list_styles()` 暴露
- 单字、单页、多页生成 API
- `handwrite.inspect_text(...)` 预检 API，用于课堂笔记真实感预检
- `StyleEngine` 推理封装，支持自动查找权重和无权重回退
- starter prototype pack，用于默认课堂笔记产品链路
- 本地 prototype library 构建脚本，可为更大覆盖做准备
- PNG 导出、单页 PDF 导出、多页 PDF 导出
- Gradio demo，支持预检报告、逐页预览浏览、整份文档下载
- 训练与评估脚本

## 环境与安装

- Python 3.9+
- 当前开发环境以 Windows 为主

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Python API 用法

### 查看内置风格

```python
import handwrite

print(handwrite.list_styles())
```

当前内置风格顺序保持稳定，但课堂笔记主链路的默认风格应理解为更自然流畅的那一档，而不是依赖风格列表顺序来推断默认值。

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

它会输出类似信息：

- 哪些字有 prototype 覆盖
- 哪些字会走模型/高质量路径
- 哪些字可能回退到较低真实感路径
- 给出建议，但**不会自动改写原文**
- 当前 active source 是内置 starter pack，还是你自己的本地 prototype pack

### 完整产品主链路

如果你想把它当成真正的课堂笔记产品来用，推荐主链路是：

1. 先用 `scripts/build_prototype_library.py` 构建一个更大的本地 prototype pack  
2. 在运行时或 demo 里通过 `prototype_pack` 指向这个 pack  
3. 先做预检，再生成/导出

```python
import handwrite

prototype_pack = "data/prototypes/my_note_pack"

report = handwrite.inspect_text(
    "今天上课主要讲了牛顿第二定律和两个例题。",
    style="行书流畅",
    prototype_pack=prototype_pack,
)
print(report["prototype_source"]["label"])

page = handwrite.generate(
    "今天上课主要讲了牛顿第二定律和两个例题。",
    style="行书流畅",
    paper="横线纸",
    layout="自然",
    font_size=80,
    prototype_pack=prototype_pack,
)
```

### 生成单字

```python
import handwrite

img = handwrite.char("你", style="行书流畅")
img.save("single.png")
```

### 生成单页课堂笔记

```python
import handwrite

page = handwrite.generate(
    "今天上课主要讲了牛顿第二定律和两个例题。",
    style="行书流畅",
    paper="横线纸",
    layout="自然",
    font_size=80,
)
page.save("note_page.png")
```

### 使用更大的本地 prototype pack

```python
import handwrite

prototype_pack = "data/prototypes/my_note_pack"

report = handwrite.inspect_text(
    "今天上课主要讲了牛顿第二定律和两个例题。",
    style="行书流畅",
    prototype_pack=prototype_pack,
)
page = handwrite.generate(
    "今天上课主要讲了牛顿第二定律和两个例题。",
    style="行书流畅",
    paper="横线纸",
    layout="自然",
    font_size=80,
    prototype_pack=prototype_pack,
)
print(report["prototype_source"]["label"])
```

这就是当前推荐的产品主链路：
1. 先用 `scripts/build_prototype_library.py` 基于 processed 数据构建更大的本地 pack
2. 再在 API 或 demo 里通过 `prototype_pack` 指向这个 pack
3. 先看预检里的 active source 和覆盖情况，再生成课堂笔记

### 生成多页课堂笔记

```python
import handwrite

pages = handwrite.generate_pages(
    "课堂笔记内容" * 120,
    style="行书流畅",
    paper="横线纸",
    layout="自然",
    font_size=80,
)

for index, page in enumerate(pages, start=1):
    page.save(f"note_{index:03d}.png")
```

### 导出 PNG / PDF

```python
import handwrite

page = handwrite.generate("导出示例。", style="行书流畅")
pages = handwrite.generate_pages("长文本内容" * 80, style="行书流畅")

handwrite.export(page, "output/page.png", format="png", dpi=300)
handwrite.export(page, "output/page.pdf", format="pdf", dpi=300)
handwrite.export_pages(pages, "output/pages", format="png", prefix="note", dpi=300)
handwrite.export_pages(pages, "output/note.pdf", format="pdf", dpi=300)
```

## CLI / 脚本说明

### 1. 本地数据准备助手

```bash
python scripts/download_data.py --scan_dir downloads --raw_dir data/raw
```

用途：把你已经手工下载好的 CASIA/HWDB zip 或已解压 split 目录整理成标准 `data/raw/<split>` 布局。

### 2. 预处理

```bash
python scripts/preprocess.py --raw_dir data/raw/HWDB1.0trn_gnt --output_dir data/processed --font_path data/fonts/NotoSerifSC-Regular.otf --charset_level 500 --min_writer_coverage 0.9
```

用途：解析 `.gnt`，归一化位图，渲染标准字图，并输出训练/构建 prototype library 所需的 `metadata.json`。

### 3. 构建更大的 prototype library

```bash
python scripts/build_prototype_library.py --metadata data/processed/metadata.json --output_dir data/prototypes/default_note
```

定位：

- 仓库自带的是 starter pack，只够支撑第一版课堂笔记产品闭环
- 如果你想往 2000+ 覆盖走，需要用本地 processed 数据构建更大的私有 prototype pack
- 构建完之后，运行时和 demo 都可以通过 `prototype_pack` 指向这个目录或其中的 `manifest.json`
- demo 现在内置了几种课堂笔记模板，可先载入示例，再改成自己的内容
- 如果 `prototype_pack` 指向的目录或 `manifest.json` 不存在，demo 会给出明确报错，而不是静默失败

### 4. 风格配置文件

`scripts/train.py` 仍然需要单独的 `selected_styles.json`。

### 5. 课堂笔记 session CLI

```bash
python scripts/note_session.py --preset 牛顿定律复习 --output_dir output/session
```

这个 CLI 入口更像产品形态：

- 支持 `--text`
- 支持 `--text_file`
- 支持 `--preset`
- 支持 `--prototype_pack`
- 一次性导出 PNG 页、PDF 和 markdown session report

### 6. 训练

```bash
python scripts/train.py --data_dir data/processed --styles_file data/processed/selected_styles.json --output_dir weights --batch_size 8 --epochs 30
```

### 7. 评估

```bash
python scripts/evaluate.py --output-dir evaluation
```

### 8. Demo

```bash
python demo/app.py
```

默认地址：

- `http://localhost:7860`

当前 demo 目标不是纯“文本转手写”，而是课堂笔记产品主链路：

- 输入课堂笔记正文
- 如有更大的本地 pack，可把路径填到 `Local Prototype Pack (optional)`
- 先看真实感预检报告
- 再生成多页手写笔记
- 逐页预览
- 下载整份 PNG / PDF

## 当前限制与务实说明

### 1. starter pack 不是完整大字库

仓库会附带一个小型 starter prototype pack，保证默认课堂笔记链路能跑通，但它不是完整 2000+ 内置资产体量。

### 2. 真正的大覆盖仍然依赖本地构建

如果你要做更大覆盖、更稳定的真实感，需要基于你自己的 processed 数据构建更大的 prototype library，或者加载真实 generator 权重。产品主链路默认鼓励你：**先构建本地 pack，再把它接到 `prototype_pack`，然后通过预检确认 active source。**

### 3. 不自动改原文

预检只做报告和建议，不会偷偷重写你的内容。产品把决定权留给用户。

### 4. 评估指标仍然偏轻量

仓库里有 Frechet-style 的轻量评估思路，但还不是完整 research-grade benchmark 流程。

## 推荐使用顺序

1. `pip install -e ".[dev]"`
2. `pytest`
3. 先用 `handwrite.inspect_text(...)` 或 demo 预检能力看文本覆盖情况
4. 直接体验 starter pack 的课堂笔记生成链路
5. 准备 `.gnt` 数据和可用中文字体
6. 如需扩覆盖，运行 `scripts/download_data.py` + `scripts/preprocess.py`
7. 运行 `scripts/build_prototype_library.py` 构建更大的本地 prototype pack
8. 如需更高质量，再准备 `selected_styles.json`、训练权重并运行训练/评估脚本
