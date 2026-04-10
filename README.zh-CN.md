# HandWrite 中文说明

## 项目简介

`HandWrite` 是一个面向中文手写体生成的 Python 项目。它的目标不是简单替换静态字体，而是逐步构建一条完整流程：

- 解析 CASIA-HWDB 手写数据集
- 构建标准字图 / 手写字图配对数据
- 训练基于 zi2zi 思路的条件生成模型
- 生成单字、整页和多页手写文档
- 提供 Python API、命令行脚本和 Gradio Demo

当前仓库已经具备从“文档设计”走到“可运行工程骨架”的主要基础模块。

## 当前状态

目前已经完成：

- 项目骨架与可安装 Python 包
- 中文风格定义与 `selected_styles.json` 读取
- `.gnt` 文件解析
- 数据预处理与 `metadata.json` 生成
- `HandwriteDataset`
- 标准字体渲染
- `Generator` / `Discriminator` 网络定义
- `StyleEngine` 推理包装
- `Composer` 页面排版
- PNG 导出与多页 PNG 导出
- 训练核心循环与训练 CLI
- 评估脚本
- `generate_pages()` 多页公开 API
- Gradio Demo 入口

当前仍然是 MVP 阶段，以下内容尚未真正完成：

- 仓库内不包含真实训练好的模型权重
- PDF 导出还没有实现
- TTF 字体导出没有实现
- 用户自定义风格还没有实现

## 适合谁使用

- 想自己训练中文手写生成模型的人
- 做竞赛 / 科研，需要把“设计说明”落成工程代码的人
- 希望基于本地环境运行，而不是依赖外部 SaaS 的人

## 环境要求

- Python 3.9+
- Windows 优先（当前开发环境就是 Windows）
- 已安装或可安装：
  - PyTorch
  - torchvision
  - Pillow
  - NumPy
  - OpenCV
  - Gradio
  - pytest

## 安装方式

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

运行测试：

```bash
pytest
```

## Python API 用法

### 1. 查看可用风格

```python
import handwrite

print(handwrite.list_styles())
```

当前内置风格：

- 工整楷书
- 圆润可爱
- 行书流畅
- 偏瘦紧凑
- 随意潦草

### 2. 生成单字

```python
import handwrite

img = handwrite.char("你", style="工整楷书")
img.save("single.png")
```

### 3. 生成单页

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

### 4. 生成多页

```python
import handwrite

pages = handwrite.generate_pages(
    "长文本内容" * 200,
    style="工整楷书",
    paper="方格纸",
    layout="自然",
    font_size=80,
)

for index, page in enumerate(pages, start=1):
    page.save(f"page_{index}.png")
```

## 命令行工具

### 1. 预处理数据

```bash
python scripts/preprocess.py \
  --raw_dir data/raw/HWDB1.0trn_gnt \
  --output_dir data/processed \
  --font_path data/fonts/NotoSerifSC-Regular.otf
```

作用：

- 读取 `.gnt`
- 过滤低质量样本
- 归一化手写图
- 渲染标准字图
- 输出 `metadata.json`

### 2. 启动训练

```bash
python scripts/train.py \
  --data_dir data/processed \
  --styles_file data/processed/selected_styles.json \
  --output_dir weights \
  --batch_size 8 \
  --epochs 30
```

训练输出包括：

- `weights/checkpoint_epoch_{N}.pt`
- `weights/samples/...png`
- `weights/train_log.csv`

### 3. 运行评估

```bash
python scripts/evaluate.py --output-dir evaluation
```

当前会生成：

- `evaluation/style_comparison.png`
- `evaluation/complex_chars.png`

### 4. 启动 Demo

```bash
python demo/app.py
```

默认地址：

- `http://localhost:7860`

## 导出工具

### 导出单页 PNG

```python
from handwrite.exporter import export_png

export_png(page, "output/page.png", dpi=300)
```

### 导出多页 PNG

```python
from handwrite.exporter import export_pages_png

export_pages_png(pages, "output/pages", prefix="page", dpi=300)
```

## StyleEngine 说明

`StyleEngine` 当前支持：

- 无权重时的 graceful fallback
- 从显式 `weights_path` 加载生成器权重
- 从 `weights/` 目录自动发现常见 checkpoint / state dict
- 权重存在时走真实 `Generator`
- 权重缺失或损坏时回退到占位/标准字渲染路径

这意味着：

- 仓库“结构上”已经支持真实推理
- 但如果你没有准备训练好的权重，当前输出仍然属于 MVP 级行为，不代表最终手写质量

## 目录结构

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

关键目录说明：

- `src/handwrite/data/`: 数据解析、字符集、字体渲染、Dataset
- `src/handwrite/engine/`: Generator、Discriminator、StyleEngine、训练核心
- `scripts/`: preprocess / train / evaluate 等命令行入口
- `demo/`: Gradio 应用
- `tests/`: 单元测试与回归测试
- `weights/`: 模型权重目录，不提交真实大文件

## 目前的限制与说明

### 1. 没有内置真实预训练权重

仓库没有附带训练好的 `.pt` / `.pth` 权重，所以：

- API 可调用
- 推理路径可工作
- 但默认输出仍然是 MVP 行为，不是最终真实手写效果

### 2. 评估脚本的 FID 是轻量近似版本

当前 `scripts/evaluate.py` 里的 `fid_score` 是轻量 Frechet-style 指标，适合早期工程验证，不等同于完整 `pytorch-fid` 流程。

### 3. PDF 还未实现

仓库目前只提供：

- 单页 PNG 导出
- 多页 PNG 导出

### 4. 大文件不要进 git

不要提交：

- 数据集
- `.gnt`
- 大模型权重
- 大体积评估图片

## 推荐使用顺序

如果你第一次接手这个项目，建议按下面顺序：

1. `pip install -e ".[dev]"`
2. `pytest`
3. 准备 `CASIA-HWDB` 和字体文件
4. 跑 `scripts/preprocess.py`
5. 准备 `selected_styles.json`
6. 跑 `scripts/train.py`
7. 跑 `scripts/evaluate.py`
8. 用 `demo/app.py` 看效果

## 项目定位总结

这不是一个“已经训好的成品手写生成器仓库”，而是一个：

- 设计清晰
- 结构完整
- 可测试
- 可继续训练和迭代

的中文手写生成工程项目。

如果你想直接拿来做最终展示，需要下一步补真实数据和权重；如果你想继续开发，这个仓库现在已经具备很强的继续扩展基础。
