# HandWrite 完整实施指南

> 本文档是交给 GPT/其他 AI 执行的详细实施手册。
> 每个任务都包含：做什么、怎么做、验收标准、预估时间。
> 环境：Windows 11, Python 3.x, PyTorch 2.6+cu124, RTX 4060 8GB, 项目根目录 D:/projects/handwrite

---

## 第一阶段：项目初始化（0.5天）

### 任务 1.1：创建项目结构

**做什么**：初始化 Python 项目骨架。

**怎么做**：
```
D:/projects/handwrite/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── handwrite/
│       ├── __init__.py
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── generator.py
│       │   ├── discriminator.py
│       │   ├── model.py
│       │   └── train.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── casia_parser.py
│       │   ├── dataset.py
│       │   ├── font_renderer.py
│       │   └── charsets.py
│       ├── composer.py
│       ├── exporter.py
│       └── styles.py
├── weights/
│   └── .gitkeep
├── demo/
│   └── app.py
├── scripts/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
└── tests/
    ├── test_engine.py
    ├── test_composer.py
    ├── test_exporter.py
    └── fixtures/
        └── sample_chars/
```

**pyproject.toml 内容**：
```toml
[project]
name = "handwrite"
version = "0.1.0"
description = "AI Chinese handwriting generator"
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
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]
```

**.gitignore 必须包含**：
```
weights/*.pt
weights/*.pth
data/raw/
data/processed/
__pycache__/
*.pyc
.venv/
```

**验收标准**：
- [ ] `pip install -e .` 成功
- [ ] `python -c "import handwrite"` 不报错
- [ ] git init 完成，首次 commit

---

## 第二阶段：数据获取与预处理（3天）

### 任务 2.1：下载 CASIA-HWDB 数据集

**做什么**：获取中科院的中文手写字符数据集。

**怎么做**：
1. 访问 http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
2. 下载 HWDB1.0、HWDB1.1 的训练集和测试集（.gnt 格式，脱机单字）
3. 如果官网访问慢，GitHub 搜索 "casia-hwdb" 找镜像
4. 将下载的 zip 文件解压到 `D:/projects/handwrite/data/raw/`

**目录结构**：
```
data/raw/
├── HWDB1.0trn_gnt/
│   ├── 001.gnt
│   ├── 002.gnt
│   └── ...
├── HWDB1.0tst_gnt/
│   └── ...
├── HWDB1.1trn_gnt/
│   └── ...
└── HWDB1.1tst_gnt/
    └── ...
```

**同时下载标准字体**：
- 思源宋体（Noto Serif SC）：https://fonts.google.com/noto/specimen/Noto+Serif+SC
- 下载 Regular 权重，放到 `data/fonts/NotoSerifSC-Regular.otf`

**验收标准**：
- [ ] `data/raw/` 下有 .gnt 文件，总大小 2-4 GB
- [ ] `data/fonts/NotoSerifSC-Regular.otf` 存在
- [ ] 能用 Python 的 `open()` 读取 .gnt 文件的前几个字节

### 任务 2.2：编写 .gnt 文件解析器

**做什么**：实现 `src/handwrite/data/casia_parser.py`，解析 .gnt 二进制文件为 (字符, 图片) 对。

**怎么做**：

.gnt 文件的二进制格式：
```
每个样本：
  - 4 字节：样本总长度（uint32, little-endian）
  - 2 字节：GBK 编码的汉字标签
  - 2 字节：图片宽度 W（uint16, little-endian）
  - 2 字节：图片高度 H（uint16, little-endian）
  - W×H 字节：灰度位图数据（逐行存储，0=白，255=黑）
```

**注意**：
- 标签是 GBK 编码，不是 Unicode。需要 `label_bytes.decode('gbk')` 转换
- 有些 .gnt 文件中可能有损坏的样本，需要 try-except 跳过
- 所有文件操作必须用 `encoding='utf-8'`（仅文本文件）或 `'rb'`（二进制文件）

**实现的函数接口**：
```python
def parse_gnt_file(gnt_path: str) -> list[tuple[str, np.ndarray]]:
    """
    解析单个 .gnt 文件。
    返回 [(字符Unicode, 灰度图np.ndarray), ...]
    """

def parse_gnt_directory(dir_path: str) -> dict[str, list[tuple[str, np.ndarray]]]:
    """
    解析目录下所有 .gnt 文件。
    返回 {writer_id: [(字符, 图片), ...], ...}
    writer_id 从文件名提取（如 "001.gnt" → "001"）
    """

def save_parsed_images(parsed_data: dict, output_dir: str):
    """
    将解析结果保存为 PNG 文件。
    输出结构：output_dir/writer_001/你.png, output_dir/writer_001/好.png, ...
    """
```

**验收标准**：
- [ ] 能正确解析至少一个 .gnt 文件
- [ ] 解析出的图片用 matplotlib 显示，能看到正确的手写汉字
- [ ] 字符标签（Unicode）与图片内容一致
- [ ] 能处理损坏样本而不崩溃
- [ ] 运行 `parse_gnt_directory` 对整个 HWDB1.0trn 目录，统计总样本数，应在几十万级别

### 任务 2.3：编写字符集定义

**做什么**：实现 `src/handwrite/data/charsets.py`，定义 500 个最常用汉字。

**怎么做**：
- 按照《现代汉语常用字表》，取频率最高的 500 个汉字
- 加上常用标点：，。！？、；：""''（）—…《》
- 加上数字 0-9、英文字母 a-z A-Z

**实现接口**：
```python
COMMON_500: list[str] = ["的", "一", "是", "了", "我", "不", "人", "在", ...]  # 500个
PUNCTUATION: list[str] = ["，", "。", "！", "？", ...]
DIGITS: list[str] = ["0", "1", ..., "9"]
LETTERS: list[str] = ["a", "b", ..., "z", "A", "B", ..., "Z"]

def get_charset(level: str = "500") -> list[str]:
    """level: '500', '1000', '3500'"""
    if level == "500":
        return COMMON_500 + PUNCTUATION + DIGITS + LETTERS
    ...
```

**验收标准**：
- [ ] `get_charset("500")` 返回约 580 个字符（500汉字+标点+数字+字母）
- [ ] 列表中无重复字符
- [ ] 涵盖"的一是了我不人在有这"等高频字

### 任务 2.4：编写标准字体渲染器

**做什么**：实现 `src/handwrite/data/font_renderer.py`，用宋体渲染标准字符图片。

**怎么做**：
```python
from PIL import Image, ImageDraw, ImageFont

def render_standard_char(
    char: str,
    font_path: str,
    image_size: int = 256,
    char_size: int = 200,  # 字符在画布上的目标大小
) -> Image.Image:
    """
    渲染单个字符的标准字体图片。
    返回 256×256 灰度 PIL.Image，白底黑字，居中放置。
    """
```

**关键细节**：
- 字符必须居中（水平+垂直），与手写图的居中策略一致
- 字符大小约占画布的 75-85%（与手写图的留白比例匹配）
- 输出灰度图（mode='L'），白底(255)黑字(0)
- 标点和数字也要正确渲染

**验收标准**：
- [ ] `render_standard_char("你", "data/fonts/NotoSerifSC-Regular.otf")` 返回 256×256 灰度图
- [ ] 字符居中，视觉上与手写图的居中方式一致
- [ ] 标点符号（如逗号、句号）能正确渲染且位置合理
- [ ] 批量渲染 500 字耗时 < 10 秒

### 任务 2.5：数据预处理主流程

**做什么**：实现 `scripts/preprocess.py`，完成从原始 .gnt 到训练数据的完整预处理。

**怎么做**：

完整流程：
1. 解析所有 .gnt 文件 → 得到 {writer_id: [(char, image), ...]}
2. 过滤质量差的样本：
   - 图片尺寸 < 32×32 或 > 300×300 → 丢弃
   - 图片几乎全白（均值 > 250）或全黑（均值 < 5）→ 丢弃
3. 统计每个书写者覆盖了多少个 COMMON_500 中的字符
4. 只保留覆盖率 ≥ 90%（至少写了 450/500 个常用字）的书写者
5. 对保留的手写图做标准化：
   - 二值化（Otsu 阈值）
   - 计算字符 bounding box，裁剪多余白边
   - 保持长宽比缩放，使字符主体占据约 200×200 像素
   - 居中放置到 256×256 白色画布上
6. 对每个 (writer_id, char)，渲染对应的标准字体图
7. 保存配对数据 + metadata.json

**输出目录**：
```
data/processed/
├── writer_001/
│   ├── 4F60_standard.png    # "你" 的 Unicode hex
│   ├── 4F60_handwrite.png
│   ├── 597D_standard.png    # "好"
│   ├── 597D_handwrite.png
│   └── ...
├── writer_002/
│   └── ...
└── metadata.json
```

**metadata.json 格式**：
```json
{
  "writers": ["001", "002", ...],
  "charset": ["你", "好", ...],
  "pairs": [
    {
      "writer_id": "001",
      "char": "你",
      "unicode_hex": "4F60",
      "standard": "data/processed/writer_001/4F60_standard.png",
      "handwrite": "data/processed/writer_001/4F60_handwrite.png"
    },
    ...
  ],
  "stats": {
    "total_pairs": 125000,
    "num_writers": 250,
    "num_chars": 500
  }
}
```

**验收标准**：
- [ ] metadata.json 生成且 stats 合理（总对数应在 5万-30万 量级）
- [ ] 随机抽取 50 对，并排显示标准图和手写图，人工确认：
  - 两张图对应的是同一个字
  - 两张图的居中方式和大小比例一致
  - 手写图质量可接受（笔画清晰、完整）
- [ ] 没有空目录或 0 字节的图片文件
- [ ] 所有书写者的覆盖率 ≥ 90%

### 任务 2.6：风格筛选

**做什么**：从预处理后的书写者中，挑选 5 个作为内置风格。

**怎么做**：
1. 编写一个可视化脚本 `scripts/select_styles.py`：
   - 对每个书写者，展示 20 个随机汉字的手写图（4×5 网格）
   - 显示书写者 ID 和覆盖率
2. 人工浏览，按以下标准挑选 5 个：

| 风格 ID | 风格名称 | 筛选标准 |
|---------|----------|----------|
| 0 | 工整楷书 | 端正规范，笔画清晰分明，结构方正 |
| 1 | 圆润可爱 | 笔画圆润饱满，结构偏圆，看起来可爱 |
| 2 | 行书流畅 | 笔画有连笔趋势，书写流畅，稍快 |
| 3 | 偏瘦紧凑 | 字形偏窄偏瘦，笔画紧凑 |
| 4 | 随意潦草 | 随意放松，笔画不太规范，稍潦草 |

3. 将选中的 5 个 writer_id 保存到 `data/processed/selected_styles.json`：
```json
{
  "styles": [
    {"id": 0, "name": "工整楷书", "writer_id": "057"},
    {"id": 1, "name": "圆润可爱", "writer_id": "123"},
    {"id": 2, "name": "行书流畅", "writer_id": "089"},
    {"id": 3, "name": "偏瘦紧凑", "writer_id": "201"},
    {"id": 4, "name": "随意潦草", "writer_id": "167"}
  ]
}
```

**验收标准**：
- [ ] `selected_styles.json` 包含 5 个风格，每个有 id/name/writer_id
- [ ] 5 种风格视觉上差异明显（并排展示同一个字的 5 种风格确认）
- [ ] 每种风格的书写者至少覆盖 475/500 个常用字

### 任务 2.7：PyTorch Dataset 实现

**做什么**：实现 `src/handwrite/data/dataset.py`，标准的 PyTorch Dataset 类。

**怎么做**：
```python
class HandwriteDataset(torch.utils.data.Dataset):
    """
    初始化参数：
        metadata_path: str - metadata.json 路径
        selected_styles_path: str - selected_styles.json 路径
        transform: Optional[callable] - 图片变换
    
    __getitem__ 返回：
        {
            "standard": Tensor(1, 256, 256),   # 标准字体图，归一化到 [-1, 1]
            "handwrite": Tensor(1, 256, 256),   # 手写图，归一化到 [-1, 1]
            "style_id": int,                     # 风格ID (0-4)
            "char": str,                         # 字符
        }
    
    归一化：灰度值从 [0, 255] 映射到 [-1, 1]
    即 pixel = (pixel / 127.5) - 1.0
    """
```

**重要**：
- 只加载 selected_styles.json 中 5 个书写者的数据
- 数据增强（可选，但推荐）：对手写图做轻微随机变换
  - 随机旋转 ±2°
  - 随机平移 ±5 像素
  - 随机缩放 0.95-1.05
- 标准字体图不做数据增强

**验收标准**：
- [ ] `len(dataset)` 返回合理数字（约 2000-2500，即 5 风格 × 450-500 字）
- [ ] `dataset[0]` 返回正确格式的字典
- [ ] standard 和 handwrite 的 tensor shape 都是 (1, 256, 256)
- [ ] 值范围在 [-1, 1]
- [ ] DataLoader(dataset, batch_size=8, shuffle=True) 能正常迭代

---

## 第三阶段：模型搭建（2天）

### 任务 3.1：寻找并适配 zi2zi-pytorch

**做什么**：在 GitHub 上找到最好的 zi2zi PyTorch 复现，fork 并适配。

**怎么做**：
1. GitHub 搜索 "zi2zi pytorch"，重点看：
   - [EuphoriaYan/zi2zi-pytorch](https://github.com/EuphoriaYan/zi2zi-pytorch)
   - [xuan-li/zi2zi-pytorch](https://github.com/xuan-li/zi2zi-pytorch)
   - 或其他 star 较多的复现
2. 克隆到本地，用少量数据（10个字 × 1个风格）跑一次训练，确认能跑通
3. 确认以下关键点：
   - 输入分辨率支持 256×256
   - 有 category embedding（风格条件）
   - 有 Label Shuffling 实现
   - 在 RTX 4060 8GB 上不 OOM（batch_size=8）

**如果找不到好的 zi2zi-pytorch**，则自己实现（见任务 3.2 和 3.3）。

**验收标准**：
- [ ] 找到了可用的 zi2zi-pytorch 代码
- [ ] 能在 RTX 4060 上用少量数据跑通一次完整训练（10个字，1个epoch）
- [ ] 训练过程中 loss 在下降
- [ ] 生成的图片有模糊的字形（不需要好看，能看出是字就行）

### 任务 3.2：实现 Generator（如果需要自己实现）

**做什么**：实现 `src/handwrite/engine/generator.py`，基于 zi2zi 的 U-Net Generator。

**网络架构详细规格**：

```
Encoder (下采样):
  e1: Conv2d(1, 64, 4, stride=2, padding=1) → LeakyReLU(0.2)          # 256→128
  e2: Conv2d(64, 128, 4, stride=2, padding=1) → BN → LeakyReLU(0.2)   # 128→64
  e3: Conv2d(128, 256, 4, stride=2, padding=1) → BN → LeakyReLU(0.2)  # 64→32
  e4: Conv2d(256, 512, 4, stride=2, padding=1) → BN → LeakyReLU(0.2)  # 32→16
  e5: Conv2d(512, 512, 4, stride=2, padding=1) → BN → LeakyReLU(0.2)  # 16→8
  e6: Conv2d(512, 512, 4, stride=2, padding=1) → BN → LeakyReLU(0.2)  # 8→4
  e7: Conv2d(512, 512, 4, stride=2, padding=1) → BN → LeakyReLU(0.2)  # 4→2
  e8: Conv2d(512, 512, 4, stride=2, padding=1) → LeakyReLU(0.2)       # 2→1

Bottleneck:
  风格 embedding (num_styles × embed_dim) 在这里与 e8 的输出 concat 或相加

Decoder (上采样 + skip connections):
  d1: ConvTranspose2d(512+embed_dim, 512, 4, stride=2, padding=1) → BN → Dropout(0.5) → ReLU  # 1→2
      + skip connect with e7
  d2: ConvTranspose2d(1024, 512, 4, stride=2, padding=1) → BN → Dropout(0.5) → ReLU  # 2→4
      + skip connect with e6
  d3: ConvTranspose2d(1024, 512, 4, stride=2, padding=1) → BN → Dropout(0.5) → ReLU  # 4→8
      + skip connect with e5
  d4: ConvTranspose2d(1024, 512, 4, stride=2, padding=1) → BN → ReLU  # 8→16
      + skip connect with e4
  d5: ConvTranspose2d(1024, 256, 4, stride=2, padding=1) → BN → ReLU  # 16→32
      + skip connect with e3
  d6: ConvTranspose2d(512, 128, 4, stride=2, padding=1) → BN → ReLU   # 32→64
      + skip connect with e2
  d7: ConvTranspose2d(256, 64, 4, stride=2, padding=1) → BN → ReLU    # 64→128
      + skip connect with e1
  d8: ConvTranspose2d(128, 1, 4, stride=2, padding=1) → Tanh           # 128→256

输入: (batch, 1, 256, 256) 标准字体图
输出: (batch, 1, 256, 256) 手写风格图，值域 [-1, 1]
```

**风格 Embedding**：
```python
self.style_embedding = nn.Embedding(num_styles=5, embedding_dim=128)
# 在 bottleneck 处，将 embedding 扩展为 (batch, 128, 1, 1) 并 concat 到 e8 输出
```

**验收标准**：
- [ ] `Generator(num_styles=5, embed_dim=128)` 可实例化
- [ ] 输入 `(8, 1, 256, 256)` 的 tensor + style_id，输出 shape `(8, 1, 256, 256)`
- [ ] 输出值范围在 [-1, 1]（Tanh）
- [ ] 参数量约 50M（`sum(p.numel() for p in model.parameters())` 验证）
- [ ] 显存占用 < 4GB（单独前向传播，batch=8）

### 任务 3.3：实现 Discriminator（如果需要自己实现）

**做什么**：实现 `src/handwrite/engine/discriminator.py`，PatchGAN 判别器。

**网络架构**：
```
输入: (batch, 1, 256, 256) — 手写图（真实或生成的）

c1: Conv2d(1, 64, 4, stride=2, padding=1) → LeakyReLU(0.2)         # 256→128
c2: Conv2d(64, 128, 4, stride=2, padding=1) → BN → LeakyReLU(0.2)  # 128→64
c3: Conv2d(128, 256, 4, stride=2, padding=1) → BN → LeakyReLU(0.2) # 64→32
c4: Conv2d(256, 512, 4, stride=1, padding=1) → BN → LeakyReLU(0.2) # 32→31

输出两个头：
  real_fake: Conv2d(512, 1, 4, stride=1, padding=1)     # 31→30, 真假判定
  category:  Conv2d(512, num_styles, 4, stride=1, padding=1)  # 31→30, 风格分类
```

**验收标准**：
- [ ] 输入 (8, 1, 256, 256)，输出 real_fake shape (8, 1, 30, 30)，category shape (8, 5, 30, 30)
- [ ] 参数量约 5M

---

## 第四阶段：训练（3天）

### 任务 4.1：实现训练脚本

**做什么**：实现 `scripts/train.py` 和 `src/handwrite/engine/train.py`。

**训练逻辑（每个 batch）**：

```
Step 1: 训练 Discriminator
  1. 取一个 batch: (standard_img, real_handwrite_img, style_id)
  2. G 生成假图: fake_img = G(standard_img, style_id)
  3. D 判断真图: real_score, real_category = D(real_handwrite_img)
  4. D 判断假图: fake_score, fake_category = D(fake_img.detach())
  5. D 损失 = BCE(real_score, 1) + BCE(fake_score, 0) + CrossEntropy(real_category, style_id)
  6. 更新 D

Step 2: 训练 Generator
  1. G 生成假图: fake_img = G(standard_img, style_id)
  2. D 判断假图: fake_score, fake_category = D(fake_img)
  3. G 损失 = BCE(fake_score, 1) + 100 * L1(fake_img, real_handwrite_img) + CrossEntropy(fake_category, style_id)
  4. 更新 G
```

**Label Shuffling 实现**（Stage 2 启用）：
```
在同一个 batch 中：
  1. 正常生成: fake_correct = G(standard_img, correct_style_id)
  2. 打乱风格: shuffled_style_id = random_permutation(style_id)
     fake_shuffled = G(standard_img, shuffled_style_id)
  3. D 需要额外判断 fake_shuffled 并给出正确的风格分类
```

**训练脚本参数**：
```bash
python scripts/train.py \
  --data_dir data/processed \
  --styles_file data/processed/selected_styles.json \
  --output_dir weights \
  --batch_size 8 \
  --epochs 30 \
  --lr 0.0002 \
  --l1_weight 100 \
  --label_shuffle_start 10 \
  --save_interval 2 \
  --sample_interval 500
```

**训练过程中必须保存的信息**：
- 每 2 个 epoch 保存 checkpoint：`weights/checkpoint_epoch_{N}.pt`
- 每 500 个 batch 生成一张样本对比图：`weights/samples/epoch{N}_batch{M}.png`
  - 内容：5 种风格 × 5 个字 = 25 张生成图，与真实图并排对比
- 训练日志：`weights/train_log.csv`，记录每个 epoch 的 G_loss, D_loss, L1_loss

**验收标准**：
- [ ] 训练脚本能启动并正常运行，无 OOM
- [ ] 训练 1 个 epoch 后 loss 在下降
- [ ] `weights/samples/` 中有对比图，能看到生成的字在逐渐变清晰
- [ ] 训练完成后（20-30 epochs），生成的字人眼可辨认（≥80% 的字能认出来）
- [ ] 不同风格生成的同一个字看起来有差异

### 任务 4.2：训练监控与问题排查

**做什么**：监控训练过程，处理常见问题。

**需要监控的指标**：
```
每个 epoch 结束时打印：
  Epoch [5/30] G_loss: 12.34  D_loss: 0.56  L1_loss: 0.089
  D_real_acc: 0.92  D_fake_acc: 0.85  Style_acc: 0.78
```

**常见问题及处理方案**：

| 症状 | 诊断 | 处理 |
|------|------|------|
| D_loss 快速降到 0，G_loss 不下降 | D 太强 | G 训练 2 步，D 训练 1 步；或降低 D 学习率到 1e-4 |
| G_loss 突然飙升 | 训练不稳定 | 回退到上一个 checkpoint，降低 lr 到 1e-4 |
| 所有输出看起来一样 | 模式崩塌 | 增大 L1 权重到 200；确认 style embedding 在正确使用 |
| 输出全是灰色/噪声 | 没有收敛 | 检查数据归一化是否正确（必须是 [-1,1]）；检查 Tanh 是否在最后一层 |
| CUDA OOM | 显存不足 | batch_size 降到 4；启用 AMP（`torch.cuda.amp`） |
| 生成的字模糊 | L1 过强 | 降低 L1 权重到 50；增加训练轮数 |
| 笔画断裂 | 细节丢失 | 加入 Sobel 边缘损失：`L_edge = L1(sobel(fake), sobel(real))`，权重 10 |

**验收标准**：
- [ ] 训练日志 CSV 存在，能画出 loss 曲线
- [ ] 最终模型的生成质量：随机取 100 个字，≥ 80% 人眼可辨认
- [ ] 5 种风格之间有可见差异

---

## 第五阶段：评估（1天）

### 任务 5.1：实现评估脚本

**做什么**：实现 `scripts/evaluate.py`，对训练好的模型做定量+定性评估。

**定量评估**：
```python
def evaluate_model(generator, dataset, device):
    """
    返回：
    {
        "l1_loss": float,          # 生成图与真实图的平均 L1 距离
        "fid_score": float,        # FID 分数（需要 pytorch-fid 库）
        "style_accuracy": float,   # 风格分类准确率
    }
    """
```

**定性评估**：
- 生成一张大图：5 种风格 × 20 个字（"永和九年岁在癸丑暮春之初会于山阴之兰亭也"），保存为 `evaluation/style_comparison.png`
- 生成复杂字测试：对 "藏赢鑫壤餐" 等复杂字生成所有风格，保存为 `evaluation/complex_chars.png`
- 生成一段完整文字的整页效果（接入 Composer 后）

**验收标准**：
- [ ] `evaluation/style_comparison.png` 存在，5 种风格差异可见
- [ ] `evaluation/complex_chars.png` 存在，复杂字 ≥ 60% 可辨认
- [ ] FID < 80（可接受），< 50（良好）
- [ ] 人眼可辨认率 ≥ 80%

---

## 第六阶段：Composer 排版（2天）

### 任务 6.1：实现排版引擎

**做什么**：实现 `src/handwrite/composer.py`，将单字拼成自然的手写页面。

**核心函数接口**：
```python
def compose_page(
    chars: list[Image.Image],   # 单字图片列表
    text: str,                   # 原始文字（用于确定换行、标点位置）
    page_size: tuple = (2480, 3508),  # A4 @ 300DPI
    font_size: int = 80,        # 每个字在页面上的大小（像素）
    margins: tuple = (200, 200, 200, 200),  # 上右下左边距
    layout: str = "自然",        # "工整" / "自然" / "潦草"
    paper: str = "白纸",         # "白纸" / "横线纸" / "方格纸" / "米字格"
) -> Image.Image:
    """返回完整的 A4 页面图片"""
```

**排版算法详细步骤**：
1. 根据 page_size、margins、font_size 计算每行字数和总行数
2. 逐字放置：
   - 将 256×256 的单字图缩放到 font_size × font_size
   - 根据 layout 预设确定随机扰动范围
   - 对每个字应用：随机缩放 → 随机旋转 → 随机偏移
   - 计算放置位置（考虑字间距 + 随机扰动）
3. 处理标点：
   - 句号、逗号等不放在行首（如果碰到行首则挤到上一行末尾）
   - 标点大小为正常字号的 50-70%
4. 处理换行：
   - `\n` 强制换行
   - 到达行尾自动换行
5. 墨水浓淡：
   - 用正弦函数生成浓度序列，周期 15 个字
   - 对每个字的 alpha 通道乘以浓度值
6. 叠加纸张背景

**纸张背景生成**：
```python
def create_paper(
    size: tuple,
    paper_type: str,  # "白纸" / "横线纸" / "方格纸" / "米字格"
    line_spacing: int = 80,  # 格线间距，与 font_size 匹配
) -> Image.Image:
```
- 横线纸：每隔 line_spacing 画一条浅灰色水平线（颜色 #CCCCCC，线宽 1px）
- 方格纸：水平 + 垂直线
- 米字格：方格 + 对角线
- 加一层高斯噪声（sigma=3）模拟纸张纤维

**验收标准**：
- [ ] 输入 "今天天气真不错" + 7 张单字图 → 输出一张 A4 图片
- [ ] 三种 layout 预设的效果差异明显
- [ ] 标点不出现在行首
- [ ] 多行文字换行正确，段首有缩进
- [ ] 四种纸张模板都能正确渲染
- [ ] 整页看起来像真人手写，不是机械等间距排列

---

## 第七阶段：导出与 API 封装（1天）

### 任务 7.1：实现 Exporter

**做什么**：实现 `src/handwrite/exporter.py`。

```python
def export_png(page: Image.Image, output_path: str, dpi: int = 300):
    """保存为 PNG，嵌入 DPI 信息"""
    page.save(output_path, dpi=(dpi, dpi))
```

**验收标准**：
- [ ] 输出的 PNG 文件能正常打开，DPI 信息正确

### 任务 7.2：封装 Python Library API

**做什么**：实现 `src/handwrite/__init__.py`，提供简洁的公开 API。

**完整 API**：
```python
# __init__.py
from handwrite.engine.model import StyleEngine
from handwrite.composer import compose_page
from handwrite.exporter import export_png
from handwrite.styles import BUILTIN_STYLES

_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        _engine = StyleEngine()  # 自动加载预训练权重
    return _engine

def char(text: str, style: str = "工整楷书") -> Image.Image:
    """生成单个字符的手写图片（256×256）"""
    engine = _get_engine()
    style_id = BUILTIN_STYLES[style]
    return engine.generate_char(text, style_id)

def generate(
    text: str,
    style: str = "工整楷书",
    paper: str = "白纸",
    layout: str = "自然",
    font_size: int = 80,
) -> Image.Image:
    """生成一页手写文档图片"""
    engine = _get_engine()
    style_id = BUILTIN_STYLES[style]
    # 逐字生成
    char_images = []
    for c in text:
        if c in ('\n', ' '):
            char_images.append(c)  # 特殊字符直接传递给 composer
        else:
            img = engine.generate_char(c, style_id)
            char_images.append(img)
    # 排版
    page = compose_page(char_images, text, font_size=font_size, layout=layout, paper=paper)
    return page

def generate_pages(text: str, **kwargs) -> list[Image.Image]:
    """生成多页手写文档"""
    # 按页面容量分割文字，每页调用 generate()
    ...

def list_styles() -> list[str]:
    """返回可用风格名称列表"""
    return list(BUILTIN_STYLES.keys())
```

**未覆盖字符的 fallback**：
```python
# 在 engine.generate_char() 中：
def generate_char(self, char: str, style_id: int) -> Image.Image:
    if char in self.supported_chars:
        # 用 GAN 生成
        return self._gan_generate(char, style_id)
    else:
        # fallback: 用开源手写字体渲染
        return self._fallback_render(char)
```

**验收标准**：
- [ ] `import handwrite; handwrite.list_styles()` 返回 5 种风格
- [ ] `handwrite.char("你")` 返回 PIL.Image，看起来像手写
- [ ] `handwrite.generate("今天天气真不错")` 返回一张完整的 A4 页面
- [ ] 输入包含未覆盖字符时不崩溃，fallback 效果可接受

---

## 第八阶段：Gradio Web Demo（0.5天）

### 任务 8.1：实现 Web Demo

**做什么**：实现 `demo/app.py`。

```python
import gradio as gr
import handwrite

def generate_handwriting(text, style, paper, layout, font_size):
    if not text.strip():
        return None
    page = handwrite.generate(
        text=text,
        style=style,
        paper=paper,
        layout=layout,
        font_size=int(font_size),
    )
    return page

demo = gr.Interface(
    fn=generate_handwriting,
    inputs=[
        gr.Textbox(label="输入文字", lines=5, placeholder="在这里输入要转换的文字..."),
        gr.Dropdown(choices=handwrite.list_styles(), value="工整楷书", label="手写风格"),
        gr.Dropdown(choices=["白纸", "横线纸", "方格纸", "米字格"], value="白纸", label="纸张类型"),
        gr.Dropdown(choices=["工整", "自然", "潦草"], value="自然", label="排版风格"),
        gr.Slider(minimum=40, maximum=120, value=80, step=10, label="字号"),
    ],
    outputs=gr.Image(label="生成结果", type="pil"),
    title="HandWrite — AI 中文手写体生成器",
    description="输入中文文字，选择手写风格，生成逼真的手写文档图片。基于条件 GAN 深度学习模型。",
    examples=[
        ["今天天气真不错，适合出去走走。", "工整楷书", "横线纸", "自然", 80],
        ["静夜思\n床前明月光，疑是地上霜。\n举头望明月，低头思故乡。", "行书流畅", "白纸", "自然", 80],
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

**验收标准**：
- [ ] `python demo/app.py` 启动后浏览器访问 localhost:7860 能看到界面
- [ ] 输入文字，选风格，点生成，能看到手写图片
- [ ] examples 中的两个示例都能正常生成
- [ ] 生成一段 100 字的文字耗时 < 30 秒

---

## 第九阶段：最终验收（0.5天）

### 端到端测试清单

| 测试 | 操作 | 期望结果 |
|------|------|----------|
| 安装测试 | `pip install -e .` | 成功，无报错 |
| 导入测试 | `import handwrite` | 成功 |
| 风格列表 | `handwrite.list_styles()` | 返回 5 种风格名 |
| 单字生成 | `handwrite.char("你")` | 返回 256×256 手写风格 PIL.Image |
| 多风格对比 | 对同一个字生成 5 种风格 | 5 张图有明显差异 |
| 整段生成 | `handwrite.generate("测试文字", style="工整楷书")` | 返回 A4 页面图片 |
| 长文多页 | `handwrite.generate_pages(500字长文)` | 返回多张页面 |
| 纸张模板 | 分别用 4 种纸张类型生成 | 背景正确（横线/方格/米字格） |
| 排版预设 | 分别用 3 种 layout 生成 | "工整"最规矩，"潦草"最随意 |
| 未覆盖字 | 输入包含生僻字 | 不崩溃，fallback 生效 |
| Web Demo | 启动 Gradio，操作一遍 | 界面正常，生成正常 |
| 性能测试 | 生成 100 字的整页文档 | < 30 秒（GPU），< 120 秒（CPU） |

### 质量标准

- 生成的手写字，随机抽 100 个，**人眼可辨认率 ≥ 80%**
- 5 种风格之间有**可见差异**
- 整页文档看起来**不像机器排版**（字间距、大小有自然变化）
- 无明显的**字形崩坏**（笔画粘连到无法辨认、部首错位）

---

## 总时间线

| 阶段 | 内容 | 时间 |
|------|------|------|
| 第一阶段 | 项目初始化 | 0.5 天 |
| 第二阶段 | 数据获取与预处理 | 3 天 |
| 第三阶段 | 模型搭建 | 2 天 |
| 第四阶段 | 训练 | 3 天（含等待训练时间） |
| 第五阶段 | 评估 | 1 天 |
| 第六阶段 | Composer 排版 | 2 天 |
| 第七阶段 | 导出与 API | 1 天 |
| 第八阶段 | Gradio Web Demo | 0.5 天 |
| 第九阶段 | 最终验收 | 0.5 天 |
| **合计** | | **~14 天（2 周）** |

## 关键依赖和安装

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 安装核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pillow numpy opencv-python gradio tqdm matplotlib

# 安装项目
pip install -e ".[dev]"

# 可选：FID 评估
pip install pytorch-fid
```

## 重要注意事项

1. **所有 Python 文件的 `open()` 调用必须显式指定 `encoding='utf-8'`**（Windows 默认编码会导致中文乱码）
2. **文件路径一律用 `pathlib.Path` 或 `/` 分隔符**，不要用 `\`
3. **模型权重文件不要提交到 git**，放在 weights/ 目录并 gitignore
4. **图表输出用 `bbox_inches='tight'`, DPI ≥ 300**
5. **CASIA-HWDB 的标签是 GBK 编码**，解析时注意 `decode('gbk')` 而非 `decode('utf-8')`
