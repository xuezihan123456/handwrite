# 创新模块 #04：自然语言控笔（NL Style Picker）

## 想解决的问题

让用户用一句话——中文或英文——就能描述想要的笔迹风格，例如「紧张焦虑的高三学生赶时间的字」或 `calm, neat elementary teacher`，系统将其翻译为可被 `composer.compose_page` 直接消费的具体参数。

## 设计思路

模块完全基于规则，不引入任何 LLM 调用：

1. **双语关键词表** (`keywords.py`)：超过 80 条中英文条目。
2. **强度修饰** (`INTENSITY_MODIFIERS`)：`极`/`非常`/`稍微`/`very`/`slightly` 等修饰词。
3. **情绪映射** (`EMOTION_TO_PARAMS`)。
4. **聚合器** (`parser._assemble`)。

## 输出结构

`StyleVector` 字段包括 `rotation_jitter`、`scale_jitter`、`ink_density`、`baseline_jitter`、`char_spacing`、`line_spacing`、`style_name`、`suggested_layout`、`mood_tags`。

## 确定性保证

同一输入永远输出同一向量；所有数值在构造时被 clamp 到合法范围。
