# 09 · 几何 + 公式 + 手写考卷

`handwrite.geometry_sheet` 把几何作图、LaTeX 公式与手写题面在 A4 试卷上整合为一种新的输出格式。

## 能力一览

- `figures.py` 暴露纯函数：`draw_circle`、`draw_triangle`、`draw_rectangle`、`draw_line`、`draw_axes`、`draw_angle_arc`、`draw_arrow`、`draw_labeled_point`。
- `problem.py` 提供数据类：`Figure` 与 `Problem`。
- `builder.py` 中的 `GeometrySheetBuilder` 负责把多道 `Problem` 排版到 A4 纸面。

## 公开 API

```python
from handwrite.geometry_sheet import (
    GeometrySheetBuilder,
    Figure,
    Problem,
    build_exam_sheet,
    draw_circle,
    draw_triangle,
)

problem = Problem(
    question_text="求圆的面积",
    figures=[Figure(image=draw_circle((320, 240), (160, 120), 80))],
    solution_steps=["S = πr²", "代入 r=10 得 S=100π"],
    answer="100π",
    formula_latex=r"S = \pi r^{2}",
)
info = build_exam_sheet([problem], "exam.pdf", style="工整楷书")
```
