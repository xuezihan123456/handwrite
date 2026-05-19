# Innovation 05 — 错题本智能复刻 (Error Notebook)

## 一句话定位
扫描错题照片 → diff 出错点 → 自动生成"原题 + 错答（红笔划掉）+ 正确解法 + 反思"的多页错题本 PDF。

## 流程
1. **录入** (`ErrorNotebookBuilder.add_entry`)。
2. **Diff** (`error_notebook.diff.diff_answers`)。
3. **渲染** (`ErrorNotebookBuilder.render`)。
4. **导出** (`.export_pdf`)。

## 公开 API
```python
from handwrite.error_notebook import (
    ErrorNotebookBuilder,
    ErrorEntry,
    build_error_notebook,
    diff_answers,
    DiffSegment,
)
```

## 用途
- 学生：扫描后导出个人错题集 PDF。
- 老师 / 教辅：批量生产带原始错答 + 标注的练习册。
- AI 数据集：作为"含手写错答的训练 / 评测样本"来源。
