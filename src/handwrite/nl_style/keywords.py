"""Bilingual (Chinese + English) keyword tables used by ``NaturalLanguageStyler``.

Each entry maps a normalised keyword to a *partial* set of style parameters
that the parser adds together. Magnitudes are intentionally modest so that
combining several keywords yields a believable composite vector without
saturation. Final clamping happens in ``StyleVector.__post_init__``.

The dictionaries below contain over 80 keyword aliases covering style,
emotion, age/persona, and writing context. All entries use snake-case
internal parameter names matching ``StyleVector`` fields.
"""

from __future__ import annotations

from .style_vector import CURSIVE_LAYOUT, NATURAL_LAYOUT, NEAT_LAYOUT


# ---------------------------------------------------------------------------
# Intensity modifiers
# ---------------------------------------------------------------------------
# These multipliers scale *all* numeric contributions of the immediately
# following keyword. Keys are lowercased (English) or kept as-is (Chinese).

INTENSITY_MODIFIERS: dict[str, float] = {
    # English
    "very": 1.6,
    "extremely": 1.9,
    "super": 1.7,
    "really": 1.4,
    "quite": 1.2,
    "slightly": 0.5,
    "a bit": 0.5,
    "a little": 0.5,
    "somewhat": 0.7,
    "kind of": 0.7,
    "kinda": 0.7,
    "mildly": 0.6,
    # Chinese (single-token and compound)
    "\u6781":           1.9,   # 极
    "\u6781\u5176":     1.8,   # 极其
    "\u975e\u5e38":     1.6,   # 非常
    "\u5341\u5206":     1.5,   # 十分
    "\u7279\u522b":     1.5,   # 特别
    "\u8d85":           1.7,   # 超
    "\u8d85\u7ea7":     1.8,   # 超级
    "\u5f88":           1.3,   # 很
    "\u633a":           1.2,   # 挺
    "\u76f8\u5f53":     1.3,   # 相当
    "\u7a0d\u5fae":     0.5,   # 稍微
    "\u7a0d\u7a0d":     0.6,   # 稍稍
    "\u6709\u70b9":     0.7,   # 有点
    "\u6709\u4e9b":     0.7,   # 有些
    "\u4e00\u70b9":     0.6,   # 一点
    "\u4e00\u4e9b":     0.7,   # 一些
    "\u4e00\u70b9\u70b9": 0.5, # 一点点
    "\u4e0d\u592a":     0.4,   # 不太
}


# ---------------------------------------------------------------------------
# Emotion mapping
# ---------------------------------------------------------------------------
# Emotions push primarily on jitter/baseline (anxious -> messy/shaky) or
# pull toward calm/neat. They contribute mood tags too.

EMOTION_TO_PARAMS: dict[str, dict[str, float | str]] = {
    # English emotions
    "anxious": {
        "rotation_jitter": 4.0,
        "baseline_jitter": 0.35,
        "scale_jitter": 0.2,
        "char_spacing": -0.1,
        "_mood": "anxious",
        "_layout": CURSIVE_LAYOUT,
    },
    "nervous": {
        "rotation_jitter": 3.5,
        "baseline_jitter": 0.3,
        "scale_jitter": 0.18,
        "_mood": "nervous",
        "_layout": NATURAL_LAYOUT,
    },
    "stressed": {
        "rotation_jitter": 4.0,
        "baseline_jitter": 0.3,
        "scale_jitter": 0.2,
        "ink_density": -0.05,
        "_mood": "stressed",
        "_layout": CURSIVE_LAYOUT,
    },
    "panic": {
        "rotation_jitter": 6.0,
        "baseline_jitter": 0.5,
        "scale_jitter": 0.3,
        "_mood": "panic",
        "_layout": CURSIVE_LAYOUT,
    },
    "calm": {
        "rotation_jitter": -2.0,
        "baseline_jitter": -0.15,
        "scale_jitter": -0.1,
        "_mood": "calm",
        "_layout": NEAT_LAYOUT,
    },
    "relaxed": {
        "rotation_jitter": -1.5,
        "baseline_jitter": -0.1,
        "_mood": "relaxed",
        "_layout": NEAT_LAYOUT,
    },
    "happy": {
        "rotation_jitter": 1.0,
        "scale_jitter": 0.1,
        "char_spacing": 0.1,
        "_mood": "happy",
        "_layout": NATURAL_LAYOUT,
    },
    "tired": {
        "ink_density": -0.2,
        "baseline_jitter": 0.2,
        "rotation_jitter": 1.5,
        "_mood": "tired",
        "_layout": CURSIVE_LAYOUT,
    },
    "angry": {
        "rotation_jitter": 4.0,
        "ink_density": 0.2,
        "scale_jitter": 0.2,
        "_mood": "angry",
        "_layout": CURSIVE_LAYOUT,
    },
    "sad": {
        "ink_density": -0.15,
        "baseline_jitter": 0.15,
        "_mood": "sad",
        "_layout": NATURAL_LAYOUT,
    },
    "focused": {
        "rotation_jitter": -2.0,
        "baseline_jitter": -0.2,
        "_mood": "focused",
        "_layout": NEAT_LAYOUT,
    },
    "confident": {
        "ink_density": 0.15,
        "rotation_jitter": -1.0,
        "_mood": "confident",
        "_layout": NEAT_LAYOUT,
    },
    # Chinese emotions
    "\u7126\u8651": {                    # 焦虑
        "rotation_jitter": 4.0,
        "baseline_jitter": 0.35,
        "scale_jitter": 0.2,
        "_mood": "anxious",
        "_layout": CURSIVE_LAYOUT,
    },
    "\u7d27\u5f20": {                    # 紧张
        "rotation_jitter": 3.5,
        "baseline_jitter": 0.3,
        "scale_jitter": 0.18,
        "_mood": "nervous",
        "_layout": NATURAL_LAYOUT,
    },
    "\u538b\u529b": {                    # 压力
        "rotation_jitter": 3.0,
        "baseline_jitter": 0.25,
        "_mood": "stressed",
        "_layout": NATURAL_LAYOUT,
    },
    "\u614c\u5f20": {                    # 慌张
        "rotation_jitter": 5.0,
        "baseline_jitter": 0.4,
        "_mood": "panic",
        "_layout": CURSIVE_LAYOUT,
    },
    "\u5e73\u9759": {                    # 平静
        "rotation_jitter": -2.0,
        "baseline_jitter": -0.15,
        "_mood": "calm",
        "_layout": NEAT_LAYOUT,
    },
    "\u51b7\u9759": {                    # 冷静
        "rotation_jitter": -2.0,
        "baseline_jitter": -0.2,
        "_mood": "calm",
        "_layout": NEAT_LAYOUT,
    },
    "\u653e\u677e": {                    # 放松
        "rotation_jitter": -1.5,
        "baseline_jitter": -0.1,
        "_mood": "relaxed",
        "_layout": NEAT_LAYOUT,
    },
    "\u5f00\u5fc3": {                    # 开心
        "rotation_jitter": 1.0,
        "scale_jitter": 0.1,
        "_mood": "happy",
        "_layout": NATURAL_LAYOUT,
    },
    "\u6109\u5feb": {                    # 愉快
        "rotation_jitter": 0.8,
        "scale_jitter": 0.08,
        "_mood": "happy",
        "_layout": NATURAL_LAYOUT,
    },
    "\u75b2\u60eb": {                    # 疲惫
        "ink_density": -0.2,
        "baseline_jitter": 0.2,
        "rotation_jitter": 1.5,
        "_mood": "tired",
        "_layout": CURSIVE_LAYOUT,
    },
    "\u56f0": {                          # 困
        "ink_density": -0.15,
        "baseline_jitter": 0.15,
        "_mood": "tired",
        "_layout": NATURAL_LAYOUT,
    },
    "\u6124\u6012": {                    # 愤怒
        "rotation_jitter": 4.0,
        "ink_density": 0.2,
        "_mood": "angry",
        "_layout": CURSIVE_LAYOUT,
    },
    "\u6c14\u6124": {                    # 气愤
        "rotation_jitter": 3.5,
        "ink_density": 0.15,
        "_mood": "angry",
        "_layout": CURSIVE_LAYOUT,
    },
    "\u4f24\u5fc3": {                    # 伤心
        "ink_density": -0.15,
        "baseline_jitter": 0.15,
        "_mood": "sad",
        "_layout": NATURAL_LAYOUT,
    },
    "\u4e13\u6ce8": {                    # 专注
        "rotation_jitter": -2.0,
        "baseline_jitter": -0.2,
        "_mood": "focused",
        "_layout": NEAT_LAYOUT,
    },
    "\u81ea\u4fe1": {                    # 自信
        "ink_density": 0.15,
        "rotation_jitter": -1.0,
        "_mood": "confident",
        "_layout": NEAT_LAYOUT,
    },
}


# ---------------------------------------------------------------------------
# Style + persona keywords
# ---------------------------------------------------------------------------
# Maps both Chinese and English keywords to partial parameter adjustments.
# Total entries (style + emotion combined) deliberately exceed 80.

KEYWORD_TO_STYLE: dict[str, dict[str, float | str]] = {
    # --- Neat / tidy family ---
    "\u5de5\u6574": {                    # 工整
        "rotation_jitter": -2.0,
        "baseline_jitter": -0.2,
        "scale_jitter": -0.1,
        "_layout": NEAT_LAYOUT,
        "_style": "neat",
    },
    "\u6574\u9f50": {                    # 整齐
        "rotation_jitter": -1.5,
        "baseline_jitter": -0.15,
        "_layout": NEAT_LAYOUT,
        "_style": "neat",
    },
    "\u6e05\u79c0": {                    # 清秀
        "rotation_jitter": -1.0,
        "ink_density": -0.05,
        "_layout": NEAT_LAYOUT,
        "_style": "elegant",
    },
    "\u7aef\u6b63": {                    # 端正
        "rotation_jitter": -2.0,
        "baseline_jitter": -0.2,
        "_layout": NEAT_LAYOUT,
        "_style": "neat",
    },
    "neat": {
        "rotation_jitter": -2.0,
        "baseline_jitter": -0.2,
        "scale_jitter": -0.1,
        "_layout": NEAT_LAYOUT,
        "_style": "neat",
    },
    "tidy": {
        "rotation_jitter": -1.5,
        "baseline_jitter": -0.15,
        "_layout": NEAT_LAYOUT,
        "_style": "neat",
    },
    "clean": {
        "rotation_jitter": -1.5,
        "baseline_jitter": -0.1,
        "_layout": NEAT_LAYOUT,
        "_style": "neat",
    },
    "careful": {
        "rotation_jitter": -1.5,
        "baseline_jitter": -0.15,
        "_layout": NEAT_LAYOUT,
        "_style": "careful",
    },
    "precise": {
        "rotation_jitter": -2.0,
        "baseline_jitter": -0.2,
        "_layout": NEAT_LAYOUT,
        "_style": "precise",
    },

    # --- Messy / sloppy / cursive ---
    "\u6f5c\u8349": {                    # 潜草
        "rotation_jitter": 4.0,
        "baseline_jitter": 0.3,
        "scale_jitter": 0.2,
        "_layout": CURSIVE_LAYOUT,
        "_layout_weight": 2.5,
        "_style": "sloppy",
    },
    "\u8349\u4e66": {                    # 草书
        "rotation_jitter": 3.5,
        "baseline_jitter": 0.25,
        "char_spacing": -0.15,
        "_layout": CURSIVE_LAYOUT,
        "_layout_weight": 2.5,
        "_style": "cursive",
    },
    "\u884c\u4e66": {                    # 行书
        "rotation_jitter": 2.0,
        "baseline_jitter": 0.15,
        "_layout": NATURAL_LAYOUT,
        "_style": "running",
    },
    "\u968f\u610f": {                    # 随意
        "rotation_jitter": 2.0,
        "baseline_jitter": 0.2,
        "scale_jitter": 0.15,
        "_layout": NATURAL_LAYOUT,
        "_style": "casual",
    },
    "\u5306\u5fd9": {                    # 匆忙
        "rotation_jitter": 3.5,
        "baseline_jitter": 0.25,
        "scale_jitter": 0.2,
        "_layout": CURSIVE_LAYOUT,
        "_layout_weight": 2.0,
        "_style": "rushed",
    },
    "\u8d76\u65f6\u95f4": {              # 赶时间
        "rotation_jitter": 4.0,
        "baseline_jitter": 0.3,
        "scale_jitter": 0.25,
        "char_spacing": -0.1,
        "_layout": CURSIVE_LAYOUT,
        "_layout_weight": 2.5,
        "_style": "rushed",
    },
    "\u5feb\u901f": {                    # 快速
        "rotation_jitter": 3.0,
        "scale_jitter": 0.18,
        "char_spacing": -0.1,
        "_layout": CURSIVE_LAYOUT,
        "_layout_weight": 2.0,
        "_style": "fast",
    },
    "\u5feb": {                          # 快
        "rotation_jitter": 2.5,
        "scale_jitter": 0.15,
        "_layout": CURSIVE_LAYOUT,
        "_style": "fast",
    },
    "messy": {
        "rotation_jitter": 4.0,
        "baseline_jitter": 0.3,
        "scale_jitter": 0.2,
        "_layout": CURSIVE_LAYOUT,
        "_layout_weight": 2.5,
        "_style": "messy",
    },
    "sloppy": {
        "rotation_jitter": 4.5,
        "baseline_jitter": 0.35,
        "scale_jitter": 0.25,
        "_layout": CURSIVE_LAYOUT,
        "_layout_weight": 2.5,
        "_style": "sloppy",
    },
    "cursive": {
        "rotation_jitter": 2.0,
        "baseline_jitter": 0.15,
        "char_spacing": -0.15,
        "_layout": CURSIVE_LAYOUT,
        "_layout_weight": 2.0,
        "_style": "cursive",
    },
    "scribble": {
        "rotation_jitter": 5.0,
        "baseline_jitter": 0.4,
        "scale_jitter": 0.3,
        "_layout": CURSIVE_LAYOUT,
        "_layout_weight": 2.5,
        "_style": "scribble",
    },
    "rushed": {
        "rotation_jitter": 4.0,
        "baseline_jitter": 0.3,
        "scale_jitter": 0.25,
        "_layout": CURSIVE_LAYOUT,
        "_layout_weight": 2.5,
        "_style": "rushed",
    },
    "fast": {
        "rotation_jitter": 3.0,
        "scale_jitter": 0.18,
        "_layout": CURSIVE_LAYOUT,
        "_layout_weight": 2.0,
        "_style": "fast",
    },
    "hurried": {
        "rotation_jitter": 3.5,
        "baseline_jitter": 0.25,
        "_layout": CURSIVE_LAYOUT,
        "_layout_weight": 2.5,
        "_style": "rushed",
    },
    "casual": {
        "rotation_jitter": 1.5,
        "baseline_jitter": 0.15,
        "_layout": NATURAL_LAYOUT,
        "_style": "casual",
    },

    # --- Elegant / refined / artistic ---
    "\u4f18\u96c5": {                    # 优雅
        "rotation_jitter": -1.0,
        "ink_density": 0.05,
        "char_spacing": 0.1,
        "line_spacing": 0.1,
        "_layout": NEAT_LAYOUT,
        "_style": "elegant",
    },
    "\u7cbe\u81f4": {                    # 精致
        "rotation_jitter": -1.5,
        "ink_density": 0.05,
        "_layout": NEAT_LAYOUT,
        "_style": "refined",
    },
    "\u7965\u548c": {                    # 祥和
        "rotation_jitter": -1.0,
        "baseline_jitter": -0.05,
        "_layout": NATURAL_LAYOUT,
        "_style": "elegant",
    },
    "\u827a\u672f": {                    # 艺术
        "rotation_jitter": 1.5,
        "ink_density": 0.1,
        "_layout": NATURAL_LAYOUT,
        "_style": "artistic",
    },
    "elegant": {
        "rotation_jitter": -1.0,
        "ink_density": 0.05,
        "char_spacing": 0.1,
        "_layout": NEAT_LAYOUT,
        "_style": "elegant",
    },
    "graceful": {
        "rotation_jitter": -1.0,
        "char_spacing": 0.1,
        "_layout": NEAT_LAYOUT,
        "_style": "elegant",
    },
    "refined": {
        "rotation_jitter": -1.5,
        "_layout": NEAT_LAYOUT,
        "_style": "refined",
    },
    "beautiful": {
        "rotation_jitter": -1.0,
        "ink_density": 0.05,
        "_layout": NEAT_LAYOUT,
        "_style": "elegant",
    },
    "artistic": {
        "rotation_jitter": 1.5,
        "ink_density": 0.1,
        "_layout": NATURAL_LAYOUT,
        "_style": "artistic",
    },

    # --- Age / persona keywords ---
    "\u7a1a\u5ae9": {                    # 稚嫩
        "rotation_jitter": 3.0,
        "baseline_jitter": 0.3,
        "scale_jitter": 0.25,
        "ink_density": -0.1,
        "_layout": NATURAL_LAYOUT,
        "_style": "childlike",
    },
    "\u513f\u7ae5": {                    # 儿童
        "rotation_jitter": 3.0,
        "baseline_jitter": 0.3,
        "scale_jitter": 0.25,
        "_layout": NATURAL_LAYOUT,
        "_style": "childlike",
    },
    "\u5b69\u5b50": {                    # 孩子
        "rotation_jitter": 2.5,
        "baseline_jitter": 0.25,
        "scale_jitter": 0.2,
        "_layout": NATURAL_LAYOUT,
        "_style": "childlike",
    },
    "\u5c0f\u5b66\u751f": {              # 小学生
        "rotation_jitter": 2.0,
        "baseline_jitter": 0.2,
        "scale_jitter": 0.15,
        "_layout": NATURAL_LAYOUT,
        "_style": "childlike",
    },
    "\u9ad8\u4e09": {                    # 高三
        "rotation_jitter": 3.0,
        "baseline_jitter": 0.2,
        "scale_jitter": 0.15,
        "char_spacing": -0.05,
        "_layout": NATURAL_LAYOUT,
        "_layout_weight": 0.4,
        "_style": "student",
    },
    "\u5b66\u751f": {                    # 学生
        "rotation_jitter": 1.5,
        "baseline_jitter": 0.1,
        "_layout": NATURAL_LAYOUT,
        "_layout_weight": 0.4,
        "_style": "student",
    },
    "\u8001\u7ec3": {                    # 老练
        "rotation_jitter": -1.5,
        "baseline_jitter": -0.1,
        "ink_density": 0.1,
        "_layout": NEAT_LAYOUT,
        "_style": "experienced",
    },
    "\u8001\u4eba": {                    # 老人
        "rotation_jitter": 2.5,
        "baseline_jitter": 0.25,
        "ink_density": -0.1,
        "_layout": NATURAL_LAYOUT,
        "_style": "elderly",
    },
    "\u8001\u5e08": {                    # 老师
        "rotation_jitter": -1.5,
        "baseline_jitter": -0.15,
        "_layout": NEAT_LAYOUT,
        "_style": "teacher",
    },
    "\u533b\u751f": {                    # 医生
        "rotation_jitter": 4.5,
        "baseline_jitter": 0.4,
        "scale_jitter": 0.3,
        "char_spacing": -0.15,
        "_layout": CURSIVE_LAYOUT,
        "_style": "doctor",
    },
    "childlike": {
        "rotation_jitter": 3.0,
        "baseline_jitter": 0.3,
        "scale_jitter": 0.25,
        "ink_density": -0.1,
        "_layout": NATURAL_LAYOUT,
        "_layout_weight": 0.6,
        "_style": "childlike",
    },
    "child": {
        "rotation_jitter": 3.0,
        "baseline_jitter": 0.3,
        "scale_jitter": 0.25,
        "_layout": NATURAL_LAYOUT,
        "_layout_weight": 0.6,
        "_style": "childlike",
    },
    "kid": {
        "rotation_jitter": 2.5,
        "baseline_jitter": 0.25,
        "scale_jitter": 0.2,
        "_layout": NATURAL_LAYOUT,
        "_layout_weight": 0.6,
        "_style": "childlike",
    },
    "elementary": {
        "rotation_jitter": 2.0,
        "baseline_jitter": 0.15,
        "_layout": NATURAL_LAYOUT,
        "_layout_weight": 0.4,
        "_style": "student",
    },
    "student": {
        "rotation_jitter": 1.5,
        "baseline_jitter": 0.1,
        "_layout": NATURAL_LAYOUT,
        "_layout_weight": 0.4,
        "_style": "student",
    },
    "elderly": {
        "rotation_jitter": 2.5,
        "baseline_jitter": 0.25,
        "ink_density": -0.1,
        "_layout": NATURAL_LAYOUT,
        "_style": "elderly",
    },
    "experienced": {
        "rotation_jitter": -1.5,
        "ink_density": 0.1,
        "_layout": NEAT_LAYOUT,
        "_style": "experienced",
    },
    "teacher": {
        "rotation_jitter": -1.5,
        "baseline_jitter": -0.15,
        "_layout": NEAT_LAYOUT,
        "_style": "teacher",
    },
    "doctor": {
        "rotation_jitter": 4.5,
        "baseline_jitter": 0.4,
        "scale_jitter": 0.3,
        "_layout": CURSIVE_LAYOUT,
        "_style": "doctor",
    },

    # --- Ink / pressure keywords ---
    "\u6d53\u91cd": {                    # 浓重
        "ink_density": 0.25,
        "_style": "heavy",
    },
    "\u6d53\u9ed1": {                    # 浓黑
        "ink_density": 0.3,
        "_style": "heavy",
    },
    "\u6dd1\u6dd1": {                    # 淑淑 (variant)
        "ink_density": -0.15,
        "_style": "light",
    },
    "\u6de1": {                          # 淡
        "ink_density": -0.2,
        "_style": "light",
    },
    "\u6de1\u8584": {                    # 淡薄
        "ink_density": -0.25,
        "_style": "light",
    },
    "\u7c97": {                          # 粗
        "ink_density": 0.2,
        "_style": "thick",
    },
    "\u7ec6": {                          # 细
        "ink_density": -0.15,
        "_style": "thin",
    },
    "heavy": {
        "ink_density": 0.25,
        "_style": "heavy",
    },
    "bold": {
        "ink_density": 0.2,
        "_style": "bold",
    },
    "thick": {
        "ink_density": 0.2,
        "_style": "thick",
    },
    "light": {
        "ink_density": -0.2,
        "_style": "light",
    },
    "thin": {
        "ink_density": -0.15,
        "_style": "thin",
    },
    "faint": {
        "ink_density": -0.25,
        "_style": "faint",
    },

    # --- Spacing / size keywords ---
    "\u7d27\u51d1": {                    # 紧凑
        "char_spacing": -0.2,
        "line_spacing": -0.15,
        "_style": "compact",
    },
    "\u758f\u6717": {                    # 疏朗
        "char_spacing": 0.2,
        "line_spacing": 0.2,
        "_style": "airy",
    },
    "\u7a00\u758f": {                    # 稀疏
        "char_spacing": 0.25,
        "line_spacing": 0.2,
        "_style": "airy",
    },
    "compact": {
        "char_spacing": -0.2,
        "line_spacing": -0.15,
        "_style": "compact",
    },
    "airy": {
        "char_spacing": 0.2,
        "line_spacing": 0.2,
        "_style": "airy",
    },
    "spacious": {
        "char_spacing": 0.25,
        "line_spacing": 0.25,
        "_style": "airy",
    },
    "spread": {
        "char_spacing": 0.2,
        "line_spacing": 0.15,
        "_style": "airy",
    },
}


def normalize_keyword(token: str) -> str:
    """Return a lowercase, stripped token suitable for dictionary lookup."""
    return token.strip().lower()


__all__ = [
    "INTENSITY_MODIFIERS",
    "EMOTION_TO_PARAMS",
    "KEYWORD_TO_STYLE",
    "normalize_keyword",
]
