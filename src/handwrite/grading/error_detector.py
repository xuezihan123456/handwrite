"""Error detector -- identify typos, grammar issues, punctuation and format problems."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ErrorType(str, Enum):
    """Categories of detected errors."""

    TYPO = "typo"
    GRAMMAR = "grammar"
    PUNCTUATION = "punctuation"
    FORMAT = "format"


@dataclass(frozen=True)
class ErrorInfo:
    """A single detected error with location and description."""

    error_type: ErrorType
    position: int            # character offset in the original text
    length: int              # length of the erroneous span
    text: str                # the erroneous text
    message: str             # human-readable description
    suggestion: Optional[str] = None  # suggested correction

    def end_position(self) -> int:
        return self.position + self.length


# ---------------------------------------------------------------------------
# Common Chinese typo pairs: wrong -> correct
# ---------------------------------------------------------------------------
_TYPO_PAIRS: list[tuple[str, str]] = [
    ("的地得", ""),  # handled specially below
    ("在再", ""),
    ("己已", ""),
    ("候侯", ""),
    ("以已", ""),
    ("象像", ""),
    ("做作", ""),
    ("坐座", ""),
    ("那哪", ""),
    ("他她它", ""),
    ("的得", ""),
    ("地得", ""),
    ("了le", ""),
    ("吗嘛", ""),
]

# Simple typo dictionary: wrong_word -> (correct_word, hint)
_TYPO_DICT: dict[str, tuple[str, str]] = {
    "按装": ("安装", "“按”应为“安”"),
    "甘败下风": ("甘拜下风", "“败”应为“拜”"),
    "自抱自弃": ("自暴自弃", "“抱”应为“暴”"),
    "针贬": ("针砭", "“贬”应为“砭”"),
    "泊来品": ("舶来品", "“泊”应为“舶”"),
    "脉博": ("脉搏", "“博”应为“搏”"),
    "松驰": ("松弛", "“驰”应为“弛”"),
    "一愁莫展": ("一筹莫展", "“愁”应为“筹”"),
    "穿流不息": ("川流不息", "“穿”应为“川”"),
    "精萃": ("精粹", "“萃”应为“粹”"),
    "重迭": ("重叠", "“迭”应为“叠”"),
    "渡假村": ("度假村", "“渡”应为“度”"),
    "防碍": ("妨碍", "“防”应为“妨”"),
    "幅射": ("辐射", "“幅”应为“辐”"),
    "一幅对联": ("一副对联", "“幅”应为“副”"),
    "气慨": ("气概", "“慨”应为“概”"),
    "一股作气": ("一鼓作气", "“股”应为“鼓”"),
    "悬梁刺骨": ("悬梁刺股", "“骨”应为“股”"),
    "粗旷": ("粗犷", "“旷”应为“犷”"),
    "食不裹腹": ("食不果腹", "“裹”应为“果”"),
    "震憾": ("震撼", "“憾”应为“撼”"),
    "凑和": ("凑合", "“和”应为“合”"),
    "侯车室": ("候车室", "“侯”应为“候”"),
    "迫不急待": ("迫不及待", "“急”应为“及”"),
    "既使": ("即使", "“既”应为“即”"),
    "一如继往": ("一如既往", "“继”应为“既”"),
    "草管人命": ("草菅人命", "“管”应为“菅”"),
    "娇揉造作": ("矫揉造作", "“娇”应为“矫”"),
    "挖墙角": ("挖墙脚", "“角”应为“脚”"),
    "一诺千斤": ("一诺千金", "“斤”应为“金”"),
    "不径而走": ("不胫而走", "“径”应为“胫”"),
    "峻工": ("竣工", "“峻”应为“竣”"),
    "不落巢臼": ("不落窠臼", "“巢”应为“窠”"),
    "烩炙人口": ("脍炙人口", "“烩”应为“脍”"),
    "打腊": ("打蜡", "“腊”应为“蜡”"),
    "死皮癞脸": ("死皮赖脸", "“癞”应为“赖”"),
    "兰天": ("蓝天", "“兰”应为“蓝”"),
    "鼎立相助": ("鼎力相助", "“立”应为“力”"),
    "再接再历": ("再接再厉", "“历”应为“厉”"),
    "老俩口": ("老两口", "“俩”应为“两”"),
    "黄梁美梦": ("黄粱美梦", "“梁”应为“粱”"),
    "了望": ("瞭望", "“了”应为“瞭”"),
    "水笼头": ("水龙头", "“笼”应为“龙”"),
    "杀戳": ("杀戮", "“戳”应为“戮”"),
    "痉孪": ("痉挛", "“孪”应为“挛”"),
    "美仑美奂": ("美轮美奂", "“仑”应为“轮”"),
    "罗嗦": ("啰嗦", "“罗”应为“啰”"),
    "蛛丝蚂迹": ("蛛丝马迹", "“蚂”应为“马”"),
    "萎糜不振": ("萎靡不振", "“糜”应为“靡”"),
    "沉缅": ("沉湎", "“缅”应为“湎”"),
    "名信片": ("明信片", "“名”应为“明”"),
    "默守成规": ("墨守成规", "“默”应为“墨”"),
    "大姆指": ("大拇指", "“姆”应为“拇”"),
    "沤心沥血": ("呕心沥血", "“沤”应为“呕”"),
    "凭添": ("平添", "“凭”应为“平”"),
    "出奇不意": ("出其不意", "“奇”应为“其”"),
    "修茸": ("修葺", "“茸”应为“葺”"),
    "亲睐": ("青睐", "“亲”应为“青”"),
    "磬竹难书": ("罄竹难书", "“磬”应为“罄”"),
    "入场卷": ("入场券", "“卷”应为“券”"),
    "声名雀起": ("声名鹊起", "“雀”应为“鹊”"),
    "发韧": ("发轫", "“韧”应为“轫”"),
    "搔痒病": ("瘙痒病", "“搔”应为“瘙”"),
    "欣尝": ("欣赏", "“尝”应为“赏”"),
    "谈笑风声": ("谈笑风生", "“声”应为“生”"),
    "人情事故": ("人情世故", "“事”应为“世”"),
    "有持无恐": ("有恃无恐", "“持”应为“恃”"),
    "额首称庆": ("额手称庆", "“首”应为“手”"),
    "追朔": ("追溯", "“朔”应为“溯”"),
    "鬼鬼崇崇": ("鬼鬼祟祟", "“崇”应为“祟”"),
    "金榜提名": ("金榜题名", "“提”应为“题”"),
    "走头无路": ("走投无路", "“头”应为“投”"),
    "趋之若骛": ("趋之若鹜", "“骛”应为“鹜”"),
    "迁徒": ("迁徙", "“徒”应为“徙”"),
    "洁白无暇": ("洁白无瑕", "“暇”应为“瑕”"),
    "九宵": ("九霄", "“宵”应为“霄”"),
    "渲泄": ("宣泄", "“渲”应为“宣”"),
    "寒喧": ("寒暄", "“喧”应为“暄”"),
    "弦律": ("旋律", "“弦”应为“旋”"),
    "膺品": ("赝品", "“膺”应为“赝”"),
    "不能自己": ("不能自已", "“己”应为“已”"),
    "尤如猛虎": ("犹如猛虎", "“尤”应为“犹”"),
    "竭泽而鱼": ("竭泽而渔", "“鱼”应为“渔”"),
    "滥芋充数": ("滥竽充数", "“芋”应为“竽”"),
    "世外桃园": ("世外桃源", "“园”应为“源”"),
    "脏款": ("赃款", "“脏”应为“赃”"),
    "醮水": ("蘸水", "“醮”应为“蘸”"),
    "蜇伏": ("蛰伏", "“蜇”应为“蛰”"),
    "装祯": ("装帧", "“祯”应为“帧”"),
    "坐阵": ("坐镇", "“阵”应为“镇”"),
    "旁证博引": ("旁征博引", "“证”应为“征”"),
    "灸手可热": ("炙手可热", "“灸”应为“炙”"),
    "九洲": ("九州", "“洲”应为“州”"),
    "床第之私": ("床笫之私", "“第”应为“笫”"),
    "姿意妄为": ("恣意妄为", "“姿”应为“恣”"),
    "编篡": ("编纂", "“篡”应为“纂”"),
    "做月子": ("坐月子", "“做”应为“坐”"),
}

# Common punctuation patterns in Chinese
_PUNCTUATION_RULES: list[tuple[str, str, str]] = [
    # (pattern, message, suggestion)
    (r"[,，]{2,}", "重复逗号", "，"),
    (r"[.。]{2,}", "重复句号", "。"),
    (r"[!！]{2,}", "重复感叹号", "！"),
    (r"[?？]{2,}", "重复问号", "？"),
    (r"[,，]\s*[,，]", "连续逗号", "，"),
    (r"[.。]\s*[.。]", "连续句号", "。"),
    (r"(?<=[\u4e00-\u9fff])[.](?=[\u4e00-\u9fff])", "中文语境中使用了英文句号", "。"),
    (r"(?<=[\u4e00-\u9fff])[,](?=[\u4e00-\u9fff])", "中文语境中使用了英文逗号", "，"),
    (r"(?<=[\u4e00-\u9fff])[?](?=[\u4e00-\u9fff])", "中文语境中使用了英文问号", "？"),
    (r"(?<=[\u4e00-\u9fff])[!](?=[\u4e00-\u9fff])", "中文语境中使用了英文感叹号", "！"),
    (r"(?<=[\u4e00-\u9fff])[:](?=[\u4e00-\u9fff])", "中文语境中使用了英文冒号", "："),
    (r"(?<=[\u4e00-\u9fff])[;](?=[\u4e00-\u9fff])", "中文语境中使用了英文分号", "；"),
]

# Format rules
_FORMAT_PATTERNS: list[tuple[str, str, str]] = [
    # (pattern, message, suggestion)
    (r"[\t]+", "使用了制表符Tab", "建议使用空格缩进"),
    (r"(?m)^\s+$", "存在空白行中的多余空格", "清除多余空格"),
    (r" {4,}", "缩进过多（超过4个空格）", "建议使用2或4个空格"),
    (r"(\r\n|\r)", "使用了非标准换行符(CR/CRLF)", "建议使用LF换行"),
]

# Grammar patterns (simplified Chinese common mistakes)
_GRAMMAR_PATTERNS: list[tuple[str, str, str]] = [
    # (pattern, message, suggestion)
    (r"通过\s*.*?使\s*.*?[，。]", "\u201c通过……使……\u201d句式缺少主语", "去掉\u201c通过\u201d或\u201c使\u201d"),
    (r"大约\s*.*?左右", "\u201c大约\u201d和\u201c左右\u201d语义重复", "保留其一"),
    (r"约\s*.*?左右", "\u201c约\u201d和\u201c左右\u201d语义重复", "保留其一"),
    (r"超过\s*.*?以上", "\u201c超过\u201d和\u201c以上\u201d语义重复", "保留其一"),
    (r"不是\s*.*?而是\s*.*?而是", "关联词\u201c不是……而是……\u201d使用错误", "检查关联词配对"),
    (r"不但\s*.*?而且\s*.*?而且", "关联词\u201c不但……而且……\u201d使用错误", "检查关联词配对"),
    (r"因为\s*.*?因此", "\u201c因为\u201d和\u201c因此\u201d不应同时使用", "改为\u201c因为……所以\u201d或\u201c由于……因此\u201d"),
    (r"虽然\s*.*?但是\s*.*?却", "\u201c虽然……但是……却\u201d关联词冗余", "保留\u201c虽然……但是\u201d或\u201c虽然……却\u201d"),
]


@dataclass
class ErrorDetector:
    """Detect errors in Chinese text.

    Supports: typos (dictionary-based), grammar patterns, punctuation, format.

    Example::

        detector = ErrorDetector()
        errors = detector.detect("他按装了软件")
        for e in errors:
            print(e)
    """

    extra_typo_dict: dict[str, tuple[str, str]] = field(default_factory=dict)
    extra_grammar_rules: list[tuple[str, str, str]] = field(default_factory=list)
    extra_punctuation_rules: list[tuple[str, str, str]] = field(default_factory=list)
    extra_format_rules: list[tuple[str, str, str]] = field(default_factory=list)

    def detect(self, text: str) -> list[ErrorInfo]:
        """Run all detection passes and return a sorted list of errors."""
        errors: list[ErrorInfo] = []
        errors.extend(self._detect_typos(text))
        errors.extend(self._detect_grammar(text))
        errors.extend(self._detect_punctuation(text))
        errors.extend(self._detect_format(text))
        # Sort by position, then by length (longer matches first for overlaps)
        errors.sort(key=lambda e: (e.position, -e.length))
        return errors

    def detect_typos(self, text: str) -> list[ErrorInfo]:
        """Detect only typo errors."""
        return sorted(self._detect_typos(text), key=lambda e: e.position)

    def detect_grammar(self, text: str) -> list[ErrorInfo]:
        """Detect only grammar errors."""
        return sorted(self._detect_grammar(text), key=lambda e: e.position)

    def detect_punctuation(self, text: str) -> list[ErrorInfo]:
        """Detect only punctuation errors."""
        return sorted(self._detect_punctuation(text), key=lambda e: e.position)

    def detect_format(self, text: str) -> list[ErrorInfo]:
        """Detect only format issues."""
        return sorted(self._detect_format(text), key=lambda e: e.position)

    # ------------------------------------------------------------------
    # Internal detection methods
    # ------------------------------------------------------------------

    def _detect_typos(self, text: str) -> list[ErrorInfo]:
        errors: list[ErrorInfo] = []
        merged = {**_TYPO_DICT, **self.extra_typo_dict}

        for wrong, (correct, hint) in merged.items():
            start = 0
            while True:
                idx = text.find(wrong, start)
                if idx == -1:
                    break
                msg = hint if hint else f"'{wrong}'应为'{correct}'"
                errors.append(
                    ErrorInfo(
                        error_type=ErrorType.TYPO,
                        position=idx,
                        length=len(wrong),
                        text=wrong,
                        message=msg,
                        suggestion=correct,
                    )
                )
                start = idx + len(wrong)

        return errors

    def _detect_grammar(self, text: str) -> list[ErrorInfo]:
        errors: list[ErrorInfo] = []
        rules = _GRAMMAR_PATTERNS + self.extra_grammar_rules

        for pattern, message, suggestion in rules:
            for m in re.finditer(pattern, text):
                errors.append(
                    ErrorInfo(
                        error_type=ErrorType.GRAMMAR,
                        position=m.start(),
                        length=m.end() - m.start(),
                        text=m.group(),
                        message=message,
                        suggestion=suggestion,
                    )
                )

        return errors

    def _detect_punctuation(self, text: str) -> list[ErrorInfo]:
        errors: list[ErrorInfo] = []
        rules = _PUNCTUATION_RULES + self.extra_punctuation_rules

        for pattern, message, suggestion in rules:
            for m in re.finditer(pattern, text):
                errors.append(
                    ErrorInfo(
                        error_type=ErrorType.PUNCTUATION,
                        position=m.start(),
                        length=m.end() - m.start(),
                        text=m.group(),
                        message=message,
                        suggestion=suggestion,
                    )
                )

        return errors

    def _detect_format(self, text: str) -> list[ErrorInfo]:
        errors: list[ErrorInfo] = []
        rules = _FORMAT_PATTERNS + self.extra_format_rules

        for pattern, message, suggestion in rules:
            for m in re.finditer(pattern, text):
                errors.append(
                    ErrorInfo(
                        error_type=ErrorType.FORMAT,
                        position=m.start(),
                        length=m.end() - m.start(),
                        text=m.group(),
                        message=message,
                        suggestion=suggestion,
                    )
                )

        return errors
