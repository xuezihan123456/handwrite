"""Character set definitions used by preprocessing and inference."""

COMMON_500 = list(
    "的一国在人了有中是年和大业不为发会工经上地市要个产这出行作生家以成到日民来我部对进多全建他公开们场展时理新方主企资实学报制政济用同于法高长现本月定化加动合品重关机分力自外者区能设后就等体下万元社过前面农也得与说之员而务利电文事可种总改三各好金第司其从平代当天水你提商十管内小技位目起海所立已通入量子问度北保心还科委都术使明着次将增基名向门应里美由规今题记点计去强两些表系办教正条最达特革收二期并程厂如道际及西口京华任调性导组东路活广意比投决交统党南安此领结营项情解义山先车然价放世间因共院步物界集把持无但城相书村求治取原处府研质信四运县军件育局干队团又造形级标联专少费效据手施权江近深更认果格几看没职服台式益想数单样只被亿老受优常销志战流很接乡头给至难观指创证织论别五协变风批见究支那查张精林每转划准做需传争税构具百或才积势举必型易视快李参回引镇首推思完消值该走装众责备州供包副极整确知贸己环话反身选亚么带采王策真女谈严斯况色打德告仅它气料神率识劳境源青护列兴许户马港则节款拉直案股光较河花根布线土克再群医清速律她族历非感占续师何影功负验望财类货约艺售连纪按讯史示象养获石食抓富模始住赛客越闻央席坚份"
)

PUNCTUATION = ["，", "。", "！", "？", "、", "；", "：", "“", "”", "（", "）", "《", "》", "—", "…"]
DIGITS = [str(i) for i in range(10)]
LETTERS = [chr(code) for code in range(ord("a"), ord("z") + 1)] + [
    chr(code) for code in range(ord("A"), ord("Z") + 1)
]


def get_charset(level: str = "500") -> list[str]:
    """Return the configured character set level."""
    if level != "500":
        raise ValueError(f"Unsupported charset level: {level}")

    charset = COMMON_500 + PUNCTUATION + DIGITS + LETTERS
    return list(dict.fromkeys(charset))
