"""Build a font-based prototype glyph pack for HandWrite.

Usage:
    python scripts/build_font_prototype_pack.py --font C:/Windows/Fonts/STKAITI.TTF --output_dir data/prototypes/font_note --size 128
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

_COMMON_500 = (
    "的一国在人了有中是年和大业不为发会工经上地市要个产这出行作生家以成到日民来我部对进多全建他公开们场展时理新方主企资实学报制政济用同于法高长现本月定化加动合品重关机分力自外者区能设后就等体下万元社过前面农也得与说之员而务利电文事可种总改三各好金第司其从平代当天水你提商十管内小技位目起海所立已通入量子问度北保心还科委都术使明着次将增基名向门应里美由规今题记点计去强两些表系办教正条最达特革收二期并程厂如道际及西口京华任调性导组东路活广意比投决交统党南安此领结营项情解义山先车然价放世间因共院步物界集把持无但城相书村求治取原处府研质信四运县军件育局干队团又造形级标联专少费效据手施权江近深更认果格几看没职服台式益想数单样只被亿老受优常销志战流很接乡头给至难观指创证织论别五协变风批见究支那查张精林每转划准做需传争税构具百或才积势举必型易视快李参回引镇首推思完消值该走装众责备州供包副极整确知贸己环话反身选亚么带采王策真女谈严斯况色打德告仅它气料神率识劳境源青护列兴许户马港则节款拉直案股光较河花根布线土克再群医清速律她族历非感占续师何影功负验望财类货约艺售连纪按讯史示象养获石食抓富模始住赛客越闻央席坚份"
)
_PUNCT = [chr(c) for c in [0xff0c,0x3002,0xff01,0xff1f,0x3001,0xff1b,0xff1a,0x201c,0x201d,0xff08,0xff09,0x300a,0x300b,0x2014,0x2026]]
_DIGITS = [str(i) for i in range(10)]
_LETTERS = [chr(c) for c in list(range(97,123)) + list(range(65,91))]

def _charset():
    return list(dict.fromkeys(list(_COMMON_500) + _PUNCT + _DIGITS + _LETTERS))

def _fit_font(char, fp, area):
    low, high, best = 1, area*2, area
    while low <= high:
        mid = (low+high)//2
        f = ImageFont.truetype(fp, size=mid)
        l,t,r,b = f.getbbox(char)
        if (r-l)<=area and (b-t)<=area: best=mid; low=mid+1
        else: high=mid-1
    return ImageFont.truetype(fp, size=best)

def _ok(char, fp, size):
    try:
        f = ImageFont.truetype(fp, size=size)
        l,t,r,b = f.getbbox(char)
        return (r-l)>0 and (b-t)>0
    except: return False

def render(char, fp, sz, seed):
    rng = random.Random(seed)
    img = Image.new("L", (sz,sz), 255)
    draw = ImageDraw.Draw(img)
    f = _fit_font(char, fp, int(sz*0.76))
    l,t,r,b = draw.textbbox((0,0), char, font=f)
    draw.text(((sz-(r-l))/2-l, (sz-(b-t))/2-t), char, font=f, fill=0)
    img = img.rotate(rng.uniform(-1.5,1.5), resample=Image.BICUBIC, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3,0.55)))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.88,1.0))
    return img

def build(fp, out, sz):
    (out/"glyphs").mkdir(parents=True, exist_ok=True)
    cs = _charset()
    print(f"字符集: {len(cs)} 个  字体: {fp}")
    glyphs, skipped = [], []
    for ch in cs:
        if not _ok(ch, fp, sz): skipped.append(ch); continue
        cp = ord(ch); fn = f"U{cp:04X}.png"
        try:
            render(ch, fp, sz, cp).save(str(out/"glyphs"/fn))
            glyphs.append({"char":ch,"file":f"glyphs/{fn}","writer_id":"font_kaiti"})
        except Exception as e:
            skipped.append(ch); print(f"  跳过 {ch!r}: {e}", file=sys.stderr)
    mp = out/"manifest.json"
    mp.write_text(json.dumps({"name":"font_note","glyphs":glyphs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成: 生成 {len(glyphs)} 个，跳过 {len(skipped)} 个")
    print(f"manifest: {mp.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="从系统手写字体批量渲染原型字形包")
    p.add_argument("--font", default="C:/Windows/Fonts/STKAITI.TTF")
    p.add_argument("--output_dir", default="data/prototypes/font_note")
    p.add_argument("--size", type=int, default=128)
    a = p.parse_args()
    fp = a.font
    for fb in ["C:/Windows/Fonts/STKAITI.TTF","C:/Windows/Fonts/simkai.ttf","C:/Windows/Fonts/STXINGKA.TTF"]:
        if not Path(fp).exists() and Path(fb).exists():
            print(f"备选字体: {fb}"); fp=fb; break
    build(fp, Path(a.output_dir), a.size)
