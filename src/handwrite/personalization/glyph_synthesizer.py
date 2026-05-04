"""Synthesize personalized glyph PNGs based on style vectors.

Generates 256x256 grayscale glyph images that match the user's
handwriting characteristics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from handwrite.personalization.style_extractor import StyleVector

_GLYPH_SIZE = 256
_DEFAULT_CHARSET = (
    "的一国在人了有中是年和大业不为发会工经上地市要个产这出行作生家以成到日民来我部对进多全建他公开们场展时理新方主企资实学报制政济用同于法高长现本月定化加动合品重关机分力自外者区能设后就等体下万元社过前面农也得与说之员而务利电文事可种总改三各好金第司其从平代当天水你提商十管内小技位目起海所立已通入量子问度北保心还科委都术使明着次将增基名向门应里美由规今题记点计去强两些表系办教正条最达特革收二期并程厂如道际及西口京华任调性导组东路活广意比投决交统党南安此领结营项情解义山先车然价放世间因共院步物界集把持无但城相书村求治取原处府研质信四运县军件育局干队团又造形级标联专少费效据手施权江近深更认果格几看没职服台式益想数单样只被亿老受优常销志战流很接乡头给至难观指创证织论别五协变风批见究支那查张精林每转划准做需传争税构具百或才积势举必型易视快李参回引镇首推思完消值该走装众责备州供包副极整确知贸己环话反身选亚么带采王策真女谈严斯况色打德告仅它气料神率识劳境源青护列兴许户马港则节款拉直案股光较河花根布线土克再群医清速律她族历非感占续师何影功负验望财类货约艺售连纪按讯史示象养获石食抓富模始住赛客越闻央席坚份"
)


class GlyphSynthesizer:
    """Synthesize personalized glyph images from a StyleVector."""

    def __init__(
        self,
        style: StyleVector,
        *,
        glyph_size: int = _GLYPH_SIZE,
        font_path: str | None = None,
    ) -> None:
        """Initialize synthesizer.

        Args:
            style: StyleVector controlling rendering parameters.
            glyph_size: Output image size (square). Default 256.
            font_path: Path to a TTF font for base rendering.
                       Falls back to common Chinese system fonts.
        """
        self._style = style
        self._glyph_size = glyph_size
        self._font_path = self._resolve_font_path(font_path)
        self._rng = np.random.RandomState(42)

    def synthesize_char(
        self,
        char: str,
        *,
        seed: int | None = None,
    ) -> Image.Image:
        """Synthesize a single glyph image.

        Args:
            char: Character to render.
            seed: Random seed for reproducibility. Uses char ordinal if None.

        Returns:
            256x256 grayscale PIL Image.
        """
        if seed is None:
            seed = ord(char)
        rng = np.random.RandomState(seed)

        sz = self._glyph_size
        style = self._style

        # Step 1: Render base character with font
        base_img = self._render_base(char, sz)

        # Step 2: Apply stroke thickness via morphological adjustment
        base_img = self._apply_stroke_thickness(base_img, style.stroke_thickness, rng)

        # Step 3: Apply slant
        if abs(style.slant_angle) > 0.5:
            base_img = base_img.rotate(
                -style.slant_angle,  # Negative because PIL rotates counter-clockwise
                resample=Image.BICUBIC,
                fillcolor=255,
                expand=False,
            )

        # Step 4: Apply cursiveness (blur for smoother edges)
        if style.cursiveness > 0.1:
            radius = 0.2 + style.cursiveness * 0.8
            base_img = base_img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Step 5: Apply ink darkness
        darkness_factor = 0.7 + (1.0 - style.ink_darkness) * 0.3
        base_img = ImageEnhance.Brightness(base_img).enhance(darkness_factor)

        # Step 6: Apply smoothness (reduce noise)
        if style.smoothness > 0.5:
            base_img = base_img.filter(
                ImageFilter.GaussianBlur(radius=0.2 + (1.0 - style.smoothness) * 0.3)
            )

        # Step 7: Add subtle natural variation
        base_img = self._add_natural_variation(base_img, rng)

        # Ensure output size
        base_img = base_img.resize((sz, sz), Image.LANCZOS)

        return base_img

    def synthesize_pack(
        self,
        output_dir: Union[str, Path],
        *,
        charset: str | None = None,
        pack_name: str = "personalized",
        writer_id: str = "personalized_user",
    ) -> Path:
        """Synthesize a full glyph pack and save as prototype library.

        Args:
            output_dir: Directory to write glyph pack.
            charset: Characters to include. Uses default common charset if None.
            pack_name: Name for the prototype pack manifest.
            writer_id: Writer identifier for manifest.

        Returns:
            Path to the generated manifest.json.
        """
        out = Path(output_dir)
        glyphs_dir = out / "glyphs"
        glyphs_dir.mkdir(parents=True, exist_ok=True)

        chars = charset or _DEFAULT_CHARSET
        glyphs = []

        for ch in chars:
            cp = ord(ch)
            filename = f"U{cp:04X}.png"
            img = self.synthesize_char(ch, seed=cp)
            img.save(str(glyphs_dir / filename))
            glyphs.append(
                {"char": ch, "file": f"glyphs/{filename}", "writer_id": writer_id}
            )

        manifest = {"name": pack_name, "glyphs": glyphs}
        manifest_path = out / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest_path

    # ------------------------------------------------------------------
    # Internal rendering pipeline
    # ------------------------------------------------------------------

    def _render_base(self, char: str, size: int) -> Image.Image:
        """Render base character onto a blank canvas."""
        img = Image.new("L", (size, size), 255)
        draw = ImageDraw.Draw(img)

        font = self._fit_font(char, size)
        bbox = draw.textbbox((0, 0), char, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (size - text_w) / 2 - bbox[0]
        y = (size - text_h) / 2 - bbox[1]
        draw.text((x, y), char, font=font, fill=0)

        return img

    def _fit_font(self, char: str, area: int) -> ImageFont.FreeTypeFont:
        """Find the largest font size that fits within the target area."""
        fp = self._font_path
        target = int(area * 0.76)
        low, high, best = 1, target * 2, target
        while low <= high:
            mid = (low + high) // 2
            try:
                font = ImageFont.truetype(fp, size=mid)
                l, t, r, b = font.getbbox(char)
                if (r - l) <= area * 0.85 and (b - t) <= area * 0.85:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            except (OSError, AttributeError):
                return ImageFont.load_default()
        try:
            return ImageFont.truetype(fp, size=best)
        except OSError:
            return ImageFont.load_default()

    def _apply_stroke_thickness(
        self, img: Image.Image, thickness: float, rng: np.random.RandomState
    ) -> Image.Image:
        """Adjust apparent stroke thickness via Gaussian blur + threshold."""
        if thickness < 0.3:
            # Thin strokes: sharpen slightly
            return img.filter(ImageFilter.SHARPEN)
        if thickness > 0.7:
            # Thick strokes: dilate
            blur_radius = 0.5 + (thickness - 0.7) * 2.0
            blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            # Re-threshold to maintain edge sharpness
            arr = np.array(blurred)
            arr = np.where(arr < 160, arr * 0.7, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="L")
        return img

    def _add_natural_variation(
        self, img: Image.Image, rng: np.random.RandomState
    ) -> Image.Image:
        """Add subtle random noise to simulate natural ink variation."""
        arr = np.array(img, dtype=np.float64)
        noise = rng.normal(0, 3.0, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    @staticmethod
    def _resolve_font_path(font_path: str | None) -> str:
        """Resolve font path, falling back to system fonts."""
        if font_path and Path(font_path).exists():
            return font_path

        # Try common Chinese handwriting/kai fonts on Windows
        candidates = [
            "C:/Windows/Fonts/STKAITI.TTF",
            "C:/Windows/Fonts/simkai.ttf",
            "C:/Windows/Fonts/STXINGKA.TTF",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate

        # Linux/Mac fallbacks
        for candidate in [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]:
            if Path(candidate).exists():
                return candidate

        return ""  # Will fall back to default PIL font
