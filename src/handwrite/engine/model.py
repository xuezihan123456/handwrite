"""Inference wrapper for handwriting generation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Collection

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from handwrite.data.charsets import get_charset
from handwrite.data.font_renderer import render_standard_char
from handwrite.engine.generator import Generator
from handwrite.prototypes import PrototypeLibrary, load_prototype_library
from handwrite.styles import BUILTIN_STYLES

_DEFAULT_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/STXINGKA.TTF"),
    Path("C:/Windows/Fonts/STKAITI.TTF"),
    Path("C:/Windows/Fonts/simkai.ttf"),
    Path(__file__).resolve().parents[3] / "data" / "fonts" / "NotoSerifSC-Regular.otf",
    Path("C:/Windows/Fonts/NotoSerifSC-VF.ttf"),
    Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simsun.ttc"),
)

_DEFAULT_WEIGHTS_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "weights" / "generator.pt",
    Path(__file__).resolve().parents[3] / "weights" / "generator.pth",
    Path(__file__).resolve().parents[3] / "weights" / "checkpoint.pt",
    Path(__file__).resolve().parents[3] / "weights" / "checkpoint.pth",
)
_DEFAULT_WEIGHT_EXTENSIONS = frozenset({".pt", ".pth"})
_DEFAULT_GENERATOR_EMBED_DIM = 128

_STYLE_VARIANTS: dict[int, tuple[float, tuple[int, int], float]] = {
    0: (0.0, (0, 0), 0.0),
    1: (-2.0, (2, -1), 0.4),
    2: (-4.0, (3, 0), 0.2),
    3: (1.5, (-1, 1), 0.0),
    4: (3.0, (-2, 1), 0.6),
}
_STYLE_NAMES_BY_ID = {style_id: name for name, style_id in BUILTIN_STYLES.items()}
_NOTE_STYLE_ID = BUILTIN_STYLES.get("行书流畅", 0)


class StyleEngine:
    """Minimal inference wrapper used by the public API."""

    def __init__(
        self,
        font_path: str | Path | None = None,
        supported_chars: Collection[str] | None = None,
        image_size: int = 256,
        char_size: int = 200,
        weights_path: str | Path | None = None,
        prototype_library: PrototypeLibrary | None = None,
        prototype_pack: str | Path | None = None,
    ) -> None:
        self.font_path = self._resolve_font_path(font_path)
        if supported_chars is None:
            self.supported_chars = frozenset(get_charset("500"))
        else:
            self.supported_chars = frozenset(supported_chars)
        self.image_size = image_size
        self.char_size = char_size
        self.valid_style_ids = frozenset(BUILTIN_STYLES.values())
        self._generator_num_styles = max(self.valid_style_ids) + 1 if self.valid_style_ids else 0
        self._generator = self._load_generator(weights_path)
        self.prototype_library = (
            prototype_library
            if prototype_library is not None
            else self._load_prototype_library(prototype_pack)
        )

    def generate_char(self, char: str, style_id: int) -> Image.Image:
        """Generate a single grayscale character image for the requested style."""
        self._validate_style_id(style_id)

        if char in self.supported_chars:
            return self._gan_generate(char, style_id)

        prototype_image = self._prototype_generate(char, style_id)
        if prototype_image is not None:
            return prototype_image
        return self._fallback_render(char)

    def inspect_char(self, char: str, style_id: int) -> dict[str, Any]:
        """Describe which realism route a character will use."""
        self._validate_style_id(style_id)

        if self._generator is not None and char in self.supported_chars:
            return {"char": char, "route": "model", "is_low_realism": False}
        prototype_library = self._prototype_library_for_style(style_id)
        if prototype_library is not None and prototype_library.has_char(char):
            return {"char": char, "route": "prototype", "is_low_realism": False}
        return {"char": char, "route": "fallback", "is_low_realism": True}

    def inspect_text(self, text: str, style_id: int) -> dict[str, Any]:
        """Summarize text coverage before generation."""
        unique_characters = list(dict.fromkeys(char for char in text if not char.isspace()))
        prototype_covered_characters: list[str] = []
        model_supported_characters: list[str] = []
        fallback_characters: list[str] = []
        for char in unique_characters:
            inspection = self.inspect_char(char, style_id)
            route = inspection["route"]
            if route == "model":
                model_supported_characters.append(char)
            elif route == "prototype":
                prototype_covered_characters.append(char)
            else:
                fallback_characters.append(char)
        return {
            "style": _STYLE_NAMES_BY_ID.get(style_id, f"style:{style_id}"),
            "total_characters": sum(1 for char in text if not char.isspace()),
            "unique_characters": unique_characters,
            "prototype_covered_characters": prototype_covered_characters,
            "model_supported_characters": model_supported_characters,
            "fallback_characters": fallback_characters,
            "prototype_pack_name": self._prototype_pack_name(style_id),
            "prototype_source": self._prototype_source(style_id),
            "prototype_source_kind": self._prototype_source_kind(style_id),
            "summary": (
                f"{len(prototype_covered_characters) + len(model_supported_characters)} / "
                f"{len(unique_characters)} characters are in a higher-realism path."
            ) if unique_characters else "No non-whitespace characters to inspect.",
        }

    def _gan_generate(self, char: str, style_id: int) -> Image.Image:
        """Use a loaded generator when available, then fall back to prototypes and note rendering."""
        model_image = self._model_generate(char, style_id)
        if model_image is not None:
            return model_image

        prototype_image = self._prototype_generate(char, style_id)
        if prototype_image is not None:
            return prototype_image

        return self._fallback_render(char)

    def _prototype_generate(self, char: str, style_id: int) -> Image.Image | None:
        prototype_library = self._prototype_library_for_style(style_id)
        if prototype_library is None or not prototype_library.has_char(char):
            return None
        image = prototype_library.get_glyph_image(char)
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        return image

    def _fallback_render(self, char: str, style_id: int | None = None) -> Image.Image:
        image = self._render_reference_char(char)
        resolved_style_id = _NOTE_STYLE_ID if style_id is None else style_id
        return self._stylize_fallback_image(image, char=char, style_id=resolved_style_id)

    def _stylize_fallback_image(self, image: Image.Image, *, char: str, style_id: int) -> Image.Image:
        seed_payload = f"{char}:{style_id}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
        rng = np.random.default_rng(seed)

        styled = image.copy()
        angle, offset, blur_radius = _STYLE_VARIANTS.get(style_id, _STYLE_VARIANTS[0])
        angle += float(rng.normal(0.0, 0.85))
        if abs(angle) > 0.05:
            styled = styled.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                fillcolor=255,
            )

        jittered_offset = (
            offset[0] + int(round(rng.uniform(-2.0, 2.0))),
            offset[1] + int(round(rng.uniform(-2.0, 2.0))),
        )
        if jittered_offset != (0, 0):
            styled = ImageChops.offset(styled, jittered_offset[0], jittered_offset[1])
            self._fill_wrapped_edges(styled, jittered_offset)

        scale = 1.0 + float(rng.uniform(-0.06, 0.015))
        if abs(scale - 1.0) > 0.01:
            resized_width = max(1, int(round(styled.width * scale)))
            resized_height = max(1, int(round(styled.height * scale)))
            resized = styled.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
            canvas = Image.new("L", (self.image_size, self.image_size), color=255)
            offset_x = (self.image_size - resized_width) // 2
            offset_y = (self.image_size - resized_height) // 2
            canvas.paste(resized, (offset_x, offset_y))
            styled = canvas

        if rng.random() < 0.5:
            styled = styled.filter(ImageFilter.MinFilter(3))
        elif rng.random() < 0.35:
            styled = styled.filter(ImageFilter.MaxFilter(3))

        if blur_radius:
            styled = styled.filter(ImageFilter.GaussianBlur(radius=max(0.1, blur_radius)))

        array = np.asarray(styled, dtype=np.int16)
        ink_mask = array < 250
        if ink_mask.any():
            noise = rng.normal(loc=0.0, scale=6.0, size=array.shape)
            array = np.where(ink_mask, np.clip(array + noise, 0, 255), array)
        return Image.fromarray(array.astype(np.uint8), mode="L")

    def _render_reference_char(self, char: str) -> Image.Image:
        if self.font_path is None:
            return self._render_placeholder_char(char)

        try:
            return render_standard_char(
                char,
                str(self.font_path),
                image_size=self.image_size,
                char_size=self.char_size,
            )
        except (FileNotFoundError, OSError, UnicodeEncodeError):
            return self._render_placeholder_char(char)

    def _validate_style_id(self, style_id: int) -> None:
        if style_id not in self._supported_style_ids():
            raise ValueError(f"Unsupported style_id: {style_id}")

    def _prototype_library_for_style(self, style_id: int) -> PrototypeLibrary | None:
        if style_id != _NOTE_STYLE_ID:
            return None
        return self.prototype_library

    def _supported_style_ids(self) -> frozenset[int]:
        if self._generator is None:
            return self.valid_style_ids
        return frozenset(
            candidate_style_id
            for candidate_style_id in self.valid_style_ids
            if 0 <= candidate_style_id < self._generator_num_styles
        )

    def _resolve_font_path(self, font_path: str | Path | None) -> Path | None:
        candidates = [Path(font_path)] if font_path is not None else list(_DEFAULT_FONT_CANDIDATES)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _load_prototype_library(prototype_pack: str | Path | None) -> PrototypeLibrary | None:
        try:
            return load_prototype_library(prototype_pack)
        except Exception:
            return None

    def _prototype_pack_name(self, style_id: int) -> str | None:
        prototype_library = self._prototype_library_for_style(style_id)
        if prototype_library is None:
            return None
        return prototype_library.name

    def _prototype_source(self, style_id: int) -> str | None:
        prototype_library = self._prototype_library_for_style(style_id)
        if prototype_library is None:
            return None
        return prototype_library.prototype_source

    def _prototype_source_kind(self, style_id: int) -> str:
        prototype_library = self._prototype_library_for_style(style_id)
        if prototype_library is None:
            return "disabled"
        return prototype_library.source_kind

    def _resolve_weights_candidates(self, weights_path: str | Path | None) -> tuple[Path, ...]:
        if weights_path is not None:
            candidate = Path(weights_path)
            return (candidate,) if candidate.exists() else ()

        resolved_candidates: list[Path] = []
        seen_paths: set[str] = set()

        for candidate in _DEFAULT_WEIGHTS_CANDIDATES:
            if candidate.exists():
                self._append_unique_path(resolved_candidates, seen_paths, candidate)

        for candidate in self._discover_default_weights():
            self._append_unique_path(resolved_candidates, seen_paths, candidate)

        resolved_candidates.sort(key=self._default_weight_sort_key)
        return tuple(resolved_candidates)

    def _load_generator(self, weights_path: str | Path | None) -> Generator | None:
        for resolved_weights_path in self._resolve_weights_candidates(weights_path):
            try:
                payload = torch.load(resolved_weights_path, map_location="cpu")
                state_dict = self._extract_generator_state_dict(payload)
                if state_dict is None:
                    continue

                generator_num_styles = self._infer_generator_num_styles(state_dict)
                generator_embed_dim = self._infer_generator_embed_dim(state_dict)
                generator = Generator(
                    num_styles=generator_num_styles,
                    embed_dim=generator_embed_dim,
                )
                generator.load_state_dict(state_dict)
                generator.eval()
                self._generator_num_styles = generator_num_styles
                return generator
            except Exception:
                continue

        return None

    def _model_generate(self, char: str, style_id: int) -> Image.Image | None:
        if self._generator is None:
            return None

        try:
            reference_array = np.asarray(self._render_reference_char(char), dtype=np.float32)
            input_tensor = torch.from_numpy(reference_array).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.div(127.5).sub(1.0)
            style_tensor = torch.tensor([style_id], dtype=torch.long)

            with torch.inference_mode():
                output_tensor = self._generator(input_tensor, style_tensor)

            if output_tensor.shape != input_tensor.shape:
                raise ValueError("Generator output must match the input image shape")

            output_array = (
                output_tensor.squeeze(0)
                .squeeze(0)
                .clamp(-1.0, 1.0)
                .add(1.0)
                .mul(127.5)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            return Image.fromarray(output_array, mode="L")
        except Exception:
            return None

    @staticmethod
    def _extract_generator_state_dict(payload: object) -> dict[str, torch.Tensor] | None:
        if isinstance(payload, dict):
            if StyleEngine._is_state_dict(payload):
                return payload

            for key in ("generator_state_dict", "state_dict", "model_state_dict"):
                nested_payload = payload.get(key)
                if StyleEngine._is_state_dict(nested_payload):
                    return nested_payload

        return None

    @staticmethod
    def _is_state_dict(payload: object) -> bool:
        return isinstance(payload, dict) and all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in payload.items()
        )

    @staticmethod
    def _append_unique_path(
        paths: list[Path], seen_paths: set[str], candidate: Path
    ) -> None:
        candidate_key = str(candidate.resolve())
        if candidate_key in seen_paths:
            return
        seen_paths.add(candidate_key)
        paths.append(candidate)

    @staticmethod
    def _discover_default_weights() -> tuple[Path, ...]:
        search_roots: list[Path] = []
        seen_roots: set[str] = set()
        for candidate in _DEFAULT_WEIGHTS_CANDIDATES:
            root = candidate.parent
            root_key = str(root.resolve())
            if root_key in seen_roots or not root.exists() or not root.is_dir():
                continue
            seen_roots.add(root_key)
            search_roots.append(root)

        discovered_paths: list[Path] = []
        for root in search_roots:
            root_candidates = [
                path
                for path in root.iterdir()
                if path.is_file()
                and path.suffix.lower() in _DEFAULT_WEIGHT_EXTENSIONS
                and StyleEngine._looks_like_weight_file(path)
            ]
            discovered_paths.extend(
                sorted(root_candidates, key=StyleEngine._default_weight_sort_key)
            )

        return tuple(discovered_paths)

    @staticmethod
    def _looks_like_weight_file(path: Path) -> bool:
        stem = path.stem.lower()
        return "generator" in stem or "checkpoint" in stem

    @staticmethod
    def _default_weight_sort_key(path: Path) -> tuple[int, int, str]:
        stem = path.stem.lower()
        has_generator = "generator" in stem
        has_checkpoint = "checkpoint" in stem
        has_best = "best" in stem
        has_state_dict = "state" in stem and "dict" in stem
        epoch = StyleEngine._extract_checkpoint_epoch(stem)
        has_epoch = epoch >= 0

        if has_best and has_generator:
            priority = 0
        elif has_checkpoint and has_epoch:
            priority = 1
        elif has_generator and has_state_dict:
            priority = 2
        elif has_generator:
            priority = 3
        elif has_checkpoint:
            priority = 4
        else:
            priority = 5

        return (priority, -epoch, path.name.lower())

    @staticmethod
    def _extract_checkpoint_epoch(stem: str) -> int:
        match = re.search(r"epoch[_-]?(\d+)", stem)
        if match is None:
            return -1
        return int(match.group(1))

    def _infer_generator_num_styles(self, state_dict: dict[str, torch.Tensor]) -> int:
        style_embedding = self._extract_style_embedding_weight(state_dict)
        if style_embedding is None or style_embedding.shape[0] < 1:
            return self._generator_num_styles
        return int(style_embedding.shape[0])

    @staticmethod
    def _infer_generator_embed_dim(state_dict: dict[str, torch.Tensor]) -> int:
        style_embedding = StyleEngine._extract_style_embedding_weight(state_dict)
        if style_embedding is None or style_embedding.shape[1] < 1:
            return _DEFAULT_GENERATOR_EMBED_DIM
        return int(style_embedding.shape[1])

    @staticmethod
    def _extract_style_embedding_weight(
        state_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        for key, value in state_dict.items():
            if key.endswith("style_embedding.weight") and value.ndim == 2:
                return value
        return None

    def _render_placeholder_char(self, char: str) -> Image.Image:
        image = Image.new("L", (self.image_size, self.image_size), color=255)
        draw = ImageDraw.Draw(image)
        inset = max(self.image_size // 10, 12)
        draw.rectangle(
            (inset, inset, self.image_size - inset, self.image_size - inset),
            outline=0,
            width=4,
        )
        draw.line(
            (inset, inset, self.image_size - inset, self.image_size - inset),
            fill=0,
            width=3,
        )
        draw.line(
            (self.image_size - inset, inset, inset, self.image_size - inset),
            fill=0,
            width=3,
        )

        label = self._placeholder_label(char)
        font = ImageFont.load_default()
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        x = (self.image_size - (right - left)) / 2 - left
        y = self.image_size - inset - (bottom - top) - 8
        draw.text((x, y), label, font=font, fill=0)
        return image

    @staticmethod
    def _placeholder_label(char: str) -> str:
        if len(char) == 1 and char.isascii():
            return char
        if char:
            return f"U+{ord(char[0]):X}"
        return "?"

    @staticmethod
    def _fill_wrapped_edges(image: Image.Image, offset: tuple[int, int]) -> None:
        x_offset, y_offset = offset
        width, height = image.size

        if x_offset > 0:
            image.paste(255, (0, 0, x_offset, height))
        elif x_offset < 0:
            image.paste(255, (width + x_offset, 0, width, height))

        if y_offset > 0:
            image.paste(255, (0, 0, width, y_offset))
        elif y_offset < 0:
            image.paste(255, (0, height + y_offset, width, height))
