"""Inference wrapper for handwriting generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Collection

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from handwrite.data.charsets import get_charset
from handwrite.data.font_renderer import render_standard_char
from handwrite.engine.generator import Generator
from handwrite.styles import BUILTIN_STYLES

_DEFAULT_FONT_CANDIDATES = (
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


class StyleEngine:
    """Minimal inference wrapper used by the public API."""

    def __init__(
        self,
        font_path: str | Path | None = None,
        supported_chars: Collection[str] | None = None,
        image_size: int = 256,
        char_size: int = 200,
        weights_path: str | Path | None = None,
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

    def generate_char(self, char: str, style_id: int) -> Image.Image:
        """Generate a single grayscale character image for the requested style."""
        self._validate_style_id(style_id)

        if char in self.supported_chars:
            return self._gan_generate(char, style_id)
        return self._fallback_render(char)

    def _gan_generate(self, char: str, style_id: int) -> Image.Image:
        """Use a loaded generator when available, otherwise apply placeholder transforms."""
        model_image = self._model_generate(char, style_id)
        if model_image is not None:
            return model_image

        image = self._render_reference_char(char)
        angle, offset, blur_radius = _STYLE_VARIANTS[style_id]

        if angle:
            image = image.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                fillcolor=255,
            )
        if offset != (0, 0):
            image = ImageChops.offset(image, offset[0], offset[1])
            self._fill_wrapped_edges(image, offset)
        if blur_radius:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return image

    def _fallback_render(self, char: str) -> Image.Image:
        return self._render_reference_char(char)

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

    def _resolve_weights_path(self, weights_path: str | Path | None) -> Path | None:
        for candidate in self._resolve_weights_candidates(weights_path):
            return candidate
        return None

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
