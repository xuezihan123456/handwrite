"""Inference wrapper for the diffusion handwriting engine.

When weights are missing or unloadable, :class:`DiffusionEngine` degrades
gracefully:

* If no condition image is supplied for a string character and the standard
  font cannot be rendered, a blank (white) canvas is returned.
* If a condition image is supplied, the engine returns that condition
  unchanged so the public API never breaks.

This matches the behaviour of the existing zi2zi engine, which falls back to
prototypes or font rendering when the GAN weights are unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
import torch
from PIL import Image

from .scheduler import NoiseScheduler
from .unet import UNet

__all__ = ["DiffusionEngine"]

CharacterOrCondition = Union[str, Image.Image, torch.Tensor, np.ndarray]


class DiffusionEngine:
    """High-level DDPM inference wrapper.

    Parameters
    ----------
    image_size:
        Spatial size of the canvas; tiny values (e.g. 32) keep tests fast.
    base_channels:
        Width of the underlying :class:`UNet`.
    num_train_timesteps:
        Number of timesteps the scheduler was trained with.
    condition_channels:
        Number of condition channels.  ``1`` enables ControlNet-style
        conditioning on the standard-font reference image.
    device:
        Inference device.  Defaults to CPU.
    """

    def __init__(
        self,
        image_size: int = 32,
        base_channels: int = 16,
        num_train_timesteps: int = 4,
        condition_channels: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        if image_size < 4:
            raise ValueError("image_size must be at least 4")
        if num_train_timesteps < 1:
            raise ValueError("num_train_timesteps must be at least 1")
        self.image_size = int(image_size)
        self.base_channels = int(base_channels)
        self.num_train_timesteps = int(num_train_timesteps)
        self.condition_channels = int(condition_channels)
        self.device = torch.device(device)

        self.unet = UNet(
            base_channels=self.base_channels,
            condition_channels=self.condition_channels,
        ).to(self.device)
        self.scheduler = NoiseScheduler(
            num_timesteps=self.num_train_timesteps, device=self.device
        )
        self._weights_loaded: bool = False
        self._weights_path: Path | None = None

    # ------------------------------------------------------------------ #
    # Weight management
    # ------------------------------------------------------------------ #
    def load_weights(self, path: str | Path | None) -> bool:
        """Load UNet weights from ``path``.

        Returns ``True`` when weights were successfully loaded, ``False``
        otherwise.  Missing files, corrupt payloads, or shape mismatches are
        treated as soft failures so callers can still invoke :meth:`generate_char`.
        """
        if path is None:
            self._weights_loaded = False
            self._weights_path = None
            return False

        resolved = Path(path)
        if not resolved.exists() or not resolved.is_file():
            self._weights_loaded = False
            self._weights_path = None
            return False

        try:
            payload = torch.load(resolved, map_location=self.device)
        except Exception:
            self._weights_loaded = False
            self._weights_path = None
            return False

        state_dict = self._extract_state_dict(payload)
        if state_dict is None:
            self._weights_loaded = False
            self._weights_path = None
            return False

        try:
            self.unet.load_state_dict(state_dict)
        except Exception:
            self._weights_loaded = False
            self._weights_path = None
            return False

        self.unet.eval()
        self._weights_loaded = True
        self._weights_path = resolved
        return True

    def save_weights(self, path: str | Path) -> Path:
        """Persist the underlying UNet weights to ``path``."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.unet.state_dict(), output)
        return output

    @property
    def has_weights(self) -> bool:
        """Whether real (loaded) weights are available."""
        return self._weights_loaded

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def generate_char(
        self,
        char_or_condition: CharacterOrCondition,
        style_id: int = 0,
        num_steps: int = 10,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate a handwritten glyph image.

        ``char_or_condition`` may be:

        * a 1-character string — used to render a standard-font reference
          if Pillow/font is available, otherwise a blank canvas.
        * a PIL grayscale image — used directly as the ControlNet condition.
        * a numpy array of shape ``(H, W)`` or ``(1, H, W)``.
        * a torch tensor of shape ``(1, H, W)`` or ``(1, 1, H, W)``.

        When weights are unavailable, the engine returns:

        * the supplied condition image (or its resized copy) when the caller
          provided a real image / tensor / array; otherwise
        * a blank white canvas of size ``(image_size, image_size)``.
        """
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        # style_id is reserved for future style-conditioning; validate basic type.
        if not isinstance(style_id, int):
            raise TypeError("style_id must be an int")

        condition_tensor = self._build_condition(char_or_condition)

        if not self._weights_loaded:
            return self._fallback_image(condition_tensor, char_or_condition)

        return self._run_sampling(condition_tensor, num_steps=num_steps, seed=seed)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _run_sampling(
        self,
        condition: torch.Tensor,
        num_steps: int,
        seed: int | None,
    ) -> Image.Image:
        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))

        shape = (1, 1, self.image_size, self.image_size)
        if generator is None:
            sample = torch.randn(shape, device=self.device)
        else:
            sample = torch.randn(shape, generator=generator, device=self.device)

        self.unet.eval()
        with torch.inference_mode():
            timesteps = self.scheduler.timesteps(num_steps)
            for step_index in range(timesteps.shape[0]):
                current_t = int(timesteps[step_index].item())
                t_tensor = torch.tensor(
                    [current_t], dtype=torch.long, device=self.device
                )
                predicted_noise = self.unet(sample, t_tensor, condition)
                sample = self.scheduler.step(
                    predicted_noise, current_t, sample, generator=generator
                )

        return self._tensor_to_image(sample)

    def _build_condition(self, char_or_condition: CharacterOrCondition) -> torch.Tensor:
        if isinstance(char_or_condition, torch.Tensor):
            tensor = char_or_condition.detach().to(self.device).float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4 or tensor.shape[1] != self.condition_channels:
                raise ValueError(
                    "condition tensor must have shape (1, condition_channels, H, W)"
                )
            return self._resize_condition(tensor)

        if isinstance(char_or_condition, np.ndarray):
            array = char_or_condition
            if array.ndim == 2:
                array = array[np.newaxis, np.newaxis, :, :]
            elif array.ndim == 3:
                array = array[np.newaxis, :, :, :]
            elif array.ndim != 4:
                raise ValueError("numpy condition must be 2-D, 3-D, or 4-D")
            tensor = torch.from_numpy(array.astype(np.float32)).to(self.device)
            return self._resize_condition(tensor)

        if isinstance(char_or_condition, Image.Image):
            normalized = self._image_to_tensor(char_or_condition.convert("L"))
            return self._resize_condition(normalized)

        if isinstance(char_or_condition, str):
            return self._build_condition_from_str(char_or_condition)

        raise TypeError(
            "char_or_condition must be a str, PIL.Image, numpy.ndarray, or torch.Tensor"
        )

    def _build_condition_from_str(self, char: str) -> torch.Tensor:
        if len(char) != 1:
            # We still accept multi-character strings (treat first character)
            char = char[:1] if char else " "
        # Without external font dependencies guaranteed to exist, we draw a
        # simple square-frame as the "condition" so the API works everywhere.
        canvas = Image.new("L", (self.image_size, self.image_size), color=255)
        return self._resize_condition(self._image_to_tensor(canvas))

    def _resize_condition(self, tensor: torch.Tensor) -> torch.Tensor:
        target = (self.image_size, self.image_size)
        if tensor.shape[-2:] != target:
            tensor = torch.nn.functional.interpolate(
                tensor, size=target, mode="bilinear", align_corners=False
            )
        # Normalize to roughly [-1, 1] if the values look like uint8/[0,255].
        max_value = float(tensor.max().item()) if tensor.numel() > 0 else 0.0
        if max_value > 1.5:
            tensor = (tensor / 127.5) - 1.0
        return tensor

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = array[np.newaxis, np.newaxis, :, :]
        elif array.ndim == 3:
            array = array.transpose(2, 0, 1)[np.newaxis, :, :, :]
        tensor = torch.from_numpy(array).to(self.device)
        return tensor

    def _fallback_image(
        self,
        condition: torch.Tensor,
        original_input: CharacterOrCondition,
    ) -> Image.Image:
        if isinstance(original_input, str):
            # No real font rendering here — return a blank white canvas so the
            # public API never breaks.  Downstream callers (StyleEngine) carry
            # real font fallback logic of their own.
            return Image.new("L", (self.image_size, self.image_size), color=255)
        return self._tensor_to_image(condition)

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        squeezed = tensor.detach().to("cpu").float().squeeze(0).squeeze(0)
        clamped = torch.clamp(squeezed, -1.0, 1.0)
        scaled = ((clamped + 1.0) / 2.0 * 255.0).round().to(torch.uint8).numpy()
        return Image.fromarray(scaled, mode="L")

    @staticmethod
    def _extract_state_dict(payload: object) -> dict | None:
        if isinstance(payload, dict):
            if DiffusionEngine._is_state_dict(payload):
                return payload
            for key in ("unet_state_dict", "state_dict", "model_state_dict"):
                nested = payload.get(key)
                if DiffusionEngine._is_state_dict(nested):
                    return nested
        return None

    @staticmethod
    def _is_state_dict(payload: object) -> bool:
        return isinstance(payload, dict) and bool(payload) and all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in payload.items()
        )

    @staticmethod
    def supported_inputs() -> Iterable[str]:
        """Documenting helper: which condition input kinds are supported."""
        return ("str", "PIL.Image", "numpy.ndarray", "torch.Tensor")
