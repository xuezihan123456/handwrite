"""Writing skill simulator.

Applies age-dependent transformations to handwriting images, simulating
the motor control development at different stages of learning to write.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .age_profiles import AgeGroup, AgeProfile, get_age_profile


class SkillSimulator:
    """Applies age-dependent handwriting distortions to character images.

    The simulator models five key developmental factors:
    - Position jitter (motor control precision)
    - Size variation (consistent letter sizing)
    - Pressure variation (pen pressure control)
    - Stroke connection (cursive tendency)
    - Baseline stability (line straightness)
    """

    def __init__(
        self,
        age_group: AgeGroup = AgeGroup.ADULT,
        profile: AgeProfile | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the skill simulator.

        Args:
            age_group: The target age group.
            profile: Optional custom profile; overrides age_group if provided.
            seed: Random seed for reproducible output.
        """
        self._age_group = age_group
        self._profile = profile if profile is not None else get_age_profile(age_group)
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed if seed is not None else 42)

    @property
    def age_group(self) -> AgeGroup:
        """Return the configured age group."""
        return self._age_group

    @property
    def profile(self) -> AgeProfile:
        """Return the active handwriting profile."""
        return self._profile

    def apply_to_image(
        self,
        image: Image.Image,
        char_size: int | None = None,
    ) -> Image.Image:
        """Apply age-dependent distortions to a character image.

        Args:
            image: Source character image (grayscale or RGBA).
            char_size: Target size for the output. If None, uses the source size.

        Returns:
            A new Image with age-dependent distortions applied.
        """
        if char_size is None:
            char_size = max(image.size)

        result = image.convert("RGBA")

        # Step 1: Apply size variation
        result = self._apply_size_variation(result, char_size)

        # Step 2: Apply jitter (position offset)
        result = self._apply_jitter(result, char_size)

        # Step 3: Apply pressure variation
        result = self._apply_pressure_variation(result)

        # Step 4: Apply tilt variation
        result = self._apply_tilt_variation(result)

        # Step 5: Apply stroke roughness for younger writers
        result = self._apply_stroke_roughness(result)

        # Step 6: Apply speed-based smoothing for older writers
        result = self._apply_speed_smoothing(result)

        # Ensure correct output size
        if result.size != (char_size, char_size):
            result = result.resize((char_size, char_size), Image.Resampling.LANCZOS)

        return result

    def apply_jitter_to_offset(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """Apply jitter to a character position offset.

        Useful for composing multiple characters with age-appropriate
        positional variation.

        Args:
            x: Original x position.
            y: Original y position.

        Returns:
            Tuple of (jittered_x, jittered_y).
        """
        profile = self._profile
        dx = self._rng.gauss(0, profile.jitter_x)
        dy = self._rng.gauss(0, profile.jitter_y)
        return (x + dx, y + dy)

    def vary_size(self, base_size: int) -> int:
        """Return a varied character size based on the age profile.

        Args:
            base_size: The nominal character size.

        Returns:
            A size value with age-appropriate variation.
        """
        variation = self._profile.size_variation
        factor = 1.0 + self._rng.gauss(0, variation)
        factor = max(0.5, min(1.5, factor))
        return max(8, int(base_size * factor))

    def _apply_size_variation(
        self,
        image: Image.Image,
        char_size: int,
    ) -> Image.Image:
        """Scale the character with age-dependent size variation."""
        varied_size = self.vary_size(char_size)
        if varied_size == char_size:
            return image

        scaled = image.resize((varied_size, varied_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (char_size, char_size), (0, 0, 0, 0))

        # Center the varied character
        offset_x = (char_size - varied_size) // 2
        offset_y = (char_size - varied_size) // 2
        canvas.paste(scaled, (offset_x, offset_y))
        return canvas

    def _apply_jitter(
        self,
        image: Image.Image,
        char_size: int,
    ) -> Image.Image:
        """Apply random positional jitter to the character."""
        profile = self._profile
        dx = int(self._rng.gauss(0, profile.jitter_x))
        dy = int(self._rng.gauss(0, profile.jitter_y))

        if dx == 0 and dy == 0:
            return image

        canvas = Image.new("RGBA", image.size, (0, 0, 0, 0))
        canvas.paste(image, (dx, dy))
        return canvas

    def _apply_pressure_variation(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """Simulate pen pressure variation by adjusting ink darkness."""
        variation = self._profile.pressure_variation
        if variation < 0.01:
            return image

        # Random pressure factor: darker or lighter
        pressure = 1.0 + self._rng.gauss(0, variation * 0.3)
        pressure = max(0.5, min(1.5, pressure))

        arr = np.array(image.convert("RGBA"), dtype=np.float32)

        # Modify RGB channels based on pressure (affect ink darkness)
        # Lower pressure = lighter ink (higher pixel values toward white)
        for c in range(3):
            channel = arr[:, :, c]
            ink_mask = channel < 240
            # Push ink pixels toward white (lighter) or keep dark (heavier)
            if pressure < 1.0:
                channel[ink_mask] = channel[ink_mask] + (255.0 - channel[ink_mask]) * (1.0 - pressure)
            arr[:, :, c] = channel

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")

    def _apply_tilt_variation(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """Apply slight rotation to simulate baseline tilt."""
        tilt = self._profile.tilt_variation
        if tilt < 0.1:
            return image

        angle = self._rng.gauss(0, tilt)
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

    def _apply_stroke_roughness(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """Apply roughness to strokes for younger writers.

        Low stability means more jittery strokes, simulated by adding
        a small amount of noise to the character edges.
        """
        profile = self._profile
        roughness = 1.0 - profile.stability
        if roughness < 0.05:
            return image

        arr = np.array(image.convert("RGBA"), dtype=np.float32)
        noise_scale = roughness * 15.0

        # Add noise to all channels, but stronger near ink edges
        noise = self._np_rng.normal(0, noise_scale, arr.shape[:2])

        # Find edge pixels (where alpha changes rapidly)
        alpha = arr[:, :, 3]
        edge_mask = alpha > 10

        for c in range(4):
            channel = arr[:, :, c]
            # Apply noise more strongly to ink pixels
            channel[edge_mask] += noise[edge_mask] * (0.5 if c == 3 else 1.0)
            arr[:, :, c] = channel

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")

    def _apply_speed_smoothing(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """Apply smoothing for faster writing (older age groups).

        Higher speed factor leads to slight Gaussian blur, simulating
        the fluid motion of fast writing.
        """
        speed = self._profile.speed_factor
        if speed < 0.8:
            return image

        blur_radius = (speed - 0.8) * 0.8  # 0.0 at speed=0.8, 0.4 at speed=1.3
        if blur_radius < 0.1:
            return image

        # Only blur the ink channels
        arr = np.array(image.convert("RGBA"), dtype=np.float32)
        ink_mask = arr[:, :, 3] > 10

        if not np.any(ink_mask):
            return image

        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred_arr = np.array(blurred.convert("RGBA"), dtype=np.float32)

        # Blend: use blurred version for ink, keep original for background
        result = arr.copy()
        result[ink_mask] = blurred_arr[ink_mask]

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGBA")


__all__ = ["SkillSimulator"]
