"""Small conditional UNet for the diffusion scaffold.

The UNet predicts noise given the noisy image, an optional ControlNet-style
condition image (the standard-font reference), and a timestep.  The design is
intentionally minimal so unit tests run on CPU in milliseconds.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["UNet", "SinusoidalTimestepEmbedding"]


class SinusoidalTimestepEmbedding(nn.Module):
    """Standard transformer-style sinusoidal timestep embedding."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be >= 2")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape (batch,)")
        half_dim = self.embedding_dim // 2
        frequency = math.log(10_000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
            * -frequency
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = nn.functional.pad(embedding, (0, 1))
        return embedding


class _ResidualBlock(nn.Module):
    """Conv block with GroupNorm + SiLU + a timestep bias added in the middle."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        timestep_dim: int,
        num_groups: int = 4,
    ) -> None:
        super().__init__()
        groups = min(num_groups, in_channels)
        if in_channels % groups != 0:
            groups = 1
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.activation1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.timestep_projection = nn.Linear(timestep_dim, out_channels)

        out_groups = min(num_groups, out_channels)
        if out_channels % out_groups != 0:
            out_groups = 1
        self.norm2 = nn.GroupNorm(out_groups, out_channels)
        self.activation2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: Tensor, timestep_embedding: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        hidden = self.conv1(self.activation1(self.norm1(inputs)))
        timestep_bias = self.timestep_projection(timestep_embedding)
        hidden = hidden + timestep_bias.unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(self.activation2(self.norm2(hidden)))
        return hidden + residual


class UNet(nn.Module):
    """Tiny conditional UNet noise predictor.

    Accepts the noisy image ``(B, 1, H, W)``, an optional condition image
    ``(B, 1, H, W)`` (the standard-font reference), and a timestep tensor
    ``(B,)``.  When ``condition_channels=1`` the condition is concatenated to
    the noisy input (ControlNet-lite style).
    """

    def __init__(
        self,
        base_channels: int = 32,
        condition_channels: int = 1,
        timestep_dim: int = 64,
    ) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be at least 1")
        if condition_channels < 0:
            raise ValueError("condition_channels must be non-negative")
        if timestep_dim < 2:
            raise ValueError("timestep_dim must be at least 2")

        self.base_channels = base_channels
        self.condition_channels = condition_channels
        self.timestep_dim = timestep_dim

        self.timestep_embedding = SinusoidalTimestepEmbedding(timestep_dim)
        self.timestep_projection = nn.Sequential(
            nn.Linear(timestep_dim, timestep_dim),
            nn.SiLU(),
            nn.Linear(timestep_dim, timestep_dim),
        )

        input_channels = 1 + condition_channels
        self.input_projection = nn.Conv2d(
            input_channels, base_channels, kernel_size=3, padding=1
        )

        # Downsampling path: 2 stages
        self.down1 = _ResidualBlock(base_channels, base_channels, timestep_dim)
        self.downsample1 = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1
        )
        self.down2 = _ResidualBlock(base_channels * 2, base_channels * 2, timestep_dim)

        # Bottleneck
        self.middle = _ResidualBlock(
            base_channels * 2, base_channels * 2, timestep_dim
        )

        # Upsampling path
        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.up1 = _ResidualBlock(base_channels * 2, base_channels, timestep_dim)

        self.output_norm = nn.GroupNorm(
            min(4, base_channels) if base_channels % min(4, base_channels) == 0 else 1,
            base_channels,
        )
        self.output_activation = nn.SiLU()
        self.output_projection = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(
        self,
        noisy_images: Tensor,
        timesteps: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        self._validate_inputs(noisy_images, timesteps, condition)

        if self.condition_channels > 0:
            if condition is None:
                condition = torch.zeros_like(noisy_images)
            inputs = torch.cat([noisy_images, condition], dim=1)
        else:
            inputs = noisy_images

        timestep_embedding = self.timestep_projection(
            self.timestep_embedding(timesteps)
        )

        hidden = self.input_projection(inputs)
        skip = self.down1(hidden, timestep_embedding)

        downsampled = self.downsample1(skip)
        downsampled = self.down2(downsampled, timestep_embedding)

        middle = self.middle(downsampled, timestep_embedding)

        upsampled = self.upsample1(middle)
        # Pad spatial dims to match skip if odd input sizes were used.
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = nn.functional.interpolate(
                upsampled, size=skip.shape[-2:], mode="nearest"
            )
        hidden_up = torch.cat([upsampled, skip], dim=1)
        hidden_up = self.up1(hidden_up, timestep_embedding)

        output = self.output_projection(
            self.output_activation(self.output_norm(hidden_up))
        )
        return output

    def _validate_inputs(
        self,
        noisy_images: Tensor,
        timesteps: Tensor,
        condition: Tensor | None,
    ) -> None:
        if noisy_images.ndim != 4 or noisy_images.shape[1] != 1:
            raise ValueError("noisy_images must have shape (batch, 1, H, W)")
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy_images.shape[0]:
            raise ValueError(
                "timesteps must have shape (batch,) matching the image batch"
            )
        if condition is not None:
            if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
                raise ValueError(
                    "condition must have shape (batch, condition_channels, H, W)"
                )
            if condition.shape[0] != noisy_images.shape[0]:
                raise ValueError(
                    "condition batch size must match noisy_images batch size"
                )
            if condition.shape[-2:] != noisy_images.shape[-2:]:
                raise ValueError(
                    "condition spatial dimensions must match noisy_images"
                )
