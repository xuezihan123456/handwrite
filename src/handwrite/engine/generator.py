"""Generator network definition."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _EncoderBlock(nn.Module):
    """Downsampling block used by the zi2zi-style U-Net encoder."""

    def __init__(self, in_channels: int, out_channels: int, *, use_norm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not use_norm,
            )
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class _DecoderBlock(nn.Module):
    """Upsampling block used by the zi2zi-style U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_norm: bool = True,
        dropout: float = 0.0,
        final_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not use_norm,
            )
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(final_activation if final_activation is not None else nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class Generator(nn.Module):
    """zi2zi-style U-Net generator with bottleneck style conditioning."""

    def __init__(self, num_styles: int = 5, embed_dim: int = 128) -> None:
        super().__init__()
        self.style_embedding = nn.Embedding(num_styles, embed_dim)

        self.e1 = _EncoderBlock(1, 64, use_norm=False)
        self.e2 = _EncoderBlock(64, 128, use_norm=True)
        self.e3 = _EncoderBlock(128, 256, use_norm=True)
        self.e4 = _EncoderBlock(256, 512, use_norm=True)
        self.e5 = _EncoderBlock(512, 512, use_norm=True)
        self.e6 = _EncoderBlock(512, 512, use_norm=True)
        self.e7 = _EncoderBlock(512, 512, use_norm=True)
        self.e8 = _EncoderBlock(512, 512, use_norm=False)

        self.d1 = _DecoderBlock(512 + embed_dim, 512, dropout=0.5)
        self.d2 = _DecoderBlock(1024, 512, dropout=0.5)
        self.d3 = _DecoderBlock(1024, 512, dropout=0.5)
        self.d4 = _DecoderBlock(1024, 512)
        self.d5 = _DecoderBlock(1024, 256)
        self.d6 = _DecoderBlock(512, 128)
        self.d7 = _DecoderBlock(256, 64)
        self.d8 = _DecoderBlock(128, 1, use_norm=False, final_activation=nn.Tanh())

    @staticmethod
    def _validate_inputs(images: Tensor, style_ids: Tensor) -> None:
        if images.ndim != 4 or images.shape[1:] != (1, 256, 256):
            raise ValueError("images must have shape (batch, 1, 256, 256)")
        if style_ids.ndim != 1 or style_ids.shape[0] != images.shape[0]:
            raise ValueError("style_ids must have shape (batch,) matching the image batch size")

    def forward(self, images: Tensor, style_ids: Tensor) -> Tensor:
        self._validate_inputs(images, style_ids)

        e1 = self.e1(images)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        style_embedding = self.style_embedding(style_ids).view(style_ids.shape[0], -1, 1, 1)
        bottleneck = torch.cat([e8, style_embedding], dim=1)

        d1 = self.d1(bottleneck)
        d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1))
        d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1))
        d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))
        return self.d8(torch.cat([d7, e1], dim=1))
