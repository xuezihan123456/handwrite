"""PatchGAN discriminator network definition."""

from __future__ import annotations

from torch import Tensor, nn


class Discriminator(nn.Module):
    """PatchGAN discriminator with a real/fake head and a style head."""

    def __init__(self, num_styles: int = 5) -> None:
        super().__init__()

        if num_styles < 1:
            raise ValueError("num_styles must be at least 1")

        self.c1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.c2 = self._block(64, 128, stride=2)
        self.c3 = self._block(128, 256, stride=2)
        self.c4 = self._block(256, 512, stride=1)

        self.real_fake = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.category = nn.Conv2d(
            512,
            num_styles,
            kernel_size=4,
            stride=1,
            padding=1,
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.c1(inputs)
        features = self.c2(features)
        features = self.c3(features)
        features = self.c4(features)

        return self.real_fake(features), self.category(features)
