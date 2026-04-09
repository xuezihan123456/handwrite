import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from scripts.evaluate import (
    evaluate_model,
    save_complex_chars,
    save_style_comparison,
)


class EchoStyleGenerator(nn.Module):
    def forward(self, images: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        style_bias = style_ids.view(-1, 1, 1, 1).float() / 10.0
        return images + style_bias


class AddConstantGenerator(nn.Module):
    def __init__(self, bias: float) -> None:
        super().__init__()
        self.bias = bias

    def forward(self, images: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        del style_ids
        return images + self.bias


class InvertGenerator(nn.Module):
    def forward(self, images: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        del style_ids
        return 1.0 - images


def _build_batches(
    standard_images: torch.Tensor,
    handwrite_images: torch.Tensor,
    style_ids: torch.Tensor,
    batch_sizes: list[int],
) -> list[dict[str, torch.Tensor]]:
    assert sum(batch_sizes) == int(style_ids.shape[0])

    batches: list[dict[str, torch.Tensor]] = []
    start = 0
    for batch_size in batch_sizes:
        end = start + batch_size
        batches.append(
            {
                "standard": standard_images[start:end],
                "handwrite": handwrite_images[start:end],
                "style_id": style_ids[start:end],
            }
        )
        start = end
    return batches


def _render_test_char(char: str, style: str) -> Image.Image:
    fill_value = min(32 + (len(style) * 15), 224)
    inset = 6 + (ord(char[0]) % 10)
    image = Image.new("L", (48, 48), color=255)
    image.paste(fill_value, (inset, inset, 48 - inset, 48 - inset))
    return image


def test_evaluate_model_returns_expected_metrics_with_finite_values() -> None:
    generator = EchoStyleGenerator()
    standard_images = torch.stack(
        [
            torch.zeros(1, 8, 8),
            torch.full((1, 8, 8), 0.25),
            torch.full((1, 8, 8), 0.5),
        ]
    )
    style_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    handwrite_images = standard_images + (style_ids.view(-1, 1, 1, 1).float() / 10.0)
    dataloader = [
        {
            "standard": standard_images,
            "handwrite": handwrite_images,
            "style_id": style_ids,
            "char": ["A", "B", "C"],
        }
    ]

    metrics = evaluate_model(generator, dataloader, device="cpu")

    assert set(metrics) == {"l1_loss", "fid_score", "style_accuracy"}
    assert metrics["l1_loss"] == pytest.approx(0.0)
    assert metrics["style_accuracy"] == pytest.approx(1.0)
    assert isinstance(metrics["fid_score"], float)
    assert math.isfinite(metrics["fid_score"])
    assert metrics["fid_score"] == pytest.approx(0.0)


def test_evaluate_model_is_stable_across_batch_groupings_without_discriminator() -> None:
    generator = AddConstantGenerator(0.7)
    standard_images = torch.zeros(4, 1, 2, 2)
    handwrite_images = (
        torch.tensor([0.0, 0.0, 1.0, 2.0], dtype=torch.float32)
        .view(-1, 1, 1, 1)
        .expand(-1, 1, 2, 2)
        .clone()
    )
    style_ids = torch.arange(4, dtype=torch.long)

    single_batch_metrics = evaluate_model(
        generator,
        _build_batches(standard_images, handwrite_images, style_ids, [4]),
        device="cpu",
    )
    paired_batch_metrics = evaluate_model(
        generator,
        _build_batches(standard_images, handwrite_images, style_ids, [2, 2]),
        device="cpu",
    )
    single_item_batch_metrics = evaluate_model(
        generator,
        _build_batches(standard_images, handwrite_images, style_ids, [1, 1, 1, 1]),
        device="cpu",
    )

    for metrics in (paired_batch_metrics, single_item_batch_metrics):
        assert metrics["l1_loss"] == pytest.approx(single_batch_metrics["l1_loss"])
        assert metrics["fid_score"] == pytest.approx(single_batch_metrics["fid_score"])
        assert metrics["style_accuracy"] == pytest.approx(single_batch_metrics["style_accuracy"])


def test_evaluate_model_uses_dataset_wide_style_proxy_without_discriminator() -> None:
    generator = InvertGenerator()
    standard_images = torch.stack(
        [
            torch.zeros(1, 4, 4),
            torch.ones(1, 4, 4),
        ]
    )
    style_ids = torch.tensor([0, 1], dtype=torch.long)
    handwrite_images = standard_images.clone()

    metrics = evaluate_model(
        generator,
        _build_batches(standard_images, handwrite_images, style_ids, [1, 1]),
        device="cpu",
    )

    assert metrics["style_accuracy"] == pytest.approx(0.0)


def test_save_style_comparison_writes_image_file(tmp_path: Path) -> None:
    output_path = tmp_path / "evaluation" / "style_comparison.png"

    saved_path = save_style_comparison(
        output_path=output_path,
        render_char_fn=_render_test_char,
        styles=["style-a", "style-b"],
        text="AB",
    )

    assert saved_path == output_path
    assert output_path.exists()
    with Image.open(output_path) as image:
        assert image.mode == "L"
        assert image.size[0] > 0
        assert image.size[1] > 0
        minimum, maximum = image.getextrema()
        assert minimum < 255
        assert minimum < maximum


def test_save_complex_chars_writes_image_file(tmp_path: Path) -> None:
    output_path = tmp_path / "evaluation" / "complex_chars.png"

    saved_path = save_complex_chars(
        output_path=output_path,
        render_char_fn=_render_test_char,
        styles=["style-a", "style-b", "style-c"],
        chars=["A", "B"],
    )

    assert saved_path == output_path
    assert output_path.exists()
    with Image.open(output_path) as image:
        assert image.mode == "L"
        assert image.size[0] > 0
        assert image.size[1] > 0
        minimum, maximum = image.getextrema()
        assert minimum < 255
        assert minimum < maximum


@pytest.mark.parametrize(
    ("pixels", "expected"),
    [
        (
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            np.array([[0, 85], [170, 255]], dtype=np.uint8),
        ),
        (
            np.array([[-4.0, -3.0], [-2.0, -1.0]], dtype=np.float32),
            np.array([[0, 85], [170, 255]], dtype=np.uint8),
        ),
    ],
)
def test_save_style_comparison_normalizes_numpy_images_using_true_extrema(
    tmp_path: Path,
    pixels: np.ndarray,
    expected: np.ndarray,
) -> None:
    output_path = tmp_path / "evaluation" / "numpy_style_comparison.png"

    def render_numpy_char(char: str, style: str) -> np.ndarray:
        del char, style
        return pixels

    save_style_comparison(
        output_path=output_path,
        render_char_fn=render_numpy_char,
        styles=["style-a"],
        text="A",
    )

    with Image.open(output_path) as image:
        cropped = np.array(image.crop((12, 12, 14, 14)))

    assert np.array_equal(cropped, expected)
