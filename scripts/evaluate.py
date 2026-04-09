"""Practical evaluation helpers for the handwriting MVP."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def evaluate_model(
    generator: nn.Module,
    dataloader: Iterable[Any],
    device: str | torch.device = "cpu",
    discriminator: nn.Module | None = None,
) -> dict[str, float]:
    """Evaluate a generator with lightweight quantitative metrics.

    The returned ``fid_score`` is a Frechet-style approximation computed on
    pooled grayscale features. It is intentionally lightweight so the repo can
    report a practical score before a full FID dependency and checkpoint flow
    are in place. Discriminator-free ``style_accuracy`` falls back to the
    nearest real sample over the full evaluation dataset.
    """

    device = torch.device(device)
    generator = generator.to(device=device)
    discriminator = None if discriminator is None else discriminator.to(device=device)

    generator_was_training = generator.training
    discriminator_was_training = None if discriminator is None else discriminator.training
    generator.eval()
    if discriminator is not None:
        discriminator.eval()

    sample_count = 0
    all_real_images: list[Tensor] = []
    all_fake_images: list[Tensor] = []
    all_style_ids: list[Tensor] = []

    try:
        with torch.no_grad():
            for batch in dataloader:
                standard_images, real_images, style_ids = _unpack_batch(batch, device)
                fake_images = generator(standard_images, style_ids)
                batch_size = int(style_ids.shape[0])

                sample_count += batch_size

                all_real_images.append(real_images.detach().cpu())
                all_fake_images.append(fake_images.detach().cpu())
                all_style_ids.append(style_ids.detach().cpu())
    finally:
        if generator_was_training:
            generator.train()
        if discriminator is not None and discriminator_was_training:
            discriminator.train()

    if sample_count == 0:
        raise ValueError("dataloader produced no batches")

    real_images = torch.cat(all_real_images, dim=0)
    fake_images = torch.cat(all_fake_images, dim=0)
    style_ids = torch.cat(all_style_ids, dim=0)
    l1_loss = float(F.l1_loss(fake_images, real_images, reduction="mean").item())

    fid_score = _frechet_distance(
        _extract_feature_vectors(real_images),
        _extract_feature_vectors(fake_images),
    )

    if discriminator is None:
        predicted_style_ids = _predict_style_ids(
            fake_images=fake_images,
            real_images=real_images,
            style_ids=style_ids,
            discriminator=None,
        )
    else:
        predicted_style_ids = _predict_style_ids(
            fake_images=fake_images.to(device=device),
            real_images=real_images,
            style_ids=style_ids,
            discriminator=discriminator,
        ).cpu()

    return {
        "l1_loss": l1_loss,
        "fid_score": fid_score,
        "style_accuracy": float((predicted_style_ids == style_ids).sum().item()) / sample_count,
    }


def save_style_comparison(
    output_path: str | Path,
    render_char_fn: Callable[[str, str], Any],
    styles: Sequence[str],
    text: str,
) -> Path:
    """Render a style-by-character comparison grid."""

    chars = [char for char in text if not char.isspace()]
    return _save_render_grid(output_path, render_char_fn, styles, chars)


def save_complex_chars(
    output_path: str | Path,
    render_char_fn: Callable[[str, str], Any],
    styles: Sequence[str],
    chars: Sequence[str],
) -> Path:
    """Render a comparison grid for a curated list of complex characters."""

    filtered_chars = [char for char in chars if char and not char.isspace()]
    return _save_render_grid(output_path, render_char_fn, styles, filtered_chars)


def _save_render_grid(
    output_path: str | Path,
    render_char_fn: Callable[[str, str], Any],
    styles: Sequence[str],
    chars: Sequence[str],
    *,
    padding: int = 12,
    gap: int = 8,
) -> Path:
    if not styles:
        raise ValueError("styles must not be empty")
    if not chars:
        raise ValueError("chars must not be empty")

    rendered_rows: list[list[Image.Image]] = []
    for style in styles:
        row = [_coerce_image(render_char_fn(char, style)) for char in chars]
        rendered_rows.append(row)

    cell_width = max(image.width for row in rendered_rows for image in row)
    cell_height = max(image.height for row in rendered_rows for image in row)
    row_count = len(rendered_rows)
    column_count = len(rendered_rows[0])
    canvas_width = (column_count * cell_width) + ((column_count - 1) * gap) + (padding * 2)
    canvas_height = (row_count * cell_height) + ((row_count - 1) * gap) + (padding * 2)

    canvas = Image.new("L", (canvas_width, canvas_height), color=255)
    for row_index, row in enumerate(rendered_rows):
        for column_index, image in enumerate(row):
            x = padding + column_index * (cell_width + gap) + ((cell_width - image.width) // 2)
            y = padding + row_index * (cell_height + gap) + ((cell_height - image.height) // 2)
            canvas.paste(image, (x, y))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file)
    return output_file


def _predict_style_ids(
    *,
    fake_images: Tensor,
    real_images: Tensor,
    style_ids: Tensor,
    discriminator: nn.Module | None,
) -> Tensor:
    if discriminator is not None:
        _, style_logits = discriminator(fake_images)
        return _style_ids_from_logits(style_logits)

    fake_vectors = fake_images.reshape(fake_images.shape[0], -1).float()
    real_vectors = real_images.reshape(real_images.shape[0], -1).float()
    distances = torch.cdist(fake_vectors, real_vectors, p=1)
    nearest_indices = distances.argmin(dim=1)
    return style_ids.index_select(0, nearest_indices)


def _style_ids_from_logits(style_logits: Tensor) -> Tensor:
    if style_logits.ndim == 2:
        pooled_logits = style_logits
    elif style_logits.ndim >= 3:
        pooled_logits = style_logits.flatten(start_dim=2).mean(dim=2)
    else:
        raise ValueError("style logits must have at least 2 dimensions")
    return pooled_logits.argmax(dim=1)


def _extract_feature_vectors(images: Tensor, output_size: int = 8) -> Tensor:
    normalized = _normalize_image_batch(images)
    grayscale = normalized.mean(dim=1, keepdim=True)
    pooled = F.adaptive_avg_pool2d(grayscale, output_size=(output_size, output_size))
    return pooled.flatten(start_dim=1).double()


def _normalize_image_batch(images: Tensor) -> Tensor:
    tensor = images.detach().float()
    minimum = float(tensor.min().item())
    maximum = float(tensor.max().item())

    if minimum >= 0.0 and maximum <= 1.0:
        normalized = tensor
    elif minimum >= -1.0 and maximum <= 1.0:
        normalized = (tensor + 1.0) / 2.0
    elif abs(maximum - minimum) < 1e-8:
        normalized = torch.full_like(tensor, 0.5)
    else:
        normalized = (tensor - minimum) / (maximum - minimum)

    return normalized.clamp(0.0, 1.0)


def _frechet_distance(real_features: Tensor, fake_features: Tensor) -> float:
    real_mean = real_features.mean(dim=0)
    fake_mean = fake_features.mean(dim=0)
    real_covariance = _covariance(real_features)
    fake_covariance = _covariance(fake_features)
    mean_term = torch.sum((real_mean - fake_mean) ** 2)
    real_sqrt = _matrix_sqrt(real_covariance)
    middle = real_sqrt @ fake_covariance @ real_sqrt
    covariance_term = torch.trace(
        real_covariance + fake_covariance - (2.0 * _matrix_sqrt(middle))
    )
    score = torch.clamp(mean_term + covariance_term, min=0.0)
    if float(score.item()) < 1e-6:
        return 0.0
    return float(score.item())


def _covariance(features: Tensor, eps: float = 1e-6) -> Tensor:
    feature_count, dimension_count = features.shape
    identity = torch.eye(dimension_count, dtype=features.dtype, device=features.device)

    if feature_count < 2:
        return identity * eps

    centered = features - features.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(feature_count - 1)
    return covariance + (identity * eps)


def _matrix_sqrt(matrix: Tensor) -> Tensor:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    clipped = torch.clamp(eigenvalues, min=0.0)
    root = torch.sqrt(clipped)
    return (eigenvectors * root.unsqueeze(0)) @ eigenvectors.T


def _coerce_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("L")

    if isinstance(value, torch.Tensor):
        array = _tensor_to_uint8_array(value)
        return Image.fromarray(array, mode="L")

    if isinstance(value, np.ndarray):
        return Image.fromarray(_array_to_uint8(value), mode="L")

    raise TypeError("render_char_fn must return a PIL image, torch tensor, or numpy array")


def _tensor_to_uint8_array(value: Tensor) -> np.ndarray:
    tensor = value.detach().cpu()
    if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
        tensor = tensor.mean(dim=0)
    elif tensor.ndim != 2:
        raise ValueError("tensor images must have shape (H, W), (1, H, W), or (3, H, W)")

    normalized = _normalize_image_batch(tensor.unsqueeze(0).unsqueeze(0))[0, 0]
    return normalized.mul(255).round().to(torch.uint8).numpy()


def _array_to_uint8(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3:
        array = array.mean(axis=0 if array.shape[0] in {1, 3} else 2)
    if array.ndim != 2:
        raise ValueError("numpy images must be 2D grayscale arrays")

    array = array.astype(np.float32, copy=False)
    minimum = float(array.min())
    maximum = float(array.max())

    if minimum >= 0.0 and maximum <= 1.0:
        normalized = array
    elif minimum >= -1.0 and maximum <= 1.0:
        normalized = (array + 1.0) / 2.0
    elif abs(maximum - minimum) < 1e-8:
        normalized = np.full_like(array, 0.5)
    else:
        normalized = (array - minimum) / (maximum - minimum)

    return np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)


def _unpack_batch(batch: Any, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(batch, Mapping):
        standard_images = _move_tensor(
            _first_mapping_value(
                batch,
                "standard",
                "standard_images",
                "standard_image",
                "source_images",
                "source",
            ),
            device=device,
            dtype=torch.float32,
            name="standard_images",
        )
        real_images = _move_tensor(
            _first_mapping_value(
                batch,
                "handwrite",
                "real_images",
                "real_image",
                "target_images",
                "target",
            ),
            device=device,
            dtype=torch.float32,
            name="real_images",
        )
        style_ids = _move_tensor(
            _first_mapping_value(batch, "style_ids", "style_id"),
            device=device,
            dtype=torch.long,
            name="style_ids",
        )
        return standard_images, real_images, style_ids

    if not isinstance(batch, (list, tuple)) or len(batch) != 3:
        raise TypeError("batch must be a mapping or a 3-item tuple/list")

    standard_images = _move_tensor(batch[0], device=device, dtype=torch.float32, name="standard_images")
    real_images = _move_tensor(batch[1], device=device, dtype=torch.float32, name="real_images")
    style_ids = _move_tensor(batch[2], device=device, dtype=torch.long, name="style_ids")
    return standard_images, real_images, style_ids


def _move_tensor(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    if value is None:
        raise TypeError(f"{name} is required")
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(device=device, dtype=dtype)


def _first_mapping_value(batch: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in batch:
            return batch[key]
    return None


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate qualitative handwriting evaluation artifacts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation"),
        help="Directory where the artifact images should be written.",
    )
    parser.add_argument(
        "--text",
        default="永和九年岁在癸丑暮春之初会于山阴之兰亭也",
        help="Sample text used for the style comparison grid.",
    )
    parser.add_argument(
        "--chars",
        nargs="*",
        default=["藏", "疆", "馨", "鬱"],
        help="Complex characters used for the complex-character grid.",
    )
    parser.add_argument(
        "--styles",
        nargs="*",
        default=None,
        help="Optional explicit style list. Defaults to handwrite.list_styles().",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    import handwrite

    styles = args.styles or handwrite.list_styles()
    output_dir = args.output_dir
    save_style_comparison(
        output_path=output_dir / "style_comparison.png",
        render_char_fn=handwrite.char,
        styles=styles,
        text=args.text,
    )
    save_complex_chars(
        output_path=output_dir / "complex_chars.png",
        render_char_fn=handwrite.char,
        styles=styles,
        chars=args.chars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
