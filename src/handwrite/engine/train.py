"""Training helpers for the handwriting engine."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn


def train_one_epoch(
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: Iterable[Any],
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
    epoch: int = 1,
    output_dir: str | Path | None = None,
    l1_weight: float = 100.0,
    sample_interval: int = 0,
    adversarial_loss_fn: nn.Module | None = None,
    reconstruction_loss_fn: nn.Module | None = None,
    category_loss_fn: nn.Module | None = None,
    label_shuffle_enabled: bool = False,
) -> dict[str, float]:
    """Run one training epoch for the handwriting GAN."""

    if epoch < 1:
        raise ValueError("epoch must be at least 1")
    if l1_weight < 0:
        raise ValueError("l1_weight must be non-negative")
    if sample_interval > 0 and output_dir is None:
        raise ValueError("output_dir is required when sample_interval is enabled")

    device = torch.device(device)
    generator = generator.to(device=device)
    discriminator = discriminator.to(device=device)
    generator.train()
    discriminator.train()

    adversarial_loss = adversarial_loss_fn or nn.BCEWithLogitsLoss()
    reconstruction_loss = reconstruction_loss_fn or nn.L1Loss()
    category_loss = category_loss_fn or nn.CrossEntropyLoss()

    metrics_total = {
        "G_loss": 0.0,
        "D_loss": 0.0,
        "L1_loss": 0.0,
    }
    sample_count = 0
    sample_root = None if output_dir is None else Path(output_dir) / "samples"

    for batch_index, batch in enumerate(dataloader, start=1):
        standard_images, real_images, style_ids = _unpack_batch(batch, device)
        batch_size = int(style_ids.shape[0])
        fake_images = generator(standard_images, style_ids)

        discriminator_optimizer.zero_grad(set_to_none=True)
        real_score, real_category = discriminator(real_images)
        fake_score, _ = discriminator(fake_images.detach())

        real_targets = torch.ones_like(real_score)
        fake_targets = torch.zeros_like(fake_score)
        style_targets = _expand_style_targets(style_ids, real_category.shape)

        d_real_loss = adversarial_loss(real_score, real_targets)
        d_fake_loss = adversarial_loss(fake_score, fake_targets)
        d_style_loss = category_loss(real_category, style_targets)
        d_loss = d_real_loss + d_fake_loss + d_style_loss
        d_loss.backward()
        discriminator_optimizer.step()

        generator_optimizer.zero_grad(set_to_none=True)
        fake_score, fake_category = discriminator(fake_images)

        generator_targets = torch.ones_like(fake_score)
        g_adv_loss = adversarial_loss(fake_score, generator_targets)
        l1_loss = reconstruction_loss(fake_images, real_images)
        g_style_loss = category_loss(
            fake_category,
            _expand_style_targets(style_ids, fake_category.shape),
        )
        g_loss = g_adv_loss + (l1_weight * l1_loss) + g_style_loss

        if label_shuffle_enabled and style_ids.shape[0] > 1:
            shuffled_style_ids = _shuffle_style_ids(style_ids)
            shuffled_fake_images = generator(standard_images, shuffled_style_ids)
            shuffled_score, shuffled_category = discriminator(shuffled_fake_images)
            shuffled_style_targets = _expand_style_targets(
                shuffled_style_ids,
                shuffled_category.shape,
            )
            g_shuffle_adv_loss = adversarial_loss(shuffled_score, generator_targets)
            g_shuffle_style_loss = category_loss(
                shuffled_category,
                shuffled_style_targets,
            )
            g_loss = g_loss + g_shuffle_adv_loss + g_shuffle_style_loss

        g_loss.backward()
        generator_optimizer.step()

        metrics_total["G_loss"] += float(g_loss.item()) * batch_size
        metrics_total["D_loss"] += float(d_loss.item()) * batch_size
        metrics_total["L1_loss"] += float(l1_loss.item()) * batch_size
        sample_count += batch_size

        if sample_interval > 0 and batch_index % sample_interval == 0:
            save_sample_grid(
                standard_images=standard_images,
                generated_images=fake_images,
                real_images=real_images,
                output_path=sample_root / f"epoch{epoch}_batch{batch_index}.png",
            )

    if sample_count == 0:
        raise ValueError("dataloader produced no batches")

    return {
        name: total / sample_count
        for name, total in metrics_total.items()
    }


def fit(
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: Iterable[Any],
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    output_dir: str | Path,
    epochs: int,
    device: str | torch.device = "cpu",
    l1_weight: float = 100.0,
    label_shuffle_start: int | None = None,
    checkpoint_interval: int = 1,
    sample_interval: int = 0,
    adversarial_loss_fn: nn.Module | None = None,
    reconstruction_loss_fn: nn.Module | None = None,
    category_loss_fn: nn.Module | None = None,
) -> list[dict[str, float]]:
    """Run full training and write checkpoints, samples, and CSV metrics."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        label_shuffle_enabled = (
            label_shuffle_start is not None and epoch >= label_shuffle_start
        )
        epoch_metrics = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloader,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            device=device,
            epoch=epoch,
            output_dir=output_path,
            l1_weight=l1_weight,
            sample_interval=sample_interval,
            adversarial_loss_fn=adversarial_loss_fn,
            reconstruction_loss_fn=reconstruction_loss_fn,
            category_loss_fn=category_loss_fn,
            label_shuffle_enabled=label_shuffle_enabled,
        )
        epoch_record = {"epoch": float(epoch), **epoch_metrics}
        history.append(epoch_record)
        _write_training_log(output_path / "train_log.csv", history)

        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            save_checkpoint(
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                epoch=epoch,
                output_dir=output_path,
                metrics=epoch_metrics,
            )

    return history


def save_checkpoint(
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: str | Path,
    metrics: Mapping[str, float] | None = None,
) -> Path:
    """Save a training checkpoint for the current epoch."""

    if epoch < 1:
        raise ValueError("epoch must be at least 1")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / f"checkpoint_epoch_{epoch}.pt"
    torch.save(
        {
            "epoch": epoch,
            "metrics": dict(metrics or {}),
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "generator_optimizer_state_dict": generator_optimizer.state_dict(),
            "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_sample_grid(
    *,
    standard_images: Tensor,
    generated_images: Tensor,
    real_images: Tensor,
    output_path: str | Path,
    max_items: int = 4,
) -> Path:
    """Save a simple comparison grid of standard, generated, and real images."""

    if max_items < 1:
        raise ValueError("max_items must be at least 1")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    item_count = min(
        max_items,
        standard_images.shape[0],
        generated_images.shape[0],
        real_images.shape[0],
    )
    if item_count == 0:
        raise ValueError("at least one image is required to save a sample grid")

    rows: list[np.ndarray] = []
    for index in range(item_count):
        tiles = [
            _tensor_to_grayscale_array(standard_images[index]),
            _tensor_to_grayscale_array(generated_images[index]),
            _tensor_to_grayscale_array(real_images[index]),
        ]
        rows.append(np.concatenate(tiles, axis=1))

    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid, mode="L").save(output_file)
    return output_file


def _write_training_log(log_path: Path, history: list[dict[str, float]]) -> None:
    fieldnames = ["epoch", "G_loss", "D_loss", "L1_loss"]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": int(row["epoch"]),
                    "G_loss": row["G_loss"],
                    "D_loss": row["D_loss"],
                    "L1_loss": row["L1_loss"],
                }
            )


def _unpack_batch(batch: Any, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(batch, Mapping):
        standard_images = _move_tensor(
            _get_first_mapping_value(
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
            _get_first_mapping_value(
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
            _get_first_mapping_value(batch, "style_ids", "style_id"),
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
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value.to(device=device, dtype=dtype)


def _get_first_mapping_value(batch: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in batch:
            return batch[key]
    return None


def _expand_style_targets(style_ids: Tensor, category_shape: torch.Size) -> Tensor:
    if style_ids.ndim != 1:
        raise ValueError("style_ids must have shape (batch,)")
    if len(category_shape) < 3:
        raise ValueError("category output must include spatial dimensions")

    spatial_shape = category_shape[2:]
    target_shape = (style_ids.shape[0], *([1] * len(spatial_shape)))
    expanded = style_ids.view(*target_shape).expand(style_ids.shape[0], *spatial_shape)
    return expanded.long()


def _shuffle_style_ids(style_ids: Tensor) -> Tensor:
    if style_ids.ndim != 1:
        raise ValueError("style_ids must have shape (batch,)")
    if style_ids.shape[0] < 2:
        return style_ids
    permutation = torch.randperm(style_ids.shape[0], device=style_ids.device)
    return style_ids.index_select(0, permutation)


def _tensor_to_grayscale_array(image: Tensor) -> np.ndarray:
    tensor = image.detach().cpu().float()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    elif tensor.ndim == 3 and tensor.shape[0] == 3:
        tensor = tensor.mean(dim=0)
    elif tensor.ndim != 2:
        raise ValueError("sample images must have shape (H, W), (1, H, W), or (3, H, W)")

    minimum = float(tensor.min().item())
    maximum = float(tensor.max().item())

    if minimum >= 0.0 and maximum <= 1.0:
        normalized = tensor
    elif minimum >= -1.0 and maximum <= 1.0:
        normalized = (tensor + 1.0) / 2.0
    elif maximum - minimum < 1e-8:
        normalized = torch.full_like(tensor, 0.5)
    else:
        normalized = (tensor - minimum) / (maximum - minimum)

    normalized = normalized.clamp(0.0, 1.0)
    return (normalized.mul(255).round().to(torch.uint8).numpy())


__all__ = [
    "fit",
    "save_checkpoint",
    "save_sample_grid",
    "train_one_epoch",
]
