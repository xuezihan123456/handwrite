"""CLI wrapper for handwriting model training."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from torch.optim import Adam
from torch.utils.data import DataLoader

from handwrite.data.dataset import HandwriteDataset
from handwrite.engine import train as engine_train
from handwrite.engine.discriminator import Discriminator
from handwrite.engine.generator import Generator


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the handwriting generation models.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing the processed dataset metadata.",
    )
    parser.add_argument(
        "--styles_file",
        type=Path,
        required=True,
        help="JSON file containing the selected styles configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for checkpoints, samples, and training logs.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--l1_weight", type=float, default=100)
    parser.add_argument("--label_shuffle_start", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=2)
    parser.add_argument("--sample_interval", type=int, default=500)
    return parser


def _resolve_training_entrypoint(module: Any) -> Callable[..., Any]:
    for attribute_name in ("fit", "train", "run", "main"):
        candidate = getattr(module, attribute_name, None)
        if callable(candidate):
            return candidate

    raise RuntimeError(
        "handwrite.engine.train does not expose a callable training entrypoint."
    )


def _infer_num_styles(dataset: Any) -> int:
    style_ids: list[int] = []

    style_by_writer = getattr(dataset, "style_by_writer", None)
    if isinstance(style_by_writer, dict):
        style_ids.extend(int(style_id) for style_id in style_by_writer.values())

    if not style_ids:
        samples = getattr(dataset, "samples", None)
        if isinstance(samples, list):
            for sample in samples:
                if not isinstance(sample, dict) or "style_id" not in sample:
                    continue
                style_ids.append(int(sample["style_id"]))

    if not style_ids:
        return 5

    return max(style_ids) + 1


def main(argv: list[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = HandwriteDataset(
        metadata_path=args.data_dir / "metadata.json",
        selected_styles_path=args.styles_file,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    num_styles = _infer_num_styles(dataset)
    generator = Generator(num_styles=num_styles)
    discriminator = Discriminator(num_styles=num_styles)
    generator_optimizer = Adam(generator.parameters(), lr=args.lr)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=args.lr)

    fit = _resolve_training_entrypoint(engine_train)
    fit(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        output_dir=args.output_dir,
        epochs=args.epochs,
        l1_weight=args.l1_weight,
        label_shuffle_start=args.label_shuffle_start,
        checkpoint_interval=args.save_interval,
        sample_interval=args.sample_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
