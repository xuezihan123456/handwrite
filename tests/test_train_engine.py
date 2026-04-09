import csv
import math

import pytest
from PIL import Image
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import handwrite.engine.train as train_module
from handwrite.engine.train import fit, train_one_epoch


class TinyGenerator(nn.Module):
    def __init__(self, num_styles: int = 3) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, 1, kernel_size=1)
        self.style_embedding = nn.Embedding(num_styles, 1)

    def forward(self, images: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        style_bias = self.style_embedding(style_ids).view(-1, 1, 1, 1)
        return torch.tanh(self.projection(images) + style_bias)


class TinyDiscriminator(nn.Module):
    def __init__(self, num_styles: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.real_fake = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.category = nn.Conv2d(4, num_styles, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(inputs)
        return self.real_fake(features), self.category(features)


class DeviceTrackingGenerator(TinyGenerator):
    def __init__(self, num_styles: int = 3) -> None:
        super().__init__(num_styles=num_styles)
        self.moved_to: list[torch.device] = []

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args:
            device = args[0]
        if device is not None:
            self.moved_to.append(torch.device(device))
        return super().to(*args, **kwargs)


class DeviceTrackingDiscriminator(TinyDiscriminator):
    def __init__(self, num_styles: int = 3) -> None:
        super().__init__(num_styles=num_styles)
        self.moved_to: list[torch.device] = []

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args:
            device = args[0]
        if device is not None:
            self.moved_to.append(torch.device(device))
        return super().to(*args, **kwargs)


class StyleRecordingGenerator(TinyGenerator):
    def __init__(self, num_styles: int = 3) -> None:
        super().__init__(num_styles=num_styles)
        self.style_calls: list[tuple[int, ...]] = []

    def forward(self, images: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        self.style_calls.append(tuple(int(style_id) for style_id in style_ids.detach().cpu().tolist()))
        return super().forward(images, style_ids)


class SideEffectGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, images: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        del style_ids
        self.forward_calls += 1
        offset = torch.full_like(images, float(self.forward_calls))
        return (images * self.scale) + offset


class RecordingDiscriminator(nn.Module):
    def __init__(self, num_styles: int = 3) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.num_styles = num_styles
        self.input_means: list[float] = []

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.input_means.append(float(inputs.mean().item()))
        score = inputs.mean(dim=(1, 2, 3), keepdim=True) + self.bias
        category = score.repeat(1, self.num_styles, 1, 1)
        return score, category


class AveragingGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        del style_ids
        return images * self.scale


class AveragingDiscriminator(nn.Module):
    def __init__(self, num_styles: int = 3) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.num_styles = num_styles

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = inputs + self.bias
        category = inputs.repeat(1, self.num_styles, 1, 1) + self.bias
        return score, category


class MeanPredictionLoss(nn.Module):
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        del targets
        return predictions.mean()


class MeanReconstructionLoss(nn.Module):
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        del targets
        return predictions.mean()


def _make_training_parts(tmp_path):
    torch.manual_seed(0)

    num_samples = 4
    image_size = 8
    num_styles = 3

    standard_images = torch.linspace(
        -1.0,
        1.0,
        steps=num_samples * image_size * image_size,
    ).reshape(num_samples, 1, image_size, image_size)
    style_ids = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    style_offsets = style_ids.float().view(-1, 1, 1, 1) / num_styles
    real_images = torch.tanh(standard_images * 0.5 + style_offsets)

    dataloader = DataLoader(
        TensorDataset(standard_images, real_images, style_ids),
        batch_size=2,
        shuffle=False,
    )

    generator = TinyGenerator(num_styles=num_styles)
    discriminator = TinyDiscriminator(num_styles=num_styles)
    generator_optimizer = Adam(generator.parameters(), lr=0.01)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=0.01)
    output_dir = tmp_path / "weights"

    return (
        generator,
        discriminator,
        dataloader,
        generator_optimizer,
        discriminator_optimizer,
        output_dir,
    )


def test_train_one_epoch_runs_on_synthetic_data(tmp_path) -> None:
    (
        generator,
        discriminator,
        dataloader,
        generator_optimizer,
        discriminator_optimizer,
        output_dir,
    ) = _make_training_parts(tmp_path)

    metrics = train_one_epoch(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        device="cpu",
        epoch=1,
        output_dir=output_dir,
        l1_weight=10.0,
        sample_interval=0,
    )

    assert set(metrics) >= {"G_loss", "D_loss", "L1_loss"}
    assert math.isfinite(metrics["G_loss"])
    assert math.isfinite(metrics["D_loss"])
    assert math.isfinite(metrics["L1_loss"])


def test_train_one_epoch_accepts_mapping_batches(tmp_path) -> None:
    (
        generator,
        discriminator,
        dataloader,
        generator_optimizer,
        discriminator_optimizer,
        output_dir,
    ) = _make_training_parts(tmp_path)

    standard_images, real_images, style_ids = next(iter(dataloader))
    mapping_batches = [
        {
            "standard_images": standard_images,
            "real_images": real_images,
            "style_ids": style_ids,
        }
    ]

    metrics = train_one_epoch(
        generator=generator,
        discriminator=discriminator,
        dataloader=mapping_batches,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        device="cpu",
        epoch=1,
        output_dir=output_dir,
        l1_weight=10.0,
        sample_interval=0,
    )

    assert set(metrics) >= {"G_loss", "D_loss", "L1_loss"}


def test_train_one_epoch_accepts_handwrite_dataset_mapping_batches(tmp_path) -> None:
    (
        generator,
        discriminator,
        dataloader,
        generator_optimizer,
        discriminator_optimizer,
        output_dir,
    ) = _make_training_parts(tmp_path)

    standard_images, real_images, style_ids = next(iter(dataloader))
    mapping_batches = [
        {
            "standard": standard_images,
            "handwrite": real_images,
            "style_id": style_ids,
            "char": ["A", "B"],
        }
    ]

    metrics = train_one_epoch(
        generator=generator,
        discriminator=discriminator,
        dataloader=mapping_batches,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        device="cpu",
        epoch=1,
        output_dir=output_dir,
        l1_weight=10.0,
        sample_interval=0,
    )

    assert set(metrics) >= {"G_loss", "D_loss", "L1_loss"}


def test_train_one_epoch_moves_models_to_requested_device(tmp_path) -> None:
    (
        _generator,
        _discriminator,
        dataloader,
        generator_optimizer,
        discriminator_optimizer,
        output_dir,
    ) = _make_training_parts(tmp_path)
    generator = DeviceTrackingGenerator()
    discriminator = DeviceTrackingDiscriminator()
    generator_optimizer = Adam(generator.parameters(), lr=0.01)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=0.01)

    train_one_epoch(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        device="cpu",
        epoch=1,
        output_dir=output_dir,
        l1_weight=10.0,
        sample_interval=0,
    )

    assert generator.moved_to == [torch.device("cpu")]
    assert discriminator.moved_to == [torch.device("cpu")]


def test_train_one_epoch_reuses_same_fake_batch_for_d_and_g_updates(tmp_path) -> None:
    generator = SideEffectGenerator()
    discriminator = RecordingDiscriminator()
    generator_optimizer = Adam(generator.parameters(), lr=0.01)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=0.01)
    dataloader = [
        (
            torch.zeros(2, 1, 2, 2),
            torch.full((2, 1, 2, 2), 5.0),
            torch.zeros(2, dtype=torch.long),
        )
    ]

    train_one_epoch(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        device="cpu",
        epoch=1,
        output_dir=tmp_path / "weights",
        l1_weight=1.0,
        sample_interval=0,
    )

    assert generator.forward_calls == 1
    assert discriminator.input_means == [5.0, 1.0, 1.0]


def test_train_one_epoch_weights_epoch_metrics_by_sample_count(tmp_path) -> None:
    generator = AveragingGenerator()
    discriminator = AveragingDiscriminator()
    generator_optimizer = Adam(generator.parameters(), lr=0.0)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=0.0)
    dataloader = [
        (
            torch.tensor([[[[1.0]]], [[[3.0]]]]),
            torch.tensor([[[[1.0]]], [[[3.0]]]]),
            torch.tensor([0, 1], dtype=torch.long),
        ),
        (
            torch.tensor([[[[9.0]]]]),
            torch.tensor([[[[9.0]]]]),
            torch.tensor([2], dtype=torch.long),
        ),
    ]

    metrics = train_one_epoch(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        device="cpu",
        epoch=1,
        output_dir=tmp_path / "weights",
        l1_weight=1.0,
        sample_interval=0,
        adversarial_loss_fn=MeanPredictionLoss(),
        reconstruction_loss_fn=MeanReconstructionLoss(),
        category_loss_fn=MeanPredictionLoss(),
    )

    assert metrics["G_loss"] == 13.0
    assert metrics["D_loss"] == 13.0
    assert math.isclose(metrics["L1_loss"], 13.0 / 3.0)


def test_train_one_epoch_requires_output_dir_before_processing_batches() -> None:
    generator = SideEffectGenerator()
    discriminator = RecordingDiscriminator()
    generator_optimizer = Adam(generator.parameters(), lr=0.01)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=0.01)
    dataloader = [
        (
            torch.ones(2, 1, 2, 2),
            torch.full((2, 1, 2, 2), 5.0),
            torch.tensor([0, 1], dtype=torch.long),
        )
    ]
    generator_state_before = {
        name: tensor.detach().clone()
        for name, tensor in generator.state_dict().items()
    }
    discriminator_state_before = {
        name: tensor.detach().clone()
        for name, tensor in discriminator.state_dict().items()
    }

    with pytest.raises(
        ValueError,
        match="output_dir is required when sample_interval is enabled",
    ):
        train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloader,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            device="cpu",
            epoch=1,
            output_dir=None,
            l1_weight=1.0,
            sample_interval=1,
        )

    assert generator.forward_calls == 0
    assert discriminator.input_means == []
    assert generator_optimizer.state_dict()["state"] == {}
    assert discriminator_optimizer.state_dict()["state"] == {}
    for name, tensor in generator.state_dict().items():
        assert torch.equal(tensor, generator_state_before[name])
    for name, tensor in discriminator.state_dict().items():
        assert torch.equal(tensor, discriminator_state_before[name])


def test_fit_writes_checkpoint_at_configured_epoch_interval(tmp_path) -> None:
    (
        generator,
        discriminator,
        dataloader,
        generator_optimizer,
        discriminator_optimizer,
        output_dir,
    ) = _make_training_parts(tmp_path)

    fit(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        device="cpu",
        output_dir=output_dir,
        epochs=2,
        l1_weight=10.0,
        checkpoint_interval=2,
        sample_interval=0,
    )

    skipped_checkpoint_path = output_dir / "checkpoint_epoch_1.pt"
    checkpoint_path = output_dir / "checkpoint_epoch_2.pt"

    assert not skipped_checkpoint_path.exists()
    assert checkpoint_path.exists()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert checkpoint["epoch"] == 2
    assert "generator_state_dict" in checkpoint
    assert "discriminator_state_dict" in checkpoint


def test_fit_writes_sample_image_at_configured_batch_interval(tmp_path) -> None:
    (
        generator,
        discriminator,
        dataloader,
        generator_optimizer,
        discriminator_optimizer,
        output_dir,
    ) = _make_training_parts(tmp_path)

    fit(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        device="cpu",
        output_dir=output_dir,
        epochs=1,
        l1_weight=10.0,
        checkpoint_interval=0,
        sample_interval=2,
    )

    skipped_sample_path = output_dir / "samples" / "epoch1_batch1.png"
    sample_path = output_dir / "samples" / "epoch1_batch2.png"

    assert not skipped_sample_path.exists()
    assert sample_path.exists()

    with Image.open(sample_path) as image:
        assert image.mode == "L"
        assert image.width > image.height
        assert image.height > 0


def test_fit_writes_csv_log_with_expected_columns_and_rows(tmp_path) -> None:
    (
        generator,
        discriminator,
        dataloader,
        generator_optimizer,
        discriminator_optimizer,
        output_dir,
    ) = _make_training_parts(tmp_path)

    history = fit(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        device="cpu",
        output_dir=output_dir,
        epochs=2,
        l1_weight=10.0,
        checkpoint_interval=0,
        sample_interval=0,
    )

    log_path = output_dir / "train_log.csv"

    assert len(history) == 2
    assert log_path.exists()

    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0].keys() >= {"epoch", "G_loss", "D_loss", "L1_loss"}
    assert rows[0]["epoch"] == "1"
    assert rows[1]["epoch"] == "2"


def test_fit_activates_label_shuffle_once_epoch_threshold_is_reached(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = StyleRecordingGenerator()
    discriminator = TinyDiscriminator()
    generator_optimizer = Adam(generator.parameters(), lr=0.0)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=0.0)
    permutation_calls: list[tuple[int, torch.device | None]] = []

    def fake_randperm(
        item_count: int,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        del args
        device = kwargs.get("device")
        permutation_calls.append((item_count, device))
        return torch.tensor([1, 2, 0], device=device)

    monkeypatch.setattr(train_module.torch, "randperm", fake_randperm)
    dataloader = [
        (
            torch.zeros(3, 1, 4, 4),
            torch.zeros(3, 1, 4, 4),
            torch.tensor([0, 1, 2], dtype=torch.long),
        )
    ]

    history = fit(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        device="cpu",
        output_dir=tmp_path / "weights",
        epochs=2,
        l1_weight=1.0,
        label_shuffle_start=2,
        checkpoint_interval=0,
        sample_interval=0,
    )

    assert len(history) == 2
    assert generator.style_calls == [
        (0, 1, 2),
        (0, 1, 2),
        (1, 2, 0),
    ]
    assert permutation_calls == [
        (3, torch.device("cpu")),
    ]
