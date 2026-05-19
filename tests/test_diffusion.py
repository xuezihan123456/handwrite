"""Tests for the diffusion-based handwriting scaffold (Innovation #3).

All tests are deliberately tiny (image_size <= 32, num_timesteps <= 4) so the
suite runs on CPU in well under 30 seconds.  Real training is **not**
performed in this repo — these tests verify the scaffold's wiring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _seed_all(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def test_module_exports_public_api() -> None:
    """Imports + public API surface defined in __init__."""
    import handwrite.diffusion as diffusion

    assert hasattr(diffusion, "DiffusionEngine")
    assert hasattr(diffusion, "UNet")
    assert hasattr(diffusion, "NoiseScheduler")
    assert hasattr(diffusion, "train_diffusion")

    expected = {"DiffusionEngine", "UNet", "NoiseScheduler", "train_diffusion"}
    assert expected.issubset(set(diffusion.__all__))


def test_unet_forward_shape_with_condition() -> None:
    """UNet returns same spatial shape as input."""
    from handwrite.diffusion import UNet

    _seed_all()
    model = UNet(base_channels=8, condition_channels=1)
    noisy = torch.randn(2, 1, 16, 16)
    condition = torch.randn(2, 1, 16, 16)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    output = model(noisy, timesteps, condition)

    assert output.shape == noisy.shape


def test_unet_forward_shape_without_condition() -> None:
    """UNet handles missing condition by zero-padding internally."""
    from handwrite.diffusion import UNet

    _seed_all()
    model = UNet(base_channels=8, condition_channels=1)
    noisy = torch.randn(1, 1, 16, 16)
    timesteps = torch.tensor([0], dtype=torch.long)

    output = model(noisy, timesteps, None)

    assert output.shape == noisy.shape


def test_unet_rejects_wrong_input_shapes() -> None:
    """UNet validates input dimensions."""
    from handwrite.diffusion import UNet

    model = UNet(base_channels=8, condition_channels=1)

    with pytest.raises(ValueError, match="noisy_images"):
        model(torch.randn(1, 3, 16, 16), torch.tensor([0], dtype=torch.long))

    with pytest.raises(ValueError, match="timesteps"):
        model(torch.randn(1, 1, 16, 16), torch.tensor([0, 1], dtype=torch.long))


def test_scheduler_linear_beta_schedule_properties() -> None:
    """Scheduler builds a valid linear beta schedule."""
    from handwrite.diffusion import NoiseScheduler

    scheduler = NoiseScheduler(num_timesteps=4)

    assert scheduler.betas.shape == (4,)
    assert torch.all(scheduler.betas > 0)
    # alphas in (0, 1), cumprod monotonically non-increasing
    assert torch.all(scheduler.alphas > 0)
    assert torch.all(scheduler.alphas < 1)
    diffs = scheduler.alpha_cumprod[1:] - scheduler.alpha_cumprod[:-1]
    assert torch.all(diffs <= 0)


def test_scheduler_add_noise_matches_closed_form() -> None:
    """add_noise(x0, eps, t) == sqrt(a_bar) x0 + sqrt(1-a_bar) eps."""
    from handwrite.diffusion import NoiseScheduler

    _seed_all()
    scheduler = NoiseScheduler(num_timesteps=4)
    x0 = torch.randn(2, 1, 8, 8)
    eps = torch.randn(2, 1, 8, 8)
    timesteps = torch.tensor([0, 3], dtype=torch.long)

    noisy = scheduler.add_noise(x0, eps, timesteps)

    sqrt_a = scheduler.sqrt_alpha_cumprod[timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_a = scheduler.sqrt_one_minus_alpha_cumprod[timesteps].view(
        -1, 1, 1, 1
    )
    expected = sqrt_a * x0 + sqrt_one_minus_a * eps
    assert torch.allclose(noisy, expected, atol=1e-6)


def test_scheduler_step_at_t_zero_is_deterministic() -> None:
    """step() at t=0 has zero variance, so it's deterministic given inputs."""
    from handwrite.diffusion import NoiseScheduler

    _seed_all()
    scheduler = NoiseScheduler(num_timesteps=4)
    sample = torch.randn(1, 1, 4, 4)
    predicted = torch.randn(1, 1, 4, 4)

    out_a = scheduler.step(predicted, 0, sample)
    out_b = scheduler.step(predicted, 0, sample)
    assert torch.allclose(out_a, out_b)


def test_scheduler_step_rejects_out_of_range_timestep() -> None:
    """Step validates the timestep argument."""
    from handwrite.diffusion import NoiseScheduler

    scheduler = NoiseScheduler(num_timesteps=4)
    sample = torch.randn(1, 1, 4, 4)
    predicted = torch.randn(1, 1, 4, 4)

    with pytest.raises(ValueError):
        scheduler.step(predicted, -1, sample)
    with pytest.raises(ValueError):
        scheduler.step(predicted, 4, sample)


def test_engine_generate_without_weights_returns_blank_for_string_input() -> None:
    """When no weights are loaded, a string input yields a blank fallback canvas."""
    from handwrite.diffusion import DiffusionEngine

    engine = DiffusionEngine(image_size=16, base_channels=8, num_train_timesteps=2)
    assert engine.has_weights is False

    image = engine.generate_char("\u4f60", style_id=0, num_steps=2)

    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    assert image.size == (16, 16)
    # blank canvas → all white
    assert min(image.getdata()) == 255
    assert max(image.getdata()) == 255


def test_engine_generate_without_weights_returns_condition_for_image_input() -> None:
    """When weights are missing, an explicit condition image is returned as-is shape."""
    from handwrite.diffusion import DiffusionEngine

    engine = DiffusionEngine(image_size=16, base_channels=8, num_train_timesteps=2)
    condition = Image.new("L", (16, 16), color=128)

    image = engine.generate_char(condition, style_id=0, num_steps=2)

    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    assert image.size == (16, 16)
    # Should be uniform mid-gray
    values = list(image.getdata())
    assert min(values) == max(values)


def test_engine_load_missing_weights_is_graceful() -> None:
    """load_weights returns False when the path doesn't exist."""
    from handwrite.diffusion import DiffusionEngine

    engine = DiffusionEngine(image_size=16, base_channels=8, num_train_timesteps=2)

    assert engine.load_weights(None) is False
    assert engine.load_weights("definitely-not-a-real-path.pt") is False
    assert engine.has_weights is False


def test_engine_load_corrupt_weights_is_graceful(tmp_path: Path) -> None:
    """load_weights returns False for unreadable / wrong-shape payloads."""
    from handwrite.diffusion import DiffusionEngine

    bad = tmp_path / "broken.pt"
    bad.write_text("not a torch checkpoint", encoding="utf-8")

    engine = DiffusionEngine(image_size=16, base_channels=8, num_train_timesteps=2)
    assert engine.load_weights(bad) is False
    assert engine.has_weights is False


def test_engine_save_and_load_round_trip(tmp_path: Path) -> None:
    """Saving + reloading recovers identical UNet weights."""
    from handwrite.diffusion import DiffusionEngine

    _seed_all(42)
    engine = DiffusionEngine(image_size=16, base_channels=8, num_train_timesteps=2)
    weights_path = tmp_path / "diffusion.pt"
    engine.save_weights(weights_path)

    other = DiffusionEngine(image_size=16, base_channels=8, num_train_timesteps=2)
    # randomize "other" first to make sure load actually overwrites
    for parameter in other.unet.parameters():
        with torch.no_grad():
            parameter.add_(1.23)

    assert other.load_weights(weights_path) is True
    assert other.has_weights is True

    for key, original_tensor in engine.unet.state_dict().items():
        assert torch.equal(other.unet.state_dict()[key], original_tensor)


def test_engine_generate_with_loaded_weights_runs_sampling(tmp_path: Path) -> None:
    """Once weights are loaded, generate_char runs the diffusion sampling path."""
    from handwrite.diffusion import DiffusionEngine

    _seed_all(7)
    engine = DiffusionEngine(image_size=16, base_channels=8, num_train_timesteps=2)
    weights_path = tmp_path / "diffusion.pt"
    engine.save_weights(weights_path)
    assert engine.load_weights(weights_path) is True

    condition = torch.zeros(1, 1, 16, 16)
    image = engine.generate_char(condition, style_id=0, num_steps=2, seed=11)

    assert isinstance(image, Image.Image)
    assert image.size == (16, 16)
    assert image.mode == "L"


def test_engine_generate_is_deterministic_with_seed(tmp_path: Path) -> None:
    """Same seed + same weights => identical output bytes."""
    from handwrite.diffusion import DiffusionEngine

    _seed_all(0)
    engine = DiffusionEngine(image_size=16, base_channels=8, num_train_timesteps=2)
    weights_path = tmp_path / "diffusion.pt"
    engine.save_weights(weights_path)
    engine.load_weights(weights_path)

    condition = torch.zeros(1, 1, 16, 16)
    image_a = engine.generate_char(condition, num_steps=2, seed=99)
    image_b = engine.generate_char(condition, num_steps=2, seed=99)

    assert image_a.tobytes() == image_b.tobytes()


def test_train_diffusion_runs_one_epoch_and_returns_metrics() -> None:
    """train_diffusion exposes the right return contract on synthetic data."""
    from handwrite.diffusion import train_diffusion

    _seed_all(0)
    target = torch.randn(2, 1, 16, 16)
    condition = torch.randn(2, 1, 16, 16)
    dataset = [(target, condition)]

    result = train_diffusion(
        dataset,
        epochs=1,
        image_size=16,
        base_channels=8,
        num_timesteps=2,
        learning_rate=1e-3,
        seed=0,
    )

    assert set(result.keys()) >= {"final_loss", "weights_path", "losses", "unet"}
    assert isinstance(result["final_loss"], float)
    assert result["weights_path"] is None
    assert len(result["losses"]) >= 1


def test_train_diffusion_loss_decreases_on_simple_overfit() -> None:
    """Training a few epochs over the same synthetic batch should trend the loss down."""
    from handwrite.diffusion import train_diffusion

    _seed_all(123)
    target = torch.randn(2, 1, 16, 16)
    condition = torch.randn(2, 1, 16, 16)
    dataset = [(target, condition)]

    # Use multiple epochs over the same batch so we accumulate enough gradient
    # steps to clearly see the loss curve drop.
    result = train_diffusion(
        dataset,
        epochs=40,
        image_size=16,
        base_channels=8,
        num_timesteps=2,
        learning_rate=1e-2,
        seed=123,
    )

    losses = result["losses"]
    assert len(losses) >= 10
    # Smoothed comparison: median of last quarter should be lower than first quarter
    # to be robust to the randomness of timestep/noise sampling each step.
    quartile = max(len(losses) // 4, 3)
    early_window = sorted(losses[:quartile])
    late_window = sorted(losses[-quartile:])
    early_median = early_window[len(early_window) // 2]
    late_median = late_window[len(late_window) // 2]
    assert late_median < early_median, (
        f"Loss did not decrease: early_median={early_median}, late_median={late_median}"
    )


def test_train_diffusion_writes_weights_when_output_path_given(tmp_path: Path) -> None:
    """train_diffusion persists weights to disk when output_path is provided."""
    from handwrite.diffusion import DiffusionEngine, train_diffusion

    _seed_all(0)
    target = torch.randn(1, 1, 16, 16)
    condition = torch.randn(1, 1, 16, 16)
    dataset = [(target, condition)]

    output_path = tmp_path / "out" / "diffusion.pt"
    result = train_diffusion(
        dataset,
        epochs=1,
        image_size=16,
        base_channels=8,
        num_timesteps=2,
        output_path=output_path,
        seed=0,
    )

    assert result["weights_path"] == output_path
    assert output_path.exists()

    # The saved file should be loadable into a fresh engine.
    engine = DiffusionEngine(image_size=16, base_channels=8, num_train_timesteps=2)
    assert engine.load_weights(output_path) is True


def test_train_diffusion_supports_mapping_batches() -> None:
    """Mapping batches with handwrite/standard keys are accepted."""
    from handwrite.diffusion import train_diffusion

    _seed_all(0)
    dataset = [
        {
            "handwrite": torch.randn(1, 1, 16, 16),
            "standard": torch.randn(1, 1, 16, 16),
        }
    ]

    result = train_diffusion(
        dataset,
        epochs=1,
        image_size=16,
        base_channels=8,
        num_timesteps=2,
        seed=0,
    )

    assert "final_loss" in result
    assert isinstance(result["final_loss"], float)


def test_train_diffusion_rejects_empty_dataset() -> None:
    """An empty dataset is an unrecoverable input error."""
    from handwrite.diffusion import train_diffusion

    with pytest.raises(ValueError):
        train_diffusion([], epochs=1)


def test_train_diffusion_rejects_zero_epochs() -> None:
    """epochs must be >= 1."""
    from handwrite.diffusion import train_diffusion

    target = torch.randn(1, 1, 16, 16)
    condition = torch.randn(1, 1, 16, 16)

    with pytest.raises(ValueError):
        train_diffusion([(target, condition)], epochs=0)
