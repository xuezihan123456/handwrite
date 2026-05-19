"""Training loop for the diffusion scaffold.

The :func:`train_diffusion` entrypoint runs a minimal but correct DDPM
training step.  Defaults are intentionally tiny so the function can be
exercised in CI on CPU in a few seconds; production-scale training would
replace ``dataset`` with a real ``(condition, target, char)`` loader and
crank the timesteps, channels, and epochs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from .scheduler import NoiseScheduler
from .unet import UNet

__all__ = ["train_diffusion"]


def train_diffusion(
    dataset: Iterable[Any],
    epochs: int = 1,
    *,
    image_size: int = 32,
    base_channels: int = 16,
    num_timesteps: int = 4,
    condition_channels: int = 1,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cpu",
    output_path: str | Path | None = None,
    unet: nn.Module | None = None,
    scheduler: NoiseScheduler | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run a tiny DDPM training loop.

    Parameters
    ----------
    dataset:
        An iterable of batches.  Each batch may be:

        * a 2-tuple ``(target, condition)``,
        * a 3-tuple ``(target, condition, _ignored_style_id)``, or
        * a mapping with keys ``"target"`` (or ``"handwrite"``) and
          ``"condition"`` (or ``"standard"``).

        Tensors must have shape ``(batch, 1, H, W)``.
    epochs:
        Number of full passes over ``dataset``.
    output_path:
        If provided, the trained UNet weights are saved here at the end.
    unet, scheduler:
        Optional pre-constructed UNet and scheduler instances.  Useful for
        tests that want to inspect the trained weights directly.
    seed:
        Optional deterministic seed.

    Returns
    -------
    dict
        ``{"final_loss": float, "weights_path": Path|None, "losses": list[float],
        "unet": UNet, "scheduler": NoiseScheduler}``.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if seed is not None:
        torch.manual_seed(int(seed))

    resolved_device = torch.device(device)

    resolved_unet: nn.Module
    if unet is None:
        resolved_unet = UNet(
            base_channels=base_channels,
            condition_channels=condition_channels,
        )
    else:
        resolved_unet = unet
    resolved_unet = resolved_unet.to(resolved_device)
    resolved_unet.train()

    if scheduler is None:
        resolved_scheduler = NoiseScheduler(
            num_timesteps=num_timesteps, device=resolved_device
        )
    else:
        resolved_scheduler = scheduler

    optimizer = torch.optim.Adam(resolved_unet.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    batches = list(dataset)
    if not batches:
        raise ValueError("dataset must yield at least one batch")

    losses: list[float] = []
    last_loss: float = float("nan")

    for _epoch in range(epochs):
        for batch in batches:
            target, condition = _unpack_batch(batch, resolved_device)

            if target.ndim != 4 or target.shape[1] != 1:
                raise ValueError("target tensors must have shape (B, 1, H, W)")
            if condition is not None and (
                condition.ndim != 4
                or condition.shape[1] != condition_channels
            ):
                raise ValueError(
                    "condition tensors must have shape (B, condition_channels, H, W)"
                )

            batch_size = target.shape[0]
            timesteps = torch.randint(
                0,
                resolved_scheduler.num_timesteps,
                (batch_size,),
                device=resolved_device,
                dtype=torch.long,
            )
            noise = torch.randn_like(target)
            noisy = resolved_scheduler.add_noise(target, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            predicted_noise = resolved_unet(noisy, timesteps, condition)
            loss = loss_fn(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            last_loss = float(loss.item())
            losses.append(last_loss)

    weights_path: Path | None = None
    if output_path is not None:
        weights_path = Path(output_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(resolved_unet.state_dict(), weights_path)

    return {
        "final_loss": last_loss,
        "weights_path": weights_path,
        "losses": losses,
        "unet": resolved_unet,
        "scheduler": resolved_scheduler,
    }


def _unpack_batch(
    batch: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, Mapping):
        target = _first_present(batch, "target", "handwrite", "real_images", "real")
        condition = _first_present(
            batch, "condition", "standard", "standard_images", "source"
        )
        if target is None:
            raise KeyError(
                "batch mapping must include a target/handwrite/real_images key"
            )
        return (
            _move(target, device, torch.float32, name="target"),
            None if condition is None else _move(
                condition, device, torch.float32, name="condition"
            ),
        )

    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            target, condition = batch
        elif len(batch) == 3:
            target, condition, _ = batch
        else:
            raise TypeError(
                "batch tuple/list must have length 2 or 3 (target, condition[, style])"
            )
        return (
            _move(target, device, torch.float32, name="target"),
            None if condition is None else _move(
                condition, device, torch.float32, name="condition"
            ),
        )

    raise TypeError("batch must be a mapping or tuple/list")


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _move(
    value: Any,
    device: torch.device,
    dtype: torch.dtype,
    *,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value.to(device=device, dtype=dtype)
