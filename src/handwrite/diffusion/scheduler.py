"""Linear-beta DDPM noise scheduler.

Implements the standard DDPM math from Ho et al. (2020) at scale-1.  Designed
for CPU testing with tiny ``num_timesteps`` defaults (e.g. 4).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["NoiseScheduler"]


class NoiseScheduler:
    """Linear beta DDPM scheduler with :meth:`add_noise` and :meth:`step`.

    Parameters
    ----------
    num_timesteps:
        Number of diffusion timesteps in the schedule.
    beta_start, beta_end:
        Endpoints of the linear beta schedule.  Default values match
        Ho et al. (2020) for full-scale training but are safe at small
        ``num_timesteps`` too.
    device:
        Optional device for cached tensors.  Defaults to CPU.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be at least 1")
        if beta_start <= 0 or beta_end <= 0:
            raise ValueError("beta_start and beta_end must be positive")
        if beta_start >= beta_end:
            raise ValueError("beta_start must be smaller than beta_end")

        self.num_timesteps = int(num_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.device = torch.device(device)

        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32, device=self.device), self.alpha_cumprod[:-1]]
        )
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """Apply forward diffusion noise at the requested timesteps.

        Implements ``x_t = sqrt(alpha_cumprod[t]) * x_0 +
        sqrt(1 - alpha_cumprod[t]) * eps``.
        """
        self._validate_timesteps(timesteps, batch_size=original_samples.shape[0])
        if noise.shape != original_samples.shape:
            raise ValueError("noise must have the same shape as original_samples")

        sqrt_alpha = self._gather(self.sqrt_alpha_cumprod, timesteps, original_samples)
        sqrt_one_minus = self._gather(
            self.sqrt_one_minus_alpha_cumprod, timesteps, original_samples
        )
        return sqrt_alpha * original_samples + sqrt_one_minus * noise

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Single reverse diffusion step.

        Given the model's predicted noise ``eps_theta`` for ``sample`` at
        ``timestep``, return ``x_{t-1}``.
        """
        if timestep < 0 or timestep >= self.num_timesteps:
            raise ValueError(
                f"timestep must be in [0, {self.num_timesteps - 1}], got {timestep}"
            )
        if model_output.shape != sample.shape:
            raise ValueError("model_output must have the same shape as sample")

        beta_t = self.betas[timestep]
        alpha_t = self.alphas[timestep]
        alpha_cumprod_t = self.alpha_cumprod[timestep]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[timestep]

        # mu_theta = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_cumprod_t)) * eps)
        coefficient = beta_t / sqrt_one_minus_alpha_cumprod_t
        mean = (1.0 / torch.sqrt(alpha_t)) * (sample - coefficient * model_output)

        if timestep == 0:
            return mean

        variance = self.posterior_variance[timestep]
        noise = self._sample_noise(sample.shape, sample.device, sample.dtype, generator)
        return mean + torch.sqrt(variance) * noise

    def timesteps(self, num_inference_steps: int | None = None) -> Tensor:
        """Return a descending tensor of inference timesteps.

        If ``num_inference_steps`` < ``num_timesteps`` the schedule is striding
        across the training timesteps, mimicking DDIM-style step skipping for
        fast inference.
        """
        if num_inference_steps is None or num_inference_steps >= self.num_timesteps:
            return torch.arange(self.num_timesteps - 1, -1, -1, device=self.device)
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1")
        stride = max(self.num_timesteps // num_inference_steps, 1)
        step_indices = torch.arange(
            self.num_timesteps - 1, -1, -stride, device=self.device
        )
        return step_indices[:num_inference_steps]

    def _validate_timesteps(self, timesteps: Tensor, batch_size: int) -> None:
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError(
                "timesteps must have shape (batch,) matching original_samples"
            )
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.num_timesteps):
            raise ValueError(
                f"timesteps must be in [0, {self.num_timesteps - 1}]"
            )

    @staticmethod
    def _gather(table: Tensor, indices: Tensor, sample: Tensor) -> Tensor:
        values = table.to(device=sample.device, dtype=sample.dtype).index_select(
            0, indices.to(device=sample.device).long()
        )
        # Broadcast to (batch, 1, 1, 1, ...) for the sample's rank
        view_shape = [values.shape[0]] + [1] * (sample.ndim - 1)
        return values.view(*view_shape)

    @staticmethod
    def _sample_noise(
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> Tensor:
        if generator is None:
            return torch.randn(shape, device=device, dtype=dtype)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
