"""Diffusion-based handwriting generation scaffold.

A scaffolded DDPM (Denoising Diffusion Probabilistic Model) with optional
ControlNet-style conditioning on a standard-font reference image.  This is
designed as a future replacement for the zi2zi GAN engine but is **not**
trained in this repository — it exposes the full API surface so downstream
training can plug in real weights without breaking callers.

Quick start::

    from handwrite.diffusion import DiffusionEngine

    engine = DiffusionEngine(image_size=32)
    image = engine.generate_char("你", style_id=0, num_steps=4)

Public API:
    - :class:`DiffusionEngine` — high level inference wrapper
    - :class:`UNet` — small conditional UNet noise predictor
    - :class:`NoiseScheduler` — linear-beta DDPM scheduler
    - :func:`train_diffusion` — tiny default training loop
"""

from .engine import DiffusionEngine
from .scheduler import NoiseScheduler
from .training import train_diffusion
from .unet import UNet

__all__ = [
    "DiffusionEngine",
    "NoiseScheduler",
    "UNet",
    "train_diffusion",
]
