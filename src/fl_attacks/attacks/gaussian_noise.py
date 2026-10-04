from __future__ import annotations

import torch
from torch import Tensor

from .base import ModelAttack, relative_rms, validate_non_negative


class GaussianNoiseAttack(ModelAttack):
    """Добавляет к честному обновлению гауссовский шум относительного масштаба."""

    def __init__(self, noise_std: float) -> None:
        self.noise_std = validate_non_negative(noise_std, "noise-std")

    def transform_delta(self, delta: Tensor) -> Tensor:
        if self.noise_std == 0.0:
            return delta

        sigma = self.noise_std * relative_rms(delta)
        noise = torch.randn_like(delta) * sigma
        return delta + noise
