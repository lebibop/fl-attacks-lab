from __future__ import annotations

import torch
from torch import Tensor

from .base import ModelAttack, validate_positive


class RandomUpdateAttack(ModelAttack):
    """Полностью заменяет честное направление случайным обновлением."""

    def __init__(self, scale: float) -> None:
        self.scale = validate_positive(scale, "random-update-scale")

    def transform_delta(self, delta: Tensor) -> Tensor:
        if delta.numel() == 0:
            return delta

        random_delta = torch.randn_like(delta)
        random_norm = torch.linalg.vector_norm(random_delta.float())
        honest_norm = torch.linalg.vector_norm(delta.detach().float())

        # При практически нулевом честном обновлении используем небольшой,
        # но ненулевой эталон, чтобы атака не превращалась в no-op.
        target_norm = self.scale * max(float(honest_norm.item()), 1e-12)

        if float(random_norm.item()) <= 0.0:
            return torch.zeros_like(delta)

        return random_delta * (target_norm / float(random_norm.item()))
