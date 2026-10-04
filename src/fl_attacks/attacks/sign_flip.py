from __future__ import annotations

from torch import Tensor

from .base import ModelAttack, validate_positive


class SignFlipAttack(ModelAttack):
    """Инвертирует направление обновления и усиливает его."""

    def __init__(self, scale: float) -> None:
        self.scale = validate_positive(scale, "model-attack-scale")

    def transform_delta(self, delta: Tensor) -> Tensor:
        return -self.scale * delta
