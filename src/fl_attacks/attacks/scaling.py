from __future__ import annotations

from torch import Tensor

from .base import ModelAttack, validate_positive


class ScalingAttack(ModelAttack):
    """Усиливает честное обновление без изменения его направления."""

    def __init__(self, factor: float) -> None:
        self.factor = validate_positive(factor, "scaling-factor")

    def transform_delta(self, delta: Tensor) -> Tensor:
        return self.factor * delta
