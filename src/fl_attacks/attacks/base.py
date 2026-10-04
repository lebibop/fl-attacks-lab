from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

StateDict = Mapping[str, Tensor]


@dataclass(frozen=True)
class AttackDiagnostics:
    """Диагностика нормы клиентского обновления до и после атаки."""

    update_l2_before: float
    update_l2_after: float

    @property
    def update_l2_ratio(self) -> float:
        """Во сколько раз изменилась L2-норма обновления."""
        if self.update_l2_before <= 1e-12:
            if self.update_l2_after <= 1e-12:
                return 0.0
            return self.update_l2_after / 1e-12
        return self.update_l2_after / self.update_l2_before


class ModelAttack(ABC):
    """Базовый контракт для атак на клиентское обновление модели."""

    @abstractmethod
    def transform_delta(self, delta: Tensor) -> Tensor:
        """Преобразует локальное обновление одного тензора параметров."""


def validate_positive(value: float, name: str) -> float:
    """Проверяет, что параметр атаки строго положительный."""
    normalized = float(value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return normalized


def validate_non_negative(value: float, name: str) -> float:
    """Проверяет, что параметр атаки неотрицательный."""
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def relative_rms(delta: Tensor, minimum: float = 1e-12) -> float:
    """Возвращает RMS обновления с нижней границей для численной устойчивости."""
    if delta.numel() == 0:
        return minimum

    rms = torch.sqrt(torch.mean(delta.detach().float().pow(2))).item()
    return max(float(rms), minimum)


def update_l2_norm(
    state: StateDict,
    global_state: StateDict,
) -> float:
    """Вычисляет общую L2-норму обновления state - global_state."""
    squared_norm = 0.0

    for name, tensor in state.items():
        if not torch.is_floating_point(tensor):
            continue

        reference = global_state[name].to(
            device=tensor.device,
            dtype=tensor.dtype,
        )
        delta = (tensor.detach() - reference).float()
        squared_norm += float(torch.sum(delta * delta).item())

    return squared_norm**0.5
