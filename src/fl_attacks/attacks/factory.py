from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from fl_attacks.constants import Attacks
from .base import ModelAttack
from .gaussian_noise import GaussianNoiseAttack
from .random_update import RandomUpdateAttack
from .scaling import ScalingAttack
from .sign_flip import SignFlipAttack

SUPPORTED_ATTACKS: Final[frozenset[str]] = frozenset(
    {
        Attacks.NONE,
        Attacks.SIGN_FLIP,
        Attacks.GAUSSIAN_NOISE,
        Attacks.SCALING,
        Attacks.RANDOM_UPDATE,
    }
)

MODEL_ATTACKS: Final[frozenset[str]] = frozenset(
    SUPPORTED_ATTACKS - {Attacks.NONE}
)


def create_attack(
    attack: str,
    config: Mapping[str, Any],
) -> ModelAttack | None:
    """Создаёт объект атаки по имени из конфигурации эксперимента."""
    normalized = attack.lower()

    if normalized == Attacks.NONE:
        return None
    if normalized == Attacks.SIGN_FLIP:
        return SignFlipAttack(scale=float(config["model-attack-scale"]))
    if normalized == Attacks.GAUSSIAN_NOISE:
        return GaussianNoiseAttack(noise_std=float(config["noise-std"]))
    if normalized == Attacks.SCALING:
        return ScalingAttack(factor=float(config["scaling-factor"]))
    if normalized == Attacks.RANDOM_UPDATE:
        return RandomUpdateAttack(scale=float(config["random-update-scale"]))

    supported = ", ".join(sorted(SUPPORTED_ATTACKS))
    raise ValueError(
        f"Unsupported attack: {attack}. Supported attacks: {supported}"
    )
