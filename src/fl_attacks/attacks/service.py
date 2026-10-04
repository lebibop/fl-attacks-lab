from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .base import AttackDiagnostics, update_l2_norm
from .factory import create_attack


def apply_model_attack(
    model: nn.Module,
    global_state: Mapping[str, torch.Tensor],
    attack: str,
    config: Mapping[str, Any],
) -> AttackDiagnostics:
    """Изменяет именно отправляемое серверу клиентское обновление модели."""
    attack_impl = create_attack(attack=attack, config=config)

    local_state = model.state_dict()
    before = update_l2_norm(local_state, global_state)

    if attack_impl is None:
        return AttackDiagnostics(
            update_l2_before=before,
            update_l2_after=before,
        )

    poisoned_state: dict[str, torch.Tensor] = {}

    for name, local_tensor in local_state.items():
        if not torch.is_floating_point(local_tensor):
            poisoned_state[name] = local_tensor
            continue

        global_tensor = global_state[name].to(
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
        delta = local_tensor - global_tensor
        attacked_delta = attack_impl.transform_delta(delta)
        poisoned_state[name] = global_tensor + attacked_delta

    model.load_state_dict(poisoned_state, strict=True)
    after = update_l2_norm(model.state_dict(), global_state)

    return AttackDiagnostics(
        update_l2_before=before,
        update_l2_after=after,
    )
