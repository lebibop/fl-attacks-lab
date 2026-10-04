from __future__ import annotations

import numpy as np


def _validate_fraction(fraction: float) -> float:
    normalized = float(fraction)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError("malicious-fraction must be in range [0.0, 1.0]")
    return normalized


def malicious_client_ids(
    num_clients: int,
    fraction: float,
    seed: int,
) -> set[int]:
    """Выбирает фиксированный воспроизводимый набор вредоносных клиентов."""
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than zero")

    normalized_fraction = _validate_fraction(fraction)
    count = int(round(num_clients * normalized_fraction))
    count = min(max(count, 0), num_clients)

    rng = np.random.default_rng(seed)
    selected = rng.permutation(num_clients)[:count]
    return {int(client_id) for client_id in selected.tolist()}


def attack_is_active(
    client_id: int,
    server_round: int,
    num_clients: int,
    malicious_fraction: float,
    attack_start_round: int,
    attack_end_round: int,
    seed: int,
) -> bool:
    """Проверяет, активна ли атака выбранного клиента в данном раунде."""
    if server_round < 1:
        raise ValueError("server_round must be greater than zero")
    if attack_start_round < 1:
        raise ValueError("attack-start-round must be greater than zero")
    if attack_end_round != -1 and attack_end_round < attack_start_round:
        raise ValueError(
            "attack-end-round must be -1 or greater than/equal to attack-start-round"
        )

    malicious_clients = malicious_client_ids(
        num_clients=num_clients,
        fraction=malicious_fraction,
        seed=seed,
    )

    attack_has_started = server_round >= attack_start_round
    attack_has_not_ended = (
        attack_end_round < 0 or server_round <= attack_end_round
    )

    return (
        client_id in malicious_clients
        and attack_has_started
        and attack_has_not_ended
    )
