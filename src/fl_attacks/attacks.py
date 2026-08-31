from __future__ import annotations

from collections.abc import Mapping, Sized
from typing import Final

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from fl_attacks.constants import Attacks

# Все поддерживаемые типы атак.
SUPPORTED_ATTACKS: Final[frozenset[str]] = frozenset(
    {
        Attacks.NONE,
        Attacks.LABEL_FLIP,
        Attacks.SIGN_FLIP,
        Attacks.GAUSSIAN_NOISE,
        Attacks.BACKDOOR,
    }
)

# Атаки, изменяющие обновление локальной модели.
MODEL_ATTACKS: Final[frozenset[str]] = frozenset(
    {
        Attacks.SIGN_FLIP,
        Attacks.GAUSSIAN_NOISE,
    }
)


def _get_targets(dataset: Dataset) -> np.ndarray:
    """Возвращает метки датасета как массив NumPy."""
    targets = getattr(dataset, "targets")
    return np.asarray(targets, dtype=np.int64)


def _dataset_length(dataset: Dataset) -> int:
    """Безопасно возвращает количество объектов в датасете."""
    if not isinstance(dataset, Sized):
        raise TypeError("Dataset must implement __len__")

    return len(dataset)


def _eligible_indices(
    targets: np.ndarray,
    source_label: int,
    target_label: int,
) -> np.ndarray:
    """
    Возвращает индексы объектов, подходящих для backdoor-атаки.

    Если source_label равен -1, выбираются все классы, кроме целевого.
    """
    if source_label < 0:
        return np.flatnonzero(targets != target_label)

    return np.flatnonzero(targets == source_label)


def _bounded_fraction_count(total: int, fraction: float) -> int:
    """Преобразует долю в количество элементов от 0 до total."""
    count = int(round(total * fraction))
    return min(max(count, 0), total)


def _random_subset(
    indices: np.ndarray,
    fraction: float,
    seed: int,
) -> set[int]:
    """Выбирает воспроизводимое случайное подмножество индексов."""
    count = _bounded_fraction_count(len(indices), fraction)

    if count == 0:
        return set()

    rng = np.random.default_rng(seed)
    selected = rng.choice(indices, size=count, replace=False)

    return {int(index) for index in selected.tolist()}


def _add_trigger(
    image: torch.Tensor,
    trigger_size: int,
    trigger_value: float,
) -> torch.Tensor:
    """Добавляет квадратный триггер в правый нижний угол изображения."""
    triggered = image.clone()

    size = min(
        trigger_size,
        triggered.shape[-2],
        triggered.shape[-1],
    )

    triggered[..., -size:, -size:] = trigger_value
    return triggered


class LabelFlipDataset(Dataset):
    """Обёртка над датасетом, подменяющая метки объектов."""

    def __init__(
        self,
        dataset: Dataset,
        source_label: int,
        target_label: int,
        num_classes: int = 10,
    ) -> None:
        """
        Инициализирует датасет с атакой подмены меток.

        При source_label = -1 метки всех классов циклически сдвигаются.
        """
        self.dataset = dataset
        self.source_label = source_label
        self.target_label = target_label
        self.num_classes = num_classes

    def __len__(self) -> int:
        """Возвращает количество объектов в исходном датасете."""
        return _dataset_length(self.dataset)

    def __getitem__(self, index: int) -> tuple[object, int]:
        """Возвращает объект и исходную либо подменённую метку."""
        image, label = self.dataset[index]
        label = int(label)

        if self.source_label < 0:
            flipped_label = (label + 1) % self.num_classes
        elif label == self.source_label:
            flipped_label = self.target_label
        else:
            flipped_label = label

        return image, flipped_label


class BackdoorDataset(Dataset):
    """Обёртка, добавляющая триггер и целевую метку части объектов."""

    def __init__(
        self,
        dataset: Dataset,
        poison_fraction: float,
        source_label: int,
        target_label: int,
        seed: int,
        trigger_size: int = 3,
        trigger_value: float = 3.0,
    ) -> None:
        """Выбирает объекты для отравления и сохраняет параметры триггера."""
        self.dataset = dataset
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value

        eligible = _eligible_indices(
            targets=_get_targets(dataset),
            source_label=source_label,
            target_label=target_label,
        )

        self.poisoned_indices = _random_subset(
            indices=eligible,
            fraction=poison_fraction,
            seed=seed,
        )

    def __len__(self) -> int:
        """Возвращает количество объектов в исходном датасете."""
        return _dataset_length(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Возвращает обычный или отравленный объект."""
        image, label = self.dataset[index]

        if index not in self.poisoned_indices:
            return image, int(label)

        poisoned = _add_trigger(
            image=image,
            trigger_size=self.trigger_size,
            trigger_value=self.trigger_value,
        )

        return poisoned, self.target_label


class TriggeredTestDataset(Dataset):
    """Формирует тестовый набор с триггерами для расчёта ASR."""

    def __init__(
        self,
        dataset: Dataset,
        source_label: int,
        target_label: int,
        trigger_size: int = 3,
        trigger_value: float = 3.0,
    ) -> None:
        """Выбирает подходящие тестовые объекты и сохраняет параметры атаки."""
        self.dataset = dataset
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value

        self.indices = _eligible_indices(
            targets=_get_targets(dataset),
            source_label=source_label,
            target_label=target_label,
        )

    def __len__(self) -> int:
        """Возвращает количество объектов, подходящих для проверки атаки."""
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Возвращает изображение с триггером и целевую метку."""
        original_index = int(self.indices[index])
        image, _ = self.dataset[original_index]

        triggered = _add_trigger(
            image=image,
            trigger_size=self.trigger_size,
            trigger_value=self.trigger_value,
        )

        return triggered, self.target_label


def malicious_client_ids(
    num_clients: int,
    fraction: float,
    seed: int,
) -> set[int]:
    """Выбирает воспроизводимый набор вредоносных клиентов."""
    count = _bounded_fraction_count(num_clients, fraction)
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
    """Проверяет, активна ли атака клиента в текущем раунде."""
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


def poison_training_dataset(
    dataset: Dataset,
    attack: str,
    source_label: int,
    target_label: int,
    poison_fraction: float,
    seed: int,
) -> Dataset:
    """Применяет атаку, изменяющую локальный обучающий датасет."""
    match attack:
        case Attacks.LABEL_FLIP:
            return LabelFlipDataset(
                dataset=dataset,
                source_label=source_label,
                target_label=target_label,
            )
        case Attacks.BACKDOOR:
            return BackdoorDataset(
                dataset=dataset,
                poison_fraction=poison_fraction,
                source_label=source_label,
                target_label=target_label,
                seed=seed,
            )
        case _:
            return dataset


def apply_model_attack(
    model: nn.Module,
    global_state: Mapping[str, torch.Tensor],
    attack: str,
    scale: float,
    noise_std: float,
) -> None:
    """
    Изменяет локальное обновление модели перед отправкой на сервер.

    Поддерживает инвертирование обновления и добавление гауссовского шума.
    """
    if attack not in MODEL_ATTACKS:
        return

    poisoned_state: dict[str, torch.Tensor] = {}

    for name, local_tensor in model.state_dict().items():
        if not torch.is_floating_point(local_tensor):
            poisoned_state[name] = local_tensor
            continue

        global_tensor = global_state[name].to(local_tensor.device)
        delta = local_tensor - global_tensor

        if attack == Attacks.SIGN_FLIP:
            poisoned_tensor = global_tensor - scale * delta
        else:
            layer_scale = delta.detach().float().std(unbiased=False).item()
            sigma = noise_std * max(layer_scale, 1e-6)
            noise = torch.randn_like(local_tensor) * sigma
            poisoned_tensor = local_tensor + noise

        poisoned_state[name] = poisoned_tensor

    model.load_state_dict(poisoned_state, strict=True)
