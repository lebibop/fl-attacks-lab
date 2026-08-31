from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from fl_attacks.attacks import _dataset_length
from fl_attacks.constants import Datasets


@dataclass(frozen=True)
class ClientLoaders:
    """Загрузчики обучающих и валидационных данных клиента."""

    train: DataLoader
    val: DataLoader


@dataclass(frozen=True)
class DatasetSpec:
    """Параметры загрузки и нормализации датасета."""

    factory: Callable[..., Dataset]
    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass(frozen=True)
class DatasetMetadata:
    """Автоматически определённые характеристики датасета."""

    in_channels: int
    num_classes: int
    class_names: tuple[str, ...]


# Настройки поддерживаемых датасетов.
DATASET_SPECS: Final[dict[str, DatasetSpec]] = {
    Datasets.MNIST: DatasetSpec(
        factory=datasets.MNIST,
        mean=(0.1307,),
        std=(0.3081,),
    ),
    Datasets.FASHION_MNIST: DatasetSpec(
        factory=datasets.FashionMNIST,
        mean=(0.2860,),
        std=(0.3530,),
    ),
    Datasets.CIFAR10: DatasetSpec(
        factory=datasets.CIFAR10,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    Datasets.CIFAR100: DatasetSpec(
        factory=datasets.CIFAR100,
        mean=(0.5071, 0.4865, 0.4409),
        std=(0.2673, 0.2564, 0.2762),
    ),
}


class IndexedSubset(Dataset):
    """Подмножество датасета с доступом к исходным индексам и меткам."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
    ) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

        all_targets = extract_targets(dataset)
        self.targets = all_targets[self.indices]

        classes = getattr(dataset, "classes", None)
        if classes is not None:
            self.classes = classes

        class_to_idx = getattr(dataset, "class_to_idx", None)
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> Any:
        original_index = int(self.indices[index])
        return self.dataset[original_index]


def _dataset_spec(dataset_name: str) -> DatasetSpec:
    """Возвращает настройки выбранного датасета."""
    normalized_name = dataset_name.lower()

    try:
        return DATASET_SPECS[normalized_name]
    except KeyError as error:
        supported = ", ".join(DATASET_SPECS)
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: {supported}"
        ) from error


def _validate_partition_count(num_clients: int) -> None:
    """Проверяет корректность количества клиентов."""
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than zero")


def _validate_partition_id(
    partition_id: int,
    num_partitions: int,
) -> None:
    """Проверяет корректность идентификатора клиентской партиции."""
    _validate_partition_count(num_partitions)

    if not 0 <= partition_id < num_partitions:
        raise ValueError(
            "partition_id must be in range "
            f"[0, {num_partitions - 1}], got {partition_id}"
        )


def dataset_transform(dataset_name: str) -> transforms.Compose:
    """Создаёт преобразование и нормализацию выбранного датасета."""
    spec = _dataset_spec(dataset_name)

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
    )


def load_dataset(
    dataset_name: str,
    root: str | Path,
    train: bool,
) -> Dataset:
    """Загружает обучающую или тестовую часть датасета."""
    spec = _dataset_spec(dataset_name)

    return spec.factory(
        root=str(Path(root)),
        train=train,
        download=True,
        transform=dataset_transform(dataset_name),
    )


def download_dataset(
    dataset_name: str,
    root: str | Path,
) -> None:
    """Заранее загружает обучающую и тестовую части датасета."""
    for train in (True, False):
        load_dataset(
            dataset_name=dataset_name,
            root=root,
            train=train,
        )


def extract_targets(dataset: Dataset) -> np.ndarray:
    """Извлекает метки из targets/labels, включая обёрнутые датасеты."""
    current: Dataset | None = dataset
    visited: set[int] = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))

        targets = getattr(current, "targets", None)
        if targets is None:
            targets = getattr(current, "labels", None)

        if targets is not None:
            if isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu().numpy()

            result = np.asarray(targets, dtype=np.int64)
            return result.reshape(-1)

        current = getattr(current, "dataset", None)

    raise TypeError("Dataset must expose a 'targets' or 'labels' attribute")


def extract_class_names(dataset: Dataset) -> tuple[str, ...]:
    """Автоматически определяет названия классов датасета."""
    current: Dataset | None = dataset
    visited: set[int] = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))

        classes = getattr(current, "classes", None)
        if classes is not None:
            names = tuple(str(value) for value in classes)
            if names:
                return names

        class_to_idx = getattr(current, "class_to_idx", None)
        if class_to_idx:
            ordered = sorted(
                class_to_idx.items(),
                key=lambda item: int(item[1]),
            )
            return tuple(name for name, _ in ordered)

        current = getattr(current, "dataset", None)

    labels = np.unique(extract_targets(dataset))
    if labels.size == 0:
        raise ValueError("Dataset does not contain class labels")

    return tuple(str(int(label)) for label in labels)


def _extract_image_tensor(sample: Any) -> torch.Tensor:
    """Извлекает изображение из элемента датасета."""
    image = sample

    if isinstance(sample, (tuple, list)):
        if not sample:
            raise TypeError("Dataset sample is empty")
        image = sample[0]

    elif isinstance(sample, Mapping):
        for key in ("image", "images", "pixel_values", "input"):
            if key in sample:
                image = sample[key]
                break
        else:
            raise TypeError("Dictionary sample does not contain an image")

    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor()(image)

    return image


def extract_in_channels(dataset: Dataset) -> int:
    """Автоматически определяет количество каналов изображения."""
    if _dataset_length(dataset) == 0:
        raise ValueError("Cannot inspect an empty dataset")

    image = _extract_image_tensor(dataset[0])

    if image.ndim == 2:
        return 1
    if image.ndim == 3:
        return int(image.shape[0])

    raise ValueError(
        "Expected image shape [H, W] or [C, H, W], "
        f"received {tuple(image.shape)}"
    )


def dataset_metadata(dataset: Dataset) -> DatasetMetadata:
    """Определяет каналы, число классов и названия классов."""
    class_names = extract_class_names(dataset)

    if len(class_names) <= 1:
        raise ValueError(
            "Classification dataset must contain at least two classes"
        )

    return DatasetMetadata(
        in_channels=extract_in_channels(dataset),
        num_classes=len(class_names),
        class_names=class_names,
    )


def iid_partitions(
    num_items: int,
    num_clients: int,
    seed: int,
) -> list[np.ndarray]:
    """Случайно и равномерно распределяет объекты между клиентами."""
    _validate_partition_count(num_clients)

    if num_items < 0:
        raise ValueError("num_items must be non-negative")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(num_items)

    return [
        partition.astype(np.int64)
        for partition in np.array_split(
            shuffled_indices,
            num_clients,
        )
    ]


def _sample_dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Создаёт один вариант Non-IID-разбиения по распределению Дирихле."""
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(np.full(num_clients, alpha))

        split_points = (np.cumsum(proportions) * len(class_indices)).astype(
            int
        )[:-1]

        class_splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(split.tolist())

    return client_indices


def _shuffle_client_partitions(
    partitions: list[list[int]],
    seed: int,
) -> list[np.ndarray]:
    """Перемешивает объекты внутри каждой клиентской партиции."""
    result: list[np.ndarray] = []

    for client_id, indices in enumerate(partitions):
        rng = np.random.default_rng(seed + 10_000 + client_id)
        partition = np.asarray(indices, dtype=np.int64)

        rng.shuffle(partition)
        result.append(partition)

    return result


def dirichlet_partitions(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    max_attempts: int = 100,
) -> list[np.ndarray]:
    """Распределяет данные между клиентами по распределению Дирихле."""
    _validate_partition_count(num_clients)

    if alpha <= 0:
        raise ValueError("dirichlet-alpha must be greater than zero")

    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")

    labels = np.asarray(labels, dtype=np.int64)

    min_required = max(
        2,
        int(0.05 * len(labels) / num_clients),
    )

    best_partitions: list[list[int]] | None = None
    best_min_size = -1

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)

        candidate = _sample_dirichlet_partition(
            labels=labels,
            num_clients=num_clients,
            alpha=alpha,
            rng=rng,
        )

        current_min_size = min(len(indices) for indices in candidate)

        if current_min_size > best_min_size:
            best_partitions = candidate
            best_min_size = current_min_size

        if current_min_size >= min_required:
            break

    if best_partitions is None:
        raise RuntimeError("Failed to create Dirichlet partitions")

    return _shuffle_client_partitions(
        partitions=best_partitions,
        seed=seed,
    )


def make_partitions(
    labels: np.ndarray,
    num_clients: int,
    partition: str,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    """Создаёт IID- или Dirichlet-разбиение датасета."""
    mode = partition.lower()

    if mode == "iid":
        return iid_partitions(
            num_items=len(labels),
            num_clients=num_clients,
            seed=seed,
        )

    if mode == "dirichlet":
        return dirichlet_partitions(
            labels=labels,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
        )

    raise ValueError("partition must be 'iid' or 'dirichlet'")


def _split_train_validation_indices(
    indices: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Разделяет клиентские индексы на обучение и валидацию."""
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    shuffled_indices = indices.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled_indices)

    if len(shuffled_indices) < 2:
        raise RuntimeError(
            "A client must receive at least two samples "
            "to create train and validation datasets"
        )

    val_size = max(
        1,
        int(round(len(shuffled_indices) * val_fraction)),
    )
    val_size = min(val_size, len(shuffled_indices) - 1)

    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]

    return train_indices, val_indices


def load_client_datasets(
    dataset_name: str,
    data_root: str | Path,
    partition_id: int,
    num_partitions: int,
    partition: str,
    dirichlet_alpha: float,
    val_fraction: float,
    seed: int,
) -> tuple[IndexedSubset, IndexedSubset]:
    """Загружает локальные обучающие и валидационные данные клиента."""
    _validate_partition_id(
        partition_id=partition_id,
        num_partitions=num_partitions,
    )

    dataset = load_dataset(
        dataset_name=dataset_name,
        root=data_root,
        train=True,
    )

    partitions = make_partitions(
        labels=extract_targets(dataset),
        num_clients=num_partitions,
        partition=partition,
        alpha=dirichlet_alpha,
        seed=seed,
    )

    client_indices = partitions[partition_id]

    try:
        train_indices, val_indices = _split_train_validation_indices(
            indices=client_indices,
            val_fraction=val_fraction,
            seed=seed + 20_000 + partition_id,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"Client {partition_id} received "
            f"only {len(client_indices)} samples; "
            "increase dirichlet-alpha or reduce num-clients"
        ) from error

    return (
        IndexedSubset(dataset, train_indices),
        IndexedSubset(dataset, val_indices),
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    """Создаёт воспроизводимый загрузчик данных."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def load_client_loaders(
    dataset_name: str,
    data_root: str | Path,
    partition_id: int,
    num_partitions: int,
    partition: str,
    dirichlet_alpha: float,
    val_fraction: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> ClientLoaders:
    """Создаёт обучающий и валидационный загрузчики клиента."""
    train_dataset, val_dataset = load_client_datasets(
        dataset_name=dataset_name,
        data_root=data_root,
        partition_id=partition_id,
        num_partitions=num_partitions,
        partition=partition,
        dirichlet_alpha=dirichlet_alpha,
        val_fraction=val_fraction,
        seed=seed,
    )

    return ClientLoaders(
        train=make_loader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + partition_id,
            device=device,
        ),
        val=make_loader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            device=device,
        ),
    )


def load_test_loader(
    dataset_name: str,
    data_root: str | Path,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    """Создаёт загрузчик общего тестового датасета."""
    dataset = load_dataset(
        dataset_name=dataset_name,
        root=data_root,
        train=False,
    )

    return make_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        device=device,
    )
