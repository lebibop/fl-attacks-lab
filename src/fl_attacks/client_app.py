from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp
from torch import nn

from .attacks import (
    SUPPORTED_ATTACKS,
    apply_model_attack,
    attack_is_active,
)
from .constants import Attacks
from .data import (
    IndexedSubset,
    dataset_metadata,
    load_client_datasets,
    make_loader,
)
from .models import create_model
from .train_utils import (
    evaluate_classification,
    seed_everything,
    train_model,
)

app = ClientApp()

Config = Mapping[str, Any]


@dataclass(frozen=True)
class ClientIdentity:
    """Идентификатор клиента и общее количество клиентов."""

    client_id: int
    num_clients: int


def _device() -> torch.device:
    """Возвращает доступное устройство для вычислений."""
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def _client_identity(context: Context) -> ClientIdentity:
    """Извлекает параметры клиента из конфигурации Flower."""
    return ClientIdentity(
        client_id=int(context.node_config["partition-id"]),
        num_clients=int(context.node_config["num-partitions"]),
    )


def _validated_attack(config: Config) -> str:
    """Нормализует и проверяет название атаки."""
    attack = str(config["attack"]).lower()

    if attack not in SUPPORTED_ATTACKS:
        supported = ", ".join(sorted(SUPPORTED_ATTACKS))
        raise ValueError(
            f"Unsupported attack: {attack}. Supported attacks: {supported}"
        )

    return attack


def _round_seed(
    base_seed: int,
    client_id: int,
    server_round: int,
) -> int:
    """Формирует воспроизводимый seed клиента для текущего раунда."""
    return base_seed + client_id + 1000 * server_round


def _load_model(
    msg: Message,
    in_channels: int,
    num_classes: int,
) -> nn.Module:
    """Создаёт модель и загружает параметры, полученные от сервера."""
    model = create_model(
        in_channels=in_channels,
        num_classes=num_classes,
    )
    received_state = msg.content["arrays"].to_torch_state_dict()

    model.load_state_dict(
        received_state,
        strict=True,
    )

    return model


def _load_local_datasets(
    config: Config,
    identity: ClientIdentity,
    dataset_name: str,
    seed: int,
) -> tuple[IndexedSubset, IndexedSubset]:
    """Загружает локальные обучающие и валидационные данные клиента."""
    return load_client_datasets(
        dataset_name=dataset_name,
        data_root=str(config["data-root"]),
        partition_id=identity.client_id,
        num_partitions=identity.num_clients,
        partition=str(config["partition"]),
        dirichlet_alpha=float(config["dirichlet-alpha"]),
        val_fraction=float(config["val-fraction"]),
        seed=seed,
    )


def _attack_active(
    config: Config,
    attack: str,
    identity: ClientIdentity,
    server_round: int,
    seed: int,
) -> bool:
    """Проверяет, должен ли клиент атаковать в текущем раунде."""
    if attack == Attacks.NONE:
        return False

    return attack_is_active(
        client_id=identity.client_id,
        server_round=server_round,
        num_clients=identity.num_clients,
        malicious_fraction=float(config["malicious-fraction"]),
        attack_start_round=int(config["attack-start-round"]),
        attack_end_round=int(config["attack-end-round"]),
        seed=seed,
    )


def _reply(
    msg: Message,
    metrics: MetricRecord,
    model: nn.Module | None = None,
) -> Message:
    """Создаёт ответ Flower с метриками и параметрами модели."""
    records: dict[str, Any] = {
        "metrics": metrics,
    }

    if model is not None:
        records["arrays"] = ArrayRecord(model.state_dict())

    return Message(
        content=RecordDict(records),
        reply_to=msg,
    )


@app.train()
def train(
    msg: Message,
    context: Context,
) -> Message:
    """Обучает локальную модель и при необходимости атакует её обновление."""
    config = context.run_config
    dataset_name = str(config["dataset"])
    attack = _validated_attack(config)

    identity = _client_identity(context)
    server_round = _server_round(msg)
    seed = int(config["seed"])
    round_seed = _round_seed(
        base_seed=seed,
        client_id=identity.client_id,
        server_round=server_round,
    )

    seed_everything(round_seed)
    device = _device()

    train_dataset, _ = _load_local_datasets(
        config=config,
        identity=identity,
        dataset_name=dataset_name,
        seed=seed,
    )
    metadata = dataset_metadata(train_dataset)

    model = _load_model(
        msg=msg,
        in_channels=metadata.in_channels,
        num_classes=metadata.num_classes,
    )

    # Эта копия — точка отсчёта Δw = w_local - w_global.
    global_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

    model.to(device)

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=int(config["batch-size"]),
        shuffle=True,
        seed=round_seed,
        device=device,
    )

    train_loss, train_accuracy = train_model(
        model=model,
        loader=train_loader,
        epochs=int(config["local-epochs"]),
        learning_rate=float(config["learning-rate"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight-decay"]),
        device=device,
    )

    active = _attack_active(
        config=config,
        attack=attack,
        identity=identity,
        server_round=server_round,
        seed=seed,
    )

    diagnostics = apply_model_attack(
        model=model,
        global_state=global_state,
        attack=attack if active else Attacks.NONE,
        config=config,
    )

    metrics = MetricRecord(
        {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "malicious": int(active),
            "update_l2_before": float(diagnostics.update_l2_before),
            "update_l2_after": float(diagnostics.update_l2_after),
            "update_l2_ratio": float(diagnostics.update_l2_ratio),
            "num-examples": len(train_dataset),
        }
    )

    return _reply(
        msg=msg,
        metrics=metrics,
        model=model,
    )


@app.evaluate()
def evaluate(
    msg: Message,
    context: Context,
) -> Message:
    """Оценивает глобальную модель на локальной валидационной выборке."""
    config = context.run_config
    dataset_name = str(config["dataset"])
    identity = _client_identity(context)
    seed = int(config["seed"])
    device = _device()

    _, val_dataset = _load_local_datasets(
        config=config,
        identity=identity,
        dataset_name=dataset_name,
        seed=seed,
    )
    metadata = dataset_metadata(val_dataset)

    model = _load_model(
        msg=msg,
        in_channels=metadata.in_channels,
        num_classes=metadata.num_classes,
    )
    model.to(device)

    val_loader = make_loader(
        dataset=val_dataset,
        batch_size=int(config["batch-size"]),
        shuffle=False,
        seed=seed,
        device=device,
    )

    classification_metrics, _, _ = evaluate_classification(
        model=model,
        loader=val_loader,
        device=device,
        num_classes=metadata.num_classes,
    )

    metrics = MetricRecord(
        {
            "eval_loss": float(classification_metrics["loss"]),
            "eval_accuracy": float(classification_metrics["accuracy"]),
            "eval_precision_macro": float(
                classification_metrics["precision_macro"]
            ),
            "eval_recall_macro": float(
                classification_metrics["recall_macro"]
            ),
            "eval_f1_macro": float(classification_metrics["f1_macro"]),
            "eval_precision_weighted": float(
                classification_metrics["precision_weighted"]
            ),
            "eval_recall_weighted": float(
                classification_metrics["recall_weighted"]
            ),
            "eval_f1_weighted": float(
                classification_metrics["f1_weighted"]
            ),
            "num-examples": len(val_dataset),
        }
    )

    return _reply(
        msg=msg,
        metrics=metrics,
    )


def _server_round(msg: Message) -> int:
    """Безопасно извлекает номер серверного раунда."""
    config_record = msg.content["config"]

    if not isinstance(config_record, ConfigRecord):
        raise TypeError("Message record 'config' must be a ConfigRecord")

    value = config_record["server-round"]

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Config value 'server-round' must be an integer")

    return value
