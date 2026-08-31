from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord,
)
from flwr.serverapp import (
    Grid,
    ServerApp,
)
from flwr.serverapp.strategy import (
    FedAdam,
    FedAvg,
    FedAvgM,
    FedMedian,
    FedTrimmedAvg,
    Strategy,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .attacks import TriggeredTestDataset
from .constants import Attacks
from .data import (
    dataset_metadata,
    load_dataset,
    make_loader,
)
from .models import create_model
from .train_utils import (
    evaluate_classification,
    evaluate_model,
    seed_everything,
)

app = ServerApp()

Config = Mapping[str, Any]
NumericMetrics = dict[str, int | float]


@dataclass(frozen=True)
class ServerSettings:
    """Основные параметры серверного эксперимента."""

    dataset_name: str
    attack: str
    seed: int
    num_clients: int
    num_rounds: int
    batch_size: int
    output_dir: Path

    @classmethod
    def from_config(cls, config: Config) -> ServerSettings:
        """Создаёт настройки сервера из конфигурации Flower."""
        return cls(
            dataset_name=str(config["dataset"]),
            attack=str(config["attack"]).lower(),
            seed=int(config["seed"]),
            num_clients=int(config["num-clients"]),
            num_rounds=int(config["num-server-rounds"]),
            batch_size=int(config["batch-size"]),
            output_dir=Path(str(config["output-dir"])),
        )


@dataclass
class EvaluationState:
    """Хранит результаты последней глобальной оценки."""

    metrics: NumericMetrics
    confusion_matrix: np.ndarray | None = None
    classification_details: dict[str, Any] | None = None


def _device() -> torch.device:
    """Возвращает доступное устройство для вычислений."""
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def _write_metrics_row(
    path: Path,
    row: NumericMetrics,
) -> None:
    """Добавляет строку серверных метрик в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def _save_confusion_matrix(
    path: Path,
    matrix: np.ndarray,
    class_names: list[str],
) -> None:
    """Сохраняет confusion matrix в CSV."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\predicted", *class_names])

        for class_name, row in zip(
            class_names,
            matrix,
            strict=True,
        ):
            writer.writerow(
                [
                    class_name,
                    *(int(value) for value in row),
                ]
            )


def _per_class_report(
    class_names: list[str],
    details: Mapping[str, Any],
) -> dict[str, dict[str, int | float]]:
    """Формирует поклассовые precision, recall, F1 и support."""
    precision_values = details["precision_per_class"]
    recall_values = details["recall_per_class"]
    f1_values = details["f1_per_class"]
    support_values = details["support_per_class"]

    return {
        class_name: {
            "precision": float(precision_values[index]),
            "recall": float(recall_values[index]),
            "f1": float(f1_values[index]),
            "support": int(support_values[index]),
        }
        for index, class_name in enumerate(class_names)
    }


def _save_classification_report(
    path: Path,
    class_names: list[str],
    details: Mapping[str, Any],
    final_metrics: Mapping[str, int | float],
) -> None:
    """Сохраняет общие и поклассовые метрики в JSON."""
    report = {
        "accuracy": float(final_metrics["clean_accuracy"]),
        "loss": float(final_metrics["clean_loss"]),
        "macro_avg": {
            "precision": float(final_metrics["clean_precision_macro"]),
            "recall": float(final_metrics["clean_recall_macro"]),
            "f1": float(final_metrics["clean_f1_macro"]),
        },
        "weighted_avg": {
            "precision": float(final_metrics["clean_precision_weighted"]),
            "recall": float(final_metrics["clean_recall_weighted"]),
            "f1": float(final_metrics["clean_f1_weighted"]),
        },
        "backdoor_asr": float(final_metrics["backdoor_asr"]),
        "per_class": _per_class_report(
            class_names=class_names,
            details=details,
        ),
    }

    path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _prepare_output_directory(output_dir: Path) -> Path:
    """Подготавливает каталог результатов и новый файл метрик."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"

    if metrics_path.exists():
        metrics_path.unlink()

    return metrics_path


def _create_triggered_loader(
    config: Config,
    settings: ServerSettings,
    test_dataset: Dataset,
    device: torch.device,
) -> DataLoader | None:
    """Создаёт тестовый загрузчик с триггерами для backdoor-атаки."""
    if settings.attack != Attacks.BACKDOOR:
        return None

    triggered_dataset = TriggeredTestDataset(
        dataset=test_dataset,
        source_label=int(config["label-source"]),
        target_label=int(config["label-target"]),
    )

    return make_loader(
        dataset=triggered_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        seed=settings.seed,
        device=device,
    )


def _load_global_model(
    in_channels: int,
    num_classes: int,
    arrays: ArrayRecord,
    device: torch.device,
) -> nn.Module:
    """Создаёт модель и загружает агрегированные параметры."""
    model = create_model(
        in_channels=in_channels,
        num_classes=num_classes,
    )

    model.load_state_dict(
        arrays.to_torch_state_dict(),
        strict=True,
    )
    model.to(device)

    return model


def _build_metrics_row(
    server_round: int,
    classification_metrics: Mapping[str, float],
    backdoor_asr: float,
) -> NumericMetrics:
    """Формирует строку метрик текущего серверного раунда."""
    return {
        "round": int(server_round),
        "clean_loss": float(classification_metrics["loss"]),
        "clean_accuracy": float(classification_metrics["accuracy"]),
        "clean_precision_macro": float(
            classification_metrics["precision_macro"]
        ),
        "clean_recall_macro": float(classification_metrics["recall_macro"]),
        "clean_f1_macro": float(classification_metrics["f1_macro"]),
        "clean_precision_weighted": float(
            classification_metrics["precision_weighted"]
        ),
        "clean_recall_weighted": float(
            classification_metrics["recall_weighted"]
        ),
        "clean_f1_weighted": float(classification_metrics["f1_weighted"]),
        "backdoor_asr": float(backdoor_asr),
    }


def _metric_record(
    row: Mapping[str, int | float],
) -> MetricRecord:
    """Преобразует серверные метрики в Flower MetricRecord."""
    return MetricRecord(
        {
            "clean_loss": float(row["clean_loss"]),
            "clean_accuracy": float(row["clean_accuracy"]),
            "clean_precision_macro": float(row["clean_precision_macro"]),
            "clean_recall_macro": float(row["clean_recall_macro"]),
            "clean_f1_macro": float(row["clean_f1_macro"]),
            "clean_precision_weighted": float(row["clean_precision_weighted"]),
            "clean_recall_weighted": float(row["clean_recall_weighted"]),
            "clean_f1_weighted": float(row["clean_f1_weighted"]),
            "backdoor_asr": float(row["backdoor_asr"]),
        }
    )


def _make_global_evaluator(
    in_channels: int,
    num_classes: int,
    device: torch.device,
    clean_test_loader: DataLoader,
    triggered_loader: DataLoader | None,
    metrics_path: Path,
    state: EvaluationState,
):
    """Создаёт функцию глобальной оценки модели после каждого раунда."""

    def global_evaluate(
        server_round: int,
        arrays: ArrayRecord,
    ) -> MetricRecord:
        """Оценивает агрегированную модель и сохраняет результаты."""
        model = _load_global_model(
            in_channels=in_channels,
            num_classes=num_classes,
            arrays=arrays,
            device=device,
        )

        (
            classification_metrics,
            confusion_matrix,
            classification_details,
        ) = evaluate_classification(
            model=model,
            loader=clean_test_loader,
            device=device,
            num_classes=num_classes,
        )

        backdoor_asr = 0.0

        if triggered_loader is not None:
            _, backdoor_asr = evaluate_model(
                model=model,
                loader=triggered_loader,
                device=device,
            )

        row = _build_metrics_row(
            server_round=server_round,
            classification_metrics=classification_metrics,
            backdoor_asr=backdoor_asr,
        )

        state.metrics.clear()
        state.metrics.update(row)
        state.confusion_matrix = confusion_matrix.copy()
        state.classification_details = classification_details

        _write_metrics_row(metrics_path, row)

        print(
            "SERVER_METRICS "
            + json.dumps(
                row,
                ensure_ascii=False,
            )
        )

        return _metric_record(row)

    return global_evaluate


def _create_strategy(
    config: Config,
    num_clients: int,
) -> Strategy:
    """Создаёт выбранную в YAML стратегию агрегации Flower."""
    strategy_name = str(config.get("strategy", "fedavg")).lower()
    minimum_nodes = min(2, num_clients)

    common_kwargs = {
        "fraction_train": float(config["fraction-train"]),
        "fraction_evaluate": float(config["fraction-evaluate"]),
        "min_train_nodes": minimum_nodes,
        "min_evaluate_nodes": minimum_nodes,
        "min_available_nodes": num_clients,
        "weighted_by_key": "num-examples",
    }

    if strategy_name == "fedavg":
        return FedAvg(**common_kwargs)

    if strategy_name == "fedavgm":
        return FedAvgM(
            **common_kwargs,
            server_learning_rate=float(config["server-learning-rate"]),
            server_momentum=float(config["server-momentum"]),
        )

    if strategy_name == "fedadam":
        return FedAdam(
            **common_kwargs,
            eta=float(config["server-learning-rate"]),
            eta_l=float(config["fedopt-client-learning-rate"]),
            beta_1=float(config["fedopt-beta-1"]),
            beta_2=float(config["fedopt-beta-2"]),
            tau=float(config["fedopt-tau"]),
        )

    if strategy_name == "fedmedian":
        return FedMedian(**common_kwargs)

    if strategy_name == "fedtrimmedavg":
        trimmed_beta = float(config["trimmed-beta"])

        if not 0.0 <= trimmed_beta < 0.5:
            raise ValueError("trimmed-beta must be in range [0.0, 0.5)")

        return FedTrimmedAvg(
            **common_kwargs,
            beta=trimmed_beta,
        )

    supported = "fedavg, fedavgm, fedadam, fedmedian, fedtrimmedavg"
    raise ValueError(
        f"Unsupported strategy: {strategy_name}. "
        f"Supported strategies: {supported}"
    )


def _save_final_model(
    arrays: ArrayRecord | None,
    output_dir: Path,
) -> None:
    """Сохраняет параметры итоговой глобальной модели."""
    if arrays is None:
        return

    final_state = arrays.to_torch_state_dict()

    torch.save(
        final_state,
        output_dir / "final_model.pt",
    )


def _save_final_reports(
    output_dir: Path,
    class_names: list[str],
    state: EvaluationState,
) -> None:
    """Сохраняет итоговую confusion matrix и classification report."""
    if state.confusion_matrix is not None:
        _save_confusion_matrix(
            path=output_dir / "confusion_matrix.csv",
            matrix=state.confusion_matrix,
            class_names=class_names,
        )

    if state.classification_details is not None and state.metrics:
        _save_classification_report(
            path=output_dir / "classification_report.json",
            class_names=class_names,
            details=state.classification_details,
            final_metrics=state.metrics,
        )


def _save_summary(
    path: Path,
    config: Config,
    settings: ServerSettings,
    final_metrics: Mapping[str, int | float],
) -> None:
    """Сохраняет параметры эксперимента и итоговые метрики."""
    summary = {
        "dataset": settings.dataset_name,
        "attack": settings.attack,
        "strategy": str(config.get("strategy", "fedavg")).lower(),
        "num_clients": settings.num_clients,
        "num_rounds": settings.num_rounds,
        "malicious_fraction": float(config["malicious-fraction"]),
        "partition": str(config["partition"]),
        "dirichlet_alpha": float(config["dirichlet-alpha"]),
        "final_metrics": dict(final_metrics),
    }

    path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


@app.main()
def main(
    grid: Grid,
    context: Context,
) -> None:
    """Запускает серверную часть федеративного эксперимента."""
    config = context.run_config
    settings = ServerSettings.from_config(config)

    seed_everything(settings.seed)

    device = _device()

    metrics_path = _prepare_output_directory(settings.output_dir)

    test_dataset = load_dataset(
        dataset_name=settings.dataset_name,
        root=str(config["data-root"]),
        train=False,
    )

    metadata = dataset_metadata(test_dataset)

    in_channels = metadata.in_channels
    num_classes = metadata.num_classes
    class_names = list(metadata.class_names)

    clean_test_loader = make_loader(
        dataset=test_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        seed=settings.seed,
        device=device,
    )

    triggered_loader = _create_triggered_loader(
        config=config,
        settings=settings,
        test_dataset=test_dataset,
        device=device,
    )

    evaluation_state = EvaluationState(metrics={})

    global_evaluate = _make_global_evaluator(
        in_channels=in_channels,
        num_classes=num_classes,
        device=device,
        clean_test_loader=clean_test_loader,
        triggered_loader=triggered_loader,
        metrics_path=metrics_path,
        state=evaluation_state,
    )

    initial_model = create_model(
        in_channels=in_channels,
        num_classes=num_classes,
    )
    initial_arrays = ArrayRecord(initial_model.state_dict())

    strategy = _create_strategy(
        config=config,
        num_clients=settings.num_clients,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=settings.num_rounds,
        train_config=ConfigRecord({}),
        evaluate_fn=global_evaluate,
    )

    _save_final_model(
        arrays=result.arrays,
        output_dir=settings.output_dir,
    )

    _save_final_reports(
        output_dir=settings.output_dir,
        class_names=class_names,
        state=evaluation_state,
    )

    _save_summary(
        path=settings.output_dir / "summary.json",
        config=config,
        settings=settings,
        final_metrics=evaluation_state.metrics,
    )
