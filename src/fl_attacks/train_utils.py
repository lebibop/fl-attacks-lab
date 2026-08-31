from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class _RunningMetrics:
    """Накапливает loss, число верных ответов и количество объектов."""

    total_loss: float = 0.0
    total_correct: int = 0
    total_examples: int = 0

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """Добавляет результаты одного батча."""
        batch_size = labels.size(0)

        self.total_loss += float(loss.item()) * batch_size
        self.total_correct += int(
            (logits.argmax(dim=1) == labels).sum().item()
        )
        self.total_examples += batch_size

    def result(self) -> tuple[float, float]:
        """Возвращает средние loss и accuracy."""
        if self.total_examples == 0:
            return 0.0, 0.0

        average_loss = self.total_loss / self.total_examples
        accuracy = self.total_correct / self.total_examples

        return average_loss, accuracy


def seed_everything(seed: int) -> None:
    """Фиксирует генераторы случайных чисел для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch_to_device(
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Переносит изображения и метки на выбранное устройство."""
    return (
        images.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
    )


def train_model(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[float, float]:
    """Обучает модель и возвращает средние loss и accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model.train()
    metrics = _RunningMetrics()

    for _ in range(epochs):
        for images, labels in loader:
            images, labels = _move_batch_to_device(
                images=images,
                labels=labels,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            metrics.update(
                logits=logits,
                labels=labels,
                loss=loss,
            )

    return metrics.result()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Оценивает модель и возвращает средние loss и accuracy."""
    criterion = nn.CrossEntropyLoss()

    model.eval()
    metrics = _RunningMetrics()

    for images, labels in loader:
        images, labels = _move_batch_to_device(
            images=images,
            labels=labels,
            device=device,
        )

        logits = model(images)
        loss = criterion(logits, labels)

        metrics.update(
            logits=logits,
            labels=labels,
            loss=loss,
        )

    return metrics.result()


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    """Выполняет поэлементное деление с заменой деления на ноль на 0."""
    result = np.zeros_like(numerator, dtype=np.float64)

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0,
    )

    return result


def _empty_classification_result(
    num_classes: int,
    confusion_matrix: np.ndarray,
) -> tuple[
    dict[str, float],
    np.ndarray,
    dict[str, Any],
]:
    """Формирует нулевой результат для пустого DataLoader."""
    metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
        "precision_macro": 0.0,
        "recall_macro": 0.0,
        "f1_macro": 0.0,
        "precision_weighted": 0.0,
        "recall_weighted": 0.0,
        "f1_weighted": 0.0,
    }

    details = {
        "precision_per_class": [0.0] * num_classes,
        "recall_per_class": [0.0] * num_classes,
        "f1_per_class": [0.0] * num_classes,
        "support_per_class": [0] * num_classes,
    }

    return metrics, confusion_matrix, details


def _classification_metrics(
    confusion_matrix: np.ndarray,
    total_loss: float,
    total_examples: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Вычисляет общие и поклассовые метрики по confusion matrix."""
    true_positive = np.diag(confusion_matrix).astype(np.float64)
    predicted_count = confusion_matrix.sum(axis=0).astype(np.float64)
    support = confusion_matrix.sum(axis=1).astype(np.float64)

    precision_per_class = _safe_divide(
        true_positive,
        predicted_count,
    )
    recall_per_class = _safe_divide(
        true_positive,
        support,
    )
    f1_per_class = _safe_divide(
        2.0 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
    )

    total_correct = float(true_positive.sum())
    class_weights = _safe_divide(
        support,
        np.asarray([support.sum()], dtype=np.float64),
    )

    metrics = {
        "loss": float(total_loss / total_examples),
        "accuracy": float(total_correct / total_examples),
        "precision_macro": float(precision_per_class.mean()),
        "recall_macro": float(recall_per_class.mean()),
        "f1_macro": float(f1_per_class.mean()),
        "precision_weighted": float(
            (precision_per_class * class_weights).sum()
        ),
        "recall_weighted": float((recall_per_class * class_weights).sum()),
        "f1_weighted": float((f1_per_class * class_weights).sum()),
    }

    details = {
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "support_per_class": support.astype(np.int64).tolist(),
    }

    return metrics, details


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[
    dict[str, float],
    np.ndarray,
    dict[str, Any],
]:
    """
    Выполняет полную оценку многоклассовой модели.

    Возвращает общие метрики, confusion matrix и поклассовые
    precision, recall, F1 и support. Строки confusion matrix
    соответствуют истинным классам, столбцы — предсказанным.
    """
    criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    total_examples = 0
    confusion_matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    for images, labels in loader:
        images, labels = _move_batch_to_device(
            images=images,
            labels=labels,
            device=device,
        )

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        labels_numpy = labels.detach().cpu().numpy().astype(np.int64)
        predictions_numpy = predictions.detach().cpu().numpy().astype(np.int64)

        np.add.at(
            confusion_matrix,
            (labels_numpy, predictions_numpy),
            1,
        )

    if total_examples == 0:
        return _empty_classification_result(
            num_classes=num_classes,
            confusion_matrix=confusion_matrix,
        )

    metrics, details = _classification_metrics(
        confusion_matrix=confusion_matrix,
        total_loss=total_loss,
        total_examples=total_examples,
    )

    return metrics, confusion_matrix, details
