from __future__ import annotations

from torch import Tensor, nn


def _validate_positive(value: int, name: str) -> int:
    """Проверяет положительный целочисленный параметр."""
    normalized = int(value)

    if normalized <= 0:
        raise ValueError(f"{name} must be greater than zero")

    return normalized


def _group_count(num_channels: int) -> int:
    """Выбирает максимальное число групп, делящее число каналов."""
    for num_groups in (8, 4, 2, 1):
        if num_channels % num_groups == 0:
            return num_groups

    return 1


def _normalization(num_channels: int) -> nn.GroupNorm:
    """Создаёт GroupNorm для заданного числа каналов."""
    return nn.GroupNorm(
        num_groups=_group_count(num_channels),
        num_channels=num_channels,
    )


class ResidualBlock(nn.Module):
    """Остаточный блок из двух свёрток."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        in_channels = _validate_positive(
            in_channels,
            "in_channels",
        )
        out_channels = _validate_positive(
            out_channels,
            "out_channels",
        )
        stride = _validate_positive(stride, "stride")

        self.main_path = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            _normalization(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _normalization(out_channels),
        )

        if stride == 1 and in_channels == out_channels:
            self.skip_path: nn.Module = nn.Identity()
        else:
            self.skip_path = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _normalization(out_channels),
            )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        """Объединяет основной и остаточный пути."""
        residual = self.skip_path(inputs)
        features = self.main_path(inputs)

        return self.activation(features + residual)


class CNN(nn.Module):
    """
    Компактная универсальная CNN с остаточными блоками.

    Архитектура рассчитана на небольшие изображения и подходит для
    MNIST, FashionMNIST, EMNIST, CIFAR-10 и CIFAR-100.
    """

    FEATURE_DIM = 128

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.in_channels = _validate_positive(
            in_channels,
            "in_channels",
        )
        self.num_classes = _validate_positive(
            num_classes,
            "num_classes",
        )

        if self.num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than one"
            )

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _normalization(32),
            nn.ReLU(inplace=True),
        )

        self.feature_extractor = nn.Sequential(
            ResidualBlock(
                in_channels=32,
                out_channels=32,
            ),
            ResidualBlock(
                in_channels=32,
                out_channels=64,
                stride=2,
            ),
            ResidualBlock(
                in_channels=64,
                out_channels=64,
            ),
            ResidualBlock(
                in_channels=64,
                out_channels=self.FEATURE_DIM,
                stride=2,
            ),
            ResidualBlock(
                in_channels=self.FEATURE_DIM,
                out_channels=self.FEATURE_DIM,
            ),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(
                self.FEATURE_DIM,
                self.num_classes,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Инициализирует обучаемые параметры сети."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.GroupNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_inputs(self, inputs: Tensor) -> None:
        """Проверяет размерность и число каналов входного тензора."""
        if inputs.ndim != 4:
            raise ValueError(
                "Expected input shape [N, C, H, W], "
                f"received {tuple(inputs.shape)}"
            )

        received_channels = int(inputs.shape[1])

        if received_channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, "
                f"received {received_channels}"
            )

    def forward(self, inputs: Tensor) -> Tensor:
        """Возвращает логиты классов."""
        self._validate_inputs(inputs)

        features = self.stem(inputs)
        features = self.feature_extractor(features)
        features = self.global_pool(features)

        return self.classifier(features)


def create_model(
    in_channels: int,
    num_classes: int,
) -> CNN:
    """Создаёт модель по автоматически определённым параметрам датасета."""
    return CNN(
        in_channels=in_channels,
        num_classes=num_classes,
    )