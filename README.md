# Запуск экспериментов

Все команды выполняются из корневой папки проекта, где находятся:

* `run.py`;
* `pyproject.toml`;
* папка `configs`.

## Установка зависимостей

Перед первым запуском выполните:

```bash
uv sync
```

Эксперименты запускаются через `run.py` в окружении `uv`.

## Быстрый запуск

```bash
pkill -f flower-supernode 2>/dev/null || true
pkill -f flower-superlink 2>/dev/null || true
pkill -f "flwr run" 2>/dev/null || true

uv run python run.py \
  --config configs/experiments.yaml \
  --experiment EXPERIMENT_NAME
```

Первые три команды завершают процессы Flower, которые могли остаться после предыдущего запуска.

## Параметры запуска

### `--config`

Указывает путь к YAML-файлу с настройками экспериментов:

```bash
--config configs/experiments.yaml
```

Можно использовать другой файл:

```bash
--config configs/my_experiments.yaml
```

Для стандартного файла `configs/experiments.yaml` параметр можно не указывать:

```bash
uv run python run.py --experiment backdoor
```

### `--experiment`

Указывает эксперимент, который нужно запустить.

Доступные значения:

* `clean` — обучение без атаки;
* `label_flip` — подмена меток;
* `sign_flip` — инвертирование обновления модели;
* `backdoor` — внедрение триггера в изображения;
* `gaussian_noise` — добавление шума к обновлениям модели;
* `all` — последовательный запуск всех экспериментов.

Название должно совпадать со значением `name` в `configs/experiments.yaml`:

```yaml
experiments:
  - name: backdoor
    attack: backdoor
```

Пример запуска:

```bash
uv run python run.py --experiment backdoor
```

### `--no-stream`

Отключает потоковый вывод логов Flower:

```bash
uv run python run.py \
  --experiment backdoor \
  --no-stream
```

## Запуск одного эксперимента

```bash
pkill -f flower-supernode 2>/dev/null || true
pkill -f flower-superlink 2>/dev/null || true
pkill -f "flwr run" 2>/dev/null || true

uv run python run.py --experiment label_flip
```

Вместо `label_flip` можно указать любое имя из раздела `experiments`.

## Запуск всех экспериментов

```bash
pkill -f flower-supernode 2>/dev/null || true
pkill -f flower-superlink 2>/dev/null || true
pkill -f "flwr run" 2>/dev/null || true

uv run python run.py --experiment all
```

Команда последовательно запускает:

```text
clean
label_flip
sign_flip
backdoor
gaussian_noise
```

## Выбор датасета

Все эксперименты выполняются на одном датасете, указанном в разделе `defaults`:

```yaml
defaults:
  dataset: fashionmnist
```

Поддерживаемые значения:

```yaml
dataset: mnist
dataset: fashionmnist
dataset: cifar10
```

Чтобы запустить все атаки на другом датасете, достаточно изменить одну строку:

```yaml
defaults:
  dataset: mnist
```

После этого команда:

```bash
uv run python run.py --experiment all
```

запустит все пять экспериментов на MNIST.

Использование одного датасета позволяет корректно сравнивать атаки. В этом случае условия обучения остаются одинаковыми, а эксперименты отличаются только типом вредоносного воздействия.

Например, результаты `sign_flip` и `backdoor` можно сравнивать напрямую, поскольку обе атаки применяются к одной модели, одному датасету и одинаковому распределению данных.

Для основного сравнения рекомендуется использовать:

```yaml
dataset: fashionmnist
```

Fashion-MNIST сложнее обычного MNIST, но требует значительно меньше ресурсов, чем CIFAR-10.


## Выбор стратегии агрегации

Стратегия задаётся параметром `strategy` в разделе `defaults` или внутри отдельного эксперимента:

```yaml
defaults:
  strategy: fedavg
```

Поддерживаются пять стратегий:

- `fedavg` — обычное взвешенное усреднение обновлений клиентов;
- `fedavgm` — FedAvg с серверным momentum;
- `fedadam` — адаптивная серверная оптимизация Adam;
- `fedmedian` — покоординатная медиана, более устойчивая к выбросам;
- `fedtrimmedavg` — усечённое среднее, отбрасывающее крайние значения.

### Общая стратегия для всех экспериментов

```yaml
defaults:
  strategy: fedavg
```

Если внутри эксперимента стратегия не указана, используется значение из `defaults`.

### Стратегия для отдельного эксперимента

```yaml
experiments:
  - name: sign_flip_median
    attack: sign_flip
    strategy: fedmedian
```

Параметр внутри эксперимента переопределяет значение из `defaults`.

### Параметры стратегий

```yaml
defaults:
  # Доступные значения:
  # fedavg | fedavgm | fedadam | fedmedian | fedtrimmedavg
  strategy: fedavg

  # Используется FedAvgM и FedAdam
  server-learning-rate: 1.0

  # Используется только FedAvgM
  server-momentum: 0.9

  # Используются только FedAdam
  fedopt-client-learning-rate: 0.1
  fedopt-beta-1: 0.9
  fedopt-beta-2: 0.99
  fedopt-tau: 0.001

  # Используется только FedTrimmedAvg
  # Допустимый диапазон: [0.0, 0.5)
  trimmed-beta: 0.2
```

Параметры, которые не относятся к выбранной стратегии, не используются.

### Примеры

FedAvg:

```yaml
experiments:
  - name: clean_fedavg
    attack: none
    strategy: fedavg
```

FedAvgM:

```yaml
experiments:
  - name: clean_fedavgm
    attack: none
    strategy: fedavgm
    server-learning-rate: 1.0
    server-momentum: 0.9
```

FedAdam:

```yaml
experiments:
  - name: clean_fedadam
    attack: none
    strategy: fedadam
    server-learning-rate: 0.1
    fedopt-client-learning-rate: 0.1
    fedopt-beta-1: 0.9
    fedopt-beta-2: 0.99
    fedopt-tau: 0.001
```

FedMedian:

```yaml
experiments:
  - name: sign_flip_median
    attack: sign_flip
    strategy: fedmedian
    model-attack-scale: 5.0
```

FedTrimmedAvg:

```yaml
experiments:
  - name: sign_flip_trimmed
    attack: sign_flip
    strategy: fedtrimmedavg
    trimmed-beta: 0.2
    model-attack-scale: 5.0
```

## Сравнение стратегий

Для корректного сравнения стратегий оставляйте одинаковыми:

- датасет;
- тип и параметры атаки;
- количество клиентов;
- количество серверных раундов;
- число локальных эпох;
- способ разбиения данных;
- долю вредоносных клиентов;
- `seed`;
- параметры локального оптимизатора.

Меняться должна только стратегия и параметры, относящиеся к ней.

Пример сравнения одной атаки:

```yaml
experiments:
  - name: sign_flip_fedavg
    attack: sign_flip
    strategy: fedavg
    model-attack-scale: 5.0

  - name: sign_flip_fedmedian
    attack: sign_flip
    strategy: fedmedian
    model-attack-scale: 5.0

  - name: sign_flip_trimmed
    attack: sign_flip
    strategy: fedtrimmedavg
    trimmed-beta: 0.2
    model-attack-scale: 5.0
```

Запуск всех сравнительных экспериментов:

```bash
uv run python run.py --experiment all
```

## Настройка экспериментов

Общие параметры задаются в разделе `defaults`:

```yaml
defaults:
  dataset: fashionmnist
  strategy: fedavg
  num-server-rounds: 15
  num-clients: 10
  local-epochs: 2
  batch-size: 64
  malicious-fraction: 0.3
```

Они применяются ко всем экспериментам.

Параметры внутри отдельного эксперимента переопределяют значения из `defaults`:

```yaml
experiments:
  - name: sign_flip
    attack: sign_flip
    model-attack-scale: 5.0
```

Для корректного сравнения атак рекомендуется оставлять одинаковыми:

* датасет;
* стратегию агрегации;
* количество клиентов;
* количество раундов;
* число локальных эпох;
* способ разбиения данных;
* долю вредоносных клиентов;
* seed;
* параметры оптимизатора.

Между экспериментами должны изменяться только тип атаки и её параметры.

## Результаты

Для каждого эксперимента создаётся отдельная папка:

```text
outputs/
└── ДАТА_ВРЕМЯ_НАЗВАНИЕ_ЭКСПЕРИМЕНТА/
```

Например:

```text
outputs/
├── 2026-07-24__12-00-00__clean/
├── 2026-07-24__12-01-00__label_flip/
├── 2026-07-24__12-02-00__sign_flip/
├── 2026-07-24__12-03-00__backdoor/
└── 2026-07-24__12-04-00__gaussian_noise/
```

Внутри папки сохраняются:

* `config.yaml` — итоговая конфигурация запуска;
* `metrics.csv` — метрики по серверным раундам;
* `metrics.png` — графики метрик;
* `confusion_matrix.csv` — итоговая confusion matrix;
* `confusion_matrix.png` — визуализация confusion matrix;
* `classification_report.json` — precision, recall, F1 и support по классам;
* `summary.json` — параметры, выбранная стратегия и итоговые метрики эксперимента;
* `final_model.pt` — веса итоговой глобальной модели.

Для backdoor-эксперимента дополнительно рассчитывается `Backdoor Attack Success Rate` — доля изображений с триггером, отнесённых моделью к целевому классу.
