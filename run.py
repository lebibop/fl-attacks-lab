from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent

# Добавляет корень проекта в путь импорта независимо от рабочей директории.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fl_attacks.data import download_dataset  # noqa: E402

Experiment = dict[str, Any]
Process = subprocess.Popen[Any]

# Локальные файлы конфигурации и журналов Flower.
FLOWER_HOME = ROOT / ".flower-local"
FLOWER_LOG_DIR = FLOWER_HOME / "logs"

# Порты локальных компонентов Flower.
SUPERLINK_FLEET_PORT = 9092
SUPERLINK_CONTROL_PORT = 9093
SUPERNODE_FIRST_PORT = 9094

# Параметры ожидания и завершения процессов.
PORT_CHECK_INTERVAL_SECONDS = 0.2
SUPERNODE_REGISTRATION_DELAY_SECONDS = 1.0
PROCESS_STOP_TIMEOUT_SECONDS = 8
PROCESS_KILL_TIMEOUT_SECONDS = 3

# Параметры старого Ray Runtime, которые не передаются в --run-config.
RUNTIME_ONLY_KEYS = {"client-cpus", "client-gpus"}


@dataclass
class FederationRuntime:
    """Хранит процессы локальной федерации и связанные с ними журналы."""

    processes: list[Process] = field(default_factory=list)
    log_files: list[TextIO] = field(default_factory=list)
    log_paths: list[Path] = field(default_factory=list)

    def stop(self) -> None:
        """Останавливает процессы федерации и закрывает файлы журналов."""
        for process in reversed(self.processes):
            stop_process(process)

        for log_file in self.log_files:
            if not log_file.closed:
                log_file.close()

    def print_log_tails(self) -> None:
        """Выводит последние строки журналов всех процессов."""
        for path in self.log_paths:
            print_log_tail(path)


def parse_args() -> argparse.Namespace:
    """Читает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Run one or all Flower federated-learning attack experiments"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiments.yaml",
        help="Path to YAML experiment configuration",
    )
    parser.add_argument(
        "--experiment",
        default="all",
        help="Experiment name or 'all'",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not stream Flower logs",
    )
    return parser.parse_args()


def load_experiments(config_path: Path) -> list[Experiment]:
    """Загружает эксперименты и объединяет их с параметрами defaults."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Корень YAML-конфигурации должен быть объектом")

    defaults_raw = raw.get("defaults", {})
    experiments_raw = raw.get("experiments", [])

    if not isinstance(defaults_raw, dict):
        raise ValueError("Поле defaults должно быть объектом")

    if not isinstance(experiments_raw, list):
        raise ValueError("Поле experiments должно быть списком")

    experiments: list[Experiment] = []

    for index, item in enumerate(experiments_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Эксперимент №{index} должен быть объектом")

        experiment = dict(defaults_raw) | dict(item)

        if "name" not in experiment:
            raise ValueError("У каждого эксперимента должно быть поле name")

        experiments.append(experiment)

    if not experiments:
        raise ValueError("В YAML не найдено ни одного эксперимента")

    return experiments


def select_experiments(
    experiments: list[Experiment],
    requested_name: str,
) -> list[Experiment]:
    """Возвращает все эксперименты или один эксперимент по имени."""
    if requested_name == "all":
        return experiments

    selected = [
        experiment
        for experiment in experiments
        if str(experiment["name"]) == requested_name
    ]

    if not selected:
        raise SystemExit(f"Эксперимент не найден: {requested_name}")

    return selected


def format_value(value: Any) -> str:
    """Преобразует значение Python в формат Flower --run-config."""
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return repr(value)

    if isinstance(value, str):
        return json.dumps(value)

    raise TypeError(
        f"Неподдерживаемый тип конфигурации: {type(value).__name__}"
    )


def build_run_config(
    resolved: Experiment,
    data_root: Path,
) -> str:
    """Формирует строку параметров для Flower --run-config."""
    items: list[str] = []

    for key, original_value in resolved.items():
        if key == "name" or key in RUNTIME_ONLY_KEYS:
            continue

        value = data_root.as_posix() if key == "data-root" else original_value
        items.append(f"{key}={format_value(value)}")

    return " ".join(items)


def read_metric_rows(metrics_path: Path) -> list[dict[str, str]]:
    """Читает строки файла с метриками."""
    with metrics_path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def metric_values(
    rows: list[dict[str, str]],
    column_name: str,
) -> list[float]:
    """Извлекает числовые значения указанной метрики."""
    return [float(row.get(column_name, 0.0) or 0.0) for row in rows]


def read_attack_name(output_dir: Path) -> str:
    """Возвращает тип атаки из сохранённой конфигурации."""
    config_path = output_dir / "config.yaml"

    if not config_path.exists():
        return "none"

    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(config_data, dict):
        return "none"

    return str(config_data.get("attack", "none"))


def configure_metric_axis(
    axis: Any,
    *,
    title: str,
    ylabel: str,
    unit_interval: bool = False,
) -> None:
    """Настраивает оси графика метрик."""
    axis.set_title(title)
    axis.set_xlabel("Server round")
    axis.set_ylabel(ylabel)

    if unit_interval:
        axis.set_ylim(0.0, 1.0)

    axis.grid(True, alpha=0.3)
    axis.legend()


def plot_metric_dashboard(
    output_dir: Path,
    rows: list[dict[str, str]],
) -> None:
    """Строит сводный график метрик по серверным раундам."""
    rounds = [int(row["round"]) for row in rows]

    clean_loss = metric_values(rows, "clean_loss")
    clean_accuracy = metric_values(rows, "clean_accuracy")
    precision_macro = metric_values(
        rows,
        "clean_precision_macro",
    )
    recall_macro = metric_values(
        rows,
        "clean_recall_macro",
    )
    f1_macro = metric_values(rows, "clean_f1_macro")
    precision_weighted = metric_values(
        rows,
        "clean_precision_weighted",
    )
    recall_weighted = metric_values(
        rows,
        "clean_recall_weighted",
    )
    f1_weighted = metric_values(
        rows,
        "clean_f1_weighted",
    )
    backdoor_asr = metric_values(rows, "backdoor_asr")

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    macro_axis, loss_axis = axes[0]
    weighted_axis, summary_axis = axes[1]

    for values, label in (
        (clean_accuracy, "Accuracy"),
        (precision_macro, "Precision macro"),
        (recall_macro, "Recall macro"),
        (f1_macro, "F1 macro"),
    ):
        macro_axis.plot(
            rounds,
            values,
            marker="o",
            label=label,
        )

    configure_metric_axis(
        macro_axis,
        title="Global classification metrics",
        ylabel="Metric value",
        unit_interval=True,
    )

    loss_axis.plot(
        rounds,
        clean_loss,
        marker="o",
        label="Clean loss",
    )
    configure_metric_axis(
        loss_axis,
        title="Global test loss",
        ylabel="Loss",
    )

    for values, label in (
        (precision_weighted, "Precision weighted"),
        (recall_weighted, "Recall weighted"),
        (f1_weighted, "F1 weighted"),
    ):
        weighted_axis.plot(
            rounds,
            values,
            marker="o",
            label=label,
        )

    configure_metric_axis(
        weighted_axis,
        title="Weighted classification metrics",
        ylabel="Metric value",
        unit_interval=True,
    )

    if read_attack_name(output_dir) == "backdoor":
        summary_axis.plot(
            rounds,
            backdoor_asr,
            marker="o",
            label="Backdoor ASR",
        )
        configure_metric_axis(
            summary_axis,
            title="Backdoor attack success rate",
            ylabel="ASR",
            unit_interval=True,
        )
    else:
        plot_final_metrics(
            summary_axis,
            accuracy=clean_accuracy[-1],
            precision=precision_macro[-1],
            recall=recall_macro[-1],
            f1=f1_macro[-1],
        )

    figure.suptitle(
        "Federated learning experiment metrics",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(
        output_dir / "metrics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_final_metrics(
    axis: Any,
    *,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
) -> None:
    """Строит столбчатую диаграмму итоговых метрик."""
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [accuracy, precision, recall, f1]
    bars = axis.bar(labels, values)

    axis.set_title("Final global metrics")
    axis.set_ylabel("Metric value")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )


def read_confusion_matrix(
    confusion_path: Path,
) -> tuple[
    list[str],
    np.ndarray[Any, np.dtype[np.int64]],
]:
    """Читает названия классов и confusion matrix из CSV."""
    with confusion_path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.reader(file))

    if len(rows) < 2:
        raise ValueError("Confusion matrix CSV is empty")

    class_names = rows[0][1:]
    matrix = np.asarray(
        [[int(value) for value in row[1:]] for row in rows[1:]],
        dtype=np.int64,
    )

    return class_names, matrix


def normalize_confusion_matrix(
    matrix: np.ndarray[Any, np.dtype[np.int64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Нормализует строки confusion matrix."""
    row_sums = matrix.sum(axis=1, keepdims=True)

    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )


def configure_confusion_axis(
    axis: Any,
    class_names: list[str],
) -> None:
    """Настраивает подписи осей confusion matrix."""
    positions = range(len(class_names))

    axis.set_xticks(positions)
    axis.set_yticks(positions)
    axis.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")


def annotate_matrix(
    axis: Any,
    matrix: np.ndarray[Any, Any],
    *,
    formatter: str,
    threshold: float,
) -> None:
    """Добавляет числовые значения в ячейки матрицы."""
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]

            axis.text(
                column_index,
                row_index,
                format(value, formatter),
                ha="center",
                va="center",
                fontsize=7,
                color=("white" if value > threshold else "black"),
            )


def plot_confusion_matrices(
    output_dir: Path,
    confusion_path: Path,
) -> None:
    """Строит исходную и нормализованную confusion matrix."""
    class_names, matrix = read_confusion_matrix(confusion_path)
    normalized_matrix = normalize_confusion_matrix(matrix)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(18, 8),
    )
    raw_axis, normalized_axis = axes

    raw_image = raw_axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )
    figure.colorbar(
        raw_image,
        ax=raw_axis,
        fraction=0.046,
        pad=0.04,
    )
    raw_axis.set_title("Confusion matrix: counts")

    normalized_image = normalized_axis.imshow(
        normalized_matrix,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    figure.colorbar(
        normalized_image,
        ax=normalized_axis,
        fraction=0.046,
        pad=0.04,
    )
    normalized_axis.set_title("Confusion matrix: normalized")

    configure_confusion_axis(
        raw_axis,
        class_names,
    )
    configure_confusion_axis(
        normalized_axis,
        class_names,
    )

    raw_threshold = matrix.max() / 2 if matrix.size else 0

    annotate_matrix(
        raw_axis,
        matrix,
        formatter="d",
        threshold=raw_threshold,
    )
    annotate_matrix(
        normalized_axis,
        normalized_matrix,
        formatter=".2f",
        threshold=0.5,
    )

    figure.suptitle(
        "Final global model confusion matrix",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(
        output_dir / "confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_metrics(output_dir: Path) -> None:
    """Создаёт графики метрик и confusion matrix эксперимента."""
    metrics_path = output_dir / "metrics.csv"

    if not metrics_path.exists():
        return

    rows = read_metric_rows(metrics_path)

    if not rows:
        return

    plot_metric_dashboard(output_dir, rows)

    confusion_path = output_dir / "confusion_matrix.csv"

    if not confusion_path.exists():
        return

    # Пустой файл матрицы не прерывает завершённый эксперимент.
    with confusion_path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        if sum(1 for _ in file) < 2:
            return

    plot_confusion_matrices(
        output_dir,
        confusion_path,
    )


def find_executable(command_name: str) -> str:
    """Возвращает путь к исполняемой команде."""
    command_path = shutil.which(command_name)

    if command_path is None:
        raise RuntimeError(
            f"Команда {command_name!r} не найдена.\n"
            "Сначала выполни uv sync, а запускай через uv run."
        )

    return command_path


def configure_flower() -> dict[str, str]:
    """Создаёт локальную конфигурацию Flower и возвращает окружение."""
    FLOWER_HOME.mkdir(
        parents=True,
        exist_ok=True,
    )
    FLOWER_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = FLOWER_HOME / "config.toml"
    config_path.write_text(
        "[superlink]\n"
        'default = "local-deployment"\n'
        "\n"
        "[superlink.local-deployment]\n"
        f'address = "127.0.0.1:{SUPERLINK_CONTROL_PORT}"\n'
        "insecure = true\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["FLWR_HOME"] = str(FLOWER_HOME)

    return env


def port_is_open(port: int) -> bool:
    """Проверяет, открыт ли локальный TCP-порт."""
    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def required_ports(num_clients: int) -> list[int]:
    """Возвращает порты, необходимые для локальной федерации."""
    supernode_ports = range(
        SUPERNODE_FIRST_PORT,
        SUPERNODE_FIRST_PORT + num_clients,
    )

    return [
        SUPERLINK_FLEET_PORT,
        SUPERLINK_CONTROL_PORT,
        *supernode_ports,
    ]


def ensure_ports_are_free(num_clients: int) -> None:
    """Проверяет доступность всех необходимых портов."""
    busy_ports = [
        port for port in required_ports(num_clients) if port_is_open(port)
    ]

    if not busy_ports:
        return

    raise RuntimeError(
        "Flower не может запуститься, потому что заняты порты: "
        f"{busy_ports}\n"
        "Останови старые процессы:\n"
        "pkill -f flower-supernode\n"
        "pkill -f flower-superlink"
    )


def wait_for_port(
    port: int,
    process: Process,
    process_name: str,
    timeout: float = 30.0,
) -> None:
    """Ожидает открытия порта запущенным процессом."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{process_name} завершился во время запуска. "
                f"Код завершения: {process.returncode}"
            )

        if port_is_open(port):
            return

        time.sleep(PORT_CHECK_INTERVAL_SECONDS)

    raise TimeoutError(f"{process_name} не открыл порт {port}")


def start_logged_process(
    command: list[str],
    log_path: Path,
    env: dict[str, str],
) -> tuple[Process, TextIO]:
    """Запускает процесс и перенаправляет его вывод в журнал."""
    log_file = log_path.open(
        "w",
        encoding="utf-8",
    )

    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise

    return process, log_file


def stop_process(process: Process) -> None:
    """Корректно завершает процесс и его дочерние процессы."""
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        process.wait(timeout=PROCESS_KILL_TIMEOUT_SECONDS)


def print_log_tail(
    path: Path,
    lines: int = 40,
) -> None:
    """Выводит последние строки указанного журнала."""
    if not path.exists():
        return

    content = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    print(f"\n--- Последние строки {path} ---")
    print("\n".join(content[-lines:]))


def register_process(
    runtime: FederationRuntime,
    command: list[str],
    log_path: Path,
    env: dict[str, str],
) -> Process:
    """Запускает процесс и регистрирует его в состоянии федерации."""
    process, log_file = start_logged_process(
        command,
        log_path,
        env,
    )

    runtime.processes.append(process)
    runtime.log_files.append(log_file)
    runtime.log_paths.append(log_path)

    return process


def build_superlink_command(
    executable: str,
) -> list[str]:
    """Формирует команду запуска Flower SuperLink."""
    return [
        executable,
        "--insecure",
        "--disable-runtime-dependency-installation",
        "--fleet-api-address",
        f"127.0.0.1:{SUPERLINK_FLEET_PORT}",
        "--control-api-address",
        f"127.0.0.1:{SUPERLINK_CONTROL_PORT}",
    ]


def build_supernode_command(
    executable: str,
    *,
    client_id: int,
    num_clients: int,
    client_port: int,
) -> list[str]:
    """Формирует команду запуска Flower SuperNode клиента."""
    return [
        executable,
        "--insecure",
        "--superlink",
        f"127.0.0.1:{SUPERLINK_FLEET_PORT}",
        "--clientappio-api-address",
        f"127.0.0.1:{client_port}",
        "--node-config",
        (f"partition-id={client_id} " f"num-partitions={num_clients}"),
    ]


def start_federation(
    num_clients: int,
    env: dict[str, str],
) -> FederationRuntime:
    """Запускает один SuperLink и по одному SuperNode на клиента."""
    ensure_ports_are_free(num_clients)

    superlink_executable = find_executable("flower-superlink")
    supernode_executable = find_executable("flower-supernode")
    runtime = FederationRuntime()

    try:
        superlink = register_process(
            runtime,
            build_superlink_command(superlink_executable),
            FLOWER_LOG_DIR / "superlink.log",
            env,
        )

        wait_for_port(
            SUPERLINK_CONTROL_PORT,
            superlink,
            "Flower SuperLink",
        )
        wait_for_port(
            SUPERLINK_FLEET_PORT,
            superlink,
            "Flower SuperLink",
        )

        for client_id in range(num_clients):
            client_port = SUPERNODE_FIRST_PORT + client_id

            supernode = register_process(
                runtime,
                build_supernode_command(
                    supernode_executable,
                    client_id=client_id,
                    num_clients=num_clients,
                    client_port=client_port,
                ),
                (FLOWER_LOG_DIR / f"supernode-{client_id}.log"),
                env,
            )

            wait_for_port(
                client_port,
                supernode,
                f"Flower SuperNode {client_id}",
            )

        # Даёт SuperNode время зарегистрироваться в SuperLink.
        time.sleep(SUPERNODE_REGISTRATION_DELAY_SECONDS)

        return runtime
    except Exception:
        runtime.print_log_tails()
        runtime.stop()
        raise


def create_output_directory(
    experiment_name: str,
) -> Path:
    """Создаёт каталог результатов с временной меткой."""
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    output_dir = ROOT / "outputs" / f"{timestamp}__{experiment_name}"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def save_resolved_config(
    output_dir: Path,
    resolved: Experiment,
) -> None:
    """Сохраняет итоговую конфигурацию эксперимента."""
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(
            resolved,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def build_flower_run_command(
    run_config: str,
    stream: bool,
) -> list[str]:
    """Формирует команду запуска Flower-эксперимента."""
    command = [
        find_executable("flwr"),
        "run",
        ".",
        "local-deployment",
        "--run-config",
        run_config,
    ]

    if stream:
        command.append("--stream")

    return command


def run_experiment(
    experiment: Experiment,
    stream: bool,
) -> None:
    """Запускает один эксперимент и сохраняет его результаты."""
    name = str(experiment["name"])
    output_dir = create_output_directory(name)

    resolved = dict(experiment)
    resolved["output-dir"] = output_dir.as_posix()

    save_resolved_config(
        output_dir,
        resolved,
    )

    dataset = str(resolved["dataset"])
    data_root = ROOT / str(resolved["data-root"])

    print(f"\n=== {name}: " f"downloading/checking {dataset} ===")
    download_dataset(dataset, data_root)

    run_config = build_run_config(
        resolved,
        data_root,
    )
    num_clients = int(resolved["num-clients"])
    env = configure_flower()
    runtime: FederationRuntime | None = None

    try:
        print(
            "=== Starting local Flower federation: "
            f"{num_clients} clients ==="
        )
        runtime = start_federation(
            num_clients=num_clients,
            env=env,
        )

        print(f"=== Running {name} ===")
        subprocess.run(
            build_flower_run_command(
                run_config,
                stream,
            ),
            cwd=ROOT,
            env=env,
            check=True,
        )
    except Exception:
        if runtime is not None:
            runtime.print_log_tails()

        raise
    finally:
        if runtime is not None:
            runtime.stop()

    plot_metrics(output_dir)

    print(f"=== Finished. Results: " f"{output_dir} ===")


def main() -> None:
    """Запускает выбранные эксперименты из YAML-конфигурации."""
    args = parse_args()

    experiments = load_experiments(args.config)
    selected = select_experiments(
        experiments,
        args.experiment,
    )

    for experiment in selected:
        run_experiment(
            experiment,
            stream=not args.no_stream,
        )


if __name__ == "__main__":
    main()
