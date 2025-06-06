import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

# Typer app
app = typer.Typer(help="MLOps project CLI")
console = Console()

# Option declarations (moved outside functions to avoid B008 warnings)
checkpoint_path_option = typer.Option(None, help="Путь к чекпоинту модели")
output_dir_option = typer.Option(None, help="Директория для сохранения результатов")
model_option = typer.Option(None, help="Модель для обучения (linear/rnn)")
experiment_name_option = typer.Option(None, help="Название эксперимента")
max_epochs_option = typer.Option(None, help="Количество эпох обучения")
mlflow_port_option = typer.Option(8080, help="Порт для MLflow UI")


def run_mlflow_ui(port: int = 8080):
    """Запускает MLflow server на указанном порту."""
    cmd = [
        "python",
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--backend-store-uri",
        "./mlruns",
        "--default-artifact-root",
        "./artifacts",
    ]
    try:
        subprocess.Popen(cmd)
        print(f"[green]MLflow server запущен на http://127.0.0.1:{port}[/green]")
    except Exception as error:
        print(f"[red]Ошибка при запуске MLflow server: {error}[/red]")


@app.command()
def train(
    model: Optional[str] = model_option,
    experiment_name: Optional[str] = experiment_name_option,
    max_epochs: Optional[int] = max_epochs_option,
):
    """Запускает обучение модели."""
    cmd = ["poetry", "run", "python", "project/my_modules/train.py"]

    if model:
        cmd.append(f"model={model}")
    if experiment_name:
        cmd.append(f"logging.experiment_name={experiment_name}")
    if max_epochs:
        cmd.append(f"training.max_epochs={max_epochs}")

    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
        print("[green]Обучение модели завершено успешно![/green]")
    except subprocess.CalledProcessError as error:
        print(f"[red]Ошибка при обучении модели: {error}[/red]")


@app.command()
def download_data():
    """Скачивает датасет."""
    try:
        subprocess.run(
            "poetry run python scripts/download_data.py", shell=True, check=True
        )
        print("[green]Данные успешно загружены![/green]")
    except subprocess.CalledProcessError as error:
        print(f"[red]Ошибка при загрузке данных: {error}[/red]")


@app.command()
def convert_to_tensorrt(checkpoint_path: Optional[str] = checkpoint_path_option):
    """Конвертирует модель в TensorRT."""
    cmd = ["poetry", "run", "python", "scripts/convert_to_tensorrt.py"]

    if checkpoint_path:
        cmd.append(f"convert.checkpoint_path={checkpoint_path}")

    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
        print("[green]Модель успешно конвертирована в TensorRT![/green]")
    except subprocess.CalledProcessError as error:
        print(f"[red]Ошибка при конвертации модели: {error}[/red]")


@app.command()
def infer(
    checkpoint_path: Optional[str] = checkpoint_path_option,
    output_dir: Optional[str] = output_dir_option,
):
    """Запускает инференс модели."""
    cmd = ["poetry", "run", "python", "project/my_modules/infer.py"]

    if checkpoint_path:
        cmd.append(f"inference.checkpoint_path={checkpoint_path}")
    if output_dir:
        cmd.append(f"inference.output_dir={output_dir}")

    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
        print("[green]Инференс успешно завершен![/green]")
    except subprocess.CalledProcessError as error:
        print(f"[red]Ошибка при инференсе: {error}[/red]")


@app.command()
def ui(port: int = mlflow_port_option):
    """Запускает MLflow server."""
    run_mlflow_ui(port)
    print(f"[green]Для доступа к MLflow откройте: http://127.0.0.1:{port}[/green]")


@app.command()
def status():
    """Показывает статус проекта."""
    table = Table(title="Статус проекта")

    table.add_column("Компонент", style="cyan")
    table.add_column("Статус", style="magenta")
    table.add_column("Путь", style="green")

    dataset_path = Path("Data/1")
    table.add_row(
        "Данные",
        "Загружены" if dataset_path.exists() else "Отсутствуют",
        str(dataset_path),
    )

    model_path = Path("model_checkpoints")
    models_exist = model_path.exists() and any(model_path.iterdir())
    table.add_row(
        "Модели", "Найдены" if models_exist else "Отсутствуют", str(model_path)
    )

    mlflow_logs_path = Path("plots/mlruns")
    table.add_row(
        "MLflow логи",
        "Существуют" if mlflow_logs_path.exists() else "Отсутствуют",
        str(mlflow_logs_path),
    )

    inference_results_path = Path("test_results")
    table.add_row(
        "Результаты",
        "Существуют" if inference_results_path.exists() else "Отсутствуют",
        str(inference_results_path),
    )

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
