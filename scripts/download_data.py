import os
import shutil
import zipfile
from pathlib import Path

import fire
import kagglehub


def download_and_extract_kaggle(
    dataset: str = "abhaymudgal/sentiment-analysis-dataset",
    target_dir: str = "Data",
) -> None:
    """
    Downloads a dataset from Kaggle using kagglehub, extracts it if necessary,
    and saves it in the specified directory.

    Args:
        dataset: Kaggle dataset identifier
        target_dir: Target directory for dataset (relative to project root)
    """
    # Получаем абсолютный путь к директории проекта (родительская директория скрипта)
    project_root = Path(__file__).parent.parent
    target_path = project_root / target_dir

    os.makedirs(target_path, exist_ok=True)

    # Download dataset
    zip_path = Path(kagglehub.dataset_download(dataset))
    print(f"Dataset downloaded to: {zip_path}")

    # Check if the downloaded file is a ZIP and extract it
    if zip_path.suffix == ".zip":
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_path)  # Распаковываем все файлы в target_path
        print(f"Extracted dataset to: {target_path}")

        # Удаляем ZIP-файл после распаковки
        os.remove(zip_path)
        print(f"Deleted ZIP file: {zip_path}")
    else:
        # Если файл не ZIP-архив, просто перемещаем его в нужную директорию
        shutil.move(zip_path, target_path / zip_path.name)
        print(f"Dataset is not a ZIP. Moved dataset to: {target_path / zip_path.name}")


if __name__ == "__main__":
    fire.Fire(download_and_extract_kaggle)
