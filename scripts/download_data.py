import os
from pathlib import Path
import kagglehub
import zipfile
import shutil
import fire

def download_and_extract_kaggle(dataset: str = "abhaymudgal/sentiment-analysis-dataset", target_dir: str = r'C:\Users\User\Desktop\MLOps\Data') -> None:
    """
    Downloads a dataset from Kaggle using kagglehub, extracts it if necessary,
    and saves it in the specified directory.
    """
    os.makedirs(target_dir, exist_ok=True)

    # Download dataset
    zip_path = Path(kagglehub.dataset_download(dataset))
    print(f"Dataset downloaded to: {zip_path}")

    # Check if the downloaded file is a ZIP and extract it
    if zip_path.suffix == ".zip":
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)  # Распаковываем все файлы в target_dir
        print(f"Extracted dataset to: {target_dir}")

        # Удаляем ZIP-файл после распаковки
        os.remove(zip_path)
        print(f"Deleted ZIP file: {zip_path}")
    else:
        # Если файл не ZIP-архив, просто перемещаем его в нужную директорию
        shutil.move(zip_path, Path(target_dir) / zip_path.name)
        print(f"Dataset is not a ZIP. Moved dataset to: {Path(target_dir) / zip_path.name}")

if __name__ == "__main__":
    fire.Fire(download_and_extract_kaggle)
