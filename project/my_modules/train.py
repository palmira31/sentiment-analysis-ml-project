import os
import subprocess

import hydra
import joblib
import matplotlib.pyplot as plt
import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from linear_model import LinearClassifier
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from model_wrapper import SentimentClassifierWrapper
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from rnn_model import RNNClassifier
from sklearn.metrics import classification_report, confusion_matrix

from data import MyDataModule


def get_git_commit() -> str:
    """Получает текущий git commit hash"""
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        return commit_hash
    except Exception:
        return "unknown"


def plot_metrics(metrics_dict, title, save_path):
    """Helper function to plot metrics"""
    plt.figure(figsize=(10, 6))
    for metric_name, values in metrics_dict.items():
        plt.plot(values, label=metric_name)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def train(cfg: DictConfig):
    # Настраиваем MLflow
    mlflow.set_tracking_uri(cfg.environment.mlflow.tracking_uri)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.environment.mlflow.tracking_uri,
    )

    # Создаем директории для сохранения графиков
    plots_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(plots_dir, exist_ok=True)

    # Создаем базовую директорию для артефактов
    artifacts_dir = os.path.join(
        cfg.environment.mlflow.artifacts_dir,
        cfg.logging.experiment_name,
        mlf_logger.run_id,
    )
    os.makedirs(artifacts_dir, exist_ok=True)

    # Логируем гиперпараметры и версию кода
    git_commit = get_git_commit()
    mlf_logger.log_hyperparams(
        {
            "model_name": cfg.model.name,
            "batch_size": cfg.training.batch_size,
            "max_epochs": cfg.training.max_epochs,
            "max_features": cfg.training.max_features,
            "git_commit": git_commit,
            "learning_rate": cfg.model.lr if hasattr(cfg.model, "lr") else None,
            "hidden_dim": (
                cfg.model.hidden_dim if hasattr(cfg.model, "hidden_dim") else None
            ),
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=artifacts_dir,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    datamodule = MyDataModule(
        data_dir=cfg.data_loading.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        max_features=cfg.training.max_features,
    )

    if cfg.model.name == "linear":
        model = LinearClassifier(input_dim=cfg.model.input_dim)
    elif cfg.model.name == "rnn":
        model = RNNClassifier(
            input_dim=cfg.model.input_dim,
            hidden_dim=cfg.model.hidden_dim,
            output_dim=cfg.model.output_dim,
            lr=cfg.model.lr,
        )
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    # Словари для хранения метрик
    train_losses = []
    val_losses = []
    val_accuracies = []

    class MetricsCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            train_loss = trainer.callback_metrics.get("train_loss")
            if train_loss is not None:
                train_losses.append(train_loss.item())
                mlf_logger.log_metrics(
                    {"train_loss": train_loss.item()}, step=trainer.current_epoch
                )

        def on_validation_epoch_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get("val_loss")
            val_acc = trainer.callback_metrics.get("val_acc")
            if val_loss is not None:
                val_losses.append(val_loss.item())
                mlf_logger.log_metrics(
                    {"val_loss": val_loss.item()}, step=trainer.current_epoch
                )
            if val_acc is not None:
                val_accuracies.append(val_acc.item())
                mlf_logger.log_metrics(
                    {"val_accuracy": val_acc.item()}, step=trainer.current_epoch
                )

    # Добавляем callback для метрик
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=mlf_logger,
        callbacks=[checkpoint_callback, metrics_callback],
        accelerator="auto",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
    )

    trainer.fit(model, datamodule=datamodule)

    # Создаем и сохраняем графики
    # 1. График функций потерь
    plot_metrics(
        {"Train Loss": train_losses, "Validation Loss": val_losses},
        "Training and Validation Loss",
        os.path.join(plots_dir, "loss_curves.png"),
    )
    mlf_logger.experiment.log_artifact(
        mlf_logger.run_id, os.path.join(plots_dir, "loss_curves.png")
    )

    # 2. График точности на валидации
    plot_metrics(
        {"Validation Accuracy": val_accuracies},
        "Validation Accuracy",
        os.path.join(plots_dir, "accuracy.png"),
    )
    mlf_logger.experiment.log_artifact(
        mlf_logger.run_id, os.path.join(plots_dir, "accuracy.png")
    )

    # 3. Матрица ошибок на валидационном наборе
    model.eval()
    val_dataloader = datamodule.val_dataloader()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            features = batch["features"]
            labels = batch["label"]
            if torch.cuda.is_available():
                features = features.cuda()
            outputs = model(features)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # Создаем матрицу ошибок
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()
    mlf_logger.experiment.log_artifact(
        mlf_logger.run_id, os.path.join(plots_dir, "confusion_matrix.png")
    )

    # Логируем отчет о классификации
    report = classification_report(all_labels, all_preds)
    with open(os.path.join(plots_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    mlf_logger.experiment.log_artifact(
        mlf_logger.run_id, os.path.join(plots_dir, "classification_report.txt")
    )

    # Сохраняем артефакты
    vectorizer = datamodule.vectorizer
    label_encoder = datamodule.label_encoder
    vectorizer_path = os.path.join(artifacts_dir, "vectorizer.pkl")
    label_encoder_path = os.path.join(artifacts_dir, "label_encoder.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, label_encoder_path)

    model_path = os.path.join(artifacts_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Определяем conda environment для serving
    conda_env = {
        "name": "sentiment_analysis_env",
        "channels": ["defaults", "conda-forge", "pytorch"],
        "dependencies": [
            f"python={cfg.environment.python_version}",
            "pytorch",
            "scikit-learn",
            "pandas",
            "numpy",
            "pip",
            {
                "pip": [
                    "mlflow",
                    "pytorch-lightning",
                ]
            },
        ],
    }

    # Определяем signature модели для MLflow
    input_schema = Schema([ColSpec(type="string", name="text")])
    output_schema = Schema([ColSpec(type="string")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Создаем пример входных данных
    example_texts = ["This is a sample text", "Another example"]
    input_example = pd.DataFrame({"text": example_texts})

    # Создаем и регистрируем модель в MLflow
    wrapper_model = SentimentClassifierWrapper(
        vectorizer=vectorizer, label_encoder=label_encoder, model=model
    )

    with mlflow.start_run(run_id=mlf_logger.run_id) as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapper_model,
            artifacts={
                "model_path": model_path,
                "vectorizer_path": vectorizer_path,
                "label_encoder_path": label_encoder_path,
            },
            conda_env=conda_env,
            signature=signature,
            input_example=input_example,
            registered_model_name=f"sentiment_classifier_{cfg.model.name}",
        )

    # Экспортируем в ONNX если нужно
    model.eval()
    batch = next(iter(datamodule.train_dataloader()))
    example_input = batch["features"]
    example_input = example_input.to(next(model.parameters()).device)

    onnx_path = os.path.join(artifacts_dir, "model.onnx")

    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Model exported to ONNX format at: {onnx_path}")
    print(f"Model registered in MLflow as: sentiment_classifier_{cfg.model.name}")
    print(f"MLflow run ID: {run.info.run_id}")


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
