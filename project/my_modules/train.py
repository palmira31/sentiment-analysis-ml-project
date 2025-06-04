import os
import subprocess

import hydra
import joblib
import pytorch_lightning as pl
import torch
from linear_model import LinearClassifier
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from rnn_model import RNNClassifier

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


def train(cfg: DictConfig):
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
    )

    mlf_logger.log_hyperparams(
        {
            "model_name": cfg.model.name,
            "batch_size": cfg.training.batch_size,
            "max_epochs": cfg.training.max_epochs,
            "max_features": cfg.training.max_features,
            "git_commit": get_git_commit(),
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.logging.checkpoint_dir, cfg.logging.experiment_name),
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

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=mlf_logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
    )

    trainer.fit(model, datamodule=datamodule)

    artifacts_dir = os.path.join(
        cfg.logging.checkpoint_dir, cfg.logging.experiment_name
    )
    os.makedirs(artifacts_dir, exist_ok=True)

    vectorizer = datamodule.vectorizer
    label_encoder = datamodule.label_encoder
    joblib.dump(vectorizer, os.path.join(artifacts_dir, "vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(artifacts_dir, "label_encoder.pkl"))

    model_path = os.path.join(artifacts_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

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


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
