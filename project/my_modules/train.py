import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from linear_model import LinearClassifier
from rnn_model import RNNClassifier
from data import MyDataModule

def main():
    """
    Запускает обучение модели.

    Параметры:
    - model_name: 'linear' или 'rnn' — выбирает модель для обучения.
    - log_name: имя папки для логов (сохранится в C:\\Users\\User\\Desktop\\MLOps\\training_logs).

    Пример запуска:
    python train.py --model rnn --log_name experiment_rnn

    Для просмотра логов TensorBoard:
    tensorboard --logdir=C:\\Users\\User\\Desktop\\MLOps\\training_logs\\experiment_rnn
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["linear", "rnn"], default="linear",
                        help="Выберите модель для тренировки")
    parser.add_argument("--log_name", type=str, default="default_run",
                        help="Имя папки для логов")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\User\Desktop\MLOps\Data\1", help="Путь к данным")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()

    base_log_dir = r"C:\Users\User\Desktop\MLOps\training_logs"

    logger = TensorBoardLogger(save_dir=base_log_dir, name=args.log_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(base_log_dir, args.log_name, "checkpoints"),
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    dm = MyDataModule(data_dir=args.data_dir, batch_size=args.batch_size)

    if args.model == "linear":
        model = LinearClassifier(input_dim=10000)  # input_dim для tfidf
    else:
        model = RNNClassifier(input_dim=10000, hidden_dim=128, output_dim=3, lr=1e-3)


    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
