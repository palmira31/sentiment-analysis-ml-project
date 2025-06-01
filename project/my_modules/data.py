from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


class MyTextDataset(Dataset):
    def __init__(
        self,
        path: str = None,
        split: str = "train",
        vectorizer: Any = None,
        label_encoder: LabelEncoder = None,
        dataframe: pd.DataFrame = None,
        max_features: int = 10000,
    ):
        self.split = split
        self.vectorizer = (
            vectorizer if vectorizer else TfidfVectorizer(max_features=max_features)
        )
        self.label_encoder = label_encoder if label_encoder else LabelEncoder()

        if dataframe is not None:
            self.data = dataframe
        elif path is not None:
            self.data = pd.read_csv(f"{path}/{split}.csv")
        else:
            raise ValueError("Either 'path' or 'dataframe' must be provided.")

        # Удаляем строки без метки
        if "gold_label" in self.data.columns:
            self.data = self.data.dropna(subset=["gold_label"])

        # Fit vectorizer на тренировке
        if split == "train":
            self.vectorizer.fit(self.data["sentence"])
            self.label_encoder.fit(self.data["gold_label"])

        # Преобразуем данные
        self.X = self.vectorizer.transform(self.data["sentence"]).toarray()

        if "gold_label" in self.data.columns:
            self.y = self.label_encoder.transform(self.data["gold_label"])
        else:
            self.y = [-1] * len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return {"features": features, "label": label}


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        max_features: int = 10000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.label_encoder = LabelEncoder()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage in (None, "fit"):
            full_df = pd.read_csv(f"{self.data_dir}/train.csv")
            full_df = full_df.dropna(subset=["gold_label"])

            train_df, val_df = train_test_split(
                full_df,
                test_size=0.2,
                random_state=42,
                stratify=full_df["gold_label"],
            )

            self.train_dataset = MyTextDataset(
                vectorizer=self.vectorizer,
                label_encoder=self.label_encoder,
                dataframe=train_df,
                split="train",
                max_features=self.max_features,
            )
            self.val_dataset = MyTextDataset(
                vectorizer=self.vectorizer,
                label_encoder=self.label_encoder,
                dataframe=val_df,
                split="val",
                max_features=self.max_features,
            )

        if stage in (None, "test"):
            self.test_dataset = MyTextDataset(
                path=self.data_dir,
                split="test",
                vectorizer=self.vectorizer,
                label_encoder=self.label_encoder,
                max_features=self.max_features,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
