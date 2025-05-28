import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["features"])
        loss = F.cross_entropy(logits, batch["label"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["features"])
        loss = F.cross_entropy(logits, batch["label"])
        acc = (logits.argmax(dim=1) == batch["label"]).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
