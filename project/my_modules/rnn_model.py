import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=3,
        lr=1e-3,
        num_layers=1,
        bidirectional=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(rnn_output_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        output, (hn, cn) = self.rnn(x)

        final_feature = (
            hn[-1]
            if not self.hparams.bidirectional
            else torch.cat((hn[-2], hn[-1]), dim=1)
        )
        final_feature = self.dropout(final_feature)
        return self.fc(final_feature)

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
