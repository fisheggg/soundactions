import sys
from pathlib import Path

import torch
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

sys.path.append(str(Path(__file__).resolve().parent))
from dgsct import load_DGSCT
from dgsct.nets.net_trans import CMBS, MMIL_Net
from dataloader import SoundActionsDataset


def cross_valid_finetune(target_label, n_splits, batch_size=16, use_wandb=True, seed=18):
    """
    k-fold cross validation finetuning
    """
    seed_everything(seed)

    # generate folds
    soundactions = SoundActionsDataset(load_mode="preload")
    splits = soundactions.gen_crossvalid_idx(target_label, n_splits)

    label_num_classes = {
        "PerceptionType": 4,
        "Enjoyable": 4,
    }

    model = LitDGSCT(
        target_label,
        pretrain=True,
        new_cls_head=True,
        num_classes=label_num_classes[target_label],
        mode="finetune",
    )

    results = []
    for i, split in enumerate(splits):
        print(f"=> Finetuning fold {i+1}/{n_splits}")
        train_loader = DataLoader(
            Subset(soundactions, split["train"]),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )
        valid_loader = DataLoader(
            Subset(soundactions, split["valid"]),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
        )

        if use_wandb:
            wandb_logger = WandbLogger(
                project="soundactions", name=f"finetune_{target_label}_{i}",
                save_dir=Path(__file__).resolve().parent / "logs/{target_label}"
            )
        es_cb = pl.callbacks.EarlyStopping(monitor="train_loss", patience=10, mode="min")
        lr_mn = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=1000,
            log_every_n_steps=19,
            logger=wandb_logger if use_wandb else None,
            callbacks=[es_cb, lr_mn],
        )

        trainer.fit(model, train_loader, valid_loader)
        wandb.finish()


class LitDGSCT(pl.LightningModule):
    def __init__(
        self,
        target_label: str,
        pretrain: bool,
        new_cls_head: bool,
        num_classes: int = None,
        mode="train",
        lr=5e-4,
    ):
        super().__init__()
        assert mode in ["train", "test", "finetune"]

        # load model
        self.model = load_DGSCT(pretrain=pretrain, mode=mode)
        if new_cls_head:
            print(f"=> Init new cls head with {num_classes} classes")
            self.model.CMBS = CMBS(opt=None, num_classes=num_classes)

        # set paramters
        self.target_label = target_label
        self.lr = lr
        self.save_hyperparameters()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, **kwargs):
        self.model(**kwargs)

    def cal_acc(self, pred, label):
        return (pred.argmax(1) == label).float().mean()

    def training_step(self, batch, batch_idx):
        label = batch["label"][self.target_label].to(self.device)
        audio = batch["audio"].to(self.device)
        video = batch["video"].to(self.device)

        _, event_scores, _, av_score, _, _ = self.model([audio], video)

        loss = self.loss(event_scores, label) + self.loss(av_score, label)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train_acc",
            self.cal_acc
            (event_scores, label),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def validation_step(self, batch, batch_idx):
        label = batch["label"][self.target_label].to(self.device)
        audio = batch["audio"].to(self.device)
        video = batch["video"].to(self.device)

        _, event_scores, _, av_score, _, _ = self.model([audio], video)
        loss = self.loss(event_scores, label) + self.loss(av_score, label)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val_acc",
            self.cal_acc(event_scores, label),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


if __name__ == "__main__":
    cross_valid_finetune("PerceptionType", 5)
