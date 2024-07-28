import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parent))
from dgsct import load_DGSCT
from dgsct.nets.net_trans import CMBS, MMIL_Net
from dataloader import SoundActionsDataset


def cross_valid_finetune(target_label, n_splits, batch_size=16):
    """
    k-fold cross validation finetuning
    """
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

        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=1000,
            log_every_n_steps=19,
        )

        trainer.fit(model, train_loader, valid_loader)


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

    def forward(self, **kwargs):
        self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        criterion = torch.nn.CrossEntropyLoss()

        _, event_scores, _, av_score, e, f = self.model(
            [batch["audio"].to(self.device)], batch["video"].to(self.device)
        )

        label = batch["label"][self.target_label].to(self.device)

        loss_event = criterion(event_scores, label)
        loss_cas = criterion(av_score, label)

        return loss_event + loss_cas

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.35)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        labels = batch["label"][self.target_label].to(self.device)
        audio = batch["audio"].to(self.device)
        video = batch["video"].to(self.device)
        _, event_scores, _, _, _, _ = self.model([audio], video)
        acc = (event_scores.argmax(1) == labels).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

if __name__ == "__main__":
    cross_valid_finetune("PerceptionType", 5)
