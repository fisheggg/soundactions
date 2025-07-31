import sys
import argparse
from pathlib import Path

import torch
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
import torchvision.transforms.v2 as TV2

sys.path.append(str(Path(__file__).resolve().parent))
from dgsct import load_DGSCT
from dgsct.nets.net_trans import CMBS, MMIL_Net
from dataloader import SoundActionsDataset
# from transform import VideoColorJitter, VideoRandomHorizontalFlip


def cross_valid_finetune(
    target_label: str,
    finetune_mode: str,
    exp_name: str,
    train_modality: str,
    valid_modality: str,
    n_splits: int,
    split_idx: int,
    batch_size: int = 16,
    use_wandb: bool = True,
    lr: int = 5e-4,
    seeds: list = [18, 19, 20, 21, 22],
):
    """
    k-fold cross validation finetuning
    """
    label_num_classes = {
        "PerceptionType": 4,
        "Enjoyable": 4,
    }
    assert target_label in label_num_classes
    assert finetune_mode in ["cls", "all"]
    assert train_modality in ["av", "a", "v"]
    assert valid_modality in ["av", "a", "v"]

    # apply video transform in finetune mode is "all"
    if finetune_mode == "all":
        video_transform = TV2.Compose(
            [
                TV2.RandomHorizontalFlip(p=0.5),
                # TV2.ColorJitter(brightness=0.5, hue=0.3, contrast=0.3, saturation=0.3),
                # TV2.ElasticTransform(alpha=30.0, sigma=5.0),
                # TV2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            ]
        )
    else:
        video_transform = None

    # generate data split using the first seed
    seed_everything(seed=seeds[0])
    # generate folds
    soundactions = SoundActionsDataset(
        load_mode="preload", modality=train_modality, video_transform=video_transform
    )
    soundactions_valid = SoundActionsDataset(
        load_mode="preload", modality=valid_modality
    )
    split = soundactions.gen_crossvalid_idx(target_label, n_splits)
    split = split[split_idx]

    # then run finetuning with different seeds
    for seed_idx, seed in enumerate(seeds):
        print(
            f"=> Finetuning with split: {split_idx}/{n_splits}, seed: {seed}, seed_idx: {seed_idx}/{len(seeds)}"
        )
        seed_everything(seed)

        model = LitDGSCT(
            target_label,
            pretrain=True,
            new_cls_head=True,
            num_classes=label_num_classes[target_label],
            mode=f"finetune_{finetune_mode}",
            lr=lr,
        )

        train_loader = DataLoader(
            Subset(soundactions, split["train"]),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )
        valid_loader = DataLoader(
            Subset(soundactions_valid, split["valid"]),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
        )

        if use_wandb:
            wandb_logger = WandbLogger(
                project="soundactions",
                name=f"{exp_name}_{finetune_mode}_{target_label}_{train_modality}_{valid_modality}_{seed_idx}_split{split_idx}_seed{seed}",
                save_dir=Path(__file__).resolve().parent
                / f"logs/{exp_name}_{finetune_mode}_{target_label}_{train_modality}_{valid_modality}",
            )
        es_cb = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, mode="min"
        )
        lr_mn = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=1000,
            log_every_n_steps=19,
            logger=wandb_logger if use_wandb else None,
            callbacks=[es_cb, lr_mn],
            enable_progress_bar=False,
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
        verbose=False,
        mode="train",
        lr=5e-4,
    ):
        super().__init__()
        assert mode in ["train", "test", "finetune", "finetune_cls", "finetune_all"], (
            f"Invalid mode: {mode}"
        )
        if mode == "finetune":
            mode = "finetune_cls"  # backward compatibility

        # load model
        self.model = load_DGSCT(pretrain=pretrain, mode=mode, verbose=verbose)
        if new_cls_head:
            if verbose:
                print(f"=> Init new cls head with {num_classes} classes")
            self.model.CMBS = CMBS(opt=None, num_classes=num_classes)

        # set paramters
        self.target_label = target_label
        self.lr = lr
        self.save_hyperparameters()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, audio, video):
        return self.model([audio], video)

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
            self.cal_acc(event_scores, label),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--target_label", type=str)
    parser.add_argument("--finetune_mode", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--train_modality", type=str)
    parser.add_argument("--valid_modality", type=str)
    parser.add_argument("--n_splits", type=int)
    parser.add_argument("--split_idx", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    print(
        f"=> Finetuning same splitwith the following parameters:\n"
        f"exp_name: {args.exp_name}\n"
        f"target_label: {args.target_label}\n"
        f"finetune_mode: {args.finetune_mode}\n"
        f"batch_size: {args.batch_size}\n"
        f"train_modality: {args.train_modality}\n"
        f"valid_modality: {args.valid_modality}\n"
        f"n_splits: {args.n_splits}\n"
        f"split_idx: {args.split_idx}\n"
        f"lr: {args.lr}\n"
    )

    cross_valid_finetune(
        exp_name=parser.exp_name,
        target_label=parser.target_label,
        finetune_mode=parser.finetune_mode,
        batch_size=parser.batch_size,
        train_modality=parser.train_modality,
        valid_modality=parser.valid_modality,
        n_splits=parser.n_splits,
        split_idx=parser.split_idx,
        lr=parser.lr,
    )
