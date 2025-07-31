import sys
import argparse
from pathlib import Path
from importlib.metadata import version

import torch
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
# models are finetuned with torch 2.3.1, but evaluated with 1.13.0 to match with original dg-sct
if version("torch") >= "2.0":
    import torchvision.transforms.v2 as TV2

sys.path.append(str(Path(__file__).resolve().parent))
from dgsct import load_DGSCT
from dgsct.nets.net_trans import CMBS, MMIL_Net
from dataloader import SoundActionsDataset, SOUNDACTIONS_MEAN, SOUNDACTIONS_STD


def cross_valid_finetune(
    target_label: str,
    finetune_mode: str,
    exp_name: str,
    train_modality: str,
    valid_modality: str,
    n_splits: int,
    dropout: float,
    adapter_layer_idx: list = None,
    batch_size: int = 16,
    use_wandb: bool = True,
    lr: int = 5e-4,
    seed: int = 18,
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

    seed_everything(seed)

    # apply video transform in finetune mode is "all"
    # if finetune_mode == "all":
    video_transform = TV2.Compose([
        TV2.RandomResizedCrop(size=(192, 192), scale=(0.75, 1.0), ratio=(0.8, 1.2)),
        TV2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=5),
        TV2.RandomPerspective(distortion_scale=0.3, p=0.3),
        TV2.RandomHorizontalFlip(p=0.5),
        # Color/appearance augmentations  
        TV2.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.2),
        TV2.RandomAutocontrast(p=0.3),
        TV2.RandomAdjustSharpness(sharpness_factor=1.3, p=0.3),
        # Deformation and blur, random
        TV2.ElasticTransform(alpha=70.0, sigma=5.0),
        TV2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        # Occlusion
        TV2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        # Normalization
        TV2.ToTensor(),
        TV2.Normalize(SOUNDACTIONS_MEAN, SOUNDACTIONS_STD),
    ])
    # else:
    # video_transform = None

    # generate folds
    soundactions = SoundActionsDataset(
        load_mode="preload",
        modality=train_modality,
        video_transform=video_transform,
        pad_mode="zero",
    )
    soundactions_valid = SoundActionsDataset(
        load_mode="preload",
        modality=valid_modality,
        pad_mode="zero",
    )
    splits = soundactions.gen_crossvalid_idx(target_label, n_splits, random_state=seed)

    for i, split in enumerate(splits):
        print(f"=> Finetuning fold {i + 1}/{n_splits}")
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
        if finetune_mode == "cls":
            model = LitDGSCT(
                target_label=target_label,
                pretrain=True,
                new_cls_head=True,
                num_classes=label_num_classes[target_label],
                lr=lr,
                dropout=dropout,
                mode="finetune_cls",
                adapter_layer_idx=None,
            )
        elif finetune_mode == "all":
            model = LitDGSCT(
                target_label=target_label,
                pretrain=True,
                new_cls_head=True,
                num_classes=label_num_classes[target_label],
                lr=lr,
                dropout=dropout,
                mode="train",
                adapter_layer_idx=adapter_layer_idx,
            )

        if use_wandb:
            wandb_logger = WandbLogger(
                project="soundactions",
                name=f"{exp_name}_{finetune_mode}_{target_label}_{train_modality}_{valid_modality}_{i}",
                save_dir=Path(__file__).resolve().parent
                / f"logs/{exp_name}_{finetune_mode}_{target_label}_{train_modality}_{valid_modality}",
            )
        es_cb = pl.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
        lr_mn = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        cp_cb = pl.callbacks.ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
            filename="{epoch}-{val_acc:.2f}",
        )

        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=1000,
            log_every_n_steps=19,
            logger=wandb_logger if use_wandb else None,
            callbacks=[es_cb, lr_mn, cp_cb],
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
        dropout: float = 0.0,
        adapter_layer_idx: list = None,
        num_classes: int = 4,
        verbose=False,
        mode="train",
        lr=5e-4,
    ):
        super().__init__()
        assert mode in ["train", "test", "finetune", "finetune_cls", "finetune_all"], (
            f"Invalid mode: {mode}"
        )
        assert target_label in ["Enjoyable", "PerceptionType"]

        if mode == "finetune":
            mode = "finetune_cls"  # backward compatibility

        # load model
        self.model = load_DGSCT(
            pretrain=pretrain,
            mode=mode,
            verbose=verbose,
            dropout=dropout,
            adapter_layer_idx=adapter_layer_idx,
        )
        if new_cls_head:
            if verbose:
                print(f"=> Init new cls head with {num_classes} classes")
            self.model.CMBS = CMBS(opt=None, num_classes=num_classes, dropout=dropout)
            self.model.CMBS.to(self.device)
            for param in self.model.CMBS.parameters():
                param.requires_grad = True

        # set paramters
        self.target_label = target_label
        self.lr = lr
        self.save_hyperparameters()
        self.loss = torch.nn.CrossEntropyLoss()
        # logging results
        self.valid_pred = []
        self.valid_gt = []
        self.train_pred = []
        self.train_gt = []
        if self.target_label == "Enjoyable":
            self.class_names = ["Yes", "Neutral", "No", "Unknown"]
        elif self.target_label == "PerceptionType":
            self.class_names = ["Impulse", "Iterative", "Sustain", "Unknown"]

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
        self.train_pred.append(event_scores.argmax(1).cpu())
        self.train_gt.append(label.cpu())

        return loss

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_pred)
        all_gt = torch.cat(self.train_gt)
        # wandb.log(
        #     {
        #         "train_confusion_matrix": wandb.plot.confusion_matrix(
        #             probs=None,
        #             preds=all_preds,
        #             y_true=all_gt,
        #             class_names=self.class_names,
        #         )
        #     }
        # )
        self.train_pred.clear()
        self.train_gt.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
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
        # print(f"=> event_scores: {event_scores}, label: {label}")
        pred = event_scores.argmax(1)
        self.valid_pred.append(pred.cpu())
        self.valid_gt.append(label.cpu())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val_acc",
            self.cal_acc(event_scores, label),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.valid_pred).tolist()
        all_gt = torch.cat(self.valid_gt).tolist()
        wandb.log(
            {
                "valid_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    preds=all_preds,
                    y_true=all_gt,
                    class_names=self.class_names,
                    title="Validation Confusion Matrix",
                )
            }
        )
        preds_count = torch.bincount(torch.tensor(all_preds), minlength=4)
        for i, count in enumerate(preds_count):
            self.log(
                f"valid_pred_count{self.class_names[i]}",
                count.item(),
                on_step=False,
                on_epoch=True,
            )
        self.valid_pred.clear()
        self.valid_gt.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--target_label", type=str)
    parser.add_argument("--finetune_mode", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--train_modality", type=str)
    parser.add_argument("--valid_modality", type=str)
    parser.add_argument("--n_splits", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--adapter_layer_idx", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()

    print(
        f"=> Finetuning with the following parameters:\n"
        f"exp_name: {args.exp_name}\n"
        f"target_label: {args.target_label}\n"
        f"finetune_mode: {args.finetune_mode}\n"
        f"batch_size: {args.batch_size}\n"
        f"train_modality: {args.train_modality}\n"
        f"valid_modality: {args.valid_modality}\n"
        f"n_splits: {args.n_splits}\n"
        f"lr: {args.lr}\n"
        f"dropout: {args.dropout}\n"
        f"adapter_layer_idx: {args.adapter_layer_idx}\n"
        f"seed: {args.seed}\n"
    )

    cross_valid_finetune(
        exp_name=args.exp_name,
        target_label=args.target_label,
        finetune_mode=args.finetune_mode,
        batch_size=args.batch_size,
        train_modality=args.train_modality,
        valid_modality=args.valid_modality,
        n_splits=args.n_splits,
        dropout=args.dropout,
        lr=args.lr,
        adapter_layer_idx=args.adapter_layer_idx,
    )
