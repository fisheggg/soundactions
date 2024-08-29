import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).resolve().parent))
from dgsct import load_DGSCT
from dataloader import SoundActionsDataset
from finetune import LitDGSCT

## DG-SCT input shape:
# => audio shape: torch.Size([2, 10, 320000]), type: torch.float32
# => video shape: torch.Size([2, 10, 3, 192, 192]), type: torch.float32
# => gt shape: torch.Size([2, 10, 29]), type: torch.float32
## DG-SCT output shape:
# is_event_scores: torch.Size([10, 2, 1])
# event_scores: torch.Size([2, 28])
# audio_visual_gate: torch.Size([10, 2, 1])
# av_score: torch.Size([2, 28])

label_color_map = {
    "PerceptionType": {
        0: "red",  # impulsive
        1: "blue",  # iterative
        2: "green",  # sustain
        3: "gray",  # Unknown
    },
    "Enjoyable": {
        0: "green",  # Yes
        1: "yellow",  # Neutral
        2: "red",  # No
        3: "gray",  # Unknown
    },
}


def plot_embeddings(
    train_modality: str,
    data_modality: str,
    plot_modality: str,
    ckpt_path: str,
    save_path=None,
    coloring_label=None,
    dr_alg="pca",
):
    assert dr_alg in ["tsne", "pca", "umap"]
    assert data_modality in ["av", "a", "v"]
    assert plot_modality in ["av", "a", "v"]

    print("=> Loading dataset...")
    soundactions = SoundActionsDataset(load_mode="online", modality=train_modality)

    print("=> Loading model...")
    if ckpt_path is None:
        model = load_DGSCT().eval().cuda()
    else:
        model = (
            LitDGSCT.load_from_checkpoint(
                ckpt_path,
                target_label=coloring_label,
                pretrain=True,
                new_cls_head=True,
                num_classes=4,
            )
            .eval()
            .cuda()
        )

    print("=> Inferencing...")
    embeds = torch.zeros(365, 256)
    colors = []
    labels = []
    for i, sample in tqdm(enumerate(soundactions)):
        out = model(
            sample["audio"].unsqueeze(0).cuda(), sample["video"].unsqueeze(0).cuda()
        )

        video_embed = out[4].detach().cpu().squeeze().mean(axis=0)
        audio_embed = out[5].detach().cpu().squeeze().mean(axis=0)

        if plot_modality == "av":
            embeds[i] = (video_embed + audio_embed) / 2
        elif plot_modality == "a":
            embeds[i] = audio_embed
        elif plot_modality == "v":
            embeds[i] = video_embed
        labels.append(sample["label"][coloring_label])
        colors.append(label_color_map[coloring_label][sample["label"][coloring_label]])

    if dr_alg == "pca":
        out = PCA(n_components=2).fit_transform(embeds)
    elif dr_alg == "tsne":
        out = TSNE(n_components=2).fit_transform(embeds)
    elif dr_alg == "umap":
        pass

    # plt.figure(figsize=(10, 10))
    # plt.scatter(out[:, 0], out[:, 1], c=colors)
    # plt.title(f"SoundActions embeddings from DG-SCT, {dr_alg}, {coloring_label}")
    # plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
        print(f"=> Saved plot to {save_path}")

    return out, labels, colors


if __name__ == "__main__":
    plot_embeddings(
        train_modality="av",
        data_modality="av",
        plot_modality="av",
        ckpt_path="/projects/ec12/jinyueg/SoundActions/soundactions/soundactions/logs/V04_all_PerceptionType_av_av/soundactions/0407syid/checkpoints/epoch=12-step=962.ckpt",
        coloring_label="PerceptionType",
        save_path="test_embeds.png",
    )
