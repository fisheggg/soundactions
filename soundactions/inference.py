
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
        "Impulse": "red",
        "Iterative": "blue",
        "Sustain": "green",
        "Unknown": "gray",
    },
    "Enjoyable": {"Yes": "green", "Neutral": "yellow", "No": "red", "Unknown": "gray"},
}


def plot_embeddings(save_path=None, pad_mode="zero", coloring_label=None, dr_alg="tsne"):
    assert dr_alg in ["tsne", "pca"]
    soundactions = SoundActionsDataset("train", pad_mode=pad_mode)

    print("=> Loading model...")
    model = load_DGSCT().eval().cuda()

    print("=> Inferencing...")
    embeds = torch.zeros(365, 256)
    colors = []
    for i, sample in tqdm(enumerate(soundactions)):
        out = model(
            [sample["audio"].unsqueeze(0).cuda()], sample["video"].unsqueeze(0).cuda()
        )
        video_embed = out[4].detach().cpu().squeeze().mean(axis=0)
        audio_embed = out[5].detach().cpu().squeeze().mean(axis=0)
        embeds[i] = (video_embed + audio_embed) / 2
        colors.append(label_color_map[coloring_label][sample["label"][coloring_label]])

    if dr_alg == "pca":
        out = PCA(n_components=2).fit_transform(embeds)
    elif dr_alg == "tsne":
        out = TSNE(n_components=2).fit_transform(embeds)

    plt.figure(figsize=(10, 10))
    plt.scatter(out[:, 0], out[:, 1], c=colors)
    plt.title(f"SoundActions embeddings from DG-SCT, {dr_alg}, {coloring_label}")
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
        print(f"=> Saved plot to {save_path}")


if __name__ == "__main__":
    plot_embeddings()
