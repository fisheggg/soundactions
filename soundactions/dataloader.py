import os
import glob

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision

from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchaudio.transforms import Resample
from torch.utils.data import DataLoader
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class SoundActionsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_mode,
        load_mode="online",
        sample_mode="full",
        root: str = "/fp/homes01/u01/ec-jinyueg/felles_/Research/Project/AMBIENT/Datasets/SoundActions/",
        video_path: str = "video-frames",
        audio_path: str = "wav",
        label_path: str = "labels/SoundActions_labeling_A.csv",
        video_transform=None,
        audio_transform=None,
        pad_mode="zero",
        size: int = None,
        orig_audio_fs: int = 48000,
    ):
        assert load_mode in ["preload", "online"]
        assert sample_mode in ["full"]

        self.root = root
        self.video_paths = sorted(glob.glob(os.path.join(root, video_path, "*")))
        self.audio_paths = sorted(glob.glob(os.path.join(root, audio_path, "*.wav")))
        assert len(self.video_paths) == len(self.audio_paths)
        self.labels = pd.read_csv(os.path.join(root, label_path))
        assert len(self.video_paths) == len(self.labels)

        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.pad_mode = pad_mode
        self.split_mode = split_mode
        self.load_mode = load_mode
        self.sample_mode = sample_mode

        self.video_standardize = Compose(
            [
                # Resize([192, 192], interpolation=Image.BICUBIC),
                Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
        self.audio_standardize = Compose(
            [
                Resample(orig_freq=orig_audio_fs, new_freq=32000),
            ]
        )

        if size is not None:
            self.video_paths = self.video_paths[:size]
            self.audio_paths = self.audio_paths[:size]
            self.labels = self.labels.iloc[:size]

        if self.load_mode == "preload":
            self.videos = []
            self.videos_metadata = []
            self.audios = []
            self.audios_fps = []
            print("=> Preloading SoundActions dataset...")
            for video_path, audio_path in zip(self.video_paths, self.audio_paths):
                self.videos.append(self._load_video(video_path))
                self.audios.append(self._load_audio(audio_path))
            print("=> Preloading done")

    def __len__(self):
        if self.sample_mode == "full":
            return len(self.video_paths)

    def __getitem__(self, index):
        data = {}
        # data["label"] = self.labels.iloc[index].to_dict()
        if self.load_mode == "online":
            video_path = self.video_paths[index]
            audio_path = self.audio_paths[index]
            data["video"] = self._load_video(video_path, pad_mode=self.pad_mode)
            data["audio"] = self._load_audio(audio_path, pad_mode=self.pad_mode)
        elif self.load_mode == "preload":
            data["video"] = self.videos[index]
            data["audio"] = self.audios[index]

        if self.video_transform is not None:
            data["video"] = self.video_transform(data["video"])
        if self.audio_transform is not None:
            data["audio"] = self.audio_transform(data["audio"])
        return data

    def _load_video(
        self,
        video_frames_dir,
        original_fps=25,
        load_fps=1,
        num_frames=10,
        pad_mode="repeat",
    ):
        assert pad_mode in ["repeat", "zero", "ninf"]

        n_frames = len(glob.glob(os.path.join(video_frames_dir, "*.png")))
        sample_frames = np.arange(num_frames) * original_fps / load_fps + 1

        total_img = []
        for frame_idx in sample_frames.astype(int):
            if frame_idx <= n_frames:
                img_path = os.path.join(video_frames_dir, f"{frame_idx:04}.png")
                tmp_img = torchvision.io.read_image(img_path).to(torch.float32) / 255.0
                tmp_img = self.video_standardize(tmp_img)
            elif pad_mode == "zero":
                tmp_img = torch.zeros_like(total_img[-1])
            elif pad_mode == "repeat":
                tmp_img = total_img[-1]
            elif pad_mode == "ninf":
                tmp_img = torch.full_like(total_img[-1], -torch.inf)
            else:
                raise ValueError(f"Invalid pad_mode: {pad_mode}")
            total_img.append(tmp_img)
        total_img = torch.stack(total_img)

        return total_img

    def _load_audio(
        self, audio_path, slice_length=32000, num_slices=10, pad_mode="repeat"
    ):
        assert pad_mode in ["repeat", "zero", "ninf"]

        audio, fs = torchaudio.load(audio_path)
        audio = self.audio_standardize(audio).squeeze()
        assert audio.ndim == 1
        if audio.shape[0] >= slice_length * num_slices:
            audio = audio[: slice_length * num_slices]
            audio = audio.view(num_slices, slice_length)
        elif pad_mode == "repeat":
            max_slices = audio.shape[0] // slice_length
            audio_new = torch.zeros((max_slices + 1) * slice_length)
            audio_new[: audio.shape[0]] = audio
            audio = audio_new.view(max_slices + 1, slice_length)
            for _ in range(num_slices - max_slices - 1):
                audio = torch.cat([audio, audio[-1].unsqueeze(0)], dim=0)
        elif pad_mode == "zero":
            max_slices = audio.shape[0] // slice_length
            audio_new = torch.zeros((max_slices + 1) * slice_length)
            audio_new[: audio.shape[0]] = audio
            audio = audio_new.view(max_slices + 1, slice_length)
            for _ in range(num_slices - max_slices - 1):
                audio = torch.cat(
                    [audio, torch.zeros_like(audio[-1]).unsqueeze(0)], dim=0
                )
        elif pad_mode == "ninf":
            max_slices = audio.shape[0] // slice_length
            audio_new = torch.zeros((max_slices + 1) * slice_length)
            audio_new[: audio.shape[0]] = audio
            audio = audio_new.view(max_slices + 1, slice_length)
            for _ in range(num_slices - max_slices - 1):
                audio = torch.cat(
                    [audio, torch.full_like(audio[-1], -torch.inf).unsqueeze(0)], dim=0
                )
        else:
            raise ValueError(f"Invalid pad_mode: {pad_mode}")

        # input shape of DGSCT is (10, 320000), each slice is repeated 10 times
        audio = audio.repeat(1, 10)

        return audio


def get_dataset_stats():
    dataset = SoundActionsDataset("train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    durations = []
    for sample in tqdm(dataloader):
        duration = sample["video"].shape[0] / sample["video_metadata"]["video_fps"]
        durations.append(duration)
        del sample

    print(f"=> durations: {durations}")


if __name__ == "__main__":
    dataset = SoundActionsDataset("train")
    # # dataset = SoundActionsDataset("train", load_mode="preload", size=10)
    print(len(dataset))
    sample = dataset[279]
    print(
        f'video shape: {sample["video"].shape}, video type: {sample["video"].dtype}, video min: {sample["video"].min()}, video max: {sample["video"].max()}'
    )
    print(
        f'audio shape: {sample["audio"].shape}, audio type: {sample["audio"].dtype}, audio min: {sample["audio"].min()}, audio max: {sample["audio"].max()}'
    )
    # print(sample["label"])

    # get_dataset_stats()
