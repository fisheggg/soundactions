import os
import glob

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision


class SoundActionsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_mode,
        load_mode="online",
        sample_mode="full",
        root: str = "/fp/homes01/u01/ec-jinyueg/felles_/Research/Project/AMBIENT/Datasets/SoundActions/",
        video_path: str = "video-HD",
        audio_path: str = "wav",
        label_path: str ="labels/SoundActions_labeling_A.csv",
        video_transform = None,
        audio_transform = None,
        size: int = None,
    ):
        assert load_mode in ["preload", "online"]
        assert sample_mode in ["full"]

        self.root = root
        self.video_paths = sorted(glob.glob(os.path.join(root, video_path, "*.mp4")))
        self.audio_paths = sorted(glob.glob(os.path.join(root, audio_path, "*.wav")))
        assert len(self.video_paths) == len(self.audio_paths)
        self.labels = pd.read_csv(os.path.join(root, label_path))
        assert len(self.video_paths) == len(self.labels)

        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.split_mode = split_mode
        self.load_mode = load_mode
        self.sample_mode = sample_mode

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
                video, _, video_metadata = torchvision.io.read_video(
                    video_path, pts_unit="sec"
                )
                audio, audio_fps = torchaudio.load(audio_path)
                self.videos.append(video)
                self.videos_metadata.append(video_metadata)
                self.audios.append(audio)
                self.audios_fps.append(audio_fps)
            print("=> Preloading done")

    def __len__(self):
        if self.sample_mode == "full":
            return len(self.video_paths)

    def __getitem__(self, index):
        data = {}
        data["label"] = self.labels.iloc[index]
        if self.load_mode == "online":
            video_path = self.video_paths[index]
            audio_path = self.audio_paths[index]
            data["video"], _, data["video_metadata"] = torchvision.io.read_video(
                video_path, pts_unit="sec"
            )
            data["audio"], data["audio_fs"] = torchaudio.load(audio_path)
        elif self.load_mode == "preload":
            data["video"] = self.videos[index]
            data["video_metadata"] = self.videos_metadata[index]
            data["audio"] = self.audios[index]
            data["audio_fs"] = self.audios_fps[index]

        if self.video_transform is not None:
            data["video"] = self.video_transform(data["video"])
        if self.audio_transform is not None:
            data["audio"] = self.audio_transform(data["audio"])
        return data


if __name__ == "__main__":
    # dataset = SoundActionsDataset("train")
    dataset = SoundActionsDataset("train", load_mode="preload", size=10)
    print(len(dataset))
    sample = dataset[0]
    print(sample["video"].shape)
    print(sample["video_metadata"])
    print(sample["audio"].shape)
    print(sample["audio_fs"])
    print(sample["label"])
