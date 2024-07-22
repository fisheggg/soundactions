import os
import glob

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class SoundActionsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_mode,
        load_mode="online",
        sample_mode="full",
        root: str = "/fp/homes01/u01/ec-jinyueg/felles_/Research/Project/AMBIENT/Datasets/SoundActions/",
        video_path: str = "video-HD",
        audio_path: str = "wav",
        label_path: str = "labels/SoundActions_labeling_A.csv",
        video_transform=None,
        audio_transform=None,
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

        self.my_normalize = Compose([
			Resize([192,192], interpolation=Image.BICUBIC),
			Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
		])

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
                self.videos.append(self.my_normalize(video))
                self.videos_metadata.append(video_metadata)
                self.audios.append(audio)
                self.audios_fps.append(audio_fps)
            print("=> Preloading done")

    def _wav2fbank(self, filename, filename2=None, idx=None):
        """
        copied from https://github.com/haoyi-duan/DG-SCT/blob/211fc57f0093e2111f43f670362a41f0a4c2322b/DG-SCT/AVE/dataloader.py#L92
        """
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0 : waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0 : waveform1.shape[1]]

            # sample lambda from uniform distribution
            # mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        if waveform.shape[1] > 16000 * (self.opt.audio_length + 0.1):
            sample_indx = np.linspace(
                0,
                waveform.shape[1] - 16000 * (self.opt.audio_length + 0.1),
                num=10,
                dtype=int,
            )
            waveform = waveform[
                :,
                sample_indx[idx] : sample_indx[idx]
                + int(16000 * self.opt.audio_length),
            ]
        ## align end ##

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=192,
            dither=0.0,
            frame_shift=5.2,
        )

        ########### ------> very important: audio normalized
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        ### <--------
        target_length = 192  ##

        # target_length = 512 ## 5s
        # target_length = 256 ## 2.5s
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

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
            data["video"] = self.my_normalize(data["video"])
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
