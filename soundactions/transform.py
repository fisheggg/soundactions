import torch
from torch import nn
from torchvision.transforms.v2 import functional as TF
from torchvision.transforms.v2 import ColorJitter


class VideoRandomHorizontalFlip(nn.Module):
    """
    Randomly flip the video horizontally with a given probability.
    If flipped, all frames in the same video will be flipped, vice versa.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0 <= p <= 1, f"p should be in [0, 1], got {p}"
        self.p = p

    def forward(self, video):
        """input shape: (T, C, H, W)"""
        assert video.dim() == 4, f"video shape: {video.shape}"
        if torch.rand(1) < self.p:
            video = TF.horizontal_flip(video)

        return video


class VideoColorJitter(nn.Module):
    """
    All frames in the same video will be applied with the same jittering parameters.
    Same API as torchvision.transforms.ColorJitter.
    """

    def __init__(
        self,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        hue: float = 0,
    ):
        super().__init__()
        self.b = brightness
        self.c = contrast
        self.s = saturation
        self.h = hue

    def forward(self, video):
        """input shape: (T, C, H, W)"""
        assert video.dim() == 4, f"video shape: {video.shape}"
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = (
            ColorJitter.get_params(self.b, self.c, self.s, self.h)
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                video = TF.adjust_brightness(video, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                video = TF.adjust_contrast(video, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                video = TF.adjust_saturation(video, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                video = TF.adjust_hue(video, hue_factor)

        return video
