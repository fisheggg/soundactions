"""
GEoPMA: Gated Ensemble of Perception Mode Adapters
Based on DG-SCT model
"""

import sys
import copy
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

sys.path.append(str(Path(__file__).resolve().parent))
from utils import wandb_config_to_pl
from dgsct import load_DGSCT
from inference import LitDGSCT
from dgsct.base_options import BaseOptions
from dgsct.nets.utils import do_mixup


class EnsembleEmbedding(nn.Module):
    def __init__(
        self,
        finetuned_ckpt_path: str,
        finetuned_config_path: str,
        beta: float,
        modality: str,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """
        Ensemble finetuned SoundActions model with original DG-SCT model

        :param beta: weight of finetuned embeddings
        """
        super().__init__()
        assert modality in ["a", "v", "av"]
        self.beta = beta
        self.modality = modality

        self.model_orig = (
            load_DGSCT(pretrain=True, mode="test", verbose=verbose).eval().to(device)
        )
        self.model_ft = (
            LitDGSCT.load_from_checkpoint(
                finetuned_ckpt_path,
                verbose=verbose,
                **wandb_config_to_pl(finetuned_config_path),
            )
            .eval()
            .model.to(device)
        )
        self.cls = self.model_orig.CMBS

    def forward(self, audio, video):
        _, _, _, _, v_feature_ft, a_feature_ft = self.model_ft(audio, video)
        _, _, _, _, v_feature, a_feature = self.model_orig(audio, video)
        if "a" in self.modality:
            a_feature = self.beta * a_feature_ft + (1 - self.beta) * a_feature
        if "v" in self.modality:
            v_feature = self.beta * v_feature_ft + (1 - self.beta) * v_feature
        return self.cls(v_feature, a_feature)


class EoPMA(nn.Module):
    def __init__(
        self,
        finetuned_ckpt_path: str,
        finetuned_config_path: str,
        beta: float,
        modality: str,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """
        EoPMA: Ensemble of Perception Mode Adapters

        :param beta: weight of finetuned embeddings
        """
        super().__init__()
        assert modality in ["a", "v", "av"]
        self.beta = beta
        self.modality = modality

        self.model_orig = (
            load_DGSCT(pretrain=True, mode="test", verbose=verbose).eval().to(device)
        )
        model_ft = (
            LitDGSCT.load_from_checkpoint(
                finetuned_ckpt_path,
                verbose=verbose,
                **wandb_config_to_pl(finetuned_config_path),
            )
            .eval()
            .model.to(device)
        )
        # only take the adapters from model_ft
        self.ft_vis_adapter_blocks_p1 = copy.deepcopy(model_ft.vis_adapter_blocks_p1)
        self.ft_vis_adapter_blocks_p2 = copy.deepcopy(model_ft.vis_adapter_blocks_p2)
        self.ft_audio_adapter_blocks_p1 = copy.deepcopy(
            model_ft.audio_adapter_blocks_p1
        )
        self.ft_audio_adapter_blocks_p2 = copy.deepcopy(
            model_ft.audio_adapter_blocks_p2
        )
        del model_ft

    def forward_swin(self, audio, vis, mixup_lambda, rand_train_idx=12, stage="eval"):
        audio = audio[0]
        audio = audio.view(audio.size(0) * audio.size(1), -1)
        waveform = audio
        bs = vis.size(0)
        vis = rearrange(vis, "b t c w h -> (b t) c w h")
        f_v = self.model_orig.swin.patch_embed(vis)

        audio = self.model_orig.htsat.spectrogram_extractor(audio)
        audio = self.model_orig.htsat.logmel_extractor(audio)
        audio = audio.transpose(1, 3)
        audio = self.model_orig.htsat.bn0(audio)
        audio = audio.transpose(1, 3)
        if self.model_orig.htsat.training:
            audio = self.model_orig.htsat.spec_augmenter(audio)
        if self.model_orig.htsat.training and mixup_lambda is not None:
            audio = do_mixup(audio, mixup_lambda)

        if (
            audio.shape[2]
            > self.model_orig.htsat.freq_ratio * self.model_orig.htsat.spec_size
        ):
            audio = self.model_orig.htsat.crop_wav(
                audio,
                crop_size=self.model_orig.htsat.freq_ratio
                * self.model_orig.htsat.spec_size,
            )
            audio = self.model_orig.htsat.reshape_wav2img(audio)
        else:  # this part is typically used, and most easy one
            audio = self.model_orig.htsat.reshape_wav2img(audio)
        frames_num = audio.shape[2]
        f_a = self.model_orig.htsat.patch_embed(audio)
        if self.model_orig.htsat.ape:
            f_a = f_a + self.model_orig.htsat.absolute_pos_embed
        f_a = self.model_orig.htsat.pos_drop(f_a)

        idx_layer = 0
        out_idx_layer = 0
        for _, (my_blk, htsat_blk) in enumerate(
            zip(self.model_orig.swin.layers, self.model_orig.htsat.layers)
        ):
            if len(my_blk.blocks) == len(htsat_blk.blocks):
                aud_blocks = htsat_blk.blocks
            else:
                aud_blocks = [
                    None,
                    None,
                    htsat_blk.blocks[0],
                    None,
                    None,
                    htsat_blk.blocks[1],
                    None,
                    None,
                    htsat_blk.blocks[2],
                    None,
                    None,
                    htsat_blk.blocks[3],
                    None,
                    None,
                    htsat_blk.blocks[4],
                    None,
                    None,
                    htsat_blk.blocks[5],
                ]
                assert len(aud_blocks) == len(my_blk.blocks)

            for blk, blk_a in zip(my_blk.blocks, aud_blocks):
                if blk_a is not None:
                    f_a_res, f_a_spatial_att_maps = self.model_orig.audio_adapter_blocks_p1[
                        idx_layer
                    ](
                        f_a.permute(0, 2, 1).unsqueeze(-1),
                        f_v.permute(0, 2, 1).unsqueeze(-1),
                    )
                    f_v_res, f_v_spatial_att_maps = self.model_orig.vis_adapter_blocks_p1[
                        idx_layer
                    ](
                        f_v.permute(0, 2, 1).unsqueeze(-1),
                        f_a.permute(0, 2, 1).unsqueeze(-1),
                    )
                    # compute embeddings from the finetuned adapters, and add them to the original embeddings
                    if "a" in self.modality:
                        f_a_res_ft, f_a_spatial_att_maps_ft = (
                            self.ft_audio_adapter_blocks_p1[idx_layer](
                                f_a.permute(0, 2, 1).unsqueeze(-1),
                                f_v.permute(0, 2, 1).unsqueeze(-1),
                            )
                        )
                        f_a_res = self.beta * f_a_res_ft + (1 - self.beta) * f_a_res
                    if "v" in self.modality:
                        f_v_res_ft, f_v_spatial_att_maps_ft = (
                            self.ft_vis_adapter_blocks_p1[idx_layer](
                                f_v.permute(0, 2, 1).unsqueeze(-1),
                                f_a.permute(0, 2, 1).unsqueeze(-1),
                            )
                        )
                        f_v_res = self.beta * f_v_res_ft + (1 - self.beta) * f_v_res

                    f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
                    f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)

                    f_a, _ = blk_a(f_a)
                    f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                    f_a_res, f_a_spatial_att_maps = self.model_orig.audio_adapter_blocks_p2[
                        idx_layer
                    ](
                        f_a.permute(0, 2, 1).unsqueeze(-1),
                        f_v.permute(0, 2, 1).unsqueeze(-1),
                    )
                    f_v_res, f_v_spatial_att_maps = self.model_orig.vis_adapter_blocks_p2[
                        idx_layer
                    ](
                        f_v.permute(0, 2, 1).unsqueeze(-1),
                        f_a.permute(0, 2, 1).unsqueeze(-1),
                    )
                    # compute embeddings from the finetuned adapters, and add them to the original embeddings
                    if "a" in self.modality:
                        f_a_res_ft, f_a_spatial_att_maps_ft = (
                            self.ft_audio_adapter_blocks_p2[idx_layer](
                                f_a.permute(0, 2, 1).unsqueeze(-1),
                                f_v.permute(0, 2, 1).unsqueeze(-1),
                            )
                        )
                        f_a_res = self.beta * f_a_res_ft + (1 - self.beta) * f_a_res
                    if "v" in self.modality:
                        f_v_res_ft, f_v_spatial_att_maps_ft = (
                            self.ft_vis_adapter_blocks_p2[idx_layer](
                                f_v.permute(0, 2, 1).unsqueeze(-1),
                                f_a.permute(0, 2, 1).unsqueeze(-1),
                            )
                        )
                        f_v_res = self.beta * f_v_res_ft + (1 - self.beta) * f_v_res

                    f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
                    f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)

                    f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                    idx_layer = idx_layer + 1

                else:
                    f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
                    f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))

            f_v = my_blk.downsample(f_v)
            if htsat_blk.downsample is not None:
                f_a = htsat_blk.downsample(f_a)

        f_v = self.model_orig.swin.norm(f_v)

        # f_v = f_v.mean(dim=1, keepdim=True)
        f_v = torch.bmm(f_v_spatial_att_maps, f_v)
        # f_a = f_a.mean(dim=1, keepdim=True)
        f_a = torch.bmm(f_a_spatial_att_maps, f_a)
        # f_v, f_a = self.audio_visual_adapter(f_v, f_a)

        ########## Temporal Attention ##########
        f_v = f_v.view(bs, 10, -1)
        f_a = f_a.view(bs, 10, -1)
        video_feature, audio_feature, audio_visual_gate = self.model_orig.temporal_attn(f_v, f_a)

        is_event_scores, event_scores, av_score = self.model_orig.CMBS(
            video_feature, audio_feature
        )

        return (
            is_event_scores,
            event_scores,
            audio_visual_gate,
            av_score,
            video_feature,
            audio_feature,
        )

    def forward(self, audio, vis, mixup_lambda=None, rand_train_idx=12, stage="eval"):
        return self.forward_swin(
            audio, vis, mixup_lambda, rand_train_idx=12, stage="eval"
        )


class GEoPMA(nn.Module):
    pass
