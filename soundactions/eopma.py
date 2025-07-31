import sys
import copy
import yaml
from pathlib import Path

import torch
from einops import rearrange
from torch import nn

sys.path.append(str(Path(__file__).resolve().parent))
from dgsct import load_DGSCT
from inference import LitDGSCT
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
        config = wandb_config_to_pl(finetuned_config_path)
        if "verbose" in config:
            config = config.pop("verbose")

        self.model_orig = (
            load_DGSCT(pretrain=True, mode="test", verbose=verbose).eval().to(device)
        )
        self.model_ft = (
            LitDGSCT.load_from_checkpoint(
                finetuned_ckpt_path,
                verbose=verbose,
                **config,
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
        beta: float,
        modality: str,
        finetuned_ckpt_path: str = None,
        finetuned_config_path: str = None,
        second_dgsct_ckpt_path: str = None,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """
        EoPMA: Ensemble of Perception Mode Adapters

        :param beta: weight of finetuned embeddings
        """
        super().__init__()
        assert modality in ["a", "v", "av"]
        if finetuned_ckpt_path is None:
            # using two dgsct
            assert second_dgsct_ckpt_path is not None
        else:
            # using soundactions finetuned
            assert second_dgsct_ckpt_path is None
            assert finetuned_config_path is not None

        self.beta = beta
        self.modality = modality
        config = wandb_config_to_pl(finetuned_config_path)
        if "verbose" in config:
            verbose = config.pop("verbose")

        self.model_orig = (
            load_DGSCT(pretrain=True, mode="test", verbose=verbose).eval().to(device)
        )
        if finetuned_ckpt_path is not None:
            # using soundactions finetuned
            model_ft = (
                LitDGSCT.load_from_checkpoint(
                    finetuned_ckpt_path,
                    verbose=verbose,
                    **config,
                )
                .eval()
                .model.to(device)
            )
        else:
            # using two dgsct
            model_ft = (
                load_DGSCT(
                    pretrain=True,
                    mode="test",
                    verbose=verbose,
                    ckpt_path=second_dgsct_ckpt_path,
                )
                .eval()
                .to(device)
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
                    f_a_res, f_a_spatial_att_maps = (
                        self.model_orig.audio_adapter_blocks_p1[idx_layer](
                            f_a.permute(0, 2, 1).unsqueeze(-1),
                            f_v.permute(0, 2, 1).unsqueeze(-1),
                        )
                    )
                    f_v_res, f_v_spatial_att_maps = (
                        self.model_orig.vis_adapter_blocks_p1[idx_layer](
                            f_v.permute(0, 2, 1).unsqueeze(-1),
                            f_a.permute(0, 2, 1).unsqueeze(-1),
                        )
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

                    f_a_res, f_a_spatial_att_maps = (
                        self.model_orig.audio_adapter_blocks_p2[idx_layer](
                            f_a.permute(0, 2, 1).unsqueeze(-1),
                            f_v.permute(0, 2, 1).unsqueeze(-1),
                        )
                    )
                    f_v_res, f_v_spatial_att_maps = (
                        self.model_orig.vis_adapter_blocks_p2[idx_layer](
                            f_v.permute(0, 2, 1).unsqueeze(-1),
                            f_a.permute(0, 2, 1).unsqueeze(-1),
                        )
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
        video_feature, audio_feature, audio_visual_gate = self.model_orig.temporal_attn(
            f_v, f_a
        )

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
    """
    Gated Ensemble of Perception Mode Adapters
    """

    def __init__(
        self,
        modality: str,
        finetuned_ckpt_path: str = None,
        finetuned_config_path: str = None,
        second_dgsct_ckpt_path: str = None,
        device: str = "cuda",
        verbose: bool = False,
    ):
        """
        EoPMA: Ensemble of Perception Mode Adapters

        :param beta: weight of finetuned embeddings
        """
        super().__init__()
        print(f"modality: {modality}")
        assert modality in ["a", "v", "av"]
        if finetuned_ckpt_path is None:
            # using two dgsct
            assert second_dgsct_ckpt_path is not None
        else:
            # using soundactions finetuned
            assert second_dgsct_ckpt_path is None
            assert finetuned_config_path is not None

        self.modality = modality

        self.model_orig = (
            load_DGSCT(pretrain=True, mode="test", verbose=verbose).eval().to(device)
        )
        if finetuned_ckpt_path is not None:
            # using soundactions finetuned
            model_ft = (
                LitDGSCT.load_from_checkpoint(
                    finetuned_ckpt_path,
                    verbose=verbose,
                    **wandb_config_to_pl(finetuned_config_path),
                )
                .eval()
                .model.to(device)
            )
        else:
            # using two dgsct
            model_ft = (
                load_DGSCT(
                    pretrain=True,
                    mode="test",
                    verbose=verbose,
                    ckpt_path=second_dgsct_ckpt_path,
                )
                .eval()
                .to(device)
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

        # get dim information of the backbones
        hidden_list, hidden_list_a = [], []
        down_in_dim, down_in_dim_a = [], []
        down_out_dim, down_out_dim_a = [], []
        conv_dim, conv_dim_a = [], []
        for idx_layer, (my_blk, my_blk_a) in enumerate(zip(self.model_orig.swin.layers, self.model_orig.htsat.layers)):
            conv_dim_tmp = (my_blk.input_resolution[0]*my_blk.input_resolution[1])
            conv_dim_tmp_a = (my_blk_a.input_resolution[0]*my_blk_a.input_resolution[1])
            if not isinstance(my_blk.downsample, nn.Identity):
                down_in_dim.append(my_blk.downsample.reduction.in_features)
                down_out_dim.append(my_blk.downsample.reduction.out_features)
            if my_blk_a.downsample is not None:
                down_in_dim_a.append(my_blk_a.downsample.reduction.in_features)
                down_out_dim_a.append(my_blk_a.downsample.reduction.out_features)
            for blk, blk_a in zip(my_blk.blocks, my_blk_a.blocks):
                hidden_d_size = blk.norm1.normalized_shape[0]
                hidden_list.append(hidden_d_size)
                conv_dim.append(conv_dim_tmp)
                hidden_d_size_a = blk_a.norm1.normalized_shape[0]
                hidden_list_a.append(hidden_d_size_a)
                conv_dim_a.append(conv_dim_tmp_a)
        gate_input_dim = [0] * len(hidden_list)
        for idx_layer in range(len(hidden_list)):
            if "a" in self.modality:
                gate_input_dim[idx_layer] += hidden_list_a[idx_layer]
            if "v" in self.modality:
                gate_input_dim[idx_layer] += hidden_list[idx_layer]
        print(f"down_in_dim: {down_in_dim}")
        print(f"down_out_dim: {down_out_dim}")
        print(f"down_in_dim_a: {down_in_dim_a}")
        print(f"down_out_dim_a: {down_out_dim_a}")
        print(f"conv_dim: {conv_dim}")
        print(f"conv_dim_a: {conv_dim_a}")

        # init gate module
        # if "a" in self.modality:
        #     self.gate_norm_a = nn.ModuleList(
        #         [
        #             nn.LayerNorm([conv_dim_a[idx_layer], hidden_list_a[idx_layer]])
        #             for idx_layer in range(len(hidden_list))
        #         ]
        #     )
        # if "v" in self.modality:
        #     self.gate_norm_v = nn.ModuleList(
        #         [
        #             nn.LayerNorm([conv_dim[idx_layer], hidden_list[idx_layer]])
        #             for idx_layer in range(len(hidden_list))
        #         ]
        #     )
        self.gate_ff_1 = nn.ModuleList(
            [
                nn.Linear(gate_input_dim[idx_layer], 2)
                for idx_layer in range(len(hidden_list))
            ]
        ).to(device)
        self.gate_ff_2 = nn.ModuleList(
            [
                nn.Linear(gate_input_dim[idx_layer], 2)
                for idx_layer in range(len(hidden_list))
            ]
        ).to(device)


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
                    # compute gate
                    if self.modality == "av":
                        # f_v_norm = self.gate_norm_v[idx_layer]
                        # f_a_norm = self.gate_norm_a[idx_layer]
                        gate_input = torch.cat(
                            [f_v.mean(dim=1), f_a.mean(dim=1)], dim=1
                        )
                    elif self.modality == "a":
                        # f_a_norm = self.gate_norm_a[idx_layer]
                        gate_input = f_a.mean(dim=1)
                    elif self.modality == "v":
                        # f_v_norm = self.gate_norm_v[idx_layer]
                        gate_input = f_v.mean(dim=1)
                    gate_weights = self.gate_ff_1[idx_layer](gate_input)
                    gate_weights = torch.softmax(gate_weights, dim=1)

                    f_a_res, f_a_spatial_att_maps = (
                        self.model_orig.audio_adapter_blocks_p1[idx_layer](
                            f_a.permute(0, 2, 1).unsqueeze(-1),
                            f_v.permute(0, 2, 1).unsqueeze(-1),
                        )
                    )
                    f_v_res, f_v_spatial_att_maps = (
                        self.model_orig.vis_adapter_blocks_p1[idx_layer](
                            f_v.permute(0, 2, 1).unsqueeze(-1),
                            f_a.permute(0, 2, 1).unsqueeze(-1),
                        )
                    )
                    # compute embeddings from the finetuned adapters, and add them to the original embeddings
                    if "a" in self.modality:
                        f_a_res_ft, f_a_spatial_att_maps_ft = (
                            self.ft_audio_adapter_blocks_p1[idx_layer](
                                f_a.permute(0, 2, 1).unsqueeze(-1),
                                f_v.permute(0, 2, 1).unsqueeze(-1),
                            )
                        )
                        f_a_res = (
                            gate_weights[:, 0].view(-1, 1, 1, 1) * f_a_res
                            + gate_weights[:, 1].view(-1, 1, 1, 1) * f_a_res_ft
                        )
                    if "v" in self.modality:
                        f_v_res_ft, f_v_spatial_att_maps_ft = (
                            self.ft_vis_adapter_blocks_p1[idx_layer](
                                f_v.permute(0, 2, 1).unsqueeze(-1),
                                f_a.permute(0, 2, 1).unsqueeze(-1),
                            )
                        )
                        f_v_res = (
                            gate_weights[:, 0].view(-1, 1, 1, 1) * f_v_res
                            + gate_weights[:, 1].view(-1, 1, 1, 1) * f_v_res_ft
                        )

                    f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
                    f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)

                    f_a, _ = blk_a(f_a)
                    f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                    # compute gate
                    if self.modality == "av":
                        # f_v_norm = self.gate_norm_v[idx_layer]
                        # f_a_norm = self.gate_norm_a[idx_layer]
                        gate_input = torch.cat(
                            [f_v.mean(dim=1), f_a.mean(dim=1)], dim=1
                        )
                    elif self.modality == "a":
                        # f_a_norm = self.gate_norm_a[idx_layer]
                        gate_input = f_a.mean(dim=1)
                    elif self.modality == "v":
                        # f_v_norm = self.gate_norm_v[idx_layer]
                        gate_input = f_v.mean(dim=1)
                    gate_weights = self.gate_ff_2[idx_layer](gate_input)
                    gate_weights = torch.softmax(gate_weights, dim=1)

                    f_a_res, f_a_spatial_att_maps = (
                        self.model_orig.audio_adapter_blocks_p2[idx_layer](
                            f_a.permute(0, 2, 1).unsqueeze(-1),
                            f_v.permute(0, 2, 1).unsqueeze(-1),
                        )
                    )
                    f_v_res, f_v_spatial_att_maps = (
                        self.model_orig.vis_adapter_blocks_p2[idx_layer](
                            f_v.permute(0, 2, 1).unsqueeze(-1),
                            f_a.permute(0, 2, 1).unsqueeze(-1),
                        )
                    )
                    # compute embeddings from the finetuned adapters, and add them to the original embeddings
                    if "a" in self.modality:
                        f_a_res_ft, f_a_spatial_att_maps_ft = (
                            self.ft_audio_adapter_blocks_p2[idx_layer](
                                f_a.permute(0, 2, 1).unsqueeze(-1),
                                f_v.permute(0, 2, 1).unsqueeze(-1),
                            )
                        )
                        f_a_res = (
                            gate_weights[:, 0].view(-1, 1, 1, 1) * f_a_res
                            + gate_weights[:, 1].view(-1, 1, 1, 1) * f_a_res_ft
                        )
                    if "v" in self.modality:
                        f_v_res_ft, f_v_spatial_att_maps_ft = (
                            self.ft_vis_adapter_blocks_p2[idx_layer](
                                f_v.permute(0, 2, 1).unsqueeze(-1),
                                f_a.permute(0, 2, 1).unsqueeze(-1),
                            )
                        )
                        f_v_res = (
                            gate_weights[:, 0].view(-1, 1, 1, 1) * f_v_res
                            + gate_weights[:, 1].view(-1, 1, 1, 1) * f_v_res_ft
                        )

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
        video_feature, audio_feature, audio_visual_gate = self.model_orig.temporal_attn(
            f_v, f_a
        )

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

def wandb_config_to_pl(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    out = {}
    for k, v in config.items():
        if "wandb" in k:
            continue
        out[k] = v["value"]
    return out


if __name__ == "__main__":
    model = GEoPMA(
        modality="av",
        finetuned_ckpt_path="/projects/ec12/jinyueg/SoundActions/soundactions/logs/W04_all_Enjoyable_av_av/soundactions/j4rfmsge/checkpoints/epoch=47-step=3504.ckpt",
        finetuned_config_path="/projects/ec12/jinyueg/SoundActions/soundactions/logs/W04_all_Enjoyable_av_av/wandb/run-20240825_142920-j4rfmsge/files/config.yaml",
        verbose=True,
    )

    audio = torch.randn(1, 10, 1, 320000).to("cuda")
    video = torch.randn(1, 10, 3, 192, 192).to("cuda")
    (
        is_event_scores,
        event_scores,
        audio_visual_gate,
        av_score,
        video_feature,
        audio_feature,
    ) = model(audio, video)
