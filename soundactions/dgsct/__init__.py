import sys
import pathlib

import torch

from .nets.net_trans import MMIL_Net
from .base_options import BaseOptions


def load_DGSCT(
    pretrain: bool,
    mode: str,
    dropout: float = 0.0,
    verbose: bool = False,
    ckpt_path: str = None,
    adapter_layer_idx: list = None,
    **kwargs,
):
    ## test: no trainable params
    ## train: train adapter + CMBS + mlp_class
    ## finetune: train CMBS + mlp_class
    assert mode in ["test", "train", "finetune_cls", "finetune_all"]
    options = BaseOptions()
    options.initialize()

    if adapter_layer_idx is not None:
        if mode != "train":
            raise ValueError("adapter_layer_idx is provided, but it will be ignored in non-train mode.")

    if mode == "test":
        args_list = [
            "--Adapter_downsample=8",
            "--accum_itr=2",
            "--batch_size=8",
            "--decay=0.35",
            "--decay_epoch=3",
            "--early_stop=20",
            "--epochs=50",
            "--is_audio_adapter_p1=1",
            "--is_audio_adapter_p2=1",
            "--is_audio_adapter_p3=0",
            "--is_before_layernorm=1",
            "--is_bn=1",
            "--is_fusion_before=1",
            "--is_gate=1",
            "--is_post_layernorm=1",
            "--is_vit_ln=0",
            "--lr=5e-04",
            "--lr_mlp=5e-06",
            "--mode=test",
            "--model=MMIL_Net",
            "--num_conv_group=2",
            "--num_tokens=32",
            "--num_workers=16",
            "--seed=43",
            "--backbone_type=audioset",
        ]
    elif mode == "train":
        args_list = [
            "--Adapter_downsample=8",
            "--accum_itr=4",
            "--batch_size=4",
            "--decay=0.35",
            "--decay_epoch=3",
            "--early_stop=20",
            "--epochs=50",
            "--is_audio_adapter_p1=1",
            "--is_audio_adapter_p2=1",
            "--is_audio_adapter_p3=0",
            "--is_before_layernorm=1",
            "--is_bn=1",
            "--is_fusion_before=1",
            "--is_gate=1",
            "--is_post_layernorm=1",
            "--lr=5e-04",
            "--lr_mlp=5e-06",
            "--mode=train",
            "--model=MMIL_Net",
            "--num_conv_group=2",
            "--num_tokens=32",
            "--num_workers=16",
            "--seed=43",
            "--backbone_type=audioset",
        ]
    elif mode.split("_")[0] == "finetune":
        args_list = [
            "--Adapter_downsample=8",
            "--accum_itr=4",
            "--batch_size=4",
            "--decay=0.35",
            "--decay_epoch=3",
            "--early_stop=20",
            "--epochs=50",
            "--is_audio_adapter_p1=1",
            "--is_audio_adapter_p2=1",
            "--is_audio_adapter_p3=0",
            "--is_before_layernorm=1",
            "--is_bn=1",
            "--is_fusion_before=1",
            "--is_gate=1",
            "--is_post_layernorm=1",
            "--lr_mlp=5e-06",
            "--mode=train",
            "--model=MMIL_Net",
            "--num_conv_group=2",
            "--num_tokens=32",
            "--num_workers=16",
            "--seed=43",
            "--backbone_type=audioset",
        ]
    for key in kwargs:
        args_list.append([f"--{key}={kwargs[key]}"])
    args = options.parser.parse_args(args_list)
    model = MMIL_Net(args, dropout=dropout)
    if pretrain:
        if verbose:
            print("=> Loading pre-trained weights for DG-SCT")
        if ckpt_path is None:
            ckpt_path = pathlib.Path(__file__) / "../../checkpoints/dg-sct/best_82.18.pt"
            ckpt_path = ckpt_path.resolve().absolute()
        else:
            ckpt_path = pathlib.Path(ckpt_path).resolve().absolute()
        model.load_state_dict(
            torch.load(ckpt_path),
            strict=False,
        )

    if mode == "test":
        model.eval()
        if verbose:
            print("=> Model set to eval mode")
    elif mode == "train":
        param_group = []
        for name, param in model.named_parameters():
            param.requires_grad = False
            tmp = 1
            for num in param.shape:
                tmp *= num
            if "ViT" in name or "swin" in name:
                param.requires_grad = False
            elif "htsat" in name:
                param.requires_grad = False
            elif "adapter_blocks" in name:
                # if adapter_layer_idx is None, train all adapters
                if adapter_layer_idx is None:
                    param.requires_grad = True
                    if verbose:
                        print("########### train layer:", name, param.shape, tmp)
                else:
                    # train the adapter layers in adapter_layer_idx
                    layer_idx = int(name.split(".")[1].split(".")[0])
                    if layer_idx in adapter_layer_idx:
                        param.requires_grad = True
                        if verbose:
                            print("########### train layer:", name, param.shape, tmp)
            elif "CMBS" in name:
                param.requires_grad = True
            elif "mlp_class" in name:
                param.requires_grad = True
            elif "temporal_attn" in name:
                param.requires_grad = True
            if "mlp_class" in name:
                param_group.append({"params": param, "lr": args.lr_mlp})
    elif mode == "finetune_cls":
        param_group = []
        for name, param in model.named_parameters():
            param.requires_grad = False
            tmp = 1
            for num in param.shape:
                tmp *= num
            if "ViT" in name or "swin" in name:
                param.requires_grad = False
            elif "htsat" in name:
                param.requires_grad = False
            elif "adapter_blocks" in name:
                param.requires_grad = False
            elif "CMBS" in name:
                param.requires_grad = True
            elif "mlp_class" in name:
                param.requires_grad = True
            elif "temporal_attn" in name:
                param.requires_grad = False
            if "mlp_class" in name:
                param_group.append({"params": param, "lr": args.lr_mlp})
            else:
                param_group.append({"params": param, "lr": args.lr})
    elif mode == "finetune_all":
        param_group = []
        for name, param in model.named_parameters():
            param.requires_grad = False
            tmp = 1
            for num in param.shape:
                tmp *= num
            if "ViT" in name or "swin" in name:
                param.requires_grad = False
            elif "htsat" in name:
                param.requires_grad = False
            elif "adapter_blocks" in name:
                param.requires_grad = True
            elif "CMBS" in name:
                param.requires_grad = True
            elif "mlp_class" in name:
                param.requires_grad = True
            elif "temporal_attn" in name:
                param.requires_grad = False
            if "mlp_class" in name:
                param_group.append({"params": param, "lr": args.lr_mlp})
            else:
                param_group.append({"params": param, "lr": args.lr})
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return model
