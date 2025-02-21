import sys
import yaml
import glob
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))
from dgsct import load_DGSCT
from dgsct.AVE.main_trans import eval
from dgsct.AVE.dataloader import AVE_dataset
from inference import LitDGSCT
from dgsct.base_options import BaseOptions
from eopma import EnsembleEmbedding, EoPMA


def run_eval(
    dgsct_dir: str,
    ckpt_path: str,
    config_path: str,
    task: str,
    inference_mode: str,
    ensemble_modality: str,
    device: str = "cuda",
    verbose: bool = False,
    **kwargs,
):
    """Test soundactions-finetuned ckpt over original DG-SCT tasks"""
    assert inference_mode in ["finetuned_only", "ensemble_embedding", "eopma"]
    soundactions = LitDGSCT.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        **wandb_config_to_pl(config_path),
    )

    if verbose:
        print(f"=> Loaded checkpoint from {ckpt_path}")
        print(f"=> Inference mode: {inference_mode}")

    if inference_mode == "finetuned_only":
        model = soundactions.model.to(device)
        # swtich to the original classifier
        model.CMBS = load_DGSCT(pretrain=True, mode="test").CMBS.to(device)
        model.eval()
    elif inference_mode == "ensemble_embedding":
        model = EnsembleEmbedding(
            finetuned_ckpt_path=ckpt_path,
            finetuned_config_path=config_path,
            beta=kwargs.get("beta"),
            modality=ensemble_modality,
            device=device,
        )
    elif inference_mode == "eopma":
        model = EoPMA(
            finetuned_ckpt_path=ckpt_path,
            finetuned_config_path=config_path,
            beta=kwargs.get("beta"),
            modality=ensemble_modality,
            device=device,
        )

    if verbose:
        print("=> Model loaded successfully")

    # load options
    opt = BaseOptions()
    opt.initialize()
    if task == "AVE":
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
            f"--root_path={dgsct_dir}",
        ]
        args = opt.parser.parse_args(args_list)

        test_dataset = AVE_dataset(opt=args, mode="test")
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )
        if verbose:
            print(f"=> Start testing on task: {task}")
        mean_acc = eval(model, test_loader, args)

        if verbose:
            print("=" * 20)
            print(f"=> Task: {task}")
            print(f"=> Checkpoint: {ckpt_path}")
            if kwargs:
                print(
                    "=> Hyperparameters: "
                    + ",".join([f"{k}={v}" for k, v in kwargs.items()])
                )
            print(f"=> Mean accuracy: {mean_acc:.2f}")
        return mean_acc


def eval_all(
    ckpt_list: list,
    config_list: list,
    inference_mode: str,
    ensemble_modality: str,
    beta_list: list,
    results_save_path: str,
):
    """Evaluate all ensemble models and save results"""
    results = []

    for ckpt, config in tqdm(zip(ckpt_list, config_list)):
        for beta in beta_list:
            try:
                mean_acc = run_eval(
                    dgsct_dir="/projects/ec12/jinyueg/DG-SCT",
                    ckpt_path=ckpt,
                    config_path=config,
                    task="AVE",
                    inference_mode=inference_mode,
                    ensemble_modality=ensemble_modality,
                    beta=beta,
                )
                print(f"=> beta: {beta}, mean_acc: {mean_acc:.2f}")
                results.append((ckpt, beta, mean_acc))
            except Exception as e:
                print(f"=> Error: {e}")
    results_df = pd.DataFrame(results, columns=["ckpt", "beta", "mean_acc"])
    results_df.to_csv(results_save_path, index=False)


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
    # run_eval(
    #     dgsct_dir="/projects/ec12/jinyueg/DG-SCT",
    #     ckpt_path='/projects/ec12/jinyueg/SoundActions/soundactions/logs/V04_all_PerceptionType_av_av/soundactions/0407syid/checkpoints/epoch=12-step=962.ckpt',
    #     config_path='/projects/ec12/jinyueg/SoundActions/soundactions/logs/V04_all_PerceptionType_av_av/wandb/run-20240825_232837-0407syid/files/config.yaml',
    #     task="AVE",
    #     inference_mode="add_final_embedding",
    #     beta=0.5,
    #     verbose=True,
    # )

    ckpt_list = glob.glob(
        "/projects/ec12/jinyueg/SoundActions/soundactions/logs/V06_all_PerceptionType_v_v/soundactions/*/checkpoints/*.ckpt"
    )
    eval_all(
        ckpt_list=ckpt_list,
        config_list=[
            "/projects/ec12/jinyueg/SoundActions/soundactions/logs/V06_all_PerceptionType_v_v/wandb/latest-run/files/config.yaml"
        ]
        * len(ckpt_list),
        inference_mode="eopma",
        ensemble_modality="v",
        beta_list=list(np.arange(0.00, 0.04, 0.001)),
        results_save_path="../results/V06_eopma_v_finebeta.csv",
    )
