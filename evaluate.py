import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SoccerWindowGraphs, SoccerWindowTensors
from models import utils
from train import run_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--model_file", type=str, default="state_dict_best.pt", help="Model file name or path")
    parser.add_argument("--data_paths", type=str, default="data/sportec/tracking_processed/J03WR9.parquet")
    parser.add_argument("--window_stride", type=int, default=1, help="Step size between windows")
    parser.add_argument("--batch_size", type=int, default=64, help="Override batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    save_path = f"saved/{args.trial:03d}"
    trial_args = utils.load_trial_args(save_path)

    np.random.seed(trial_args.get("seed", 128))
    torch.manual_seed(trial_args.get("seed", 128))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = utils.build_model(trial_args, device=device)
    model_path = utils.resolve_model_path(save_path, args.model_file)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)

    if os.path.isdir(args.data_paths):
        data_paths = [os.path.join(args.data_paths, f) for f in os.listdir(args.data_paths) if f.endswith(".parquet")]
        data_paths.sort()
    elif os.path.isfile(args.data_paths):
        data_paths = [args.data_paths]
    else:
        raise FileNotFoundError(f"Data path not found: {args.data_paths}")

    dataset_args = {
        "node_in_dim": trial_args.get("node_in_dim", 8),
        "edge_in_dim": trial_args.get("edge_in_dim", 0),
        "team_size": trial_args.get("team_size", 11),
        "fps": trial_args.get("fps", 25.0),
        "sample_freq": trial_args.get("sample_freq", 5),
        "window_seconds": trial_args.get("window_seconds", 10.0),
        "window_stride": args.window_stride,
        "self_loops": not trial_args.get("no_self_loops", False),
        "flip_pitch": trial_args.get("flip_pitch", False),
    }

    agent_model = trial_args.get("agent_model")
    if agent_model == "gat":
        dataset = SoccerWindowGraphs(data_paths, verbose=True, **dataset_args)
        collate_fn = utils.collate_window_batch
    else:
        dataset = SoccerWindowTensors(data_paths, verbose=True, **dataset_args)
        collate_fn = None

    batch_size = args.batch_size or trial_args.get("batch_size", 8)
    num_workers = trial_args.get("num_workers", 4)
    pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    rloss_weight = trial_args.get("rloss_weight", 0) if agent_model and agent_model.startswith("set") else 0
    crf_weight = trial_args.get("crf_weight", 0)
    if "pretrain" in os.path.basename(model_path):
        crf_weight = 0

    losses = run_epoch(
        model,
        loader,
        device,
        macro_weight=trial_args.get("macro_weight", 0),
        rloss_weight=rloss_weight,
        crf_weight=crf_weight,
        use_emit_loss=trial_args.get("emit_loss", True),
        use_src_loss=trial_args.get("src_loss", False),
        use_dst_loss=trial_args.get("dst_loss", False),
        train=False,
    )

    total_loss = sum([value for key, value in losses.items() if key.endswith("loss")])
    print(utils.loss_str(losses))
