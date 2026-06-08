import argparse
import os
import re

import numpy as np
import pandas as pd
import torch

from datatools.evaluate_events import align_true_and_pred
from datatools.postprocess import detect_events
from inference import inference
from models.utils import build_model, load_trial_args, resolve_model_path

SOURCE_CFG = {"sportec": {"fps": 25.0}, "kleague": {"fps": 10.0}, "tracab": {"fps": 25.0}}
PERF_KEYS = [
    "macro_acc",
    "edge_acc",
    "src_acc",
    "dst_acc",
    "forbid_rate",
    "ev_prec",
    "ev_recall",
    "ev_f1",
]
TIME_KEYS = [
    "params",
    "epoch_time_mean",
    "epoch_time_std",
    "inference_fps",
    "match_latency",
    "ep_duration",
    "ep_latency",
]


def resolve_data_paths(trial_args, eval_source, file_start, file_count):
    """Determine which tracking parquet files to evaluate on."""
    src_dir = f"data/{eval_source}/tracking_processed"
    all_files = sorted(os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".parquet"))
    train_source = trial_args.get("sources", ["sportec"])[0]

    if trial_args.get("lomocv", False) and train_source == eval_source:
        fold_idx = trial_args["cv_fold_idx"]
        assert fold_idx < len(all_files), f"cv_fold_idx {fold_idx} >= num files {len(all_files)}"
        return [all_files[fold_idx]]
    else:
        end = file_start + file_count if file_count > 0 else len(all_files)
        selected = all_files[file_start:end]
        assert len(selected) > 0, f"No files selected: source={eval_source}, start={file_start}, count={file_count}"
        return selected


def has_first_half_red_card(tracking_path, team_size=11):
    """Check if a red card occurred in the first half (period_id == 1)."""
    tracking = pd.read_parquet(tracking_path)
    first_half = tracking[tracking["period_id"] == 1]
    for phase in first_half["phase_id"].unique():
        if phase == 0:
            continue
        pt = first_half[first_half["phase_id"] == phase].dropna(axis=1, how="all")
        n_home = len([c for c in pt.columns if c.startswith("home_") and c.endswith("_x") and "_pad_" not in c])
        n_away = len([c for c in pt.columns if c.startswith("away_") and c.endswith("_x") and "_pad_" not in c])
        if n_home != team_size or n_away != team_size:
            return True
    return False


def load_events(tracking_path, eval_source):
    """Load ground-truth event parquet corresponding to a tracking file."""
    match_id = os.path.splitext(os.path.basename(tracking_path))[0]
    event_path = f"data/{eval_source}/event_processed/{match_id}.parquet"
    if os.path.isfile(event_path):
        return pd.read_parquet(event_path)
    return None


def parse_training_stats(save_path: str) -> dict:
    """Parse training log for epoch times and args for param count."""
    result = {"params": np.nan, "epoch_time_mean": np.nan, "epoch_time_std": np.nan}

    # Read total_params from args.json
    trial_args = load_trial_args(save_path)
    result["params"] = trial_args.get("total_params", np.nan)

    # Parse epoch times from log.txt
    log_path = os.path.join(save_path, "log.txt")
    if not os.path.isfile(log_path):
        return result

    epoch_times = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"(?:Time|Epoch time):\s+([\d.]+)\s*s", line)
            if m:
                epoch_times.append(float(m.group(1)))

    if epoch_times:
        result["epoch_time_mean"] = np.mean(epoch_times)
        result["epoch_time_std"] = np.std(epoch_times)

    return result


def evaluate_single_trial(trial_id, eval_source, file_start, file_count, model_file, device):
    """Evaluate a single trial: run inference per match, compute edge + event + time metrics."""
    save_path = f"saved/{trial_id:03d}"
    trial_args = load_trial_args(save_path)

    np.random.seed(trial_args.get("seed", 128))
    torch.manual_seed(trial_args.get("seed", 128))

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = build_model(trial_args, device=device)
    model_path = resolve_model_path(save_path, model_file)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)

    data_paths = resolve_data_paths(trial_args, eval_source, file_start, file_count)

    src_cfg = SOURCE_CFG[eval_source]
    eval_fps = src_cfg["fps"]
    target_fps = trial_args.get("target_fps", 5.0)
    eval_sample_freq = int(eval_fps / target_fps)

    use_crf = trial_args.get("crf_weight", 0) > 0
    window_seconds = trial_args.get("window_seconds", None)
    ball_feature = trial_args.get("ball_feature", False)

    # Per-match inference and evaluation
    match_metrics = []
    all_ep_latencies = []
    all_ep_frame_counts = []
    total_inference_time = 0.0
    total_frame_count = 0

    for tracking_path in data_paths:
        match_id = os.path.splitext(os.path.basename(tracking_path))[0]
        tracking = pd.read_parquet(tracking_path)

        _, _, pred_poss, stats = inference(
            model,
            tracking,
            ball_feature=ball_feature,
            use_crf=use_crf,
            decode="indep",
            correct_episode_lasts=True,
            window_seconds=window_seconds,
            fps=eval_fps,
            sample_freq=eval_sample_freq,
            skip_red_phases=True,
        )

        metrics = {
            "match_id": match_id,
            "macro_acc": stats.get("macro_acc", np.nan),
            "edge_acc": stats.get("edge_acc", np.nan),
            "src_acc": stats.get("src_acc", np.nan),
            "dst_acc": stats.get("dst_acc", np.nan),
            "forbid_rate": stats.get("forbid_rate", np.nan),
        }

        # Event-level evaluation (if ground-truth events exist)
        events = load_events(tracking_path, eval_source)
        if events is not None:
            if "edge_src" in pred_poss.columns:
                valid_frames = pred_poss.index[pred_poss["edge_src"].notna()]
            else:
                valid_frames = pred_poss.index[pred_poss.notna().any(axis=1)]
            pred_poss_eval = pred_poss.loc[valid_frames]
            tracking_eval = tracking[tracking["frame_id"].isin(valid_frames)]
            events_eval = events[events["frame_id"].isin(valid_frames)]

            pred_events = detect_events(tracking_eval, pred_poss_eval)
            event_stats, _ = align_true_and_pred(events_eval, pred_events, threshold=1.0)
            total = event_stats.loc["total"]
            metrics["ev_prec"] = total["precision"]
            metrics["ev_recall"] = total["recall"]
            metrics["ev_f1"] = total["f1"]
        else:
            metrics["ev_prec"] = np.nan
            metrics["ev_recall"] = np.nan
            metrics["ev_f1"] = np.nan

        # Collect time-related metrics
        total_inference_time += stats.get("inference_time", 0)
        total_frame_count += stats.get("n_frames", 0)
        all_ep_latencies.extend(stats.get("ep_latencies", []))
        all_ep_frame_counts.extend(stats.get("ep_frame_counts", []))

        match_metrics.append(metrics)
        print(f"{match_id}: " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if k != "match_id") + "\n")

    # Average performance across matches
    avg = {}
    for key in PERF_KEYS:
        values = [m[key] for m in match_metrics if not np.isnan(m[key])]
        avg[key] = np.mean(values) if values else np.nan

    # Time-related metrics
    training_stats = parse_training_stats(save_path)
    avg["params"] = training_stats["params"]
    avg["epoch_time_mean"] = training_stats["epoch_time_mean"]
    avg["epoch_time_std"] = training_stats["epoch_time_std"]
    avg["match_latency"] = total_inference_time / len(data_paths) if data_paths else np.nan
    avg["inference_fps"] = total_frame_count / total_inference_time if total_inference_time > 0 else np.nan
    avg["ep_duration"] = np.mean(all_ep_frame_counts) / eval_fps if all_ep_frame_counts else np.nan
    avg["ep_latency"] = np.mean(all_ep_latencies) if all_ep_latencies else np.nan

    # Attach efficiency metrics to each match for print_summary
    efficiency = {k: avg[k] for k in TIME_KEYS if k in avg}
    for m in match_metrics:
        m.update(efficiency)

    print_summary(match_metrics)
    return avg


def print_summary(all_results):
    """Print mean ± std summary of results."""

    print("[Performance]")
    for key in PERF_KEYS:
        values = [r[key] for r in all_results if not np.isnan(r.get(key, np.nan))]
        if values:
            print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("\n[Efficiency]")
    for key in TIME_KEYS:
        values = [r[key] for r in all_results if not np.isnan(r.get(key, np.nan))]
        if not values:
            continue
        if key == "params":
            print(f"  params: {int(values[0]):,}")
        elif key == "epoch_time_mean":
            stds = [r["epoch_time_std"] for r in all_results if not np.isnan(r.get("epoch_time_std", np.nan))]
            print(f"  epoch_time: {np.mean(values):.2f} ± {np.mean(stds):.2f} s")
        elif key == "epoch_time_std":
            continue
        else:
            print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, nargs="+", required=True, help="start and end of a range if n = 2")
    parser.add_argument("--eval_source", type=str, default="sportec", choices=list(SOURCE_CFG.keys()))
    parser.add_argument("--file_start", type=int, default=6, help="Start index of files to evaluate")
    parser.add_argument("--file_count", type=int, default=1, help="Number of files to evaluate (0 means all)")
    parser.add_argument("--model_file", type=str, default="state_dict_best_acc.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if len(args.trials) == 2 and args.trials[1] > args.trials[0]:
        trial_ids = list(range(args.trials[0], args.trials[1] + 1))
    else:
        trial_ids = args.trials

    all_results = []
    for trial_id in trial_ids:
        print(f"\n========== Model {trial_id} ==========")

        # Skip LOMOCV folds where the test match has a first-half red card
        trial_args = load_trial_args(f"saved/{trial_id:03d}")
        if trial_args.get("lomocv", False) and trial_args.get("sources", ["sportec"])[0] == args.eval_source:
            data_paths = resolve_data_paths(trial_args, args.eval_source, args.file_start, args.file_count)
            if any(has_first_half_red_card(p, trial_args.get("team_size", 11)) for p in data_paths):
                print("  Skipped (first-half red card in test match)")
                continue

        result = evaluate_single_trial(
            trial_id=trial_id,
            eval_source=args.eval_source,
            file_start=args.file_start,
            file_count=args.file_count,
            model_file=args.model_file,
            device=args.device,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print("\n=========== Overall ===========")
        print_summary(all_results)
