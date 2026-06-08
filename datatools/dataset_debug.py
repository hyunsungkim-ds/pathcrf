import os
import sys
from pprint import pprint
from typing import List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import SoccerWindowTensors
from models.dynamic_sparse_crf import DynamicSparseCRF
from models.utils import build_edge_compression_maps


def scan_invalid_poss(dataset: SoccerWindowTensors, max_report: int = 20) -> List[dict]:
    """Detect samples where poss_seq[:, 0] or poss_seq[:, 1] is outside [0, N)."""
    reports = []
    for idx in tqdm(range(len(dataset)), desc="Scanning out-of-range poss"):
        s = dataset.samples[idx]
        poss = s["poss"]
        input_seq = s["input"]
        if poss is None:
            continue

        n_nodes = input_seq.shape[1]
        poss_arr = np.asarray(poss, dtype=np.float64)
        src = poss_arr[:, 0]
        dst = poss_arr[:, 1]

        bad_mask = np.isnan(src) | np.isnan(dst) | (src < 0) | (src >= n_nodes) | (dst < 0) | (dst >= n_nodes)
        if bad_mask.any():
            bad_indices = np.where(bad_mask)[0]
            meta = s["meta"]
            reports.append(
                {
                    "sample_idx": idx,
                    "n_nodes": n_nodes,
                    "bad_timesteps": int(bad_mask.sum()),
                    "bad_values": [
                        (int(t), float(src[t]), float(dst[t]), meta["start_frame_id"] + t * dataset.sample_freq)
                        for t in bad_indices[:5]
                    ],
                    "meta": {
                        "file": os.path.basename(meta["file"]),
                        "phase_id": meta["phase_id"],
                        "episode_id": meta["episode_id"],
                        "node_order": meta["node_order"],
                    },
                }
            )
            if len(reports) >= max_report:
                break
    return reports


def build_padded_order(meta: pd.Series, team_size: int) -> List[str]:
    node_order = meta["node_order"]
    left_team = meta["left_team"]
    right_team = meta["right_team"]
    left = [p for p in node_order if p.startswith(left_team)]
    right = [p for p in node_order if p.startswith(right_team)]
    outside = [p for p in node_order if p.startswith("out_")]
    # pad to fixed size
    left = left + [f"pad_L{i}" for i in range(team_size - len(left))]
    right = right + [f"pad_R{i}" for i in range(team_size - len(right))]
    outside = outside + [f"pad_O{i}" for i in range(4 - len(outside))]
    return left + right + outside


def scan_forbidden_trans(dataset: SoccerWindowTensors, max_report: int = 20) -> List[dict]:
    team_size = dataset.team_size
    n_nodes = team_size * 2 + 4
    _, orig2comp, comp_src, comp_dst = build_edge_compression_maps(team_size * 2, 4)
    orig2comp = orig2comp.cpu().numpy()

    crf = DynamicSparseCRF(comp_src, comp_dst, edge_embed_dim=16, team_size=team_size)
    inc_idx = crf.inc_idx
    inc_mask = crf.inc_mask

    reports = []
    for idx in tqdm(range(len(dataset)), desc="Scanning forbidden edge transitions"):
        s = dataset.samples[idx]
        poss = s["poss"]
        if poss is None:
            continue
        poss = np.asarray(poss, dtype=np.int64)
        T = poss.shape[0]

        src = poss[:, 0]
        dst = poss[:, 1]
        edge_ids = src * n_nodes + dst
        comp_ids = orig2comp[edge_ids]

        meta = s["meta"]
        padded_order = build_padded_order(meta, team_size)
        start_frame = meta["start_frame_id"]
        sample_freq = dataset.sample_freq

        for t in range(1, T):
            prev_comp = comp_ids[t - 1]
            curr_comp = comp_ids[t]

            prev_src, prev_dst = int(src[t - 1]), int(dst[t - 1])
            curr_src, curr_dst = int(src[t]), int(dst[t])

            # invalid edge itself (outside -> non-self etc.)
            if prev_comp < 0 or curr_comp < 0:
                reports.append(
                    {
                        "sample_idx": idx,
                        "t": t,
                        "frame_id": start_frame + t * sample_freq,
                        "prev_edge": f"{padded_order[prev_src]}->{padded_order[prev_dst]}",
                        "curr_edge": f"{padded_order[curr_src]}->{padded_order[curr_dst]}",
                        "meta": {
                            "file": os.path.basename(meta["file"]),
                            "phase_id": meta["phase_id"],
                            "episode_id": meta["episode_id"],
                            "start_frame_id": meta["start_frame_id"],
                        },
                    }
                )
                if len(reports) >= max_report:
                    return reports
                continue

            # transition validity check via inc_idx/inc_mask
            allowed = ((inc_idx[curr_comp] == prev_comp) & inc_mask[curr_comp]).any().item()
            if not allowed:
                reports.append(
                    {
                        "sample_idx": idx,
                        "t": t,
                        "frame_id": start_frame + t * sample_freq,
                        "prev_edge": f"{padded_order[prev_src]}->{padded_order[prev_dst]}",
                        "curr_edge": f"{padded_order[curr_src]}->{padded_order[curr_dst]}",
                        "meta": {
                            "file": os.path.basename(meta["file"]),
                            "phase_id": meta["phase_id"],
                            "episode_id": meta["episode_id"],
                            "start_frame_id": meta["start_frame_id"],
                        },
                    }
                )
                if len(reports) >= max_report:
                    return reports

    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="kleague", choices=["kleague", "sportec"])
    parser.add_argument("--n_matches", type=int, default=3)
    parser.add_argument("--scan", type=str, default="features", choices=["features", "labels", "trans"])
    args = parser.parse_args()

    SOURCE_CFG = {
        "sportec": {"dir": "data/sportec/tracking_processed", "fps": 25.0, "sample_freq": 5},
        "kleague": {"dir": "data/kleague/tracking_processed", "fps": 10.0, "sample_freq": 2},
    }
    cfg = SOURCE_CFG[args.source]

    data_paths = sorted(f"{cfg['dir']}/{f}" for f in os.listdir(cfg["dir"]) if f.endswith(".parquet"))[: args.n_matches]

    dataset = SoccerWindowTensors(
        data_paths,
        node_in_dim=8,
        team_size=11,
        fps=cfg["fps"],
        sample_freq=cfg["sample_freq"],
        window_seconds=10.0,
        window_stride=50,
        self_loops=True,
        flip_pitch=False,
        verbose=True,
    )

    if args.scan == "features":
        count = 0
        for idx, s in enumerate(tqdm(dataset.samples, desc="Scanning NaN in input features")):
            input_arr = s["input"]
            if np.isnan(input_arr).any():
                nan_count = int(np.isnan(input_arr).sum())
                T, N, F = input_arr.shape
                # Find which nodes have NaN
                nan_nodes = set()
                for n in range(N):
                    if np.isnan(input_arr[:, n, :]).any():
                        nan_nodes.add(n)
                meta = s["meta"]
                node_order = meta["node_order"]
                nan_names = [node_order[n] if n < len(node_order) else f"pad_{n}" for n in sorted(nan_nodes)]
                print(
                    f"  sample {idx}: {nan_count} NaN values, file={os.path.basename(meta['file'])}, "
                    f"phase={meta['phase_id']}, episode={meta['episode_id']}, nan_nodes={nan_names}"
                )
                count += 1
        print(f"\nFound {count} samples with NaN in input features")

    if args.scan == "labels":
        reports = scan_invalid_poss(dataset, max_report=0)
        print(f"\nFound {len(reports)} samples with out-of-range poss labels")
        for r in reports:
            print(f"\n  sample {r['sample_idx']}: n_nodes={r['n_nodes']}, bad_timesteps={r['bad_timesteps']}")
            print(f"    file={r['meta']['file']}, phase={r['meta']['phase_id']}, episode={r['meta']['episode_id']}")
            print(f"    node_order={r['meta']['node_order']}")
            for t, src, dst, frame_id in r["bad_values"]:
                print(f"    t={t}, frame_id={frame_id}: src={src}, dst={dst}")

    elif args.scan == "trans":
        violations = scan_forbidden_trans(dataset, max_report=20)
        file_frames = []
        print(f"\nFound {len(violations)} forbidden transitions")
        for v in violations:
            file = v["meta"]["file"]
            frame = v["frame_id"]
            if (file, frame) not in file_frames:
                print([v["meta"], v["frame_id"], v["prev_edge"], v["curr_edge"]])
                file_frames.append((file, frame))
