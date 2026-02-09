import os
import sys
from pprint import pprint
from typing import List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import SoccerWindowDataset, SoccerWindowTensors
from models.edge_embed_crf import EdgeEmbedCRF
from models.utils import build_edge_compression_maps


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


def scan_forbidden(dataset: SoccerWindowTensors, max_report: int = 20) -> List[dict]:
    team_size = dataset.team_size
    n_nodes = team_size * 2 + 4
    _, orig2comp, comp_src, comp_dst = build_edge_compression_maps(team_size * 2, 4)
    orig2comp = orig2comp.cpu().numpy()

    # CRF transition rules
    crf = EdgeEmbedCRF(comp_src, comp_dst, edge_embed_dim=16, team_size=team_size)
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


def build_edge_df(dataset: SoccerWindowDataset, sample_freq: int = 5) -> pd.DataFrame:
    """
    Build a DataFrame of possession edges per window sampled by window_stride.
    Columns: file, episode_id, frame_id, poss_edge (tuple of src, dst).
    """
    rows = []
    for s in tqdm(dataset.samples, desc="Building edge sequence DF"):
        poss = s["poss"]
        if poss is None:
            continue
        meta = s["meta"]
        start_frame_id = meta["start_frame_id"]
        node_order = meta["node_order"]
        for t in range(poss.shape[0]):
            src = int(poss[t, 0])
            dst = int(poss[t, 1])
            rows.append(
                {
                    "file": os.path.basename(meta["file"]),
                    "episode_id": meta["episode_id"],
                    "frame_id": int(start_frame_id + t * sample_freq),
                    "poss_edge": (node_order[src], node_order[dst]),
                }
            )

    return pd.DataFrame(rows).drop_duplicates()


if __name__ == "__main__":
    TRACKING_DIR = "data/sportec/tracking_processed"
    data_paths = np.sort([f"{TRACKING_DIR}/{f}" for f in os.listdir(TRACKING_DIR)])[:5]
    data_paths.sort()

    dataset = SoccerWindowTensors(
        data_paths,
        node_in_dim=8,
        team_size=11,
        fps=25.0,
        sample_freq=5,
        window_seconds=10.0,
        window_stride=50,
        self_loops=True,
        flip_pitch=False,
        verbose=True,
    )

    violations = scan_forbidden(dataset, max_report=40)
    file_frames = []
    print(f"found {len(violations)} violations")
    for v in violations:
        file = v["meta"]["file"]
        frame = v["frame_id"]
        if (file, frame) not in file_frames:
            print([v["meta"], v["frame_id"], v["prev_edge"], v["curr_edge"]])
            file_frames.append((file, frame))
