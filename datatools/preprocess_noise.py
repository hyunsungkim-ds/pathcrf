import os
import re
import sys
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())


from datatools import config, utils  # noqa: E402

NOISE_SEED = 42

SCENARIOS = []  # ["bias", "outlier"]
LEVELS = ["low", "mid", "high"]

OCCLUSION_DIST = {"low": 1.0, "mid": 2.0, "high": 3.0}
BIAS_SIGMA = {"low": 0.5, "mid": 1.0, "high": 2.0}
OUTLIER_PROB = {"low": 0.001, "mid": 0.01, "high": 0.04}
OUTLIER_SIGMA = 5.0


def _player_ids(tracking: pd.DataFrame) -> List[str]:
    x_cols = [c for c in tracking.columns if re.match(r"(home|away)_\d+_x", c)]
    return sorted({c[:-2] for c in x_cols})


def _clip_player_xy(tracking: pd.DataFrame, player_ids: List[str]) -> pd.DataFrame:
    out = tracking.copy()
    for p in player_ids:
        x_col, y_col = f"{p}_x", f"{p}_y"
        if x_col in out.columns:
            out[x_col] = pd.to_numeric(out[x_col], errors="coerce").clip(0.0, config.PITCH_X)
        if y_col in out.columns:
            out[y_col] = pd.to_numeric(out[y_col], errors="coerce").clip(0.0, config.PITCH_Y)
    return out


def _noise_occlusion(tracking: pd.DataFrame, player_ids: List[str], level: str) -> pd.DataFrame:
    merge_dist = OCCLUSION_DIST[level]
    out = tracking.copy()
    if not player_ids:
        return out

    xy_cols = [f"{p}_x" for p in player_ids] + [f"{p}_y" for p in player_ids]
    player_count = len(player_ids)

    for idx in out.index:
        row = out.loc[idx, xy_cols].to_numpy(dtype=float)
        x = row[:player_count]
        y = row[player_count:]
        coords = np.stack([x, y], axis=1)
        valid = np.isfinite(coords).all(axis=1)
        if valid.sum() < 2:
            continue

        valid_ids = np.where(valid)[0]
        valid_coords = coords[valid_ids]
        dmat = np.linalg.norm(valid_coords[:, None, :] - valid_coords[None, :, :], axis=2)
        iu = np.triu_indices(len(valid_ids), k=1)
        close_pairs = np.flatnonzero(dmat[iu] <= merge_dist)
        if len(close_pairs) == 0:
            continue

        # Build connected components on "close-enough" player graph and merge
        # each component to one deterministic anchor location.
        n_valid = len(valid_ids)
        parent = np.arange(n_valid)
        rank = np.zeros(n_valid, dtype=np.int8)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        iu0, iu1 = iu
        for k in close_pairs:
            union(int(iu0[k]), int(iu1[k]))

        groups = {}
        for local_id in range(n_valid):
            root = find(local_id)
            groups.setdefault(root, []).append(local_id)

        for members in groups.values():
            if len(members) < 2:
                continue
            member_global_ids = [int(valid_ids[m]) for m in members]
            anchor_xy = coords[member_global_ids].mean(axis=0)
            for g in member_global_ids:
                coords[g] = anchor_xy

        out.loc[idx, [f"{p}_x" for p in player_ids]] = coords[:, 0]
        out.loc[idx, [f"{p}_y" for p in player_ids]] = coords[:, 1]

    return _clip_player_xy(out, player_ids)


def _noise_bias(
    tracking: pd.DataFrame,
    player_ids: List[str],
    level: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sigma = BIAS_SIGMA[level]
    out = tracking.copy()
    if not player_ids:
        return out

    if "episode_id" not in out.columns:
        out, _ = utils.label_frames_and_episodes(out)

    episode_ids = pd.Series(out["episode_id"]).dropna().astype(int).unique().tolist()
    for ep in tqdm(episode_ids, desc="Adding a bias per episode-player pair"):
        ep_mask = out["episode_id"] == ep
        if not ep_mask.any():
            continue
        for p in player_ids:
            x_col, y_col = f"{p}_x", f"{p}_y"
            if x_col not in out.columns or y_col not in out.columns:
                continue
            valid = ep_mask & out[x_col].notna() & out[y_col].notna()
            idx = out.index[valid]
            if len(idx) == 0:
                continue
            noise = rng.normal(0.0, sigma, size=(len(idx), 2))
            out.loc[idx, x_col] = out.loc[idx, x_col].to_numpy(dtype=float) + noise[:, 0]
            out.loc[idx, y_col] = out.loc[idx, y_col].to_numpy(dtype=float) + noise[:, 1]

    return _clip_player_xy(out, player_ids)


def _noise_outlier(
    tracking: pd.DataFrame,
    player_ids: List[str],
    level: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    prob = OUTLIER_PROB[level]
    out = tracking.copy()
    if not player_ids:
        return out

    for p in tqdm(player_ids, desc="Adding sporadic outliers to player trajectories"):
        x_col, y_col = f"{p}_x", f"{p}_y"
        if x_col not in out.columns or y_col not in out.columns:
            continue
        valid_mask = out[x_col].notna() & out[y_col].notna()
        valid_idx = out.index[valid_mask]
        if len(valid_idx) == 0:
            continue
        hit_mask = rng.random(len(valid_idx)) < prob
        hit_idx = valid_idx[hit_mask]
        if len(hit_idx) == 0:
            continue
        offsets = rng.normal(0.0, OUTLIER_SIGMA, size=(len(hit_idx), 2))
        out.loc[hit_idx, x_col] = out.loc[hit_idx, x_col].to_numpy(dtype=float) + offsets[:, 0]
        out.loc[hit_idx, y_col] = out.loc[hit_idx, y_col].to_numpy(dtype=float) + offsets[:, 1]

    return _clip_player_xy(out, player_ids)


def inject_player_noise(tracking: pd.DataFrame, scenario: str, level: str) -> pd.DataFrame:
    if scenario == "none":
        return tracking.copy()

    player_ids = _player_ids(tracking)
    if not player_ids:
        return tracking.copy()

    rng = np.random.default_rng(NOISE_SEED)
    if scenario == "occlusion":
        return _noise_occlusion(tracking, player_ids, level)
    if scenario == "bias":
        return _noise_bias(tracking, player_ids, level, rng)
    if scenario == "outlier":
        return _noise_outlier(tracking, player_ids, level, rng)
    raise ValueError(f"Unsupported scenario={scenario!r}. Use one of: none, {', '.join(SCENARIOS)}")


if __name__ == "__main__":
    IN_TRACKING_DIR = "data/sportec/tracking_processed"
    OUT_ROOT_DIR = "data/sportec/tracking_noisy"
    fps = 25.0

    match_files = sorted(f for f in os.listdir(IN_TRACKING_DIR) if f.endswith(".parquet"))

    # Unfiltered baseline: clean tracking with denoise=False (no noise injection)
    out_dir = os.path.join(OUT_ROOT_DIR, "unfiltered")
    os.makedirs(out_dir, exist_ok=True)
    print("\nScenario: unfiltered")

    for i, fname in enumerate(match_files):
        match_id = fname.split(".")[0]
        print(f"[{i + 1}/{len(match_files)}] {match_id}")
        tracking = pd.read_parquet(f"{IN_TRACKING_DIR}/{fname}")

        # Drop existing derived features so they get recomputed
        derived_cols = [c for c in tracking.columns if re.search(r"_(vx|vy|speed|accel)$", c)]
        tracking = tracking.drop(columns=derived_cols)

        tracking = utils.calculate_running_features(tracking, fps=fps, denoise=False)
        tracking.to_parquet(f"{out_dir}/{fname}")

    # Noisy variants: scenario × level grid
    for scenario in SCENARIOS:
        for level in LEVELS:
            out_dir = os.path.join(OUT_ROOT_DIR, f"{scenario}_{level}")
            os.makedirs(out_dir, exist_ok=True)
            print(f"\nScenario: {scenario}-{level}")

            for i, fname in enumerate(match_files):
                match_id = fname.split(".")[0]
                print(f"[{i + 1}/{len(match_files)}] {match_id}")
                tracking = pd.read_parquet(f"{IN_TRACKING_DIR}/{fname}")

                # Drop existing derived features so they get recomputed from noisy positions
                derived_cols = [c for c in tracking.columns if re.search(r"_(vx|vy|speed|accel)$", c)]
                tracking = tracking.drop(columns=derived_cols)

                tracking = inject_player_noise(tracking, scenario=scenario, level=level)
                tracking = utils.calculate_running_features(tracking, fps=fps, denoise=True)

                tracking.to_parquet(f"{out_dir}/{fname}")
