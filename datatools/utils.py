import os
import re
import sys
from typing import Callable, List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from tqdm import tqdm

from datatools import config


def detect_keepers(period_tracking: pd.DataFrame, left_first=False):
    home_x_cols = [c for c in period_tracking.columns if re.match(r"home_.*_x", c)]
    away_x_cols = [c for c in period_tracking.columns if re.match(r"away_.*_x", c)]

    home_gk = (period_tracking[home_x_cols].mean() - config.PITCH_X / 2).abs().idxmax()[:-2]
    away_gk = (period_tracking[away_x_cols].mean() - config.PITCH_Y / 2).abs().idxmax()[:-2]

    home_gk_x = period_tracking[f"{home_gk}_x"].mean()
    away_gk_x = period_tracking[f"{away_gk}_x"].mean()

    if left_first:
        return (home_gk, away_gk) if home_gk_x < away_gk_x else (away_gk, home_gk)
    else:
        return (home_gk, away_gk)


def find_active_players(traces: pd.DataFrame, frame: int = None, team: str = None, include_goals=False) -> dict:
    if pd.isna(frame):
        snapshot = traces.dropna(how="all", axis=1).copy()
    else:
        snapshot = traces.loc[frame:frame].dropna(how="all", axis=1).copy()

    if include_goals:
        home_players = [c[:-2] for c in snapshot.columns if re.match(r"home_.*_x", c)]
        away_players = [c[:-2] for c in snapshot.columns if re.match(r"away_.*_x", c)]
    else:
        home_players = [c[:-2] for c in snapshot.columns if re.match(r"home_\d+_x", c)]
        away_players = [c[:-2] for c in snapshot.columns if re.match(r"away_\d+_x", c)]

    if not pd.isna(frame):
        team = team or traces.at[frame, "ball_owning_home_away"]
    else:
        team = team or "home"

    if team == "home":
        players = [home_players, away_players]
    else:
        players = [away_players, home_players]

    return players


def seconds_to_timestamp(total_seconds: float) -> str:
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{int(seconds):02d}{f'{seconds % 1:.2f}'[1:]}"


def timestamp_to_seconds(timestamp: str) -> float:
    minutes, seconds = timestamp.split(":")
    return float(minutes) * 60 + float(seconds)


def series_to_seconds(series: pd.Series) -> pd.Series:
    def _to_sec(val):
        if isinstance(val, (int, float, np.floating)) and not np.isnan(val):
            return float(val)
        return timestamp_to_seconds(val)

    return series.apply(_to_sec)


def compute_abs_seconds(frames: pd.DataFrame, half_seconds: float = 45 * 60) -> pd.Series:
    if "period_id" in frames.columns and "timestamp" in frames.columns:
        period = frames["period_id"].fillna(1).astype(float)
        ts = pd.to_numeric(frames["timestamp"], errors="coerce")
        return ts + (period - 1) * half_seconds
    if "seconds" in frames.columns:
        return pd.to_numeric(frames["seconds"], errors="coerce")
    return pd.Series(np.nan, index=frames.index)


def assign_zone(df: pd.DataFrame, grid: tuple[int, int] = (6, 4)) -> pd.DataFrame:
    gx, gy = grid
    df = df.copy()
    df["cell_x"] = np.floor(df["ball_x"] / (config.PITCH_X / gx)).astype(int)
    df["cell_y"] = np.floor(df["ball_y"] / (config.PITCH_Y / gy)).astype(int)
    df["cell_x"] = df["cell_x"].clip(0, gx - 1)
    df["cell_y"] = df["cell_y"].clip(0, gy - 1)
    return df


def label_frames_and_episodes(
    tracking: pd.DataFrame, events: pd.DataFrame = None, fps: float = 25.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tracking = tracking.copy().sort_values(["period_id", "timestamp"], ignore_index=True)

    if "frame_id" not in tracking.columns:
        tracking["frame_id"] = (tracking["timestamp"] * fps).round().astype(int)
        n_prev_frames = 0

        for i in tracking["period_id"].unique():
            period_tracking = tracking[tracking["period_id"] == i]
            tracking.loc[period_tracking.index, "frame_id"] += n_prev_frames
            n_prev_frames += len(period_tracking)

    if "episode_id" not in tracking.columns:
        tracking["episode_id"] = 0
        n_prev_episodes = 0

        for i in tracking["period_id"].unique():
            period_tracking = tracking[tracking["period_id"] == i].copy()
            alive_tracking = period_tracking[period_tracking["ball_state"] == "alive"].copy()

            frame_diffs = np.diff(alive_tracking["frame_id"].values, prepend=-5)
            period_episode_ids = (frame_diffs >= 5).astype(int).cumsum() + n_prev_episodes
            tracking.loc[alive_tracking.index, "episode_id"] = period_episode_ids

            n_prev_episodes = period_episode_ids.max()

    tracking = tracking.set_index("frame_id")

    if events is not None and "episode_id" not in events.columns:
        events = events.copy()
        events["episode_id"] = 0

        for i in events.index:
            frame_id = events.at[i, "frame_id"]
            if not pd.isna(frame_id):
                events.at[i, "episode_id"] = tracking.at[frame_id, "episode_id"]

    return tracking.reset_index(), events


def label_phases(tracking: pd.DataFrame) -> pd.DataFrame:
    phases = summarize_phases(tracking)

    tracking = tracking.copy()
    tracking["phase_id"] = 0

    for i in phases.index:
        start_frame = phases.at[i, "start_frame_id"]
        end_frame = phases.at[i, "end_frame_id"]
        phase_mask = tracking["frame_id"].between(start_frame, end_frame)
        tracking.loc[phase_mask, "phase_id"] = i

    return tracking


def summarize_playing_times(tracking: pd.DataFrame) -> pd.DataFrame:
    if "frame_id" in tracking.columns:
        tracking = tracking.copy().set_index("frame_id")

    players = [c[:-2] for c in tracking.columns if c[:4] in ["home", "away"] and c.endswith("_x")]
    play_records = dict()

    for p in players:
        player_x = tracking[f"{p}_x"].dropna()
        if not player_x.empty:
            play_records[p] = {"in_frame_id": player_x.index[0], "out_frame_id": player_x.index[-1]}

    return pd.DataFrame(play_records).T


def summarize_phases(tracking: pd.DataFrame, keepers: List[str] = None) -> pd.DataFrame:
    if "frame_id" in tracking:
        tracking = tracking.copy().set_index("frame_id")

    keepers = [] if keepers is None else list(keepers)

    play_records = summarize_playing_times(tracking)
    player_in_frames = play_records["in_frame_id"].unique().tolist()
    player_out_frames = (play_records["out_frame_id"].unique() + 1).tolist()
    period_start_frames = tracking.reset_index().groupby("period_id")["frame_id"].first().values.tolist()
    phase_changes = np.sort(np.unique(player_in_frames + player_out_frames + period_start_frames))

    phases = []

    for i, start_frame in enumerate(phase_changes[:-1]):
        end_frame = phase_changes[i + 1] - 1
        alive_tracking = tracking[tracking["ball_state"] == "alive"].loc[start_frame:end_frame].copy()
        if len(alive_tracking) < 100:
            continue

        active_players = find_active_players(alive_tracking)
        home_keepers = [p for p in keepers if p in active_players[0]]
        away_keepers = [p for p in keepers if p in active_players[1]]
        home_x_cols = [f"{p}_x" for p in active_players[0]]
        away_x_cols = [f"{p}_x" for p in active_players[1]]
        home_keeper = home_keepers[0] if home_keepers else alive_tracking[home_x_cols].mean().idxmin()[:-2]
        away_keeper = away_keepers[0] if away_keepers else alive_tracking[away_x_cols].mean().idxmax()[:-2]

        phase_dict = {
            "period_id": alive_tracking["period_id"].iloc[0],
            "start_frame_id": start_frame,
            "end_frame_id": end_frame,
            "active_players": active_players[0] + active_players[1],
            "active_keepers": [home_keeper, away_keeper],
        }
        phases.append(phase_dict)

    phases = pd.DataFrame(phases)
    phases.index.name = "phase"
    phases.index += 1

    return phases


def calculate_running_features(tracking: pd.DataFrame, fps=25) -> pd.DataFrame:
    from scipy.signal import savgol_filter

    tracking = tracking.copy()

    if "episode_id" not in tracking.columns:
        tracking = label_frames_and_episodes(tracking)

    if "phase_id" not in tracking.columns:
        tracking = label_phases(tracking)

    home_players = [c[:-2] for c in tracking.dropna(axis=1, how="all").columns if re.match(r"home_.*_x", c)]
    away_players = [c[:-2] for c in tracking.dropna(axis=1, how="all").columns if re.match(r"away_.*_x", c)]
    objects = home_players + away_players + ["ball"]
    physical_features = ["x", "y", "vx", "vy", "speed", "accel"]

    state_cols = ["frame_id", "period_id", "timestamp", "phase_id", "episode_id", "ball_state", "ball_owning_team_id"]
    feature_cols = [f"{p}_{f}" for p in objects for f in physical_features]
    if "ball_z" in tracking.columns:
        feature_cols.append("ball_z")

    if "player_id" in tracking.columns:
        state_cols.append("player_id")

    for p in tqdm(objects, desc="Calculating running features per player"):
        new_cols = [f"{p}_{x}" for x in physical_features[2:]]
        new_features = pd.DataFrame(np.nan, index=tracking.index, columns=new_cols)

        # Drop pre-existing columns to avoid duplicate column names during concat/assign
        tracking = tracking.drop(columns=[c for c in new_cols if c in tracking.columns], errors="ignore")
        tracking = pd.concat([tracking, new_features], axis=1)

        for i in tracking["period_id"].unique():
            x: pd.Series = tracking.loc[tracking["period_id"] == i, f"{p}_x"].dropna()
            y: pd.Series = tracking.loc[tracking["period_id"] == i, f"{p}_y"].dropna()
            if x.empty:
                continue

            vx = savgol_filter(np.diff(x.values) * fps, window_length=15, polyorder=2)
            vy = savgol_filter(np.diff(y.values) * fps, window_length=15, polyorder=2)
            ax = savgol_filter(np.diff(vx) * fps, window_length=9, polyorder=2)
            ay = savgol_filter(np.diff(vy) * fps, window_length=9, polyorder=2)

            tracking.loc[x.index[1:], f"{p}_vx"] = vx
            tracking.loc[x.index[1:], f"{p}_vy"] = vy
            tracking.loc[x.index[1:], f"{p}_speed"] = np.sqrt(vx**2 + vy**2)
            tracking.loc[x.index[1:-1], f"{p}_accel"] = np.sqrt(ax**2 + ay**2)

            tracking.at[x.index[0], f"{p}_vx"] = tracking.at[x.index[1], f"{p}_vx"]
            tracking.at[x.index[0], f"{p}_vy"] = tracking.at[x.index[1], f"{p}_vy"]
            tracking.at[x.index[0], f"{p}_speed"] = tracking.at[x.index[1], f"{p}_speed"]
            tracking.loc[[x.index[0], x.index[-1]], f"{p}_accel"] = 0

    return tracking[state_cols + feature_cols].copy()


def linear_scoring_func(min_input: float, max_input: float, increasing=False) -> Callable:
    assert min_input < max_input

    def func(x: float) -> float:
        if increasing:
            return (x - min_input) / (max_input - min_input)
        else:
            return 1 - (x - min_input) / (max_input - min_input)

    return lambda x: np.maximum(0, np.minimum(1, func(x)))


# Scoring functions for ELASTIC
player_dist_func = linear_scoring_func(1, 3, increasing=False)
ball_accel_func = linear_scoring_func(0, 20, increasing=True)
kick_dist_func = linear_scoring_func(0, 5, increasing=True)


def score_nw(features: pd.Series | pd.DataFrame, player_id: str, kick_dist_col: str) -> float | np.ndarray:
    if isinstance(features, pd.DataFrame):
        scores = np.zeros(len(features), dtype=float)
        if scores.size == 0:
            return scores

        mask = features["player_id"] == player_id
        if not mask.any():
            return scores

        ball_accel_score = 100 / 3 * ball_accel_func(features.loc[mask, "ball_accel"].to_numpy())
        player_dist_score = 100 / 3 * player_dist_func(features.loc[mask, "player_dist"].to_numpy())
        kick_dist_score = 100 / 3 * kick_dist_func(features.loc[mask, kick_dist_col].to_numpy())
        scores[mask.to_numpy()] = ball_accel_score + player_dist_score + kick_dist_score
        return scores

    elif features["player_id"] == player_id:  # isinstance(features, pd.Series)
        ball_accel_score = 100 / 3 * ball_accel_func(features["ball_accel"])
        player_dist_score = 100 / 3 * player_dist_func(features["player_dist"])
        kick_dist_score = 100 / 3 * kick_dist_func(features[kick_dist_col])
        return ball_accel_score + player_dist_score + kick_dist_score

    else:
        return 0.0


def score_nw_duel(features: pd.Series | pd.DataFrame, player_id: str, kick_dist_col: str) -> float | np.ndarray:
    if isinstance(features, pd.DataFrame):
        scores = np.zeros(len(features), dtype=float)
        if scores.size == 0:
            return scores

        mask = features["player_id"] == player_id
        if not mask.any():
            return scores

        ball_accel_score = 25 * ball_accel_func(features.loc[mask, "ball_accel"].to_numpy())
        player_dist_score = 25 * player_dist_func(features.loc[mask, "player_dist"].to_numpy())
        oppo_dist_score = 25 * player_dist_func(features.loc[mask, "oppo_dist"].to_numpy())
        kick_dist_score = 25 * kick_dist_func(features.loc[mask, kick_dist_col].to_numpy())
        scores[mask.to_numpy()] = ball_accel_score + player_dist_score + oppo_dist_score + kick_dist_score
        return scores

    elif features["player_id"] == player_id:  # isinstance(features, pd.Series)
        ball_accel_score = 25 * ball_accel_func(features["ball_accel"])
        player_dist_score = 25 * player_dist_func(features["player_dist"])
        oppo_dist_score = 25 * player_dist_func(features["oppo_dist"])
        kick_dist_score = 25 * kick_dist_func(features[kick_dist_col])
        return ball_accel_score + player_dist_score + oppo_dist_score + kick_dist_score

    else:
        return 0.0


def plot_player_probs(
    prob_df: pd.DataFrame,
    max_rows: int = 1000,
    valid_prob: float = 0.01,
    shade_prob: float = 0.5,
    shade_alpha: float = 0.25,
    ax=None,
    show_legend: bool = False,
):
    """Plot per-player probability time series for the first `max_rows` rows.

    - Use `timestamp` column for x if present, otherwise `frame_id`, otherwise index.
    - Plot only columns starting with `home_` or `away_` whose max prob >= `valid_prob`.
    - Fix y-axis to [0, 1].
    """
    import matplotlib.pyplot as plt

    plot_df = prob_df.head(max_rows).copy()

    if "timestamp" in plot_df.columns:
        x = plot_df["timestamp"].to_numpy()
        xlabel = "timestamp"
    elif "frame_id" in plot_df.columns:
        x = plot_df["frame_id"].to_numpy()
        xlabel = "frame_id"
    else:
        x = plot_df.index.to_numpy()
        xlabel = "index"

    player_cols = [c for c in plot_df.columns if c.startswith("home_") or c.startswith("away_")]
    plot_df = plot_df[player_cols].dropna(axis=1, how="all")
    if plot_df.empty:
        raise ValueError("prob_df has no non-empty player columns to plot.")

    max_probs = plot_df.max(axis=0, skipna=True)
    plot_cols = [c for c in plot_df.columns if max_probs.get(c, 0) >= valid_prob]
    if not plot_cols:
        raise ValueError(f"No player columns with max probability >= {valid_prob}.")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    line_colors = {}
    for col in plot_cols:
        line = ax.plot(x, plot_df[col].to_numpy(), linewidth=1, label=col)[0]
        line_colors[col] = line.get_color()

    # Shade spans where argmax prob >= shade_prob
    if shade_prob is not None:
        max_probs = plot_df[plot_cols].max(axis=1, skipna=True)
        valid_mask = max_probs >= shade_prob
        argmax_cols = plot_df[plot_cols].fillna(0.0).idxmax(axis=1)

        if valid_mask.any():
            # Find contiguous segments of same argmax where valid_mask is True
            start_idx = None
            prev_col = None
            for i in range(len(plot_df)):
                if valid_mask.iat[i]:
                    curr_col = argmax_cols.iat[i]
                    if start_idx is None:
                        start_idx = i
                        prev_col = curr_col
                    elif curr_col != prev_col:
                        ax.axvspan(
                            x[start_idx],
                            x[i - 1],
                            color=line_colors.get(prev_col, "gray"),
                            alpha=shade_alpha,
                            linewidth=0,
                        )
                        start_idx = i
                        prev_col = curr_col
                else:
                    if start_idx is not None:
                        ax.axvspan(
                            x[start_idx],
                            x[i - 1],
                            color=line_colors.get(prev_col, "gray"),
                            alpha=shade_alpha,
                            linewidth=0,
                        )
                        start_idx = None
                        prev_col = None

            if start_idx is not None:
                ax.axvspan(
                    x[start_idx],
                    x[len(plot_df) - 1],
                    color=line_colors.get(prev_col, "gray"),
                    alpha=shade_alpha,
                    linewidth=0,
                )

    ax.set_ylim(0, 1)
    if len(x) > 0:
        ax.set_xlim(x[0], x[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(ncol=2, fontsize=10, frameon=False)

    return ax
