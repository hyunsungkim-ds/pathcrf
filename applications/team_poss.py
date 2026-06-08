"""
Team possession timeline for the tutorial's downstream analysis.

Infers per-frame team possession from events and
plots the true vs. predicted home-possession share over time.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datatools import utils


def compute_team_poss(events: pd.DataFrame, tracking: pd.DataFrame, mode: str = "contested") -> pd.DataFrame:
    """
    Assign per-frame team possession using event episodes and two rules:
    - mode="contested": gaps, episode boundaries, and team-change kicks are labeled "contested"
    - mode="split": gaps/edges assigned to event teams; team-change kicks are split half/half
    """
    events = events.copy()
    events["seconds"] = events["timestamp"].apply(utils.timestamp_to_seconds)
    events["team"] = events["player_id"].map(lambda x: x.split("_")[0])

    tracking = tracking.copy()
    tracking["team"] = pd.Series(np.nan, dtype=str)
    tracking["seconds"] = tracking["timestamp"]

    for episode_id, ep_tracking in tracking.groupby("episode_id"):
        ep_tracking = ep_tracking.sort_values("seconds", kind="stable")
        ep_events = events[events["episode_id"] == episode_id].sort_values("seconds", kind="stable")

        if episode_id == 0 or ep_events.empty or ep_tracking.empty:
            continue

        t_start = float(ep_tracking["seconds"].min())
        t_end = float(ep_tracking["seconds"].max())
        times = ep_events["seconds"].to_numpy(dtype=float)
        teams = ep_events["team"].to_numpy(dtype=object)

        segments = []
        if mode == "contested":
            segments.append((t_start, times[0], "contested"))
            for i in range(len(times) - 1):
                label = teams[i] if teams[i] == teams[i + 1] else "contested"
                segments.append((times[i], times[i + 1], label))
            segments.append((times[-1], t_end, "contested"))
        else:
            segments.append((t_start, times[0], teams[0]))
            for i in range(len(times) - 1):
                if teams[i] == teams[i + 1]:
                    segments.append((times[i], times[i + 1], teams[i]))
                else:
                    mid = (times[i] + times[i + 1]) / 2.0
                    segments.append((times[i], mid, teams[i]))
                    segments.append((mid, times[i + 1], teams[i + 1]))
            segments.append((times[-1], t_end, teams[-1]))

        seg_ends = np.array([seg[1] for seg in segments], dtype=float)
        seg_labels = np.array([seg[2] for seg in segments], dtype=object)

        secs = ep_tracking["seconds"].to_numpy(dtype=float)
        idx = np.searchsorted(seg_ends, secs, side="right")
        idx = np.clip(idx, 0, len(seg_labels) - 1)
        tracking.loc[ep_tracking.index, "team"] = seg_labels[idx]

    return tracking["team"].copy()


def compute_true_pred_poss(
    true_events: pd.DataFrame,
    pred_events: pd.DataFrame,
    tracking: pd.DataFrame,
    mode: str = "split",
) -> pd.DataFrame:
    """Per-frame true/pred team possession over in-play frames."""
    frame_cols = ["frame_id", "period_id", "timestamp", "episode_id"]
    frames = tracking.loc[tracking["episode_id"] > 0, frame_cols].copy()
    frames["true_team"] = compute_team_poss(true_events, tracking, mode=mode)
    frames["pred_team"] = compute_team_poss(pred_events, tracking, mode=mode)
    frames = frames.dropna(subset="true_team").copy()
    return frames


def home_poss_by_time(frames: pd.DataFrame, bin_minutes: int = 1) -> pd.DataFrame:
    """Absolute-minute home share (True/Pred) with %p error."""
    if frames.empty:
        return pd.DataFrame(columns=["minute", "true_home_poss", "pred_home_poss", "error_pp", "valid_frames"])

    assert "true_team" in frames.columns and "pred_team" in frames.columns

    frames = frames.copy()
    frames["abs_seconds"] = utils.compute_abs_seconds(frames)
    frames["minute"] = ((frames["abs_seconds"] // (60 * bin_minutes)) * bin_minutes).astype(int)

    rows = []
    for minute, minute_df in frames.groupby("minute"):
        total = len(minute_df)
        true_share = (minute_df["true_team"] == "home").sum() / total
        pred_share = (minute_df["pred_team"] == "home").sum() / total
        rows.append(
            {
                "minute": int(minute),
                "true_home_poss": float(true_share),
                "pred_home_poss": float(pred_share),
                "error_pp": float(abs(pred_share - true_share) * 100.0),
            }
        )
    result = pd.DataFrame(rows).sort_values("minute")
    counts = frames.groupby("minute").size().rename("valid_frames").reset_index()
    result = result.merge(counts, on="minute", how="left")

    return result


def plot_true_pred_poss(
    values_true,
    values_pred,
    title: Optional[str] = None,
    x: Optional[list[int]] = None,
):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, values_true * 100, marker="o", color="tab:green", linewidth=2.2, label="True")
    ax.plot(x, values_pred * 100, marker="o", color="darkorange", linewidth=2.2, label="Pred")
    ax.set_xlabel("Time (min)", fontdict={"size": 18})
    ax.set_ylabel("Home Possession (%)", fontdict={"size": 18})
    ax.set_ylim(0, 100)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="both", color="gray", linestyle="--", linewidth=1)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16)
    plt.show()
    return fig, ax
