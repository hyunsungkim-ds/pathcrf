from __future__ import annotations

import numpy as np
import pandas as pd

from datatools import evaluate_events, event_postprocessing, utils


def infer_team_poss(events: pd.DataFrame, tracking: pd.DataFrame, mode: str = "contested") -> pd.DataFrame:
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


def _merge_frames(frames_gt: pd.DataFrame, frames_pred: pd.DataFrame) -> pd.DataFrame:
    gt_cols = [c for c in ["frame_id", "team", "seconds", "period_id", "timestamp"] if c in frames_gt.columns]
    pred_cols = [c for c in ["frame_id", "team", "seconds", "period_id", "timestamp"] if c in frames_pred.columns]
    gt = frames_gt[gt_cols].rename(columns={"team": "team_gt", "seconds": "seconds_gt"})
    pred = frames_pred[pred_cols].rename(columns={"team": "team_pred", "seconds": "seconds_pred"})
    merged = gt.merge(pred, on="frame_id", how="outer")
    if "seconds_gt" in merged.columns or "seconds_pred" in merged.columns:
        merged["seconds"] = merged.get("seconds_gt").combine_first(merged.get("seconds_pred"))
    if "period_id" not in merged.columns:
        for df in [gt, pred]:
            if "period_id" in df.columns:
                merged = merged.merge(df[["frame_id", "period_id"]], on="frame_id", how="left")
                break
    if "timestamp" not in merged.columns:
        for df in [gt, pred]:
            if "timestamp" in df.columns:
                merged = merged.merge(df[["frame_id", "timestamp"]], on="frame_id", how="left")
                break
    return merged


def _merge_frames_with_xy(frames_gt: pd.DataFrame, frames_pred: pd.DataFrame) -> pd.DataFrame:
    gt_cols = [c for c in ["frame_id", "team", "ball_x", "ball_y"] if c in frames_gt.columns]
    pred_cols = [c for c in ["frame_id", "team"] if c in frames_pred.columns]
    gt = frames_gt[gt_cols].rename(columns={"team": "team_gt"})
    pred = frames_pred[pred_cols].rename(columns={"team": "team_pred"})
    return gt.merge(pred, on="frame_id", how="outer")


def calc_team_possession_accuracy(frames_gt: pd.DataFrame, frames_pred: pd.DataFrame) -> float:
    """Frame-level team possession accuracy (missing side counts as incorrect)."""
    merged = _merge_frames(frames_gt, frames_pred)
    if merged.empty:
        return 0.0
    match = merged["team_gt"].notna() & merged["team_pred"].notna() & (merged["team_gt"] == merged["team_pred"])
    return float(match.mean())


def calc_possession_accuracy_by_minute(frames: pd.DataFrame) -> pd.DataFrame:
    """Minute-level possession accuracy on absolute match time (period-offset)."""
    if frames.empty:
        return pd.DataFrame(columns=["minute", "accuracy"])
    frames["abs_seconds"] = utils.compute_abs_seconds(frames)
    frames = frames[frames["abs_seconds"].notna()].copy()
    frames["minute"] = (frames["abs_seconds"] // 60).astype(int)
    frames["match"] = (
        frames["true_team"].notna() & frames["pred_team"].notna() & (frames["true_team"] == frames["pred_team"])
    )
    out = frames.groupby("minute")["match"].mean().reset_index()
    out.columns = ["minute", "accuracy"]
    return out


def home_poss_by_time(frames: pd.DataFrame, bin_minutes: int = 1) -> pd.DataFrame:
    """Absolute-minute home share (True/Pred) with %p error.

    Args:
        frames_gt: Ground truth frames
        frames_pred: Predicted frames
        bin_minutes: Bin size in minutes (e.g., 1 for per-minute, 5 for 5-minute bins)
    """
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


def zone_accuracy_grid(
    frames_gt: pd.DataFrame,
    frames_pred: pd.DataFrame,
    grid: tuple[int, int] = (6, 4),
) -> np.ndarray:
    """Zone-level possession accuracy grid."""
    merged = _merge_frames_with_xy(frames_gt, frames_pred)
    merged = utils.assign_zone(merged, grid=grid)
    merged = merged.dropna(subset=["cell_x", "cell_y"]).copy()
    merged["match"] = (
        merged["team_gt"].notna() & merged["team_pred"].notna() & (merged["team_gt"] == merged["team_pred"])
    )
    gx, gy = grid
    grid_vals = np.full((gy, gx), np.nan, dtype=float)
    grouped = merged.groupby(["cell_x", "cell_y"])["match"].mean()
    for (cx, cy), val in grouped.items():
        grid_vals[int(cy), int(cx)] = float(val)
    return grid_vals


def zone_accuracy_timeline(
    frames_gt: pd.DataFrame,
    frames_pred: pd.DataFrame,
    grid: tuple[int, int] = (6, 4),
) -> pd.DataFrame:
    """Zone-level possession accuracy timeline (absolute minutes)."""
    merged = _merge_frames_with_xy(frames_gt, frames_pred)
    merged = utils.assign_zone(merged, grid=grid)
    merged = merged.dropna(subset=["cell_x", "cell_y"]).copy()
    merged["abs_seconds"] = utils.compute_abs_seconds(merged)
    merged = merged[merged["abs_seconds"].notna()].copy()
    merged["minute"] = (merged["abs_seconds"] // 60).astype(int)
    merged["match"] = (
        merged["team_gt"].notna() & merged["team_pred"].notna() & (merged["team_gt"] == merged["team_pred"])
    )
    grouped = merged.groupby(["minute", "cell_x", "cell_y"])["match"].mean().reset_index()
    grouped = grouped.rename(columns={"match": "accuracy"})
    return grouped


def zone_home_share_error_grid(
    frames_gt: pd.DataFrame,
    frames_pred: pd.DataFrame,
    grid: tuple[int, int] = (6, 4),
) -> pd.DataFrame:
    """Zone-level home share %p error data."""
    merged = _merge_frames_with_xy(frames_gt, frames_pred)
    merged = utils.assign_zone(merged, grid=grid)
    merged = merged.dropna(subset=["cell_x", "cell_y"]).copy()

    rows = []
    for (cx, cy), mdf in merged.groupby(["cell_x", "cell_y"]):
        total = len(mdf)
        gt_home = (mdf["team_gt"] == "home").sum() / total
        pred_home = (mdf["team_pred"] == "home").sum() / total
        rows.append(
            {
                "cell_x": int(cx),
                "cell_y": int(cy),
                "true_home_poss": float(gt_home),
                "pred_home_poss": float(pred_home),
                "error_pp": float(abs(pred_home - gt_home) * 100.0),
                "count": int(total),
            }
        )

    return pd.DataFrame(rows)


def _get_start_xy(row: pd.Series) -> tuple[float, float]:
    if "start_x" in row and "start_y" in row:
        return float(row.get("start_x")), float(row.get("start_y"))
    return float(row.get("x")), float(row.get("y"))


def _get_end_xy(row: pd.Series) -> tuple[float, float]:
    if "end_x" in row and "end_y" in row:
        return float(row.get("end_x")), float(row.get("end_y"))
    return float(row.get("x")), float(row.get("y"))


def _append_if_finite(values: list[float], val: float) -> None:
    if np.isfinite(val):
        values.append(float(val))


def calc_pass_alignment_metrics(events_gt: pd.DataFrame, events_pred: pd.DataFrame) -> dict[str, float]:
    """NW-based kick alignment metrics (mean over matched kick pairs)."""
    gt = events_gt.copy()
    pred = events_pred.copy()

    gt = event_postprocessing.add_end_xy(gt, max_gap_sec=10.0)
    pred = event_postprocessing.add_end_xy(pred, max_gap_sec=10.0)

    if "seconds" not in gt.columns:
        gt["seconds"] = utils.series_to_seconds(gt["timestamp"])
    if "seconds" not in pred.columns:
        pred["seconds"] = utils.series_to_seconds(pred["timestamp"])

    if "episode_id" in gt.columns:
        gt = gt[gt["episode_id"] > 0].copy()
    if "episode_id" in pred.columns:
        pred = pred[pred["episode_id"] > 0].copy()

    pass_types = {"pass", "kick"}
    gt = gt[gt["event_type"].isin(pass_types)].copy()
    pred = pred[pred["event_type"].isin(pass_types)].copy()

    if "x" not in gt.columns and "start_x" in gt.columns:
        gt = gt.assign(x=gt["start_x"], y=gt["start_y"])
    if "x" not in pred.columns and "start_x" in pred.columns:
        pred = pred.assign(x=pred["start_x"], y=pred["start_y"])

    if gt.empty or pred.empty:
        return {
            "mean_time_distance": 0.0,
            "mean_start_distance": 0.0,
            "mean_end_distance": 0.0,
            "mean_kick_length_abs_error": 0.0,
            "mean_kick_length_mape": 0.0,
            "mean_direction_cosine": 0.0,
            "pair_count": 0.0,
        }

    pair_count = 0
    time_diffs: list[float] = []
    start_dists: list[float] = []
    end_dists: list[float] = []
    length_abs: list[float] = []
    length_mape: list[float] = []
    direction_cos: list[float] = []

    episode_ids = np.sort(pd.Index(gt["episode_id"]).union(pred["episode_id"]).unique())
    for ep in episode_ids:
        gt_ep = gt[gt["episode_id"] == ep].reset_index(drop=True)
        pred_ep = pred[pred["episode_id"] == ep].reset_index(drop=True)
        if gt_ep.empty and pred_ep.empty:
            continue
        alignments = evaluate_events.needleman_wunsch(gt_ep, pred_ep)
        for true_idx, pred_idx, _move in alignments:
            if true_idx is None or pred_idx is None:
                continue
            true_event = gt_ep.iloc[true_idx]
            pred_event = pred_ep.iloc[pred_idx]

            pair_count += 1

            t_diff = abs(float(pred_event["seconds"]) - float(true_event["seconds"]))
            _append_if_finite(time_diffs, t_diff)

            sx_gt, sy_gt = _get_start_xy(true_event)
            sx_pr, sy_pr = _get_start_xy(pred_event)
            _append_if_finite(start_dists, float(np.hypot(sx_pr - sx_gt, sy_pr - sy_gt)))

            ex_gt, ey_gt = _get_end_xy(true_event)
            ex_pr, ey_pr = _get_end_xy(pred_event)
            _append_if_finite(end_dists, float(np.hypot(ex_pr - ex_gt, ey_pr - ey_gt)))

            len_gt = float(np.hypot(ex_gt - sx_gt, ey_gt - sy_gt))
            len_pr = float(np.hypot(ex_pr - sx_pr, ey_pr - sy_pr))
            _append_if_finite(length_abs, abs(len_pr - len_gt))
            if np.isfinite(len_gt) and len_gt > 0:
                _append_if_finite(length_mape, abs(len_pr - len_gt) / len_gt)

            if np.isfinite(len_gt) and np.isfinite(len_pr) and len_gt > 0 and len_pr > 0:
                vec_gt = np.array([ex_gt - sx_gt, ey_gt - sy_gt], dtype=float)
                vec_pr = np.array([ex_pr - sx_pr, ey_pr - sy_pr], dtype=float)
                denom = np.linalg.norm(vec_gt) * np.linalg.norm(vec_pr)
                if denom > 0:
                    _append_if_finite(direction_cos, float(np.dot(vec_gt, vec_pr) / denom))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "mean_time_distance": _mean(time_diffs),
        "mean_start_distance": _mean(start_dists),
        "mean_end_distance": _mean(end_dists),
        "mean_kick_length_abs_error": _mean(length_abs),
        "mean_kick_length_mape": _mean(length_mape),
        "mean_direction_cosine": _mean(direction_cos),
        "pair_count": float(pair_count),
    }
