import os
from typing import Tuple

import numpy as np
import pandas as pd

from datatools import evaluate_events, utils


def load_match_data(match_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tracking_path = f"data/sportec/tracking_processed/{match_id}.parquet"
    gt_path = f"data/output_events/event_rdp/{match_id}.parquet"
    pred_path = f"data/output_events/event_rdp_pred/{match_id}.parquet"

    if not os.path.exists(tracking_path):
        raise FileNotFoundError(f"Tracking not found: {tracking_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT events not found: {gt_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Pred events not found: {pred_path}")

    tracking = pd.read_parquet(tracking_path)
    events_gt = pd.read_parquet(gt_path)
    events_pred = pd.read_parquet(pred_path)

    if "phase_id" in tracking.columns:
        tracking = tracking[tracking["phase_id"] != 0].copy()

    return tracking, events_gt, events_pred


def _map_episode_id(events: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    if "episode_id" in events.columns and not events["episode_id"].fillna(0).eq(0).all():
        return events

    if "frame_id" not in tracking.columns:
        return events

    events = events.copy()
    if events["frame_id"].isnull().any():
        events = events.dropna(subset=["frame_id"])

    ep_lookup = tracking.set_index("frame_id")["episode_id"].to_dict()
    events["episode_id"] = events["frame_id"].map(lambda x: ep_lookup.get(int(x), -1))

    events = events[events["episode_id"] != -1]
    return events


def prepare_events(events: pd.DataFrame, tracking: pd.DataFrame, map_episode_id: bool = True) -> pd.DataFrame:
    events = events.copy()

    if map_episode_id and ("episode_id" not in events.columns or events["episode_id"].fillna(0).eq(0).all()):
        events = _map_episode_id(events, tracking)

    events = evaluate_events.relabel_events(events, drop_fouls=True, keep_shots=False)
    events = add_end_xy(events, max_gap_sec=10.0)
    return events


def add_end_xy(events: pd.DataFrame, max_gap_sec: float = 10.0) -> pd.DataFrame:
    """
    Add end_x/end_y when missing by using the next event within the same episode,
    constrained by a maximum time gap. Existing end_x/end_y are preserved.
    """
    events = events.copy()

    if "x" not in events.columns or "y" not in events.columns:
        return events

    if "seconds" not in events.columns:
        events["seconds"] = utils.series_to_seconds(events["timestamp"])

    events["_orig_index"] = events.index
    sort_cols = []
    if "episode_id" in events.columns:
        sort_cols.append("episode_id")
    if "frame_id" in events.columns:
        sort_cols.append("frame_id")
    elif "seconds" in events.columns:
        sort_cols.append("seconds")

    if sort_cols:
        events = events.sort_values(sort_cols, kind="stable")

    if "episode_id" in events.columns:
        group = events.groupby("episode_id", sort=False)
    else:
        events["_episode_id"] = 0
        group = events.groupby("_episode_id", sort=False)

    next_x = group["x"].shift(-1)
    next_y = group["y"].shift(-1)
    next_t = group["seconds"].shift(-1)
    dt = next_t - events["seconds"]
    valid = dt.notna() & (dt <= max_gap_sec)

    end_x = events["end_x"] if "end_x" in events.columns else pd.Series(np.nan, index=events.index)
    end_y = events["end_y"] if "end_y" in events.columns else pd.Series(np.nan, index=events.index)

    end_x = end_x.where(end_x.notna(), next_x.where(valid))
    end_y = end_y.where(end_y.notna(), next_y.where(valid))

    events["end_x"] = end_x
    events["end_y"] = end_y

    events = events.sort_values("_orig_index", kind="stable").drop(columns=["_orig_index"], errors="ignore")
    if "_episode_id" in events.columns:
        events = events.drop(columns=["_episode_id"])
    return events
