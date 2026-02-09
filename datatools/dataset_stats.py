import os
import sys
from typing import Iterable, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

from datatools import utils


def count_dataset_stats(
    match_ids: Iterable[str],
    fps: float = 25.0,
    sample_freq: int = 5,
    window_seconds: float = 10.0,
    window_stride: int = 1,
    drop_short_episodes: bool = False,
) -> Tuple[int, int, int]:
    TRACKING_DIR = "data/sportec/tracking_processed"
    EVENT_DIR = "data/sportec/event_rdp"

    window_size = int(round(window_seconds * fps / sample_freq))
    window_stride = int(window_stride)

    episode_count = 0
    event_count = 0
    frame_count = 0
    window_count = 0

    for match_id in match_ids:
        events = pd.read_parquet(f"{EVENT_DIR}/{match_id}.parquet")
        tracking = pd.read_parquet(f"{TRACKING_DIR}/{match_id}.parquet")

        for episode in tracking["episode_id"].unique():
            ep_events = events[(events["episode_id"] == episode) & (events["event_type"] != "foul")]
            ep_tracking = tracking[tracking["episode_id"] == episode]

            if episode == 0:
                continue

            if drop_short_episodes and len(ep_tracking) < window_seconds * fps:
                continue

            if ep_tracking["player_id"].dropna().empty:
                continue

            episode_count += 1
            event_count += len(ep_events)
            frame_count += len(ep_tracking)

            down_idx = np.arange(0, len(ep_tracking), sample_freq, dtype=np.int64)
            if len(down_idx) == 0 or down_idx[-1] != len(ep_tracking) - 1:
                down_idx = np.append(down_idx, len(ep_tracking) - 1)

            starts = list(range(0, len(down_idx) - window_size + 1, window_stride))
            starts = starts[:-1] + [len(down_idx) - window_size]
            window_count += len(starts)

    print(f"{episode_count} episodes, {event_count} events, {frame_count} frames, {window_count} windows")


if __name__ == "__main__":
    TRACKING_DIR = "data/sportec/tracking_processed"
    match_ids = np.sort([f.split(".")[0] for f in os.listdir(TRACKING_DIR)])[5:6]
    print(match_ids)
    count_dataset_stats(match_ids, drop_short_episodes=True)
