import os
import re
import sys
from typing import List, Optional, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from rdp import rdp
from tqdm import tqdm

from datatools import config, utils


class Preprocessor:
    def __init__(self, events: pd.DataFrame, tracking: pd.DataFrame):
        events = events.copy().dropna(subset="frame_id", ignore_index=True)
        events = events.drop(columns=["start_x", "start_y", "utc_timestamp"], errors="ignore")

        tracking_ = tracking.set_index("frame_id")
        events["x"] = events["frame_id"].apply(lambda x: tracking_.at[x, "ball_x"])
        events["y"] = events["frame_id"].apply(lambda x: tracking_.at[x, "ball_y"])

        self.tracking, self.events = utils.label_frames_and_episodes(tracking, events)

        self.home_players = [c[:-2] for c in tracking.columns if re.match(r"home_\d+_x", c)]
        self.away_players = [c[:-2] for c in tracking.columns if re.match(r"away_\d+_x", c)]
        self.team_players = {"home": self.home_players, "away": self.away_players}

    @staticmethod
    def simplify_trajectory(
        xy: pd.DataFrame,
        frame_scale=config.RDP_FRAME_SCALE,
        min_angle=config.RDP_MIN_ANGLE,
    ) -> np.ndarray:
        fxy = xy.reset_index().copy()  # Columns: [frame_id, ball_x, ball_y]
        fxy["frame_id"] *= frame_scale  # To eliminate the influence of frame_id in RDP
        rdp_points = np.array(rdp(fxy, epsilon=0.5))

        dirs = np.diff(rdp_points[:, 1:], axis=0)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs_norm = dirs / (norms + 1e-8)

        v1 = dirs_norm[:-1]
        v2 = dirs_norm[1:]
        cos_theta = np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)
        angles = np.arccos(cos_theta)

        change_idx = np.where(angles >= np.deg2rad(min_angle))[0] + 1
        change_idx = np.concatenate([[0], change_idx, [len(rdp_points) - 1]])
        rdp_points = rdp_points[change_idx]
        rdp_points[:, 0] = (rdp_points[:, 0] / frame_scale).round()

        return rdp_points

    @staticmethod
    def enrich_path_info(path: pd.DataFrame, events: pd.DataFrame, rdp_points: np.ndarray) -> pd.DataFrame:
        path["event_idx"] = np.nan
        path["event_frame_id"] = np.nan
        path["rdp_frame_id"] = np.nan

        event_mask = path["event_pos"].notna()
        if event_mask.any():
            event_pos = path.loc[event_mask, "event_pos"].astype(int).to_numpy()
            path.loc[event_mask, "event_idx"] = events.index.to_numpy()[event_pos]
            if "frame_id" in events.columns:
                path.loc[event_mask, "event_frame_id"] = events.iloc[event_pos]["frame_id"].values

        rdp_mask = path["rdp_pos"].notna()
        if rdp_mask.any():
            rdp_pos = path.loc[rdp_mask, "rdp_pos"].astype(int).to_numpy()
            path.loc[rdp_mask, "rdp_frame_id"] = rdp_points[rdp_pos, 0]

        return path

    @staticmethod
    def align_event_rdp(
        events: pd.DataFrame,
        rdp_points: np.ndarray,
        max_dist=config.POSS_MAX_DIST,
        gap_rdp=-10.0,
        gap_event=-10.0,
        repeat_penalty=-20.0,
        repeat_threshold=50.0,
    ) -> pd.DataFrame:

        if events.empty and len(rdp_points) == 0:
            path = pd.DataFrame(columns=["event_pos", "rdp_pos", "move", "dist"])

        if events.empty:
            rdp_pos = np.arange(len(rdp_points), dtype=float)
            path = pd.DataFrame({"event_pos": np.nan, "rdp_pos": rdp_pos, "move": "left", "dist": np.nan})

        if len(rdp_points) == 0:
            event_pos = np.arange(len(events), dtype=float)
            path = pd.DataFrame({"event_pos": event_pos, "rdp_pos": np.nan, "move": "up", "dist": np.nan})

        else:
            ball_xy = rdp_points[:, 1:]
            event_xy = events[["x", "y"]].to_numpy()
            dists = np.linalg.norm(event_xy[:, None, :] - ball_xy[None, :, :], axis=2)
            score_mat = 100.0 * (1.0 - dists / max_dist)
            score_mat = np.clip(score_mat, -100.0, 100.0)
            score_mat[dists >= max_dist] = -1.0e9

            n_events, n_rdp = dists.shape
            dp_mat = np.zeros((n_events + 1, n_rdp + 1), dtype=float)
            trace = np.zeros((n_events + 1, n_rdp + 1), dtype=np.int8)

            for i in range(1, n_events + 1):
                dp_mat[i, 0] = dp_mat[i - 1, 0] + gap_event
                trace[i, 0] = 1
            for j in range(1, n_rdp + 1):
                dp_mat[0, j] = dp_mat[0, j - 1] + gap_rdp
                trace[0, j] = 2

            for i in range(1, n_events + 1):
                for j in range(1, n_rdp + 1):
                    diag = dp_mat[i - 1, j - 1] + score_mat[i - 1, j - 1]
                    up_gap = dp_mat[i - 1, j] + gap_event
                    left_gap = dp_mat[i, j - 1] + gap_rdp
                    up_match = -np.inf
                    if score_mat[i - 1, j - 1] >= repeat_threshold:
                        up_match = dp_mat[i - 1, j] + score_mat[i - 1, j - 1] + repeat_penalty
                    if diag >= up_gap and diag >= left_gap and diag >= up_match:
                        dp_mat[i, j] = diag
                        trace[i, j] = 0
                    elif up_match >= up_gap and up_match >= left_gap:
                        dp_mat[i, j] = up_match
                        trace[i, j] = 3
                    elif up_gap >= left_gap:
                        dp_mat[i, j] = up_gap
                        trace[i, j] = 1
                    else:
                        dp_mat[i, j] = left_gap
                        trace[i, j] = 2

            path_rows = []
            i = n_events
            j = n_rdp
            while i > 0 or j > 0:
                if i > 0 and j > 0 and trace[i, j] in (0, 3):
                    event_pos = i - 1
                    rdp_pos = j - 1
                    move = "diag" if trace[i, j] == 0 else "repeat"
                    dist = dists[event_pos, rdp_pos]
                    if trace[i, j] == 0:
                        i -= 1
                        j -= 1
                    else:
                        i -= 1
                elif i > 0 and (j == 0 or trace[i, j] == 1):
                    event_pos = i - 1
                    rdp_pos = np.nan
                    move = "up"
                    dist = np.nan
                    i -= 1
                else:
                    event_pos = np.nan
                    rdp_pos = j - 1
                    move = "left"
                    dist = np.nan
                    j -= 1

                path_rows.append(
                    {
                        "event_pos": event_pos,
                        "rdp_pos": rdp_pos,
                        "move": move,
                        "dist": dist,
                    }
                )

            path_rows.reverse()
            path = pd.DataFrame(path_rows)
            return Preprocessor.enrich_path_info(path, events, rdp_points)

    def detect_touches_episode(
        self,
        episode_id: int,
        rdp_min_angle=config.RDP_MIN_ANGLE,
        max_dist=config.POSS_MAX_DIST,
    ) -> pd.DataFrame:
        ep_tracking = self.tracking[self.tracking["episode_id"] == episode_id].copy().set_index("frame_id")
        ep_events = self.events[self.events["episode_id"] == episode_id]
        frame2sec = ep_tracking["timestamp"].to_dict()

        if len(ep_tracking) < 25 or ep_events.empty:
            return None

        ball_xy = ep_tracking[["ball_x", "ball_y"]].copy()
        rdp_points = Preprocessor.simplify_trajectory(ball_xy, min_angle=rdp_min_angle)

        touches = Preprocessor.align_event_rdp(ep_events, rdp_points, max_dist=max_dist)

        touches["period_id"] = int(ep_events["period_id"].iloc[0])
        touches["episode_id"] = int(episode_id)

        event_mask = touches["event_frame_id"].notna()
        touches["frame_id"] = touches["event_frame_id"].where(event_mask, touches["rdp_frame_id"]).astype(int)
        touches["timestamp"] = touches["frame_id"].map(frame2sec).apply(utils.seconds_to_timestamp)
        touches["player_id"] = pd.Series(index=touches.index, dtype="object")
        touches.loc[event_mask, "player_id"] = ep_events["player_id"].values
        touches.loc[event_mask, "event_type"] = ep_events["spadl_type"].values

        touches["x"] = touches["frame_id"].apply(lambda x: ep_tracking.at[x, "ball_x"])
        touches["y"] = touches["frame_id"].apply(lambda x: ep_tracking.at[x, "ball_y"])

        for i, row in touches[~event_mask].iterrows():
            frame_id = int(round(row["frame_id"]))
            tracking_row: pd.Series = ep_tracking.loc[frame_id]
            cand_players = self.home_players + self.away_players

            prev_events = touches.loc[(touches.index < i) & event_mask]
            next_events = touches.loc[(touches.index > i) & event_mask]

            if not prev_events.empty and not next_events.empty:
                prev_idx = prev_events.index[-1]
                next_idx = next_events.index[0]

                prev_frame = int(round(touches.at[prev_idx, "frame_id"]))
                next_frame = int(round(touches.at[next_idx, "frame_id"]))

                if frame_id < prev_frame + 5 or frame_id > next_frame - 5:
                    continue

                prev_team = touches.at[prev_idx, "player_id"][:4]
                next_team = touches.at[next_idx, "player_id"][:4]

                if next_frame - prev_frame < 50 and prev_team == next_team and prev_team in self.team_players:
                    cand_players = self.team_players[prev_team]

            player_x_cols = [f"{p}_x" for p in cand_players]
            player_y_cols = [f"{p}_y" for p in cand_players]
            player_x = tracking_row[player_x_cols].dropna().astype(float)
            player_y = tracking_row[player_y_cols].dropna().astype(float)

            dists = np.sqrt((player_x.values - row["x"]) ** 2 + (player_y.values - row["y"]) ** 2)
            min_idx = dists.argmin()

            if dists[min_idx] >= max_dist:
                continue

            touches.at[i, "player_id"] = player_x.index[min_idx][:-2]

        if event_mask.any():
            last_event_frame = touches.loc[event_mask, "frame_id"].max()
            pitch_mask = (touches["x"].between(1, config.PITCH_X - 1)) & (touches["y"].between(1, config.PITCH_Y - 1))
            out_points = touches[(touches["frame_id"] > last_event_frame) & ~pitch_mask]
            if not out_points.empty:
                out_idx = out_points.index[0]
                shot_mask = ep_events["spadl_type"] == "shot"
                last_shot_success = ep_events.loc[shot_mask, "success"].iloc[-1] if shot_mask.any() else False
                touches.at[out_idx, "player_id"] = "goal" if last_shot_success else "out"
                touches = touches[touches["frame_id"] <= touches.at[out_idx, "frame_id"]].copy()

        touches = touches.dropna(subset="player_id").sort_values(["frame_id", "event_idx"], ignore_index=True)

        event_rows = touches[touches["event_frame_id"].notna()]
        if not event_rows.empty:
            drop_indices: List[int] = []
            prev_player_id = None

            for frame_id, group in event_rows.groupby("frame_id", sort=False):
                group = group.sort_values("event_idx")
                if len(group) == 1:
                    prev_player_id = group.iloc[-1]["player_id"]
                    continue
                if prev_player_id is not None and (group["player_id"] == prev_player_id).any():
                    player_group = group[group["player_id"] == prev_player_id]
                    keep_idx = player_group["event_idx"].astype(float).idxmax()
                    drop_indices.extend([idx for idx in group.index if idx != keep_idx])
                    prev_player_id = group.loc[keep_idx, "player_id"]
                else:
                    keep_idx = group["event_idx"].astype(float).idxmax()
                    drop_indices.extend([idx for idx in group.index if idx != keep_idx])
                    prev_player_id = group.loc[keep_idx, "player_id"]

            if drop_indices:
                touches = touches.drop(index=drop_indices)

        none_type = touches["event_type"].isna()
        same_prev = touches["player_id"].eq(touches["player_id"].shift(1))
        same_next = touches["player_id"].eq(touches["player_id"].shift(-1))
        touches = touches.loc[~(none_type & same_prev & same_next)].copy()

        player_mask = touches["player_id"].astype(str).str.startswith(("home_", "away_"), na=False)
        frame_ids = touches.loc[player_mask, "frame_id"].round().astype(int).to_numpy()
        player_ids = touches.loc[player_mask, "player_id"].astype(str).to_numpy()
        row_idx = ep_tracking.index.get_indexer(frame_ids)
        x_col_idx = ep_tracking.columns.get_indexer(player_ids + "_x")
        y_col_idx = ep_tracking.columns.get_indexer(player_ids + "_y")
        touches.loc[player_mask, "x"] = ep_tracking.to_numpy()[row_idx, x_col_idx].astype(float).round(2)
        touches.loc[player_mask, "y"] = ep_tracking.to_numpy()[row_idx, y_col_idx].astype(float).round(2)

        return touches

    def detect_touches(self, rdp_min_angle=config.RDP_MIN_ANGLE, max_dist=config.POSS_MAX_DIST) -> pd.DataFrame:
        touches: List[pd.DataFrame] = []

        for episode in tqdm(self.tracking["episode_id"].unique(), desc="Detecting touches per episode"):
            if episode == 0:
                continue

            ep_touches = self.detect_touches_episode(episode, rdp_min_angle, max_dist)

            if ep_touches is not None and not ep_touches.empty:
                touches.append(ep_touches)

        touches: pd.DataFrame = pd.concat(touches, ignore_index=True).dropna(subset="player_id")

        out_mask = touches["player_id"] == "out"
        goal_mask = touches["player_id"] == "goal"
        touches.loc[out_mask, "event_type"] = "out"
        touches.loc[goal_mask, "event_type"] = "goal"

        ratio = config.PITCH_Y / config.PITCH_X
        diag1 = touches["y"] - ratio * touches["x"]
        diag2 = touches["y"] + ratio * touches["x"] - config.PITCH_Y

        out_l = out_mask & (diag1 > 0) & (diag2 < 0)
        out_r = out_mask & (diag1 < 0) & (diag2 > 0)
        out_b = out_mask & (diag1 < 0) & (diag2 < 0)
        out_t = out_mask & (diag1 > 0) & (diag2 > 0)
        goal_l = goal_mask & (touches["x"] < 10)
        goal_r = goal_mask & (touches["x"] > config.PITCH_X - 10)

        touches.loc[out_l, "player_id"] = "out_left"
        touches.loc[out_r, "player_id"] = "out_right"
        touches.loc[out_b, "player_id"] = "out_bottom"
        touches.loc[out_t, "player_id"] = "out_top"
        touches.loc[goal_l, "player_id"] = "goal_left"
        touches.loc[goal_r, "player_id"] = "goal_right"

        return touches[config.TOUCH_COLS]

    def merge_tracking_poss(self, touches: pd.DataFrame) -> pd.DataFrame:
        tracking_poss = pd.merge(self.tracking, touches[["frame_id", "player_id"]], how="left")

        for episode, ep_tracking in tracking_poss.groupby("episode_id"):
            if episode == 0:
                continue

            ep_tracking = ep_tracking.copy()
            valid_poss = ep_tracking["player_id"].dropna()

            if not valid_poss.empty:
                prev_poss = ep_tracking["player_id"].ffill()
                next_poss = ep_tracking["player_id"].bfill()
                ep_tracking["player_id"] = np.where(prev_poss == next_poss, prev_poss, np.nan)

                first_poss = valid_poss.index[0]
                last_poss = valid_poss.index[-1]
                ep_tracking.loc[:first_poss, "player_id"] = valid_poss.at[first_poss]
                ep_tracking.loc[last_poss:, "player_id"] = valid_poss.at[last_poss]

                tracking_poss.loc[ep_tracking.index, "player_id"] = ep_tracking["player_id"]

        return tracking_poss


if __name__ == "__main__":
    IN_EVENT_DIR = "data/sportec/event_synced"
    IN_TRACKING_DIR = "data/sportec/tracking_parquet"
    OUT_EVENT_DIR = "data/sportec/event_rdp"
    OUT_TRACKING_DIR = "data/sportec/tracking_processed"

    os.makedirs(OUT_EVENT_DIR, exist_ok=True)
    os.makedirs(OUT_TRACKING_DIR, exist_ok=True)
    match_ids = [f.split(".")[0] for f in os.listdir(IN_EVENT_DIR)]

    for i, match_id in enumerate(match_ids):
        print()
        print(f"[{i + 1}] {match_id}")

        events = pd.read_parquet(f"{IN_EVENT_DIR}/{match_id}.parquet")
        tracking = pd.read_parquet(f"{IN_TRACKING_DIR}/{match_id}.parquet")
        tracking[["timestamp", "ball_x", "ball_y"]] = tracking[["timestamp", "ball_x", "ball_y"]].round(2)

        proc = Preprocessor(events, tracking)
        touches = proc.detect_touches()
        touches.to_parquet(f"{OUT_EVENT_DIR}/{match_id}.parquet")

        tracking = proc.merge_tracking_poss(touches)
        tracking_processed = utils.calculate_running_features(tracking)
        tracking_processed.to_parquet(f"{OUT_TRACKING_DIR}/{match_id}.parquet")
