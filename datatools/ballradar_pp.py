import numpy as np
import pandas as pd
import scipy.signal as signal
from tqdm import tqdm

from datatools import config, utils


class Postprocessor:
    def __init__(self, tracking: pd.DataFrame, poss_probs: pd.DataFrame, pred_ball_xy: pd.DataFrame):
        tracking = tracking.copy()
        tracking = tracking.set_index("frame_id", drop=False)

        self.poss_probs = poss_probs.dropna(axis=0, how="all").copy()
        self.tracking = tracking.loc[self.poss_probs.index].copy()

        if "out_left_x" not in self.tracking.columns:
            outside_nodes = Postprocessor._outside_nodes()
            for k, (x, y) in outside_nodes.items():
                self.tracking[f"{k}_x"] = x
                self.tracking[f"{k}_y"] = y

        pred_ball_xy = pred_ball_xy.copy()
        if "frame_id" in pred_ball_xy.columns:
            pred_ball_xy = pred_ball_xy.set_index("frame_id", drop=False)
        pred_ball_xy = pred_ball_xy.reindex(self.tracking.index)
        self.tracking["pred_ball_x"] = pred_ball_xy["ball_x"]
        self.tracking["pred_ball_y"] = pred_ball_xy["ball_y"]

        self.player_ids = Postprocessor._infer_player_ids(self.tracking)
        self.poss_scores = pd.DataFrame(index=self.tracking.index, columns=poss_probs.columns, dtype=float)

        output_cols = ["carrier", "ball_x", "ball_y", "focus_x", "focus_y"]
        self.output = pd.DataFrame(index=self.tracking.index, columns=output_cols)
        self.output[output_cols[1:]] = self.output[output_cols[1:]].astype(float)

    @staticmethod
    def _infer_player_ids(tracking: pd.DataFrame) -> list:
        cols = set(tracking.columns)
        player_ids = []
        for c in tracking.columns:
            if (c.startswith("home_") or c.startswith("away_")) and c.endswith("_x"):
                pid = c[:-2]
                if f"{pid}_y" in cols:
                    player_ids.append(pid)
        return player_ids

    @staticmethod
    def _is_player(node: object) -> bool:
        return isinstance(node, str) and (node.startswith("home_") or node.startswith("away_"))

    @staticmethod
    def _outside_nodes() -> dict:
        return {
            "out_left": (0.0, config.PITCH_Y / 2),
            "out_right": (config.PITCH_X, config.PITCH_Y / 2),
            "out_bottom": (config.PITCH_X / 2, 0.0),
            "out_top": (config.PITCH_X / 2, config.PITCH_Y),
        }

    @staticmethod
    def _get_node_xy(tracking: pd.DataFrame, frame_id: int, node: object) -> tuple[float, float]:
        if Postprocessor._is_player(node):
            return (float(tracking.at[frame_id, f"{node}_x"]), float(tracking.at[frame_id, f"{node}_y"]))
        else:
            return Postprocessor._outside_nodes().get(node, (np.nan, np.nan))

    def _carry_records_to_events(self, carry_records: pd.DataFrame) -> pd.DataFrame:
        events: list[dict] = []
        if carry_records is None or carry_records.empty:
            return pd.DataFrame(
                columns=[
                    "frame_id",
                    "period_id",
                    "episode_id",
                    "timestamp",
                    "player_id",
                    "receiver_id",
                    "event_type",
                    "start_x",
                    "start_y",
                    "end_x",
                    "end_y",
                ]
            )

        carry_records = carry_records.copy()
        if "episode_id" not in carry_records.columns:
            carry_records["episode_id"] = carry_records["start_idx"].map(self.tracking["episode_id"])

        carry_records = carry_records.sort_values(["episode_id", "start_idx"]).reset_index(drop=True)
        for episode_id, group in carry_records.groupby("episode_id", sort=False):
            group = group.reset_index(drop=True)
            for i, row in group.iterrows():
                carrier = row["carrier"]
                if pd.isna(carrier):
                    continue

                start_frame = int(row["start_idx"])
                end_frame = int(row["end_idx"])
                next_carrier = group.at[i + 1, "carrier"] if i + 1 < len(group) else None

                if not Postprocessor._is_player(carrier):
                    start_x, start_y = Postprocessor._get_node_xy(self.tracking, start_frame, carrier)
                    events.append(
                        {
                            "frame_id": start_frame,
                            "period_id": self.tracking.at[start_frame, "period_id"],
                            "episode_id": self.tracking.at[start_frame, "episode_id"],
                            "timestamp": utils.seconds_to_timestamp(self.tracking.at[start_frame, "timestamp"]),
                            "player_id": carrier,
                            "receiver_id": carrier,
                            "event_type": "out",
                            "start_x": start_x,
                            "start_y": start_y,
                            "end_x": np.nan,
                            "end_y": np.nan,
                        }
                    )
                    continue

                start_x, start_y = Postprocessor._get_node_xy(self.tracking, start_frame, carrier)
                events.append(
                    {
                        "frame_id": start_frame,
                        "period_id": self.tracking.at[start_frame, "period_id"],
                        "episode_id": self.tracking.at[start_frame, "episode_id"],
                        "timestamp": utils.seconds_to_timestamp(self.tracking.at[start_frame, "timestamp"]),
                        "player_id": carrier,
                        "receiver_id": carrier,
                        "event_type": "control",
                        "start_x": start_x,
                        "start_y": start_y,
                        "end_x": np.nan,
                        "end_y": np.nan,
                    }
                )

                receiver_id = next_carrier if next_carrier is not None else carrier

                end_x0, end_y0 = Postprocessor._get_node_xy(self.tracking, end_frame, carrier)
                events.append(
                    {
                        "frame_id": end_frame,
                        "period_id": self.tracking.at[end_frame, "period_id"],
                        "episode_id": self.tracking.at[end_frame, "episode_id"],
                        "timestamp": utils.seconds_to_timestamp(self.tracking.at[end_frame, "timestamp"]),
                        "player_id": carrier,
                        "receiver_id": receiver_id,
                        "event_type": "kick",
                        "start_x": end_x0,
                        "start_y": end_y0,
                        "end_x": np.nan,
                        "end_y": np.nan,
                    }
                )

        events_df = pd.DataFrame(events)
        if events_df.empty:
            return events_df

        events_df = events_df.sort_values(["episode_id", "frame_id"]).reset_index(drop=True)
        for episode_id, group in events_df.groupby("episode_id", sort=False):
            if group.empty:
                continue
            idxs = group.index.to_list()
            for i in range(len(idxs) - 1):
                curr_idx = idxs[i]
                next_idx = idxs[i + 1]
                next_frame = int(events_df.at[next_idx, "frame_id"])
                receiver_id = events_df.at[curr_idx, "receiver_id"]
                end_x, end_y = Postprocessor._get_node_xy(self.tracking, next_frame, receiver_id)
                events_df.at[curr_idx, "end_x"] = end_x
                events_df.at[curr_idx, "end_y"] = end_y

            last_idx = idxs[-1]
            last_frame = int(events_df.at[last_idx, "frame_id"])
            last_receiver = events_df.at[last_idx, "receiver_id"]
            end_x, end_y = Postprocessor._get_node_xy(self.tracking, last_frame, last_receiver)
            events_df.at[last_idx, "end_x"] = end_x
            events_df.at[last_idx, "end_y"] = end_y

        return events_df

    def _events_to_edge_seq(self, pred_events: pd.DataFrame) -> pd.DataFrame:
        edge_seq = pd.DataFrame(index=self.tracking.index, columns=["sender_id", "receiver_id"])
        if pred_events is None or pred_events.empty:
            return edge_seq

        pred_events = pred_events.sort_values(["episode_id", "frame_id"]).reset_index(drop=True)
        for episode_id, ep_events in pred_events.groupby("episode_id", sort=False):
            ep_tracking = self.tracking[self.tracking["episode_id"] == episode_id]
            if ep_tracking.empty:
                continue
            default_end = int(ep_tracking.index.max()) + 1

            ep_events = ep_events.reset_index(drop=True)
            for i in range(len(ep_events)):
                frame_i = int(ep_events.at[i, "frame_id"])
                event_type = ep_events.at[i, "event_type"]
                player_id = ep_events.at[i, "player_id"]
                next_frame = int(ep_events.at[i + 1, "frame_id"]) if i + 1 < len(ep_events) else default_end

                if event_type == "control":
                    next_kick_frame = None
                    for j in range(i + 1, len(ep_events)):
                        if ep_events.at[j, "event_type"] == "kick":
                            next_kick_frame = int(ep_events.at[j, "frame_id"])
                            break
                    end_frame = next_kick_frame if next_kick_frame is not None else default_end
                    sender_id = player_id
                    receiver_id = player_id
                elif event_type == "kick":
                    end_frame = next_frame
                    sender_id = player_id
                    receiver_id = ep_events.at[i + 1, "player_id"] if i + 1 < len(ep_events) else player_id
                elif event_type == "out":
                    end_frame = next_frame
                    sender_id = player_id
                    receiver_id = player_id
                else:
                    end_frame = next_frame
                    sender_id = player_id
                    receiver_id = player_id

                mask = (
                    (self.tracking["episode_id"] == episode_id)
                    & (self.tracking.index >= frame_i)
                    & (self.tracking.index < end_frame)
                )
                edge_seq.loc[mask, "sender_id"] = sender_id
                edge_seq.loc[mask, "receiver_id"] = receiver_id

        return edge_seq

    @staticmethod
    def calc_ball_features(ball_tracking: pd.DataFrame) -> pd.DataFrame:
        W_LEN = 7
        P_ORDER = 2

        ball_tracking = ball_tracking.dropna(subset=["pred_ball_x"])
        times = ball_tracking["timestamp"].values
        dt = np.median(np.diff(times)) if len(times) > 1 else 0.04

        x = ball_tracking["pred_ball_x"].values
        y = ball_tracking["pred_ball_y"].values
        x = signal.savgol_filter(x, window_length=W_LEN, polyorder=P_ORDER)
        y = signal.savgol_filter(y, window_length=W_LEN, polyorder=P_ORDER)

        vx = np.diff(x, prepend=x[0]) / dt
        vy = np.diff(y, prepend=y[0]) / dt
        vx = signal.savgol_filter(vx, window_length=W_LEN, polyorder=P_ORDER)
        vy = signal.savgol_filter(vy, window_length=W_LEN, polyorder=P_ORDER)
        speeds = np.sqrt(vx**2 + vy**2)

        accels = np.diff(speeds, prepend=speeds[-1]) / dt
        accels[:2] = 0
        accels[-2:] = 0
        accels = signal.savgol_filter(accels, window_length=W_LEN, polyorder=P_ORDER)

        cols = ["timestamp", "x", "y", "vx", "vy", "speed", "accel"]
        ball_tracking_arr = np.stack((times, x, y, vx, vy, speeds, accels), axis=1)
        return pd.DataFrame(ball_tracking_arr, index=ball_tracking.index, columns=cols)

    @staticmethod
    def calc_ball_dists(tracking: pd.DataFrame, players: list) -> pd.DataFrame:
        # Calculate distances from the ball to the players
        player_xy_cols = [f"{p}{t}" for p in players for t in ["_x", "_y"]]
        player_xy = tracking[player_xy_cols].values.reshape(tracking.shape[0], -1, 2)
        pred_xy = tracking[["pred_ball_x", "pred_ball_y"]].values[:, np.newaxis, :]
        ball_dists = np.linalg.norm(pred_xy - player_xy, axis=-1)
        ball_dists = pd.DataFrame(ball_dists, index=tracking.index, columns=players)

        # Calculate distances from the ball to the pitch lines
        ball_dists["out_left"] = tracking["pred_ball_x"].abs()
        ball_dists["out_right"] = (config.PITCH_X - tracking["pred_ball_x"]).abs()
        ball_dists["out_bottom"] = tracking["pred_ball_y"].abs()
        ball_dists["out_top"] = (config.PITCH_Y - tracking["pred_ball_y"]).abs()

        return ball_dists

    @staticmethod
    def calc_poss_scores(
        poss_probs: pd.DataFrame,
        ball_dists: pd.DataFrame,
        players: list,
        max_dist: float = 10,
    ) -> pd.DataFrame:
        score_cols = [c for c in poss_probs.columns if c in ball_dists.columns]
        player_cols = [c for c in players if c in score_cols]
        ball_dists = ball_dists[score_cols].copy()
        if player_cols:
            ball_dists[player_cols] = ball_dists[player_cols].where(ball_dists[player_cols] < max_dist, 100)

        poss_scores = poss_probs[score_cols] / (np.sqrt(ball_dists[score_cols]) + 1e-6)
        poss_scores["idxmax"] = poss_scores[score_cols].idxmax(axis=1)
        poss_scores["max"] = poss_scores[score_cols].max(axis=1)

        return poss_scores

    @staticmethod
    def generate_carry_records(carriers: pd.Series):
        carriers_prev = carriers.ffill()
        carriers_next = carriers.bfill()
        carriers = carriers_prev.where(carriers_prev == carriers_next, np.nan)

        poss_changes = carriers.notna().astype(int).diff().fillna(0)
        start_idxs = poss_changes[poss_changes > 0].index.values.tolist()
        end_idxs = poss_changes[poss_changes < 0].index.values.tolist()
        if not start_idxs:
            start_idxs = [carriers.index[0]]
        if not end_idxs:
            end_idxs = [carriers.index[-1]]

        if start_idxs[0] > end_idxs[0]:
            start_idxs.insert(0, carriers.index[0])
        if start_idxs[-1] > end_idxs[-1]:
            end_idxs.append(carriers.index[-1])

        carry_records = pd.DataFrame(np.stack([start_idxs, end_idxs], axis=1), columns=["start_idx", "end_idx"])
        carry_records["carrier"] = carriers.loc[start_idxs].values.tolist()

        return carry_records

    @staticmethod
    def detect_carries_by_accel(
        ball_feats: pd.DataFrame, ball_dists: pd.DataFrame = None, poss_scores: pd.DataFrame = None, max_accel=5
    ) -> tuple[pd.Series, pd.DataFrame]:
        assert ball_dists is not None or poss_scores is not None

        accels = ball_feats[["accel"]].copy()
        for k in np.arange(2) + 1:
            accels[f"prev{k}"] = accels["accel"].shift(k, fill_value=0)
            accels[f"next{k}"] = accels["accel"].shift(-k, fill_value=0)

        max_flags = (accels["accel"] == accels.max(axis=1)) & (accels["accel"] > max_accel)
        min_flags = (accels["accel"] == accels.min(axis=1)) & (accels["accel"] < -max_accel)
        max_idxs = accels[max_flags].index.tolist()
        min_idxs = accels[min_flags].index.tolist()

        if ball_feats.index[0] in max_idxs:
            max_idxs.pop(0)
        if ball_feats.index[-1] in min_idxs:
            min_idxs.pop(-1)

        if not min_idxs:
            min_idxs.insert(0, ball_feats.index[0])
        if not max_idxs:
            max_idxs.append(ball_feats.index[-1])

        if min_idxs[0] > max_idxs[0]:
            min_idxs.insert(0, ball_feats.index[0])
        if min_idxs[-1] > max_idxs[-1]:
            max_idxs.append(ball_feats.index[-1])

        max_idxs_grouped = []
        min_idxs_grouped = []
        carry_records = []

        while max_idxs:
            # let the local minima belong to the same group if there is no local maximum between them
            min_group = []
            while min_idxs and min_idxs[0] < max_idxs[0]:
                min_group.append(min_idxs.pop(0))
            min_idxs_grouped.append(min_group)

            # let the local maxima belong to the same group if there is no local minimum between them
            max_group = []
            if min_idxs:
                while max_idxs and max_idxs[0] < min_idxs[0]:
                    max_group.append(max_idxs.pop(0))
            else:
                while max_idxs:
                    max_group.append(max_idxs.pop(0))
            max_idxs_grouped.append(max_group)

        for i in range(len(max_idxs_grouped)):
            start_idx = ball_feats.loc[min_idxs_grouped[i], "accel"].idxmin()
            end_idx = ball_feats.loc[max_idxs_grouped[i], "accel"].idxmax()
            carry_record = [start_idx, end_idx]

            if i == 0 or ball_dists.loc[start_idx:end_idx].min().min() < 4:
                carrier = ball_dists.loc[start_idx:end_idx].mean().idxmin()
                ball_feats.loc[start_idx:end_idx, "carrier"] = carrier
                carry_record.append(carrier)

            carry_records.append(carry_record)

        # carry_records = pd.DataFrame(carry_records, columns=["start_idx", "end_idx", "carrier"])
        return Postprocessor.generate_carry_records(ball_feats["carrier"])

    @staticmethod
    def detect_carries_by_poss_score(poss_scores: pd.DataFrame, thres_touch=0.1, thres_carry=0.4) -> pd.DataFrame:
        poss_ids = (poss_scores["idxmax"] != poss_scores["idxmax"].shift(1)).cumsum()

        grouper = poss_scores.groupby(poss_ids)
        peak_idxs = grouper["max"].idxmax().rename("index")
        carriers = grouper["idxmax"].first().rename("carrier")
        max_scores = grouper["max"].max().rename("max_score")
        peaks = pd.concat([peak_idxs, carriers, max_scores], axis=1).set_index("index")
        peaks = peaks[peaks["max_score"] > thres_touch].copy()

        carry_records = []
        for i, curr_idx in enumerate(peaks.index):
            carrier = peaks.at[curr_idx, "carrier"]
            prev_idx = peaks.index[i - 1] if i > 1 else poss_scores.index[0]
            next_idx = peaks.index[i + 1] if i < len(peaks) - 1 else poss_scores.index[-1]

            if peaks.at[curr_idx, "max_score"] > thres_carry:
                curr_poss_scores = poss_scores.loc[prev_idx:next_idx]
                curr_poss_scores = curr_poss_scores[curr_poss_scores["idxmax"] == carrier]
                start_idx = curr_poss_scores[curr_poss_scores["max"] > thres_carry].index[0]
                end_idx = curr_poss_scores.index[-1]
                carry_records.append([start_idx, end_idx, carrier])
            else:
                carry_records.append([curr_idx, curr_idx + 1, carrier])

        carry_records = pd.DataFrame(carry_records, columns=["start_idx", "end_idx", "carrier"])
        carry_records.at[0, "start_idx"] = poss_scores.index[0]
        carry_records.at[len(carry_records) - 1, "end_idx"] = poss_scores.index[-1]

        return carry_records

    @staticmethod
    def detect_carries_for_imputation(
        tracking: pd.DataFrame, poss_scores: pd.DataFrame, thres_touch=0.3
    ) -> pd.DataFrame:
        poss_scores = poss_scores.reset_index().copy()
        poss_ids = (poss_scores["idxmax"] != poss_scores["idxmax"].shift(1)).cumsum()

        grouper = poss_scores.groupby(poss_ids)
        peak_idxs = grouper["index"].last()
        carriers = grouper["idxmax"].first().rename("carrier")
        max_scores = grouper["max"].max().rename("max_score")
        peaks = pd.concat([peak_idxs, carriers, max_scores], axis=1).set_index("index")

        peaks = peaks[peaks["max_score"] > thres_touch].copy()
        for i in peaks.index:
            tracking.at[i, "carrier"] = peaks.at[i, "carrier"]

        carriers_prev = tracking["carrier"].ffill()
        carriers_next = tracking["carrier"].bfill()
        tracking["carrier"] = carriers_prev.where(carriers_prev == carriers_next)

        carry_ids = pd.Series(0, index=tracking.index)
        valid_carriers = tracking["carrier"].dropna().copy()
        carry_ids.loc[valid_carriers.index] = (valid_carriers != valid_carriers.shift(1)).cumsum()

        grouper = tracking.groupby(carry_ids)
        start_idxs = grouper["frame_id"].first().rename("start_idx") - 1
        end_idxs = grouper["frame_id"].last().rename("end_idx") - 1
        carriers = grouper["carrier"].first()

        return pd.concat([start_idxs, end_idxs, carriers], axis=1)[1:].copy()

    @staticmethod
    def finetune_ball_trace(
        tracking: pd.DataFrame, carry_records: pd.DataFrame = None, focus: bool = True
    ) -> pd.DataFrame:
        output_cols = ["carrier", "ball_x", "ball_y", "focus_x", "focus_y"]
        output = pd.DataFrame(index=tracking.index, columns=output_cols)
        output[output_cols[1:]] = output[output_cols[1:]].astype(float)

        # Reconstruct the ball trace
        for i in carry_records.index:
            start_idx = carry_records.at[i, "start_idx"]
            end_idx = carry_records.at[i, "end_idx"]
            carrier = carry_records.at[i, "carrier"]

            output.loc[start_idx:end_idx, "carrier"] = carrier
            if not carrier.startswith("out"):
                output.loc[start_idx:end_idx, "ball_x"] = tracking.loc[start_idx:end_idx, f"{carrier}_x"]
                output.loc[start_idx:end_idx, "ball_y"] = tracking.loc[start_idx:end_idx, f"{carrier}_y"]
            elif carrier in ["out_left", "out_right"]:
                output.loc[start_idx:end_idx, "ball_x"] = 0 if carrier == "out_left" else config.PITCH_X
                output.loc[start_idx:end_idx, "ball_y"] = tracking.loc[start_idx:end_idx, "pred_ball_y"].mean()
            else:  # carrier in ["out_bottom", "out_top"]
                output.loc[start_idx:end_idx, "ball_x"] = tracking.loc[start_idx:end_idx, "pred_ball_x"].mean()
                output.loc[start_idx:end_idx, "ball_y"] = 0 if carrier == "out_bottom" else config.PITCH_Y

        output[["ball_x", "ball_y"]] = output[["ball_x", "ball_y"]].interpolate(limit_direction="both")

        # Calculate xy coordinates to center on when zooming in the panoramic match video
        if focus:
            carry_records["trans_prev"] = 0.0

            for i in carry_records.index:
                if i == carry_records.index[0]:
                    send_idx = output.index[0]
                else:
                    send_idx = carry_records.at[i - 1, "end_idx"]
                receive_idx = carry_records.at[i, "start_idx"]

                trans_x = abs(output.at[receive_idx, "ball_x"] - output.at[send_idx, "ball_x"])
                trans_y = abs(output.at[receive_idx, "ball_y"] - output.at[send_idx, "ball_y"])
                carry_records.at[i, "trans_prev"] = max(trans_x, trans_y * 0.5)

            carry_records["trans_next"] = carry_records["trans_prev"].shift(-1).fillna(0)
            carry_records["trans_dur"] = carry_records["end_idx"] - carry_records["start_idx"] + 1
            carry_records["focus"] = (carry_records["trans_dur"] > 5) + (
                carry_records[["trans_prev", "trans_next"]].min(axis=1) > 15
            )

            for i in carry_records.index:
                if i > carry_records.index[0] and carry_records.at[i - 1, "focus"] and not carry_records.at[i, "focus"]:
                    continue

                else:
                    carry_records.at[i, "focus"] = True

                    start_idx = carry_records.at[i, "start_idx"]
                    end_idx = carry_records.at[i, "end_idx"]
                    if i > carry_records.index[0] and i < carry_records.index[-1]:
                        start_idx += min(5, carry_records.at[i, "trans_dur"] - 1)
                    if i == carry_records.index[-1]:
                        end_idx -= 1

                    carrier = carry_records.at[i, "carrier"]
                    output.at[start_idx, "focus_x"] = tracking.at[start_idx, f"{carrier}_x"]
                    output.at[start_idx, "focus_y"] = tracking.at[start_idx, f"{carrier}_y"]
                    output.at[end_idx, "focus_x"] = tracking.at[end_idx, f"{carrier}_x"]
                    output.at[end_idx, "focus_y"] = tracking.at[end_idx, f"{carrier}_y"]

            output[["focus_x", "focus_y"]] = output[["focus_x", "focus_y"]].interpolate(limit_direction="both")
            output["focus_x"] = output["focus_x"].clip(0, 108)
            output["focus_y"] = output["focus_y"].clip(18, 54)

        return output

    def run(self, method="ball_accel", max_accel=5, thres_touch=0.1, thres_carry=0.4, evaluate=False):
        # Note: evaluate argument is retained for API compatibility but ignored.
        carry_records_list = []

        for phase in tqdm(self.tracking["phase_id"].unique(), desc="Postprocessing"):
            phase_tracking = self.tracking[self.tracking["phase_id"] == phase].copy()
            phase_poss_probs = self.poss_probs.loc[phase_tracking.index]

            players = self.player_ids
            for p in players:
                if p in phase_poss_probs.columns:
                    phase_poss_probs[p] = signal.savgol_filter(phase_poss_probs[p], window_length=11, polyorder=2)
            self.poss_probs.loc[phase_tracking.index] = phase_poss_probs

            if method == "ball_accel":
                episodes = [e for e in phase_tracking["episode_id"].unique() if e > 0]

                for episode in episodes:
                    ep_tracking = self.tracking[self.tracking["episode_id"] == episode].copy()
                    ball_feats = Postprocessor.calc_ball_features(ep_tracking)
                    ball_dists = Postprocessor.calc_ball_dists(ep_tracking, players)
                    carry_records = Postprocessor.detect_carries_by_accel(ball_feats, ball_dists, max_accel=max_accel)
                    carry_records["phase"] = phase
                    carry_records_list.append(carry_records)

            elif method == "poss_score":
                ball_dists = Postprocessor.calc_ball_dists(phase_tracking, players)
                poss_scores = Postprocessor.calc_poss_scores(phase_poss_probs, ball_dists, players)
                carry_records = Postprocessor.detect_carries_by_poss_score(poss_scores, thres_touch, thres_carry)
                carry_records["phase"] = phase
                carry_records_list.append(carry_records)

            elif method == "imputation":
                episodes = [e for e in phase_tracking["episode_id"].unique() if e > 0]

                for episode in episodes:
                    ep_tracking = self.tracking[self.tracking["episode_id"] == episode].copy()
                    ep_poss_probs = self.poss_probs.loc[ep_tracking.index]

                    ep_tracking["bias_x"] = ep_tracking["pred_ball_x"] - ep_tracking["masked_ball_x"]
                    ep_tracking["bias_y"] = ep_tracking["pred_ball_y"] - ep_tracking["masked_ball_y"]
                    ep_tracking["bias_x"] = ep_tracking["bias_x"].interpolate(limit_direction="both")
                    ep_tracking["bias_y"] = ep_tracking["bias_y"].interpolate(limit_direction="both")
                    ep_tracking["pred_ball_x"] = ep_tracking["pred_ball_x"] - ep_tracking["bias_x"]
                    ep_tracking["pred_ball_y"] = ep_tracking["pred_ball_y"] - ep_tracking["bias_y"]

                    ball_dists = Postprocessor.calc_ball_dists(ep_tracking, players)
                    ball_dists[players] = ball_dists[players].where(ep_poss_probs > 0.1, 100)
                    ball_dists["idxmin"] = ball_dists[players].idxmin(axis=1)
                    ball_dists["min"] = ball_dists[players].min(axis=1)
                    ep_tracking["carrier"] = np.where(ball_dists["min"] < 0.5, ball_dists["idxmin"], np.nan)

                    poss_scores = Postprocessor.calc_poss_scores(ep_poss_probs, ball_dists, players)
                    carry_records = Postprocessor.detect_carries_for_imputation(ep_tracking, poss_scores, thres_touch)
                    carry_records["phase"] = phase
                    carry_records_list.append(carry_records)

        carry_records = pd.concat(carry_records_list, ignore_index=True)
        self.carry_records = carry_records
        pred_events = self._carry_records_to_events(carry_records)
        if not evaluate:
            return pred_events

        pred_edges = self._events_to_edge_seq(pred_events)
        true_sender = self.tracking["player_id"].ffill().bfill()
        true_receiver = self.tracking["player_id"].bfill().ffill()

        valid_mask = (
            (self.tracking["episode_id"] > 0)
            & true_sender.notna()
            & true_receiver.notna()
            & pred_edges["sender_id"].notna()
            & pred_edges["receiver_id"].notna()
        )
        if valid_mask.any():
            sender_acc = (pred_edges.loc[valid_mask, "sender_id"] == true_sender.loc[valid_mask]).mean()
            receiver_acc = (pred_edges.loc[valid_mask, "receiver_id"] == true_receiver.loc[valid_mask]).mean()
            edge_acc = (
                (pred_edges.loc[valid_mask, "sender_id"] == true_sender.loc[valid_mask])
                & (pred_edges.loc[valid_mask, "receiver_id"] == true_receiver.loc[valid_mask])
            ).mean()
        else:
            sender_acc = 0.0
            receiver_acc = 0.0
            edge_acc = 0.0

        stats = {
            "edge_acc": float(edge_acc),
            "sender_acc": float(sender_acc),
            "receiver_acc": float(receiver_acc),
            "n_frames": int(valid_mask.sum()),
        }
        return pred_events, stats

    @staticmethod
    def detect_false_poss_segments(
        tracking: pd.DataFrame, true_col: str = "player_id", pred_col: str = "pred_player_id"
    ) -> pd.DataFrame:
        true_poss = tracking[true_col].bfill().ffill()
        pred_poss = tracking[pred_col]

        false_idxs = true_poss.index[true_poss != pred_poss]
        time_diffs = pd.Series(false_idxs.diff().fillna(10).values, index=false_idxs)
        segment_ids = (time_diffs > 3).astype(int).cumsum().rename("segment_id").reset_index()

        start_idxs = segment_ids.groupby("segment_id")["index"].first().rename("start_idx")
        end_idxs = segment_ids.groupby("segment_id")["index"].last().rename("end_idx")
        false_segments = pd.concat([start_idxs, end_idxs], axis=1)

        false_segments["miss"] = False
        false_segments["false_alarm"] = False

        for i in false_segments.index:
            i0 = false_segments.at[i, "start_idx"]
            i1 = false_segments.at[i, "end_idx"]

            true_players = true_poss.loc[i0:i1].unique()
            pred_players = pred_poss.loc[i0:i1].unique()
            true_players_ext = true_poss.loc[i0 - 10 : i1 + 10].unique()
            pred_players_ext = pred_poss.loc[i0 - 10 : i1 + 10].unique()

            false_segments.at[i, "miss"] = len(set(true_players) - set(pred_players_ext)) != 0
            false_segments.at[i, "false_alarm"] = len(set(pred_players) - set(true_players_ext)) != 0

        return false_segments
