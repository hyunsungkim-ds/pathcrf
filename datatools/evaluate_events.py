from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from datatools import config, shot_detection, utils

SIMPLE_TYPES = ["kick", "control", "out"]
SHOT_TYPES = ["shot", "shot_block"]
OUTGOING_TYPES = config.PASS_LIKE_OPEN + config.SET_PIECE + ["tackle", "bad_touch", "dispossessed", "kick"]
INCOMING_TYPES = config.INCOMING

DEFAULT_SCORING = {
    "time_weight": 25.0,
    "dist_weight": 25.0,
    "player_weight": 25.0,
    "type_match_weight": 25.0,
    "shot_pass_weight": 20.0,
    "max_td": 3.0,
    "max_dist": 25.0,
}


def relabel_events(events: pd.DataFrame, drop_fouls: bool = True, keep_shots: bool = False) -> pd.DataFrame:
    """
    Reclassify events into pass/control/out using the next event in the same episode.

    Rules:
    - If event_type is "out" -> "out"
    - Else if next event is performed by the same player -> "control"
    - Else if next event is performed by another player -> "kick"
    Events without a next event (and not "out") are dropped.
    """
    events = events.copy()
    if drop_fouls:
        events = events[events["event_type"] != "foul"].copy()

    if "start_x" in events.columns:
        events = events.rename(columns={"start_x": "x", "start_y": "y"}).copy()

    events["seconds"] = events["timestamp"].apply(utils.timestamp_to_seconds)
    events = events.sort_values("frame_id", ignore_index=True, kind="stable")

    next_player_ids = events.groupby("episode_id")["player_id"].shift(-1)

    # Auto-detect shots if requested and no labels exist
    # (Note: relabel_events renames start_x to x, so we rely on caller or check both)
    # But optimal place for detection is BEFORE renaming.
    # Here we assume detection already happened IF columns exist.
    # If not, shot_mask will be empty unless it's GT.

    shot_mask = events["event_type"].isin(SHOT_TYPES) if keep_shots else pd.Series(False, index=events.index)
    if keep_shots and "is_pred_shot_union" in events.columns:
        shot_mask = shot_mask | events["is_pred_shot_union"].astype(bool)

    simple_types = pd.Series(index=events.index, dtype=object)
    simple_types[events["event_type"] == "out"] = "out"

    non_out_mask = events["event_type"] != "out"
    same_player = events["player_id"] == next_player_ids
    simple_types[non_out_mask & same_player] = "control"
    simple_types[non_out_mask & ~same_player & next_player_ids.notna()] = "kick"

    last_mask = non_out_mask & next_player_ids.isna()
    simple_types[last_mask & events["event_type"].isna()] = "kick"
    simple_types[last_mask & events["event_type"].isin(OUTGOING_TYPES)] = "kick"
    simple_types[last_mask & events["event_type"].isin(INCOMING_TYPES)] = "control"

    if keep_shots:
        simple_types[shot_mask] = "shot"

    events["event_type"] = simple_types
    if keep_shots:
        events = events[events["event_type"].isin(SIMPLE_TYPES + SHOT_TYPES)].copy()
    else:
        events = events[events["event_type"].isin(SIMPLE_TYPES)].copy()
    return events


def _pair_score(true_event: pd.Series, pred_event: pd.Series, scoring: dict | None = None) -> float:
    cfg = {**DEFAULT_SCORING, **(scoring or {})}

    max_td = cfg.get("max_td", 3.0)
    max_dist = cfg.get("max_dist", 25.0)

    if "frame_id" in true_event and "frame_id" in pred_event and true_event["frame_id"] == pred_event["frame_id"]:
        time_score = cfg["time_weight"]
    else:
        td = abs(true_event["seconds"] - pred_event["seconds"]).round(2)
        time_score = cfg["time_weight"] * (1.0 - min(td, max_td) / max_td)
        if td >= max_td:
            time_score = 0.0

    dist = ((true_event["x"] - pred_event["x"]) ** 2 + (true_event["y"] - pred_event["y"]) ** 2) ** 0.5
    xy_score = cfg["dist_weight"] * (1.0 - min(dist, max_dist) / max_dist)

    player_score = cfg["player_weight"] if true_event["player_id"] == pred_event["player_id"] else 0.0

    true_type = true_event["event_type"]
    pred_type = pred_event["event_type"]
    shot_pass_bridge = (true_type == "shot" and pred_type == "pass") or (pred_type == "shot" and true_type == "pass")

    if true_type == pred_type:
        type_score = cfg["type_match_weight"]
    elif shot_pass_bridge:
        type_score = cfg["shot_pass_weight"]
    else:
        type_score = 0.0
    return time_score + xy_score + player_score + type_score


def needleman_wunsch(
    true_events: pd.DataFrame,
    pred_events: pd.DataFrame,
    gap_penalty=-10.0,
    scoring: dict | None = None,
):
    n_true = len(true_events)
    n_pred = len(pred_events)
    if n_true == 0 and n_pred == 0:
        return []
    if n_true == 0:
        return [(None, j, "gap_true") for j in range(n_pred)]
    if n_pred == 0:
        return [(i, None, "gap_pred") for i in range(n_true)]

    dp = [[0.0] * (n_pred + 1) for _ in range(n_true + 1)]
    ptr = [[0] * (n_pred + 1) for _ in range(n_true + 1)]

    for i in range(1, n_true + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
        ptr[i][0] = 1
    for j in range(1, n_pred + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty
        ptr[0][j] = 2

    for i in range(1, n_true + 1):
        true_event = true_events.iloc[i - 1]
        for j in range(1, n_pred + 1):
            pred_event = pred_events.iloc[j - 1]
            match_score = dp[i - 1][j - 1] + _pair_score(true_event, pred_event, scoring)
            delete_score = dp[i - 1][j] + gap_penalty
            insert_score = dp[i][j - 1] + gap_penalty

            if match_score >= delete_score and match_score >= insert_score:
                dp[i][j] = match_score
                ptr[i][j] = 0
            elif delete_score >= insert_score:
                dp[i][j] = delete_score
                ptr[i][j] = 1
            else:
                dp[i][j] = insert_score
                ptr[i][j] = 2

    pairs = []
    i, j = n_true, n_pred
    while i > 0 or j > 0:
        move = ptr[i][j]
        if i > 0 and j > 0 and move == 0:
            pairs.append((i - 1, j - 1, "match"))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 1):
            pairs.append((i - 1, None, "gap_pred"))
            i -= 1
        else:
            pairs.append((None, j - 1, "gap_true"))
            j -= 1

    pairs.reverse()
    return pairs


def add_alignment_details(
    alignments: pd.DataFrame, true_events: pd.DataFrame, pred_events: pd.DataFrame
) -> pd.DataFrame:
    result = alignments.copy()
    true_events = true_events.copy()

    if "receiver_id" not in true_events.columns:
        next_players = true_events.groupby("episode_id")["player_id"].shift(-1)
        true_events["receiver_id"] = next_players.fillna(true_events["player_id"])

    result["_true_frame_id"] = result["true_index"].map(true_events["frame_id"])
    result["_pred_frame_id"] = result["pred_index"].map(pred_events["frame_id"])
    result["_true_player_id"] = result["true_index"].map(true_events["player_id"])
    result["_pred_player_id"] = result["pred_index"].map(pred_events["player_id"])
    result["_true_receiver_id"] = result["true_index"].map(true_events["receiver_id"])
    result["_pred_receiver_id"] = result["pred_index"].map(pred_events["receiver_id"])
    result["_true_event_type"] = result["true_index"].map(true_events["event_type"])
    result["_pred_event_type"] = result["pred_index"].map(pred_events["event_type"])
    result["_true_x"] = result["true_index"].map(true_events["x"])
    result["_true_y"] = result["true_index"].map(true_events["y"])
    result["_pred_x"] = result["pred_index"].map(pred_events["x"])
    result["_pred_y"] = result["pred_index"].map(pred_events["y"])

    result["frame_diff"] = (result["_true_frame_id"] - result["_pred_frame_id"]).abs()
    result["same_player"] = result["_true_player_id"] == result["_pred_player_id"]
    result["same_receiver"] = result["_true_receiver_id"] == result["_pred_receiver_id"]
    result["same_type"] = result["_true_event_type"] == result["_pred_event_type"]

    dist_x = result["_true_x"] - result["_pred_x"]
    dist_y = result["_true_y"] - result["_pred_y"]
    result["xy_dist"] = (dist_x**2 + dist_y**2) ** 0.5
    result.loc[result["_pred_event_type"] == "out", "xy_dist"] = np.nan

    result = result.drop(columns=[c for c in result.columns if c[:5] in ["_true", "_pred"]])
    return result


def align_true_and_pred(
    true_events: pd.DataFrame,
    pred_events: pd.DataFrame,
    threshold: float = 1.0,
    keep_shots: bool = False,
    gap_penalty: float = -10.0,
    scoring_config: dict | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Auto-detect shots on Prediction data if needed
    if keep_shots and "is_pred_shot_union" not in pred_events.columns:
        # Check if we have necessary columns for detection
        required = ["start_x", "start_y", "player_id"]
        if all(col in pred_events.columns for col in required):
            print("  [evaluate_events] Auto-detecting shots in prediction data...")
            pred_events = shot_detection.detect_shots(pred_events)

    true_events = relabel_events(true_events, drop_fouls=True, keep_shots=keep_shots)
    pred_events = relabel_events(pred_events, drop_fouls=False, keep_shots=keep_shots)

    event_types = SIMPLE_TYPES + SHOT_TYPES if keep_shots else SIMPLE_TYPES
    type_stats = {t: {"true_count": 0, "pred_count": 0, "matched": 0} for t in event_types}
    true_counts = true_events["event_type"].value_counts()
    pred_counts = pred_events["event_type"].value_counts()
    for event_type in event_types:
        type_stats[event_type]["true_count"] = int(true_counts.get(event_type, 0))
        type_stats[event_type]["pred_count"] = int(pred_counts.get(event_type, 0))

    alignments = []
    episode_ids = np.sort(pd.Index(true_events["episode_id"]).union(pred_events["episode_id"]).unique())
    for episode_id in tqdm(episode_ids):
        ep_true = true_events[true_events["episode_id"] == episode_id]
        ep_pred = pred_events[pred_events["episode_id"] == episode_id]

        ep_alignment = needleman_wunsch(ep_true, ep_pred, gap_penalty=gap_penalty, scoring=scoring_config)
        for true_idx, pred_idx, move in ep_alignment:
            true_event = ep_true.iloc[true_idx] if true_idx is not None else None
            pred_event = ep_pred.iloc[pred_idx] if pred_idx is not None else None

            alignments.append(
                {
                    "episode_id": episode_id,
                    "true_index": (int(true_event.name) if true_event is not None else np.nan),
                    "pred_index": (int(pred_event.name) if pred_event is not None else np.nan),
                    "move": move,
                }
            )

            if (
                true_event is not None
                and pred_event is not None
                and true_event["player_id"] == pred_event["player_id"]
                and true_event["event_type"] == pred_event["event_type"]
                and abs(true_event["seconds"] - pred_event["seconds"]) <= threshold
            ):
                type_stats[true_event["event_type"]]["matched"] += 1

    stats = []
    for event_type in event_types:
        stats.append(
            {
                "event_type": event_type,
                "pred_count": int(type_stats[event_type]["pred_count"]),
                "true_count": int(type_stats[event_type]["true_count"]),
                "matched": int(type_stats[event_type]["matched"]),
            }
        )

    stats = pd.DataFrame(stats).set_index("event_type")
    stats.loc["total"] = stats.sum()

    stats["precision"] = np.where(stats["pred_count"] > 0, stats["matched"] / stats["pred_count"], 0.0)
    stats["recall"] = np.where(stats["true_count"] > 0, stats["matched"] / stats["true_count"], 0.0)
    stats["f1"] = (stats[["precision", "recall"]].prod(axis=1) * 2) / (stats[["precision", "recall"]].sum(axis=1))
    stats[["precision", "recall", "f1"]] = stats[["precision", "recall", "f1"]].fillna(0).round(4)

    alignments = add_alignment_details(pd.DataFrame(alignments), true_events, pred_events)
    return stats, alignments


def evaluate_time_location(
    alignments: pd.DataFrame, thres_frames: float = 25, thres_dist: float = 5
) -> Tuple[float, float, float]:
    """
    Evaluate precision/recall/F1 using an aligned events DF.

    A pair is counted as correct if:
    - both true_index and pred_index are present, AND
    - frame_diff <= thres_frames, AND
    - xy_dist <= thres_dist
    (player/type can differ).
    """
    required = ["true_index", "pred_index", "frame_diff", "xy_dist"]
    missing = [col for col in required if col not in alignments.columns]
    if missing:
        raise ValueError(f"alignments is missing required columns: {missing}")

    true_mask = alignments["true_index"].notna()
    pred_mask = alignments["pred_index"].notna()
    pair_mask = true_mask & pred_mask

    matched = (pair_mask & (alignments["frame_diff"] <= thres_frames) & (alignments["xy_dist"] <= thres_dist)).sum()
    true_count = true_mask.sum()
    pred_count = pred_mask.sum()

    precision = float(matched / pred_count) if pred_count > 0 else 0.0
    recall = float(matched / true_count) if true_count > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return round(precision, 4), round(recall, 4), round(f1, 4)


def evaluate_time_location_grid(
    alignments: pd.DataFrame,
    thres_frames_list: Optional[List[float]] = None,
    thres_dist_list: Optional[List[float]] = None,
    visualize: bool = False,
    font_scale: float = 1.3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate precision/recall/F1 across a grid of time and distance thresholds.

    Returns three DataFrames (precision, recall, f1) with:
    - index: thres_frames
    - columns: thres_dist
    """
    if thres_frames_list is None:
        thres_frames_list = [12.5, 25, 37.5, 50]
    if thres_dist_list is None:
        thres_dist_list = [2, 4, 6, 8, 10]

    frames = [float(v) for v in thres_frames_list]
    dists = [float(v) for v in thres_dist_list]

    prec_df = pd.DataFrame(index=frames, columns=dists, dtype=float)
    recall_df = pd.DataFrame(index=frames, columns=dists, dtype=float)
    f1_df = pd.DataFrame(index=frames, columns=dists, dtype=float)

    for tf in frames:
        for td in dists:
            prec, rec, f1 = evaluate_time_location(alignments, thres_frames=tf, thres_dist=td)
            prec_df.loc[tf, td] = prec
            recall_df.loc[tf, td] = rec
            f1_df.loc[tf, td] = f1

    for df in (prec_df, recall_df, f1_df):
        df.index.name = "thres_frames"
        df.columns.name = "thres_dist"

    if visualize:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for visualize=True") from exc

        try:
            import seaborn as sns
        except ImportError:
            sns = None

        if font_scale is not None:
            plt.rcParams.update(
                {
                    "font.size": 10 * font_scale,
                    "axes.titlesize": 12 * font_scale,
                    "axes.labelsize": 11 * font_scale,
                    "xtick.labelsize": 10 * font_scale,
                    "ytick.labelsize": 10 * font_scale,
                }
            )
            if sns is not None:
                sns.set_context("notebook", font_scale=font_scale)

        titles = ["Precision", "Recall", "F1"]
        dfs = [prec_df, recall_df, f1_df]
        yticklabels = [f"{v / 25:.1f}" for v in frames]

        for title, df in zip(titles, dfs):
            fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
            if sns is not None:
                sns.heatmap(
                    df,
                    ax=ax,
                    cmap="YlGnBu",
                    vmin=0.5,
                    vmax=1,
                    annot=True,
                    fmt=".1%",
                    cbar=True,
                    xticklabels=dists,
                    yticklabels=yticklabels,
                )
            else:
                im = ax.imshow(df.values, vmin=0, vmax=1, cmap="YlGnBu")
                for i in range(len(frames)):
                    for j in range(len(dists)):
                        val = df.values[i, j]
                        ax.text(j, i, f"{val * 100:.1f}%", ha="center", va="center", color="black")
                ax.set_xticks(np.arange(len(dists)))
                ax.set_xticklabels(dists)
                ax.set_yticks(np.arange(len(frames)))
                ax.set_yticklabels(yticklabels)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # ax.set_title(title)
            ax.set_xlabel("Spatial tolerance (m)")
            ax.set_ylabel("Temporal tolerance (s)")
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
            ax.tick_params(axis="y", labelrotation=0)

            plt.show()
            plt.close(fig)

    return prec_df, recall_df, f1_df
