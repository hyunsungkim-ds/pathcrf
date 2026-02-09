from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from tqdm import tqdm

import datatools.matplotsoccer as mps
from datatools import config, utils

OUTSIDE_NODES = {
    "out_left": (0.0, config.PITCH_Y / 2),
    "out_right": (config.PITCH_X, config.PITCH_Y / 2),
    "out_bottom": (config.PITCH_X / 2, 0.0),
    "out_top": (config.PITCH_X / 2, config.PITCH_Y),
}


def _is_player(node: object) -> bool:
    return isinstance(node, str) and (node.startswith("home_") or node.startswith("away_"))


def _get_node_xy(tracking: pd.DataFrame, frame_id: int, node: str) -> Tuple[float, float]:
    if _is_player(node):
        return (float(tracking.at[frame_id, f"{node}_x"]), float(tracking.at[frame_id, f"{node}_y"]))
    else:
        return OUTSIDE_NODES.get(node, (np.nan, np.nan))


def _overwrite_short_self_loops(edge_tuples: pd.Series, min_dur: int) -> pd.Series:
    if min_dur == 0 or edge_tuples.empty:
        return edge_tuples

    adjusted = edge_tuples.copy()
    event_id = edge_tuples.ne(edge_tuples.shift()).cumsum()

    for _, group in edge_tuples.groupby(event_id):
        edge = group.iloc[0]
        if not edge[0] == edge[1] or len(group) >= min_dur:
            continue
        last_pos = edge_tuples.index.get_loc(group.index[-1])
        if last_pos + 1 >= len(edge_tuples):
            continue
        next_edge = edge_tuples.iloc[last_pos + 1]
        adjusted.loc[group.index] = [next_edge] * len(group)

    return adjusted


def edge_probs_to_seq(edge_probs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-frame edge probabilities (columns "src-dst") into edge_src/edge_dst labels.

    This matches the use_crf=True output format (edge_src, edge_dst) but uses per-frame argmax.
    """
    if edge_probs is None:
        raise ValueError("edge_probs must be a DataFrame, got None.")
    if edge_probs.empty:
        return pd.DataFrame(index=edge_probs.index, columns=["edge_src", "edge_dst"])

    edge_cols = list(edge_probs.columns)
    if not edge_cols:
        return pd.DataFrame(index=edge_probs.index, columns=["edge_src", "edge_dst"])

    src_list: List[str] = []
    dst_list: List[str] = []
    for col in edge_cols:
        if not isinstance(col, str) or "-" not in col:
            raise ValueError(f"Invalid edge column name: {col!r}. Expected format 'src-dst'.")
        src, dst = col.split("-", 1)
        src_list.append(src)
        dst_list.append(dst)

    values = edge_probs.to_numpy(dtype=float, copy=False)
    valid_ents = np.isfinite(values)
    valid_rows = valid_ents.any(axis=1)

    argmax_idx = np.where(valid_ents, values, 0.0).argmax(axis=1)
    argmax_idx[~valid_rows] = 0

    edge_src = np.array(src_list, dtype=object)[argmax_idx]
    edge_dst = np.array(dst_list, dtype=object)[argmax_idx]
    edge_src = np.where(valid_rows, edge_src, np.nan)
    edge_dst = np.where(valid_rows, edge_dst, np.nan)

    return pd.DataFrame({"edge_src": edge_src, "edge_dst": edge_dst}, index=edge_probs.index)


def detect_events(tracking: pd.DataFrame, edge_seq: pd.DataFrame, min_dur: int = 10) -> pd.DataFrame:
    """
    Detect events from tracking and predicted edge sequences.

    Output columns:
        frame_id, period_id, episode_id, timestamp, player_id, event_type, start_x, start_y, end_x, end_y
    """
    tracking = tracking.copy()
    edge_seq = edge_seq.copy()

    if "frame_id" in tracking.columns:
        tracking = tracking.set_index("frame_id").sort_index()
    if "frame_id" in edge_seq.columns:
        edge_seq = edge_seq.set_index("frame_id").sort_index()
    if not {"edge_src", "edge_dst"}.issubset(edge_seq.columns):
        edge_seq = edge_probs_to_seq(edge_seq)
    assert (tracking.index == edge_seq.index).all()

    events: List[Dict[str, object]] = []

    for episode_id in tqdm(sorted(tracking["episode_id"].unique())):
        if episode_id == 0:
            continue

        ep_tracking = tracking[tracking["episode_id"] == episode_id]
        ep_edges = edge_seq.loc[ep_tracking.index]
        edge_tuples = pd.Series(list(zip(ep_edges["edge_src"], ep_edges["edge_dst"])), index=ep_edges.index)
        edge_tuples = _overwrite_short_self_loops(edge_tuples, min_dur)

        edge_codes = pd.Series(pd.factorize(edge_tuples)[0], index=edge_tuples.index)
        change_mask = edge_codes.diff().fillna(0) != 0
        if not edge_tuples.empty:
            first_edge = edge_tuples.iloc[0]
            change_mask.iloc[0] = first_edge[0] != first_edge[1]
        change_frames = edge_tuples.index[change_mask]

        ep_events: List[Dict[str, object]] = []

        for frame_id in change_frames:
            prev_edge = edge_tuples.shift(1).at[frame_id]
            curr_edge = edge_tuples.at[frame_id]
            if curr_edge == prev_edge:
                continue

            if curr_edge[0] == curr_edge[1] and prev_edge is None:
                continue
            elif curr_edge[0] == curr_edge[1]:  # Self-loop where prev_edge is not None
                event_type = "control" if _is_player(curr_edge[0]) else "out"
            elif _is_player(curr_edge[0]):  # Edge to another node
                event_type = "kick"
            else:
                print(f"Invalid non-self edge from an outside node at frame {frame_id}")
                continue

            start_x, start_y = _get_node_xy(tracking, frame_id, curr_edge[0])
            ep_events.append(
                {
                    "frame_id": frame_id,
                    "period_id": tracking.at[frame_id, "period_id"],
                    "episode_id": tracking.at[frame_id, "episode_id"],
                    "timestamp": utils.seconds_to_timestamp(tracking.at[frame_id, "timestamp"]),
                    "player_id": curr_edge[0],
                    "receiver_id": curr_edge[1],
                    "event_type": event_type,
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": np.nan,
                    "end_y": np.nan,
                }
            )

        if not ep_events:
            continue

        for i in range(len(ep_events) - 1):
            next_frame = ep_events[i + 1]["frame_id"]
            receiver_id = ep_events[i]["receiver_id"]
            end_x, end_y = _get_node_xy(tracking, next_frame, receiver_id)
            ep_events[i]["end_x"] = end_x
            ep_events[i]["end_y"] = end_y

        valid_edges = ep_edges.dropna(subset=["edge_src", "edge_dst"])
        last_frame = int(valid_edges.index[-1])
        last_receiver_id = str(valid_edges.iloc[-1]["edge_dst"])
        end_x, end_y = _get_node_xy(tracking, last_frame, last_receiver_id)
        ep_events[-1]["end_x"] = end_x
        ep_events[-1]["end_y"] = end_y

        events.extend(ep_events)

    return pd.DataFrame(events)


def classify_setpieces(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    x_col = "x" if "x" in events.columns else "start_x"
    y_col = "y" if "y" in events.columns else "start_y"

    origin_x = float(mps.pitch_config["origin_x"])
    origin_y = float(mps.pitch_config["origin_y"])
    pitch_length = float(mps.pitch_config["length"])
    pitch_width = float(mps.pitch_config["width"])

    corners = [
        (origin_x, origin_y),
        (origin_x, origin_y + pitch_width),
        (origin_x + pitch_length, origin_y),
        (origin_x + pitch_length, origin_y + pitch_width),
    ]
    corner_margin = 2.0

    gk_box_length = float(mps.pitch_config["six_yard_box_length"])
    gk_box_width = float(mps.pitch_config["six_yard_box_width"])
    gk_box_x_left = origin_x + gk_box_length
    gk_box_x_right = origin_x + pitch_length - gk_box_length
    gk_box_y_min = origin_y + pitch_width / 2.0 - gk_box_width / 2.0
    gk_box_y_max = origin_y + pitch_width / 2.0 + gk_box_width / 2.0
    goalkick_margin = 3.0

    for idx in events.index:
        x = float(events.at[idx, x_col])
        y = float(events.at[idx, y_col])
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        player_id = events.at[idx, "player_id"] if "player_id" in events.columns else None
        is_home = isinstance(player_id, str) and player_id.startswith("home_")
        is_away = isinstance(player_id, str) and player_id.startswith("away_")

        if is_home and any((x - cx) ** 2 + (y - cy) ** 2 <= corner_margin**2 for cx, cy in corners[2:]):
            events.at[idx, "event_type"] = "corner"
            continue
        if is_away and any((x - cx) ** 2 + (y - cy) ** 2 <= corner_margin**2 for cx, cy in corners[:2]):
            events.at[idx, "event_type"] = "corner"
            continue

        if y < origin_y + 1.0 or y > origin_y + pitch_width - 1.0:
            events.at[idx, "event_type"] = "throw_in"
            continue

        near_x_left = (gk_box_x_left - goalkick_margin) <= x <= (gk_box_x_left + goalkick_margin)
        near_x_right = (gk_box_x_right - goalkick_margin) <= x <= (gk_box_x_right + goalkick_margin)
        near_y = gk_box_y_min <= y <= gk_box_y_max
        if near_y and ((is_home and near_x_left) or (is_away and near_x_right)):
            events.at[idx, "event_type"] = "goalkick"

    return events


def classify_episode_starts(events: pd.DataFrame) -> pd.DataFrame:
    """Extract episode start events, classify set pieces, and update event_type."""
    events = events.copy()
    episode_starts = classify_setpieces(events.groupby("episode_id").head(1))
    events.loc[episode_starts.index, "event_type"] = episode_starts["event_type"]
    return events


def _team_color(player_id: object) -> str:
    if isinstance(player_id, str):
        if player_id.startswith("home_"):
            return "tab:red"
        if player_id.startswith("away_"):
            return "tab:blue"
    return "tab:gray"


def _render_event_table(
    events: pd.DataFrame,
    fig: plt.Figure,
    table_ax: plt.Axes,
    alignments: Optional[pd.DataFrame] = None,
) -> None:
    table_ax.axis("off")
    table_ax.set_xlim(0, 1)
    table_ax.set_ylim(0, 1)

    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    ax_h_px = table_ax.get_position().height * fig_h_px
    row_h_px = 15 * 1.2
    header_lines = 2
    available_px = max(0, ax_h_px - row_h_px * header_lines)
    max_rows = max(1, int(available_px / row_h_px))
    line_step = row_h_px / ax_h_px if ax_h_px > 0 else 0.05

    header_text = "timestamp | sender  | receiver | event_type"
    header_artist = table_ax.text(
        0,
        1,
        header_text,
        fontsize=13,
        fontfamily="monospace",
        ha="left",
        va="top",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = header_artist.get_window_extent(renderer=renderer)
    x0, _ = table_ax.transAxes.inverted().transform((bbox.x0, bbox.y0))
    x1, _ = table_ax.transAxes.inverted().transform((bbox.x1, bbox.y0))
    header_len = max(len(header_text), 1)
    char_w = (x1 - x0) / header_len
    base_x = x0
    table_ax.plot(
        [x0, x1],
        [1 - line_step * 1.3, 1 - line_step * 1.3],
        transform=table_ax.transAxes,
        color="black",
        linewidth=1.0,
        clip_on=False,
    )

    events = events.copy()
    display_ts = events["timestamp"].astype(str).str.strip()

    sender_arr = events["player_id"].to_numpy()
    receiver_arr = events["receiver_id"].to_numpy()
    event_types = events["event_type"].to_numpy()

    if "episode_id" in events.columns:
        same_episode = events["episode_id"].shift(1) == events["episode_id"]
    else:
        same_episode = pd.Series(True, index=events.index)
    invalid_mask = (events["receiver_id"].shift(1) != events["player_id"]) & same_episode
    if len(invalid_mask) > 0:
        invalid_mask.iloc[0] = False

    events_index = events.index.to_numpy()
    align_mismatch = None
    if alignments is not None and not alignments.empty:
        required_cols = {"pred_index", "same_player", "same_receiver", "same_type"}
        if required_cols.issubset(alignments.columns):
            align = alignments.copy()
            if "episode_id" in align.columns and "episode_id" in events.columns:
                align = align[align["episode_id"].isin(events["episode_id"].unique())]
            pred_index = pd.to_numeric(align["pred_index"], errors="coerce")
            align = align[pred_index.notna()].copy()
            if not align.empty:
                align["pred_index"] = pred_index[pred_index.notna()].astype(int)
                mismatch = ~align[["same_player", "same_receiver", "same_type"]].all(axis=1)
                align_map = dict(zip(align["pred_index"].astype(int), mismatch.astype(bool)))
                align_mismatch = np.array([align_map.get(idx, False) for idx in events_index], dtype=bool)

    start_y = 1 - line_step * 1.6
    n_rows = min(max_rows, len(events))
    start_idx = max(0, len(events) - n_rows)
    row_positions: Dict[int, float] = {}
    sender_display_width = 7
    receiver_display_width = 8
    sender_highlight_width = 7
    receiver_highlight_width = 7
    sender_offset = 9 + 3
    receiver_offset = sender_offset + sender_display_width + 3

    def _fmt_pid(pid: object) -> str:
        if pid is None:
            return ""
        if _is_player(pid):
            return str(pid)
        return str(pid).split("_")[-1]

    for i in range(n_rows):
        idx = start_idx + i
        time_str = display_ts.iloc[idx]
        y_pos = start_y - i * line_step
        row_positions[idx] = y_pos

        sender_id = sender_arr[idx] if idx < len(sender_arr) else ""
        sender_id = _fmt_pid(sender_id)
        receiver_id = receiver_arr[idx] if idx < len(receiver_arr) else ""
        receiver_id = _fmt_pid(receiver_id)

        event_type = event_types[idx] if idx < len(event_types) else ""
        row_text = (
            f"{time_str:9s} | {sender_id:{sender_display_width}s} | "
            f"{receiver_id:{receiver_display_width}s} | {event_type}"
        )
        is_invalid = bool(invalid_mask.iloc[idx])
        is_mismatch = bool(align_mismatch[idx]) if align_mismatch is not None else False
        if is_invalid or is_mismatch:
            bbox = {"facecolor": "gold", "edgecolor": "none", "boxstyle": "square,pad=0.1"}
        else:
            bbox = None
        table_ax.text(
            base_x,
            y_pos,
            row_text,
            fontsize=13,
            fontfamily="monospace",
            ha="left",
            va="top",
            bbox=bbox,
        )

    red_bbox = {"facecolor": "red", "edgecolor": "none", "boxstyle": "square,pad=0.1"}
    for idx, y_pos in row_positions.items():
        if not bool(invalid_mask.iloc[idx]):
            continue
        sender_id = sender_arr[idx] if idx < len(sender_arr) else ""
        sender_text = f"{_fmt_pid(sender_id):{sender_highlight_width}s}"
        table_ax.text(
            base_x + char_w * sender_offset,
            y_pos,
            sender_text,
            fontsize=13,
            fontfamily="monospace",
            ha="left",
            va="top",
            color="white",
            bbox=red_bbox,
            zorder=5,
        )

        prev_idx = idx - 1
        if prev_idx not in row_positions:
            continue
        prev_receiver = receiver_arr[prev_idx] if prev_idx < len(receiver_arr) else ""
        receiver_text = f"{_fmt_pid(prev_receiver):{receiver_highlight_width}s}"
        table_ax.text(
            base_x + char_w * receiver_offset,
            row_positions[prev_idx],
            receiver_text,
            fontsize=13,
            fontfamily="monospace",
            ha="left",
            va="top",
            color="white",
            bbox=red_bbox,
            zorder=5,
        )


def plot_events(
    events: pd.DataFrame,
    alignments: Optional[pd.DataFrame] = None,
    focus_box: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10.8, 7.2),
    show_table: bool = True,
):
    events = events.copy()

    if "receiver_id" not in events.columns:
        next_players = events.groupby("episode_id")["player_id"].shift(-1)
        events["receiver_id"] = next_players.fillna(events["player_id"])

    if fig is None or ax is None:
        if show_table:
            fig = plt.figure(figsize=(figsize[0] * 1.35, figsize[1]))
            gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.5], wspace=0.05)
            ax = fig.add_subplot(gs[0, 0])
            table_ax = fig.add_subplot(gs[0, 1])
        else:
            fig, ax = plt.subplots(figsize=figsize)
            table_ax = None
    else:
        table_ax = None

    mps.field("green", config.PITCH_X, config.PITCH_Y, fig, ax, show=False)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()

    if focus_box is not None:
        (tl_x, tl_y), (br_x, br_y) = focus_box
        x_min = min(tl_x, br_x)
        x_max = max(tl_x, br_x)
        y_min = min(tl_y, br_y)
        y_max = max(tl_y, br_y)
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2.0,
            edgecolor="tab:red",
            facecolor="none",
            zorder=5,
        )
        ax.add_patch(rect)

    end_offset = 1.2
    for _, row in events.iterrows():
        start_x, start_y = float(row["start_x"]), float(row["start_y"])
        end_x, end_y = row.get("end_x", np.nan), row.get("end_y", np.nan)
        if not np.isfinite(start_x) or not np.isfinite(start_y):
            continue
        color = _team_color(row.get("player_id"))

        if np.isfinite(end_x) and np.isfinite(end_y):
            end_x = float(end_x)
            end_y = float(end_y)
            dx = end_x - start_x
            dy = end_y - start_y
            dist = np.hypot(dx, dy)
            if dist > 1e-6:
                scale = max((dist - end_offset) / dist, 0.0)
                adj_end_x = start_x + dx * scale
                adj_end_y = start_y + dy * scale
            else:
                adj_end_x = end_x
                adj_end_y = end_y

            if str(row.get("event_type", "")).lower() == "control":
                ax.plot(
                    [start_x, adj_end_x],
                    [start_y, adj_end_y],
                    linestyle="--",
                    color=color,
                    lw=2,
                    alpha=0.8,
                    zorder=2,
                )
            else:
                arrow = patches.FancyArrowPatch(
                    (start_x, start_y),
                    (adj_end_x, adj_end_y),
                    arrowstyle="->",
                    mutation_scale=14,
                    color=color,
                    lw=2,
                    alpha=0.8,
                    zorder=2,
                )
                ax.add_patch(arrow)

        if _is_player(row["player_id"]):
            ax.scatter(
                [start_x],
                [start_y],
                s=600,
                c=color,
                linewidths=1.5,
                zorder=3,
            )

            ax.text(
                start_x,
                start_y,
                row["player_id"].split("_")[-1],
                ha="center",
                va="center",
                color="white",
                fontsize=20,
                fontweight="bold",
                zorder=4,
            )

    if show_table and table_ax is not None:
        _render_event_table(events, fig, table_ax, alignments=alignments)

    return fig, ax
