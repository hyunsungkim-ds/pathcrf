import os
import re
import sys
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import math

# --- Embedded Matplotsoccer Logic ---
from matplotlib.patches import Arc
from matplotlib.pyplot import cm
from scipy.optimize import linear_sum_assignment

import datatools.matplotsoccer as mps
from datatools import config, event_postprocessing, utils


def load_data(match_name="J03WR9"):
    # File Paths
    TRACKING_PATH = f"data/sportec/tracking_processed/{match_name}.parquet"
    EVENT_ACTUAL_PATH = f"data/output_events/event_rdp/{match_name}.parquet"
    EVENT_PRED_PATH = f"data/output_events/event_rdp_pred/{match_name}.parquet"

    # Check paths
    if not os.path.exists(TRACKING_PATH):
        raise FileNotFoundError(f"Tracking: {TRACKING_PATH}")
    if not os.path.exists(EVENT_ACTUAL_PATH):
        raise FileNotFoundError(f"Act Events: {EVENT_ACTUAL_PATH}")
    if not os.path.exists(EVENT_PRED_PATH):
        raise FileNotFoundError(f"Pred Events: {EVENT_PRED_PATH}")

    print("Loading data...")
    tracking = pd.read_parquet(TRACKING_PATH)
    events_act = pd.read_parquet(EVENT_ACTUAL_PATH)
    events_pred = pd.read_parquet(EVENT_PRED_PATH)

    # Remove Phase 0 (Dead time)
    tracking = tracking[tracking["phase_id"] != 0]

    # Ensure phase_id is int
    tracking["phase_id"] = tracking["phase_id"].astype(int)

    unique_phases = sorted(tracking["phase_id"].unique())
    print(f"Loaded data. Unique phases after filter: {unique_phases}")

    events_act = event_postprocessing.prepare_events(events_act, tracking)
    events_pred = event_postprocessing.prepare_events(events_pred, tracking)

    return tracking, events_act, events_pred


def get_node_positions_per_phase(tracking):
    """
    Calculate average position of each player per phase.
    **IMPROVED**: Only consider 'episode_id > 0' (Active Play).
    """
    phase_nodes = {}
    phases = sorted(tracking["phase_id"].unique())

    # Filter for active play only
    if "episode_id" in tracking.columns:
        active_tracking = tracking[tracking["episode_id"] > 0]
    else:
        active_tracking = tracking

    for phase in phases:
        p_data = active_tracking[active_tracking["phase_id"] == phase]
        if p_data.empty:
            continue

        phase_nodes[phase] = {}

        active_cols = [
            c for c in p_data.columns if c.endswith("_x") and (c.startswith("home_") or c.startswith("away_"))
        ]

        player_stats = []
        for col in active_cols:
            pid = col.replace("_x", "")
            x_col, y_col = f"{pid}_x", f"{pid}_y"

            valid = p_data[[x_col, y_col]].dropna()
            if not valid.empty:
                avg_x = valid[x_col].mean()
                avg_y = valid[y_col].mean()
                count = len(valid)
                player_stats.append({"id": pid, "x": avg_x, "y": avg_y, "count": count})

        # Top 11 Filter
        home_players = sorted(
            [p for p in player_stats if p["id"].startswith("home_")],
            key=lambda x: x["count"],
            reverse=True,
        )[:11]
        away_players = sorted(
            [p for p in player_stats if p["id"].startswith("away_")],
            key=lambda x: x["count"],
            reverse=True,
        )[:11]

        for p in home_players + away_players:
            phase_nodes[phase][p["id"]] = (p["x"], p["y"])

    return phase_nodes


def get_edges_per_phase(events, tracking, is_pred=False):
    """
    Logic: Treat `event_type in {'pass','kick'}` with same-team next event as a pass.
    """
    phase_edges = {}

    # Map Phase ID
    if "phase_id" not in events.columns:
        frame_phase_map = tracking.set_index("frame_id")["phase_id"].to_dict()
        events["phase_id"] = events["frame_id"].map(frame_phase_map)

    # Valid Phase Only
    events = events.dropna(subset=["phase_id"])
    phases = sorted(events["phase_id"].unique())

    events = events.sort_values(["frame_id"]).reset_index(drop=True)

    for phase in phases:
        phase_edges[phase] = {}
        p_events = events[events["phase_id"] == phase]

        indices = p_events.index

        for i in range(len(indices) - 1):
            curr = events.loc[indices[i]]
            next_ev = events.loc[indices[i + 1]]

            # --- [CRITICAL UPDATE] Stricter Pass Definition ---

            # 1. Episode Consistency (If available)
            # If episode changes, it's not a continuous play (e.g. foul -> free kick)
            if "episode_id" in curr and "episode_id" in next_ev:
                if curr["episode_id"] != next_ev["episode_id"]:
                    continue

            # 2. Time Gap Check
            # If gap is too large (>5s), it's likely not a direct pass
            if "timestamp" in curr and "timestamp" in next_ev:
                try:
                    time_diff = float(next_ev["timestamp"]) - float(curr["timestamp"])
                    if time_diff > 10.0:  # Threshold: 10 seconds
                        continue
                except ValueError:
                    pass

            # --------------------------------------------------

            if str(curr.get("event_type", "")).lower() not in {"pass", "kick"}:
                continue

            # Team Check
            def get_team(pid):
                if isinstance(pid, str) and "_" in pid:
                    return pid.split("_")[0]
                return None

            team_c = get_team(curr["player_id"])
            team_n = get_team(next_ev["player_id"])

            if team_c and team_n and team_c == team_n:
                # Same Team Sequence
                sender = curr["player_id"]
                receiver = next_ev["player_id"]

                # Exclude self-pass (dribble) for pass map?
                # User said: "Continuous events by same team... as PASS."
                # Usually self-pass is dribble. Pass map implies Connection.
                # Let's EXCLUDE self-loops `if sender != receiver`.
                if sender != receiver:
                    edge = (sender, receiver)
                    phase_edges[phase][edge] = phase_edges[phase].get(edge, 0) + 1

    return phase_edges


def aggregate_phases_hungarian(phase_nodes, phase_edges, team_prefix="home"):
    # (Existing Logic - Preserved)
    phases = sorted(phase_nodes.keys())
    global_nodes = {}
    current_mapping = {}
    mapping_history = {}

    if not phases:
        return {}, {}, {}

    first_phase = phases[0]
    # Filter prefix
    initial = {pid: pos for pid, pos in phase_nodes[first_phase].items() if pid.startswith(team_prefix)}

    active_pos_ids = []
    next_pos_id = 0

    for pid, pos in initial.items():
        global_nodes[next_pos_id] = {
            "sum_x": pos[0],
            "sum_y": pos[1],
            "count": 1,
            "history": {pid: 1},
        }
        current_mapping[pid] = next_pos_id
        active_pos_ids.append(next_pos_id)
        next_pos_id += 1

    mapping_history[first_phase] = current_mapping.copy()

    for i in range(len(phases) - 1):
        next_p = phases[i + 1]
        next_players = {p: pos for p, pos in phase_nodes[next_p].items() if p.startswith(team_prefix)}
        n_pids = list(next_players.keys())

        if not n_pids:
            mapping_history[next_p] = {}
            continue

        pos_coords = []
        for pid in active_pos_ids:
            node = global_nodes[pid]
            pos_coords.append((node["sum_x"] / node["count"], node["sum_y"] / node["count"]))

        cost_mtx = np.zeros((len(active_pos_ids), len(n_pids)))
        for r, p_coord in enumerate(pos_coords):
            for c, n_pid in enumerate(n_pids):
                p2 = next_players[n_pid]
                cost_mtx[r, c] = np.sqrt((p_coord[0] - p2[0]) ** 2 + (p_coord[1] - p2[1]) ** 2)

        row_ind, col_ind = linear_sum_assignment(cost_mtx)

        new_mapping = {}
        for r, c in zip(row_ind, col_ind):
            pos_id = active_pos_ids[r]
            n_pid = n_pids[c]
            new_mapping[n_pid] = pos_id

            # Update stats
            global_nodes[pos_id]["sum_x"] += next_players[n_pid][0]
            global_nodes[pos_id]["sum_y"] += next_players[n_pid][1]
            global_nodes[pos_id]["count"] += 1
            global_nodes[pos_id]["history"][n_pid] = global_nodes[pos_id]["history"].get(n_pid, 0) + 1

        current_mapping = new_mapping
        mapping_history[next_p] = current_mapping.copy()

    final_nodes = {}
    for pos_id, data in global_nodes.items():
        if data["count"] > 0:
            top_player = max(data["history"], key=data["history"].get)
            label = top_player.split("_")[1]
            final_nodes[pos_id] = {
                "x": data["sum_x"] / data["count"],
                "y": data["sum_y"] / data["count"],
                "label": label,
                "player_counts": data["history"],
            }

    return final_nodes, global_nodes, mapping_history


def aggregate_edges(phase_edges, phase_nodes, global_nodes, mapping_history, team_prefix="home"):
    final_edges = {}
    for phase, edges in phase_edges.items():
        if phase not in mapping_history:
            continue
        mapping = mapping_history[phase]

        for (u, v), count in edges.items():
            # Check prefix
            if not u.startswith(team_prefix):
                continue

            p_u = mapping.get(u)
            p_v = mapping.get(v)

            if p_u is not None and p_v is not None and p_u != p_v:
                key = (p_u, p_v)
                final_edges[key] = final_edges.get(key, 0) + count
    return final_edges


def compute_edge_errors(edges_act, edges_pred):
    errors = {}
    all_edges = set(edges_act.keys()) | set(edges_pred.keys())
    for edge in all_edges:
        act = edges_act.get(edge, 0)
        pred = edges_pred.get(edge, 0)
        if act > 0:
            err = abs(pred - act) / act
        else:
            err = 1.0 if pred > 0 else 0.0
        errors[edge] = min(err, 1.0)
    return errors


def compute_node_errors(edges_act, edges_pred):
    nodes = set([u for u, _ in edges_act] + [u for u, _ in edges_pred])
    act_deg = {}
    pred_deg = {}
    for (u, v), w in edges_act.items():
        act_deg[u] = act_deg.get(u, 0) + w
    for (u, v), w in edges_pred.items():
        pred_deg[u] = pred_deg.get(u, 0) + w

    errors = {}
    for n in nodes:
        act = act_deg.get(n, 0)
        pred = pred_deg.get(n, 0)
        if act > 0:
            err = abs(pred - act) / act
        else:
            err = 1.0 if pred > 0 else 0.0
        errors[n] = min(err, 1.0)
    return errors


def draw_pass_map(
    nodes: Dict[int, Dict[str, Any]],
    edges: Dict[Tuple[int, int], int],
    ax: Axes,
    title: str = "Home, True",  # {Home|Away}, {True|Predicted}
    crop_x: int | None = None,
    crop_y: int | None = None,
):
    def _styled_title(ax, title: str):
        title = str(title)
        lower = title.lower()
        if lower.startswith("home"):
            team_color_title = "#8b0000"
            team_len = 4
        elif lower.startswith("away"):
            team_color_title = "#0b3d91"
            team_len = 4
        else:
            ax.set_title(title, color="black", fontsize=30, fontweight="bold", pad=20)
            return

        team_text = title[:team_len]
        rest_text = title[team_len:]
        parts = [
            TextArea(team_text, textprops={"color": team_color_title, "weight": "bold", "size": 30}),
            TextArea(rest_text, textprops={"color": "black", "weight": "bold", "size": 30}),
        ]
        box = HPacker(children=parts, align="center", pad=0, sep=1)
        ab = AnnotationBbox(
            box,
            (0.5, 1.02),
            xycoords="axes fraction",
            frameon=False,
            box_alignment=(0.5, 0),
        )
        ax.add_artist(ab)

    # REPLACED: Use datatools.matplotsoccer 'green' field
    fig = ax.get_figure()
    mps.field("green", config.PITCH_X, config.PITCH_Y, ax=ax, fig=fig, show=False)
    ax.set_axis_off()
    ax.set_aspect("equal")

    if crop_x is not None and crop_y is not None and nodes:
        mean_x = float(np.mean([d["x"] for d in nodes.values()]))
        mean_y = float(np.mean([d["y"] for d in nodes.values()]))

        def _clamp_window(center: float, half_size: float, min_v: float, max_v: float):
            full = max_v - min_v
            window = 2 * half_size
            if window >= full:
                return min_v, max_v
            start = center - half_size
            end = center + half_size
            if start < min_v:
                end += min_v - start
                start = min_v
            if end > max_v:
                start -= end - max_v
                end = max_v
            return start, end

        x_min, x_max = _clamp_window(mean_x, crop_x, 0.0, float(config.PITCH_X))
        y_min, y_max = _clamp_window(mean_y, crop_y, 0.0, float(config.PITCH_Y))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    scale = 1 if crop_x is None else config.PITCH_X / crop_x / 2
    team_color = "tab:red" if title.lower().startswith("home") else "tab:blue"
    max_count = max(edges.values()) if edges else 1

    for (u, v), count in edges.items():
        if u not in nodes or v not in nodes:
            continue
        start = nodes[u]
        end = nodes[v]

        width = (count / max_count) * 15 * scale
        alpha = 0.7
        ax.plot(
            [start["x"], end["x"]],
            [start["y"], end["y"]],
            color="black",
            linewidth=width,
            alpha=alpha,
            zorder=2,
        )

    for pid, data in nodes.items():
        # Out-degree size
        out_deg = sum([c for (u, v), c in edges.items() if u == pid])
        size = (500 + out_deg * 20) * scale

        ax.scatter(
            data["x"],
            data["y"],
            s=size,
            color=team_color,
            edgecolors="white",
            linewidth=2,
            zorder=3,
        )
        ax.text(
            data["x"],
            data["y"],
            data["label"],
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=18 * scale,
            zorder=4,
        )

    _styled_title(ax, title)


def compare_edge_counts(edges_act, edges_pred, team_name):
    print(f"\n--- {team_name} Stats ---")
    total_act = sum(edges_act.values())
    total_pred = sum(edges_pred.values())
    print(f"Total Passes: True={total_act}, Pred={total_pred}")

    # Calculate MAE
    all_k = set(edges_act.keys()) | set(edges_pred.keys())
    diffs = []
    for k in all_k:
        diffs.append(abs(edges_act.get(k, 0) - edges_pred.get(k, 0)))
    print(f"Edge MAE: {np.mean(diffs):.2f}")


def draw_formation_check(tracking, phase_nodes):
    print("\nGenerating Formation Check (Average Position by Phase)...")
    phases = sorted(phase_nodes.keys())

    # Sort by start frame
    phase_meta = []
    for p in phases:
        p_data = tracking[tracking["phase_id"] == p]
        if p_data.empty:
            continue
        start_f = p_data["frame_id"].min()
        end_f = p_data["frame_id"].max()
        period = int(p_data["period_id"].iloc[0])
        start_ts = p_data["timestamp"].min()
        end_ts = p_data["timestamp"].max()
        phase_meta.append(
            {
                "pid": p,
                "start_f": start_f,
                "end_f": end_f,
                "period": period,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }
        )

    # Sort chronologically
    phase_meta.sort(key=lambda x: x["start_f"])

    # Grid calculation
    n = len(phase_meta)
    cols = 4
    rows = (n // cols) + (1 if n % cols > 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), facecolor="white")
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    prev_home = set()
    prev_away = set()

    for i, meta in enumerate(phase_meta):
        ax: Axes = axes[i]
        phase = meta["pid"]
        nodes = phase_nodes[phase]

        # Determine Period Label
        p_map = {1: "1H", 2: "2H", 3: "E1", 4: "E2"}
        period_str = p_map.get(meta["period"], "P?")

        # Duration info
        s_min, s_sec = divmod(int(meta["start_ts"]), 60)
        e_min, e_sec = divmod(int(meta["end_ts"]), 60)
        time_str = f"{period_str} {s_min}:{s_sec:02d} ~ {e_min}:{e_sec:02d}"

        # Detect Changes
        curr_home = {k for k in nodes.keys() if k.startswith("home")}
        curr_away = {k for k in nodes.keys() if k.startswith("away")}

        sub_text = ""
        # Only compare if specific conditions met (e.g. same match, not first phase)
        # Assuming consecutive phases in same match
        if i > 0:
            home_in = sorted([x.split("_")[1] for x in (curr_home - prev_home)])
            home_out = sorted([x.split("_")[1] for x in (prev_home - curr_home)])
            away_in = sorted([x.split("_")[1] for x in (curr_away - prev_away)])
            away_out = sorted([x.split("_")[1] for x in (prev_away - curr_away)])

            changes = []
            if home_in or home_out:
                changes.append(f"Home: IN {','.join(home_in)} / OUT {','.join(home_out)}")
            if away_in or away_out:
                changes.append(f"Away: IN {','.join(away_in)} / OUT {','.join(away_out)}")

            if changes:
                sub_text = "\n".join(changes)

        prev_home = curr_home
        prev_away = curr_away

        # Draw Pitch
        fig_ref = ax.get_figure()
        mps.field("green", config.PITCH_X, config.PITCH_Y, ax=ax, fig=fig_ref, show=False)
        ax.set_axis_off()
        ax.set_aspect("equal")

        # Plot Nodes
        for pid, pos in nodes.items():
            color = "tab:red" if pid.startswith("home") else "tab:blue"
            label = pid.split("_")[1]
            ax.scatter(pos[0], pos[1], color=color, s=200, edgecolors="white", zorder=3)
            ax.text(
                pos[0],
                pos[1],
                label,
                color="white",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                zorder=4,
            )

        title = f"Phase {phase}\n{time_str}"
        if sub_text:
            title += f"\n{sub_text}"

        ax.set_title(title, color="black", fontsize=16, fontweight="bold")

    # Hide unused
    for j in range(len(phase_meta), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig("application/passmap/result/average_position_by_phase.png", facecolor="white")
    print("Saved average_position_by_phase.png")


def main():
    try:
        tracking, true_events, pred_events = load_data()
    except Exception as e:
        print(f"Error: {e}")
        return

    phase_nodes = get_node_positions_per_phase(tracking)

    # Process Edges (New Logic)
    true_edges = get_edges_per_phase(true_events, tracking)
    pred_edges = get_edges_per_phase(pred_events, tracking)

    # Home
    nodes_h, glob_h, hist_h = aggregate_phases_hungarian(phase_nodes, {}, "home")
    true_edges_h = aggregate_edges(true_edges, phase_nodes, glob_h, hist_h, "home")
    pred_edges_h = aggregate_edges(pred_edges, phase_nodes, glob_h, hist_h, "home")

    # Away
    nodes_a, glob_a, hist_a = aggregate_phases_hungarian(phase_nodes, {}, "away")
    true_edges_a = aggregate_edges(true_edges, phase_nodes, glob_a, hist_a, "away")
    pred_edges_a = aggregate_edges(pred_edges, phase_nodes, glob_a, hist_a, "away")

    # Visualize - Side by Side
    os.makedirs("application/passmap/result", exist_ok=True)

    # Adjusted figsize to reduce vertical whitespace (Pitch is ~105x68, x2 ~= 3:1 aspect ratio)
    # Using (24, 10) gives a much wider aspect ratio, fitting the pitches better.
    fig_h, ax_h = plt.subplots(1, 2, figsize=(24, 10), facecolor="white")
    draw_pass_map(nodes_h, true_edges_h, ax_h[0], "tab:red", "Home True")
    draw_pass_map(nodes_h, pred_edges_h, ax_h[1], "tab:red", "Home Predicted")
    fig_h.subplots_adjust(wspace=0.04, top=0.9)
    fig_h.savefig("application/passmap/result/passmap_home_final.png", facecolor="white")
    print("Saved Home Split")

    fig_a, ax_a = plt.subplots(1, 2, figsize=(24, 10), facecolor="white")
    draw_pass_map(nodes_a, true_edges_a, ax_a[0], "tab:blue", "Away True")
    draw_pass_map(nodes_a, pred_edges_a, ax_a[1], "tab:blue", "Away Predicted")
    fig_a.subplots_adjust(wspace=0.04, top=0.9)
    fig_a.savefig("application/passmap/result/passmap_away_final.png", facecolor="white")
    print("Saved Away Split")

    # Stats
    compare_edge_counts(true_edges_h, pred_edges_h, "Home")
    compare_edge_counts(true_edges_a, pred_edges_a, "Away")

    # Draw Formation
    draw_formation_check(tracking, phase_nodes)


if __name__ == "__main__":
    main()
