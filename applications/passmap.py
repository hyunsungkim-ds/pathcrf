"""
Pass maps for the tutorial's downstream analysis.

Builds per-phase player nodes and pass edges, aggregates them across phases via
Hungarian matching, computes graph-similarity stats, and renders pass maps.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon

import datatools.matplotsoccer as mps
from datatools import config

EdgeDict = Dict[Tuple[int, int], float]
NodeDict = Dict[int, Dict[str, Any]]


def get_node_positions_per_phase(tracking):
    """Average on-pitch position of each player per phase (active play only)."""
    phase_nodes = {}
    phases = sorted(tracking["phase_id"].unique())

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

        # top 11 by appearance count
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
    """Count directed pass edges (kick/pass to a same-team teammate) per phase."""
    phase_edges = {}

    if "phase_id" not in events.columns:
        frame_phase_map = tracking.set_index("frame_id")["phase_id"].to_dict()
        events["phase_id"] = events["frame_id"].map(frame_phase_map)

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

            if "episode_id" in curr and "episode_id" in next_ev:
                if curr["episode_id"] != next_ev["episode_id"]:
                    continue

            # skip large time gaps (not a direct pass)
            if "timestamp" in curr and "timestamp" in next_ev:
                try:
                    time_diff = float(next_ev["timestamp"]) - float(curr["timestamp"])
                    if time_diff > 10.0:  # 10s
                        continue
                except ValueError:
                    pass

            if str(curr.get("event_type", "")).lower() not in {"pass", "kick"}:
                continue

            def get_team(pid):
                if isinstance(pid, str) and "_" in pid:
                    return pid.split("_")[0]
                return None

            team_c = get_team(curr["player_id"])
            team_n = get_team(next_ev["player_id"])

            if team_c and team_n and team_c == team_n:
                sender = curr["player_id"]
                receiver = next_ev["player_id"]

                # exclude self-loops (dribbles) so that a pass map shows only sender-receiver edges
                if sender != receiver:
                    edge = (sender, receiver)
                    phase_edges[phase][edge] = phase_edges[phase].get(edge, 0) + 1

    return phase_edges


def aggregate_phases_hungarian(phase_nodes, phase_edges, team_prefix="home"):
    """Match per-phase player positions into stable nodes across phases (Hungarian)."""
    phases = sorted(phase_nodes.keys())
    global_nodes = {}
    current_mapping = {}
    mapping_history = {}

    if not phases:
        return {}, {}, {}

    first_phase = phases[0]
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


def aggregate_edges(phase_edges, mapping_history, team_prefix="home"):
    """Remap per-phase pass edges onto aggregated node ids for one team."""
    final_edges = {}
    for phase, edges in phase_edges.items():
        if phase not in mapping_history:
            continue
        mapping = mapping_history[phase]

        for (u, v), count in edges.items():
            if not u.startswith(team_prefix):
                continue

            p_u = mapping.get(u)
            p_v = mapping.get(v)

            if p_u is not None and p_v is not None and p_u != p_v:
                key = (p_u, p_v)
                final_edges[key] = final_edges.get(key, 0) + count
    return final_edges


def _node_list(nodes: NodeDict, edges_true: EdgeDict, edges_pred: EdgeDict) -> list:
    """Sorted union of node ids across nodes and both edge sets."""
    node_ids = set(nodes.keys())
    for u, v in edges_true.keys():
        node_ids.add(u)
        node_ids.add(v)
    for u, v in edges_pred.keys():
        node_ids.add(u)
        node_ids.add(v)
    return sorted(node_ids, key=lambda x: str(x))


def _out_degrees(edges: EdgeDict, node_ids: Iterable) -> Dict[int, float]:
    """Out-degree (summed outgoing edge weight) per node."""
    deg = {n: 0.0 for n in node_ids}
    for (u, _v), w in edges.items():
        if u in deg:
            deg[u] += float(w)
    return deg


def _adjacency_matrix(node_ids: list, edges: EdgeDict) -> np.ndarray:
    """Weighted adjacency matrix over node_ids."""
    idx = {n: i for i, n in enumerate(node_ids)}
    mat = np.zeros((len(node_ids), len(node_ids)), dtype=float)
    for (u, v), w in edges.items():
        if u in idx and v in idx:
            mat[idx[u], idx[v]] += float(w)
    return mat


def compute_jensen_shannon(nodes: NodeDict, edges_true: EdgeDict, edges_pred: EdgeDict) -> float:
    """Jensen-Shannon divergence between true/pred weighted adjacency matrices."""
    node_ids = _node_list(nodes, edges_true, edges_pred)
    if not node_ids:
        return 0.0

    a_true = _adjacency_matrix(node_ids, edges_true).flatten()
    a_pred = _adjacency_matrix(node_ids, edges_pred).flatten()

    sum_true = float(a_true.sum())
    sum_pred = float(a_pred.sum())

    if sum_true == 0.0 and sum_pred == 0.0:
        return 0.0
    if sum_true == 0.0 or sum_pred == 0.0:
        return 1.0

    p = a_true / sum_true
    q = a_pred / sum_pred
    return float(jensenshannon(p, q, base=2.0) ** 2)


def compute_spectral_distance(nodes: NodeDict, edges_true: EdgeDict, edges_pred: EdgeDict) -> float:
    """Spectral distance from normalized-Laplacian eigenvalues."""
    node_ids = _node_list(nodes, edges_true, edges_pred)
    if not node_ids:
        return 0.0

    def _normalized_laplacian(edges: EdgeDict) -> np.ndarray:
        a = _adjacency_matrix(node_ids, edges)
        a = 0.5 * (a + a.T)
        deg = a.sum(axis=1)
        with np.errstate(divide="ignore"):
            inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        d_inv_sqrt = np.diag(inv_sqrt)
        ident = np.eye(len(node_ids))
        return ident - d_inv_sqrt @ a @ d_inv_sqrt

    laplacian_true = _normalized_laplacian(edges_true)
    laplacian_pred = _normalized_laplacian(edges_pred)

    eigen_true = np.linalg.eigvalsh(laplacian_true)
    eigen_pred = np.linalg.eigvalsh(laplacian_pred)

    return float(np.mean(np.abs(eigen_true - eigen_pred)))


def compute_passmap_stats(nodes: NodeDict, true_edges: EdgeDict, pred_edges: EdgeDict) -> Dict[str, float]:
    """Node/edge/JS/spectral similarity metrics between true and predicted pass maps."""
    node_ids = _node_list(nodes, true_edges, pred_edges)

    true_deg = _out_degrees(true_edges, node_ids)
    pred_deg = _out_degrees(pred_edges, node_ids)

    if node_ids:
        mean_node_degree = float(np.mean([true_deg[n] for n in node_ids]))
        mae_node_degree = float(np.mean([abs(true_deg[n] - pred_deg[n]) for n in node_ids]))
    else:
        mean_node_degree = 0.0
        mae_node_degree = 0.0

    mean_edge_weight = float(np.mean(list(true_edges.values()))) if true_edges else 0.0

    all_edges = set(true_edges.keys()) | set(pred_edges.keys())
    if all_edges:
        mae_edge_weight = float(np.mean([abs(true_edges.get(e, 0.0) - pred_edges.get(e, 0.0)) for e in all_edges]))
    else:
        mae_edge_weight = 0.0

    js_div = compute_jensen_shannon(nodes, true_edges, pred_edges)
    spectral_dist = compute_spectral_distance(nodes, true_edges, pred_edges)

    return {
        "mean_node_degree": mean_node_degree,
        "mae_node_degree": mae_node_degree,
        "mean_edge_weight": mean_edge_weight,
        "mae_edge_weight": mae_edge_weight,
        "jensen_shannon": js_div,
        "spectral_dist": spectral_dist,
    }


def draw_passmap(
    nodes: Dict[int, Dict[str, Any]],
    edges: Dict[Tuple[int, int], int],
    ax: Axes,
    title: str = "Home, True",  # {Home|Away}, {True|Predicted}
    crop_x: int | None = None,
    crop_y: int | None = None,
):
    """Render one team's pass map (nodes + weighted edges) onto an axis."""

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


def compute_passmaps(
    true_events: pd.DataFrame,
    pred_events: pd.DataFrame,
    tracking: pd.DataFrame,
    team: str = "home",
) -> Dict[str, Any]:
    """Aggregate one team's pass map into {team, nodes, true_edges, pred_edges, stats}."""
    phase_nodes = get_node_positions_per_phase(tracking)
    true_phase_edges = get_edges_per_phase(true_events, tracking)
    pred_phase_edges = get_edges_per_phase(pred_events, tracking)

    nodes, _, mapping_history = aggregate_phases_hungarian(phase_nodes, {}, team)
    true_edges = aggregate_edges(true_phase_edges, mapping_history, team)
    pred_edges = aggregate_edges(pred_phase_edges, mapping_history, team)

    stats = compute_passmap_stats(nodes, true_edges, pred_edges)
    return {
        "team": team,
        "nodes": nodes,
        "true_edges": true_edges,
        "pred_edges": pred_edges,
        "stats": stats,
    }


def plot_true_pred_passmaps(passmap: Dict[str, Any], crop_x: int = 36, crop_y: int = 24, figsize=(18, 9)):
    """Draw a team's true vs predicted pass maps side by side (from compute_passmaps)."""
    team_label = str(passmap["team"]).title()
    fig, ax = plt.subplots(1, 2, figsize=figsize, facecolor="white")
    draw_passmap(passmap["nodes"], passmap["true_edges"], ax[0], f"{team_label}, True", crop_x, crop_y)
    draw_passmap(passmap["nodes"], passmap["pred_edges"], ax[1], f"{team_label}, Predicted", crop_x, crop_y)
    fig.subplots_adjust(wspace=0.05, top=0.9)
    plt.show()
    return fig, ax
