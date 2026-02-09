import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from application.passmap.generate_visualization import (
    aggregate_edges,
    aggregate_phases_hungarian,
    compute_edge_errors,
    compute_node_errors,
    get_edges_per_phase,
    get_node_positions_per_phase,
    load_data,
)


def get_degrees(edges_dict):
    deg = {}
    for (u, v), w in edges_dict.items():
        deg[u] = deg.get(u, 0) + w
    return deg


def draw_node_error_bar(team_name, nodes, act_deg, pred_deg, save_path):
    all_pids = sorted(list(set(act_deg.keys()) | set(pred_deg.keys())))

    data = []
    errors = []

    for pid in all_pids:
        act = act_deg.get(pid, 0)
        pred = pred_deg.get(pid, 0)

        label = nodes[pid]["label"] if pid in nodes else str(pid)

        if act > 0:
            err = abs(pred - act) / act
            data.append({"Label": label, "Error": err, "Act": act, "Pred": pred})
            errors.append(err)

    if not data:
        print(f"No valid data for {team_name} node error.")
        return

    df = pd.DataFrame(data)
    df = df.sort_values(by="Error", ascending=False)
    avg_error = np.mean(errors) if errors else 0.0

    plt.figure(figsize=(12, 6))
    bars = plt.bar(df["Label"], df["Error"], color="skyblue", edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height*100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.axhline(
        y=avg_error,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Average Error: {avg_error*100:.1f}%",
    )

    plt.title(
        f"{team_name} - Node Pass Volume Error (|Pred-True|/True)",
        fontsize=24,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Player Label", fontsize=16, fontweight="bold")
    plt.ylabel("Relative Error", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, max(df["Error"].max() * 1.2, avg_error * 1.2))
    plt.legend(fontsize=14)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Node Error Bar Chart to {save_path}")


def draw_edge_error_matrix(team_name, nodes, edges_act, edges_pred, save_path):
    sorted_pids = sorted(list(nodes.keys()), key=lambda x: str(nodes[x]["label"]))
    labels = [nodes[pid]["label"] for pid in sorted_pids]
    n = len(labels)

    error_matrix = np.zeros((n, n))
    annot_matrix = np.empty((n, n), dtype=object)

    for i, u_pid in enumerate(sorted_pids):
        for j, v_pid in enumerate(sorted_pids):
            if u_pid == v_pid:
                error_matrix[i, j] = 0
                annot_matrix[i, j] = ""
                continue

            act = edges_act.get((u_pid, v_pid), 0)
            pred = edges_pred.get((u_pid, v_pid), 0)
            diff = abs(pred - act)

            if act > 0:
                rel_err = diff / act
            else:
                rel_err = 1.0 if pred > 0 else 0.0

            error_matrix[i, j] = min(rel_err, 1.0)
            annot_matrix[i, j] = "" if (act == 0 and pred == 0) else f"{act}/{pred}"

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        error_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="Reds",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={},
        annot_kws={"size": 12, "weight": "bold"},
    )

    cbar = ax.collections[0].colorbar
    cbar.set_label("Relative Error (Capped at 1.0)", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=14)

    plt.title(
        f"{team_name} - Edge Error Matrix (True / Pred)",
        fontsize=24,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Receiver", fontsize=16, fontweight="bold")
    plt.ylabel("Sender", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Edge Error Matrix to {save_path}")


def run_team_visuals(team_name, nodes, edges_act, edges_pred):
    act_deg = get_degrees(edges_act)
    pred_deg = get_degrees(edges_pred)

    save_path_bar = f"application/passmap/result/error_node_bar_{team_name}.png"
    draw_node_error_bar(team_name, nodes, act_deg, pred_deg, save_path_bar)

    save_path_matrix = f"application/passmap/result/error_edge_matrix_{team_name}.png"
    draw_edge_error_matrix(team_name, nodes, edges_act, edges_pred, save_path_matrix)


def get_error_maps(edges_act, edges_pred):
    return {
        "edge_errors": compute_edge_errors(edges_act, edges_pred),
        "node_errors": compute_node_errors(edges_act, edges_pred),
    }


def main():
    try:
        tracking, ev_act, ev_pred = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    phase_nodes = get_node_positions_per_phase(tracking)
    edges_act = get_edges_per_phase(ev_act, tracking)
    edges_pred = get_edges_per_phase(ev_pred, tracking)

    os.makedirs("application/passmap/result", exist_ok=True)

    final_nodes_h, glob_nodes_h, map_hist_h = aggregate_phases_hungarian(
        phase_nodes, {}, team_prefix="home"
    )
    final_edges_act_h = aggregate_edges(edges_act, phase_nodes, glob_nodes_h, map_hist_h, "home")
    final_edges_pred_h = aggregate_edges(edges_pred, phase_nodes, glob_nodes_h, map_hist_h, "home")
    run_team_visuals("HOME", final_nodes_h, final_edges_act_h, final_edges_pred_h)

    final_nodes_a, glob_nodes_a, map_hist_a = aggregate_phases_hungarian(
        phase_nodes, {}, team_prefix="away"
    )
    final_edges_act_a = aggregate_edges(edges_act, phase_nodes, glob_nodes_a, map_hist_a, "away")
    final_edges_pred_a = aggregate_edges(edges_pred, phase_nodes, glob_nodes_a, map_hist_a, "away")
    run_team_visuals("AWAY", final_nodes_a, final_edges_act_a, final_edges_pred_a)


if __name__ == "__main__":
    main()
