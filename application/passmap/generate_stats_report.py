import numpy as np
from scipy.stats import spearmanr

from application.passmap.generate_visualization import (
    aggregate_edges,
    aggregate_phases_hungarian,
    get_edges_per_phase,
    get_node_positions_per_phase,
    load_data,
)


def analyze_team_stats(team_name, nodes, edges_act, edges_pred):
    print(f"\n{'='*20} {team_name} TEAM STATISTICS {'='*20}")

    # Helper to get node degrees
    def get_degrees(edges_dict):
        deg = {}
        for (u, v), w in edges_dict.items():
            deg[u] = deg.get(u, 0) + w
        return deg

    act_deg = get_degrees(edges_act)
    pred_deg = get_degrees(edges_pred)

    # 1. Total Passes
    total_act = sum(edges_act.values())
    total_pred = sum(edges_pred.values())
    print(f"Total Passes: Actual={total_act}, Predicted={total_pred} (Diff: {total_pred - total_act})")

    # 2. Edge Correlation & Error
    all_edges = set(edges_act.keys()) | set(edges_pred.keys())
    act_vals = []
    pred_vals = []

    abs_errors = []

    for e in all_edges:
        a = edges_act.get(e, 0)
        p = edges_pred.get(e, 0)
        act_vals.append(a)
        pred_vals.append(p)
        abs_errors.append(abs(a - p))

    act_vals = np.array(act_vals)
    pred_vals = np.array(pred_vals)

    mae = np.mean(abs_errors)
    corr = np.corrcoef(act_vals, pred_vals)[0, 1] if len(act_vals) > 1 else 0

    print("Metrics:")
    print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  - Correlation (Pearson):     {corr:.4f}")

    # 3. Standardized Graph Topology Metrics (Absolute Scale 0~1)
    # A. Cosine Similarity (Structure Identity)
    # Cos(A, B) = (A . B) / (||A|| * ||B||)

    norm_act = np.linalg.norm(act_vals)
    norm_pred = np.linalg.norm(pred_vals)

    if norm_act > 0 and norm_pred > 0:
        cosine_sim = np.dot(act_vals, pred_vals) / (norm_act * norm_pred)
    else:
        cosine_sim = 0.0

    # B. Centrality Rank Correlation (Hub Identification)
    # Spearman Rank Correlation of Node Degrees
    # "Did we identify key players correctly?"

    # Calculate degrees
    all_players = sorted(list(set(nodes.keys()) | set(act_deg.keys())))
    deg_act_vec = [act_deg.get(p, 0) for p in all_players]
    deg_pred_vec = [pred_deg.get(p, 0) for p in all_players]

    spearman_rho, _ = spearmanr(deg_act_vec, deg_pred_vec)

    print(f"  - Graph Cosine Similarity:   {cosine_sim:.4f} (Max 1.0)")
    print(f"  - Centrality Rank Corr:      {spearman_rho:.4f} (Max 1.0)")

    # 4. Relative Error (New Metric: |P-T|/T)
    # Critical View: Only valid where T > 0.

    # Node MRE
    node_errs = []
    node_labels_err = []

    # Remove redundant code later
    all_pids = set(act_deg.keys()) | set(pred_deg.keys())

    for pid in all_pids:
        act = act_deg.get(pid, 0)
        pred = pred_deg.get(pid, 0)

        if act > 0:
            err = abs(pred - act) / act
            node_errs.append(err)

            # Get label
            label = nodes[pid]["label"] if pid in nodes else str(pid)
            node_labels_err.append((label, act, pred, err))

    avg_node_mre = np.mean(node_errs) if node_errs else 0.0

    print(f"  - Node MRE (Out-Degree):     {avg_node_mre*100:.1f}%")

    # Print Top 5 Node Errors
    print("\n  [Node Error Breakdown - Top 5 Worst]")
    node_labels_err.sort(key=lambda x: x[3], reverse=True)
    for label, a, p, e in node_labels_err[:5]:
        print(f"    - Player {label:>2}: Act={a:<3} Pred={p:<3} (Err: {e*100:.1f}%)")

    print("\n  [Node Error Breakdown - Top 5 Best]")
    node_labels_err.sort(key=lambda x: x[3])
    for label, a, p, e in node_labels_err[:5]:
        print(f"    - Player {label:>2}: Act={a:<3} Pred={p:<3} (Err: {e*100:.1f}%)")

    # Edge MRE (Only for edges existing in Actual)
    edge_errs = []
    for (u, v), act_w in edges_act.items():
        if act_w > 0:
            pred_w = edges_pred.get((u, v), 0)
            err = abs(pred_w - act_w) / act_w
            edge_errs.append(err)

    avg_edge_mre = np.mean(edge_errs) if edge_errs else 0.0

    print(f"  - Node MRE (Out-Degree):     {avg_node_mre*100:.1f}%")
    print(f"  - Edge MRE (Existing):       {avg_edge_mre*100:.1f}%")

    # 5. Top 5 Edges Comparison
    print("\n[Top 5 Frequent Connections]")

    sorted_act = sorted(edges_act.items(), key=lambda x: x[1], reverse=True)[:5]
    sorted_pred = sorted(edges_pred.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"{'Rank':<4} | {'Actual (Src->Dst: Count)':<30} | {'Predicted (Src->Dst: Count)':<30}")
    print("-" * 70)

    for i in range(5):

        def fmt_edge(item):
            if not item:
                return "-"
            (u, v), c = item
            # Get Labels
            l_u = nodes[u]["label"]
            l_v = nodes[v]["label"]
            return f"{l_u:>2} -> {l_v:<2} : {c}"

        row_act = sorted_act[i] if i < len(sorted_act) else None
        row_pred = sorted_pred[i] if i < len(sorted_pred) else None

        print(f"#{i+1:<3} | {fmt_edge(row_act):<30} | {fmt_edge(row_pred):<30}")


def main():
    try:
        tracking, ev_act, ev_pred = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Process phases
    phase_nodes = get_node_positions_per_phase(tracking)
    edges_act = get_edges_per_phase(ev_act, tracking)
    edges_pred = get_edges_per_phase(ev_pred, tracking)

    # --- HOME TEAM ---
    final_nodes_h, glob_nodes_h, map_hist_h = aggregate_phases_hungarian(phase_nodes, {}, team_prefix="home")
    final_edges_act_h = aggregate_edges(edges_act, phase_nodes, glob_nodes_h, map_hist_h, "home")
    final_edges_pred_h = aggregate_edges(edges_pred, phase_nodes, glob_nodes_h, map_hist_h, "home")

    analyze_team_stats("HOME", final_nodes_h, final_edges_act_h, final_edges_pred_h)

    # --- AWAY TEAM ---
    final_nodes_a, glob_nodes_a, map_hist_a = aggregate_phases_hungarian(phase_nodes, {}, team_prefix="away")
    final_edges_act_a = aggregate_edges(edges_act, phase_nodes, glob_nodes_a, map_hist_a, "away")
    final_edges_pred_a = aggregate_edges(edges_pred, phase_nodes, glob_nodes_a, map_hist_a, "away")

    analyze_team_stats("AWAY", final_nodes_a, final_edges_act_a, final_edges_pred_a)


if __name__ == "__main__":
    main()
