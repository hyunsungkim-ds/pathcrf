from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon

EdgeDict = Dict[Tuple[int, int], float]
NodeDict = Dict[int, Dict[str, Any]]


def _node_list(nodes: NodeDict, edges_true: EdgeDict, edges_pred: EdgeDict) -> list:
    node_ids = set(nodes.keys())
    for u, v in edges_true.keys():
        node_ids.add(u)
        node_ids.add(v)
    for u, v in edges_pred.keys():
        node_ids.add(u)
        node_ids.add(v)
    return sorted(node_ids, key=lambda x: str(x))


def _out_degrees(edges: EdgeDict, node_ids: Iterable) -> Dict[int, float]:
    deg = {n: 0.0 for n in node_ids}
    for (u, _v), w in edges.items():
        if u in deg:
            deg[u] += float(w)
    return deg


def _adjacency_matrix(node_ids: list, edges: EdgeDict) -> np.ndarray:
    idx = {n: i for i, n in enumerate(node_ids)}
    mat = np.zeros((len(node_ids), len(node_ids)), dtype=float)
    for (u, v), w in edges.items():
        if u in idx and v in idx:
            mat[idx[u], idx[v]] += float(w)
    return mat


def compute_jensen_shannon(nodes: NodeDict, edges_true: EdgeDict, edges_pred: EdgeDict) -> float:
    """
    Jensen-Shannon divergence between weighted adjacency matrices.
    """
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
    """
    Graph spectral distance using normalized Laplacian eigenvalues.
    """
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
