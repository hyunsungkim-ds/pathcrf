import os
import shutil
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import axes
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

import datatools.matplotsoccer as mps
from datatools import config, event_postprocessing


def _infer_team_from_player(player_id: object) -> str:
    if player_id is None or (isinstance(player_id, float) and np.isnan(player_id)):
        return "unknown"
    s = str(player_id).lower()
    if s.startswith("home_"):
        return "home"
    if s.startswith("away_"):
        return "away"
    if s.startswith("out"):
        return "out"
    return "unknown"


def _event_xy(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    if "x" not in out.columns and "start_x" in out.columns:
        out["x"] = out["start_x"]
    if "y" not in out.columns and "start_y" in out.columns:
        out["y"] = out["start_y"]
    return out


def _filter_inplay_events(events: pd.DataFrame, team: str = None) -> pd.DataFrame:
    events = _event_xy(events)
    teams = ["home", "away"] if team is None else [team]
    events["team"] = events["player_id"].apply(_infer_team_from_player)
    events = events[events["team"].isin(teams)]
    events = events[events["event_type"].isin(["kick", "control"])].copy()
    events = events[(events["x"].between(0, config.PITCH_X)) & (events["y"].between(0, config.PITCH_Y))]
    return events


def _hist2d_prob(
    df: pd.DataFrame,
    bins: tuple[int, int] = (12, 8),
    x_col: str = "x",
    y_col: str = "y",
) -> np.ndarray:
    gx, gy = bins
    if df.empty:
        return np.zeros((gy, gx), dtype=float)
    hist, _, _ = np.histogram2d(
        df[x_col].to_numpy(),
        df[y_col].to_numpy(),
        bins=[gx, gy],
        range=[[0, config.PITCH_X], [0, config.PITCH_Y]],
    )
    hist = hist.T
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def _kde_from_hist(prob_map: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if prob_map.sum() <= 0:
        return prob_map.copy()
    smooth = gaussian_filter(prob_map, sigma=sigma)
    total = smooth.sum()
    return smooth / total if total > 0 else smooth


def _kde_from_points(
    df: pd.DataFrame,
    bins: tuple[int, int] = (120, 80),
    sigma: float = 2.4,
    x_col: str = "x",
    y_col: str = "y",
) -> np.ndarray:
    base = _hist2d_prob(df, bins=bins, x_col=x_col, y_col=y_col)
    return _kde_from_hist(base, sigma=sigma)


def _distribution_intersection(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(float)
    q = q.astype(float)
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    return float(np.minimum(p, q).sum())


def _wasserstein_xy(p: np.ndarray, q: np.ndarray) -> float:
    gy, gx = p.shape
    p = p.astype(float)
    q = q.astype(float)
    ps = p.sum()
    qs = q.sum()
    if ps <= 0 or qs <= 0:
        return 0.0
    p = p / ps
    q = q / qs
    x_coords = (np.arange(gx) + 0.5) * (config.PITCH_X / gx)
    y_coords = (np.arange(gy) + 0.5) * (config.PITCH_Y / gy)
    p_x = p.sum(axis=0)
    q_x = q.sum(axis=0)
    p_y = p.sum(axis=1)
    q_y = q.sum(axis=1)
    wx = wasserstein_distance(x_coords, x_coords, p_x, q_x)
    wy = wasserstein_distance(y_coords, y_coords, p_y, q_y)
    return float((wx + wy) * 0.5)


def _emd2d(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(float)
    q = q.astype(float)
    ps = p.sum()
    qs = q.sum()
    if ps <= 0 and qs <= 0:
        return 0.0
    if ps <= 0 or qs <= 0:
        return float("nan")

    p = p / ps
    q = q / qs
    gy, gx = p.shape
    n = gx * gy

    xs = (np.arange(gx) + 0.5) * (config.PITCH_X / gx)
    ys = (np.arange(gy) + 0.5) * (config.PITCH_Y / gy)
    cell_xy = np.array([(x, y) for y in ys for x in xs], dtype=float)
    cost = cdist(cell_xy, cell_xy, metric="euclidean")

    supply = p.ravel()
    demand = q.ravel()
    c_vec = cost.ravel()

    eye_n = sparse.eye(n, format="csr")
    ones_row = np.ones((1, n))
    a_rows = sparse.kron(eye_n, ones_row, format="csr")
    a_cols = sparse.kron(ones_row, eye_n, format="csr")
    a_eq = sparse.vstack([a_rows, a_cols], format="csr")
    b_eq = np.concatenate([supply, demand])

    # NOTE: HiGHS presolve can misclassify some feasible OT instances as infeasible.
    # Use robust options first, then retry with presolve disabled.
    res = linprog(
        c_vec,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-6,
            "dual_feasibility_tolerance": 1e-6,
        },
    )
    if not res.success:
        res = linprog(
            c_vec,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=(0, None),
            method="highs",
            options={
                "presolve": False,
                "primal_feasibility_tolerance": 1e-6,
                "dual_feasibility_tolerance": 1e-6,
            },
        )
    if not res.success:
        return float("nan")
    return float(res.fun)


def _coarsen_prob_map(prob_map: np.ndarray, target_bins: tuple[int, int] = (12, 8)) -> np.ndarray:
    gy_t, gx_t = target_bins[1], target_bins[0]
    gy, gx = prob_map.shape
    if gy <= gy_t and gx <= gx_t:
        out = prob_map.astype(float)
        total = out.sum()
        return out / total if total > 0 else out

    y_edges = np.linspace(0, gy, gy_t + 1).astype(int)
    x_edges = np.linspace(0, gx, gx_t + 1).astype(int)
    out = np.zeros((gy_t, gx_t), dtype=float)
    for iy in range(gy_t):
        for ix in range(gx_t):
            y0, y1 = y_edges[iy], y_edges[iy + 1]
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            if y1 > y0 and x1 > x0:
                out[iy, ix] = float(prob_map[y0:y1, x0:x1].sum())
    total = out.sum()
    return out / total if total > 0 else out


def _metrics_text(true_map: np.ndarray, pred_map: np.ndarray) -> str:
    true_c = _coarsen_prob_map(true_map, target_bins=(12, 8))
    pred_c = _coarsen_prob_map(pred_map, target_bins=(12, 8))
    inter = _distribution_intersection(true_c, pred_c)
    w1 = _wasserstein_xy(true_c, pred_c)
    emd = _emd2d(true_c, pred_c)
    return f"Intersection {inter:.3f} | W1 {w1:.3f} | EMD {emd:.3f}"


def _plot_pitch_heatmap(
    ax: axes.Axes,
    mat: np.ndarray,
    title: str | None = None,
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
    interpolation: str = "nearest",
):
    mps.field(color="white", fig=ax.figure, ax=ax, show=False)
    im = ax.imshow(
        mat,
        extent=[0, config.PITCH_X, 0, config.PITCH_Y],
        origin="lower",
        cmap=cmap,
        alpha=0.72,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        zorder=7000,
    )
    if title is not None:
        ax.set_title(title, fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def _plot_single_heatmap_figure(
    mat: np.ndarray,
    title: str,
    cbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    interpolation: str,
    metrics_text: str,
):
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    im = _plot_pitch_heatmap(
        ax,
        mat,
        title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label(cbar_label)
    fig.text(0.5, 0.02, metrics_text, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def show_event_heatmap(
    events: pd.DataFrame,
    target_id: str,
    title: Optional[str] = None,
    use_kde: bool = True,
    bins: tuple[int, int] = (int(config.PITCH_X), int(config.PITCH_Y)),
    sigma: float = 2.6,
    vmax: Optional[float] = None,
    show_cbar: bool = False,
):
    bins = (int(bins[0]), int(bins[1]))

    if target_id in {"home", "away"}:
        filtered = _filter_inplay_events(events, target_id)
    else:
        filtered = _filter_inplay_events(events)
        filtered = filtered[filtered["player_id"].astype(str) == str(target_id)]

    if use_kde:
        mat = _kde_from_points(filtered, bins=bins, sigma=sigma, x_col="x", y_col="y")
        interpolation = "bicubic"
    else:
        mat = _hist2d_prob(filtered, bins=bins, x_col="x", y_col="y")
        interpolation = "nearest"

    vmax = max(float(mat.max()), 1e-6) if vmax is None else float(vmax)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = _plot_pitch_heatmap(
        ax,
        mat,
        title=title,
        cmap="turbo",
        vmin=0.0,
        vmax=vmax,
        interpolation=interpolation,
    )
    if show_cbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label("Density")
    fig.subplots_adjust(wspace=0, top=0.9)
    plt.show()


def compare_event_heatmaps(
    true_events: pd.DataFrame,
    pred_events: pd.DataFrame,
    target_id: str,
    use_kde: bool = True,
    bins: tuple[int, int] = (int(config.PITCH_X), int(config.PITCH_Y)),
    sigma: float = 2.6,
    vmax: Optional[float] = None,
    show_cbar: bool = True,
):
    def _styled_title(ax, parts: list[dict]):
        areas = [
            TextArea(
                part["text"],
                textprops={"color": part.get("color", "black"), "weight": part.get("weight", "normal"), "size": 35},
            )
            for part in parts
        ]
        box = HPacker(children=areas, align="center", pad=0, sep=1)
        ab = AnnotationBbox(
            box,
            (0.5, 0.98),
            xycoords="axes fraction",
            frameon=False,
            box_alignment=(0.5, 0),
        )
        ax.add_artist(ab)

    bins = (int(bins[0]), int(bins[1]))
    target_str = str(target_id).lower()

    if target_str in {"home", "away"}:
        true_filtered = _filter_inplay_events(true_events, target_str)
        pred_filtered = _filter_inplay_events(pred_events, target_str)
        target_display = target_str.title()
    else:
        true_filtered = _filter_inplay_events(true_events)
        pred_filtered = _filter_inplay_events(pred_events)
        true_filtered = true_filtered[true_filtered["player_id"].astype(str) == str(target_id)]
        pred_filtered = pred_filtered[pred_filtered["player_id"].astype(str) == str(target_id)]
        target_parts = str(target_id).split("_", 1)
        if len(target_parts) == 2:
            team, number = target_parts
            target_display = f"{team.title()} {number}"
        else:
            target_display = str(target_id)

    if use_kde:
        true_map = _kde_from_points(true_filtered, bins=bins, sigma=sigma, x_col="x", y_col="y")
        pred_map = _kde_from_points(pred_filtered, bins=bins, sigma=sigma, x_col="x", y_col="y")
        interpolation = "bicubic"
    else:
        true_map = _hist2d_prob(true_filtered, bins=bins, x_col="x", y_col="y")
        pred_map = _hist2d_prob(pred_filtered, bins=bins, x_col="x", y_col="y")
        interpolation = "nearest"

    vmax = max(float(true_map.max()), float(pred_map.max()), 1e-6) if vmax is None else float(vmax)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    im0 = _plot_pitch_heatmap(
        axes[0],
        true_map,
        title=None,
        cmap="turbo",
        vmin=0.0,
        vmax=vmax,
        interpolation=interpolation,
    )
    im1 = _plot_pitch_heatmap(
        axes[1],
        pred_map,
        title=None,
        cmap="turbo",
        vmin=0.0,
        vmax=vmax,
        interpolation=interpolation,
    )

    if target_str.startswith("home"):
        target_color = "#8b0000"
    elif target_str.startswith("away"):
        target_color = "#0b3d91"
    else:
        target_color = "black"

    _styled_title(
        axes[0],
        [
            {"text": target_display, "color": target_color, "weight": "bold"},
            {"text": ", True", "weight": "bold"},
        ],
    )
    _styled_title(
        axes[1],
        [
            {"text": target_display, "color": target_color, "weight": "bold"},
            {"text": ", Predicted", "weight": "bold"},
        ],
    )

    if show_cbar:
        fig.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.01).set_label("Density")
        fig.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.01).set_label("Density")

    fig.tight_layout()
    plt.show()


def _select_key_players(true_events: pd.DataFrame) -> list[str]:
    onball = _filter_inplay_events(true_events).copy()
    if onball.empty:
        return []
    agg = onball.groupby(["player_id", "team"]).size().reset_index(name="event_count")
    agg["score"] = agg["event_count"]

    key_players: list[str] = []
    for team in ["home", "away"]:
        team_df = agg[agg["team"] == team].sort_values("score", ascending=False)
        if team == "away":
            team_df = team_df[team_df["player_id"].astype(str) != "away_1"]
        if not team_df.empty:
            key_players.append(str(team_df.iloc[0]["player_id"]))
    if len(key_players) < 2:
        for pid in agg.sort_values("score", ascending=False)["player_id"].tolist():
            if pid not in key_players:
                key_players.append(str(pid))
            if len(key_players) == 2:
                break
    return key_players[:2]


def save_event_heatmaps(true_events: pd.DataFrame, pred_events: pd.DataFrame, save_fig):
    bins = (12, 8)
    dense_bins = (config.PITCH_X, config.PITCH_Y)

    team_maps: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, str]] = {}
    for team in ["home", "away"]:
        true_team = _filter_inplay_events(true_events, team)
        pred_team = _filter_inplay_events(pred_events, team)

        true_disc = _hist2d_prob(true_team, bins=bins, x_col="x", y_col="y")
        pred_disc = _hist2d_prob(pred_team, bins=bins, x_col="x", y_col="y")
        true_kde = _kde_from_points(true_team, bins=dense_bins, sigma=2.6, x_col="x", y_col="y")
        pred_kde = _kde_from_points(pred_team, bins=dense_bins, sigma=2.6, x_col="x", y_col="y")
        team_maps[(team, "discrete")] = (true_disc, pred_disc, "nearest")
        team_maps[(team, "continuous")] = (true_kde, pred_kde, "bicubic")

    key_players = _select_key_players(true_events)
    onball_true = _filter_inplay_events(true_events)
    onball_pred = _filter_inplay_events(pred_events)
    player_maps: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for pid in key_players:
        true_p = onball_true[onball_true["player_id"] == pid]
        pred_p = onball_pred[onball_pred["player_id"] == pid]
        true_map = _kde_from_points(true_p, bins=dense_bins, sigma=2.6, x_col="x", y_col="y")
        pred_map = _kde_from_points(pred_p, bins=dense_bins, sigma=2.6, x_col="x", y_col="y")
        player_maps[pid] = (true_map, pred_map, "bicubic")

    def _shared_scale(maps: list[tuple[np.ndarray, np.ndarray, str]], diff_scale: float = 2.0) -> tuple[float, float]:
        if not maps:
            return 1e-6, 1e-6
        vmax = max(max(float(gt.max()), float(pr.max())) for gt, pr, _ in maps)
        diff_max = max(float(np.abs(pr - gt).max()) for gt, pr, _ in maps)
        return max(vmax, 1e-6), max(diff_max * diff_scale, 1e-6)

    discrete_maps = [team_maps[(team, "discrete")] for team in ["home", "away"]]
    continuous_team_maps = [team_maps[(team, "continuous")] for team in ["home", "away"]]
    player_continuous_maps = [player_maps[pid] for pid in key_players]
    discrete_vmax, discrete_diff_v = _shared_scale(discrete_maps)
    continuous_team_vmax, continuous_team_diff_v = _shared_scale(continuous_team_maps)
    player_vmax, player_diff_v = _shared_scale(player_continuous_maps)

    for team in ["home", "away"]:
        true_disc, pred_disc, interp_disc = team_maps[(team, "discrete")]
        metrics_disc = _metrics_text(true_disc, pred_disc)
        diff_disc = pred_disc - true_disc
        fig = _plot_single_heatmap_figure(
            true_disc,
            title=f"Event Heatmap ({team.upper()}, discrete, True)",
            cbar_label="density",
            cmap="turbo",
            vmin=0.0,
            vmax=discrete_vmax,
            interpolation=interp_disc,
            metrics_text=metrics_disc,
        )
        save_fig(fig, f"event_heatmap_{team}_discrete_true.png")
        fig = _plot_single_heatmap_figure(
            pred_disc,
            title=f"Event Heatmap ({team.upper()}, discrete, Pred)",
            cbar_label="density",
            cmap="turbo",
            vmin=0.0,
            vmax=discrete_vmax,
            interpolation=interp_disc,
            metrics_text=metrics_disc,
        )
        save_fig(fig, f"event_heatmap_{team}_discrete_pred.png")
        fig = _plot_single_heatmap_figure(
            diff_disc,
            title=f"Event Heatmap ({team.upper()}, discrete, Pred - True)",
            cbar_label="diff",
            cmap="RdBu_r",
            vmin=-discrete_diff_v,
            vmax=discrete_diff_v,
            interpolation=interp_disc,
            metrics_text=metrics_disc,
        )
        save_fig(fig, f"event_heatmap_{team}_discrete_diff.png")

        true_kde, pred_kde, interp_kde = team_maps[(team, "continuous")]
        metrics_kde = _metrics_text(true_kde, pred_kde)
        diff_kde = pred_kde - true_kde
        fig = _plot_single_heatmap_figure(
            true_kde,
            title=f"Event Heatmap ({team.upper()}, continuous, True)",
            cbar_label="density",
            cmap="turbo",
            vmin=0.0,
            vmax=continuous_team_vmax,
            interpolation=interp_kde,
            metrics_text=metrics_kde,
        )
        save_fig(fig, f"event_heatmap_{team}_continuous_true.png")
        fig = _plot_single_heatmap_figure(
            pred_kde,
            title=f"Event Heatmap ({team.upper()}, continuous, Pred)",
            cbar_label="density",
            cmap="turbo",
            vmin=0.0,
            vmax=continuous_team_vmax,
            interpolation=interp_kde,
            metrics_text=metrics_kde,
        )
        save_fig(fig, f"event_heatmap_{team}_continuous_pred.png")
        fig = _plot_single_heatmap_figure(
            diff_kde,
            title=f"Event Heatmap ({team.upper()}, continuous, Pred - True)",
            cbar_label="diff",
            cmap="RdBu_r",
            vmin=-continuous_team_diff_v,
            vmax=continuous_team_diff_v,
            interpolation=interp_kde,
            metrics_text=metrics_kde,
        )
        save_fig(fig, f"event_heatmap_{team}_continuous_diff.png")

    for pid in key_players:
        true_map, pred_map, interp = player_maps[pid]
        metrics_player = _metrics_text(true_map, pred_map)
        diff_player = pred_map - true_map
        fig = _plot_single_heatmap_figure(
            true_map,
            title=f"On-ball Heatmap ({pid}, True)",
            cbar_label="density",
            cmap="turbo",
            vmin=0.0,
            vmax=player_vmax,
            interpolation=interp,
            metrics_text=metrics_player,
        )
        save_fig(fig, f"event_heatmap_player_{pid}_true.png")
        fig = _plot_single_heatmap_figure(
            pred_map,
            title=f"On-ball Heatmap ({pid}, Pred)",
            cbar_label="density",
            cmap="turbo",
            vmin=0.0,
            vmax=player_vmax,
            interpolation=interp,
            metrics_text=metrics_player,
        )
        save_fig(fig, f"event_heatmap_player_{pid}_pred.png")
        fig = _plot_single_heatmap_figure(
            diff_player,
            title=f"On-ball Heatmap ({pid}, Pred - True)",
            cbar_label="diff",
            cmap="RdBu_r",
            vmin=-player_diff_v,
            vmax=player_diff_v,
            interpolation=interp,
            metrics_text=metrics_player,
        )
        save_fig(fig, f"event_heatmap_player_{pid}_diff.png")


def generate_event_heatmap_report(
    tracking: Optional[pd.DataFrame] = None,
    true_events: Optional[pd.DataFrame] = None,
    pred_events: Optional[pd.DataFrame] = None,
    output_dir: str = "application/event_heatmap",
):
    true_events = event_postprocessing.prepare_events(true_events, tracking)
    pred_events = event_postprocessing.prepare_events(pred_events, tracking)

    os.makedirs(output_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, "result")
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    def save_fig(fig, name: str):
        path = os.path.join(result_dir, name)
        fig.savefig(path, facecolor="white")
        plt.close(fig)

    save_event_heatmaps(true_events, pred_events, save_fig)
