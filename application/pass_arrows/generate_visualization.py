import os
import shutil
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

import datatools.matplotsoccer as mps
from application.possession import possession_metrics as metrics_possession
from datatools import config, event_postprocessing, utils, visualization


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


def _onball_events(events: pd.DataFrame) -> pd.DataFrame:
    ev = _event_xy(events)
    ev["team"] = ev["player_id"].apply(_infer_team_from_player)
    ev["event_type_norm"] = ev["event_type"].astype(str).str.lower()
    ev = ev[ev["event_type_norm"].isin(["kick", "pass", "control"])].copy()
    ev = ev[ev["team"].isin(["home", "away"])]
    ev = ev.dropna(subset=["x", "y"])
    ev = ev[(ev["x"] >= 0) & (ev["x"] <= config.PITCH_X) & (ev["y"] >= 0) & (ev["y"] <= config.PITCH_Y)]
    return ev


def _to_seconds_series(series: pd.Series) -> pd.Series:
    def _to_sec(val: object) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        if isinstance(val, str):
            try:
                return float(utils.timestamp_to_seconds(val))
            except Exception:
                return np.nan
        try:
            return float(val)
        except Exception:
            return np.nan

    return series.apply(_to_sec)


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


def _metrics_text(true_map: np.ndarray, pred_map: np.ndarray) -> str:
    inter = _distribution_intersection(true_map, pred_map)
    w1 = _wasserstein_xy(true_map, pred_map)
    emd = _emd2d(true_map, pred_map)
    return f"Intersection {inter:.3f} | W1 {w1:.3f} | EMD {emd:.3f}"


def _select_key_players(events_gt: pd.DataFrame) -> list[str]:
    onball = _onball_events(events_gt).copy()
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


def _build_pass_table(events: pd.DataFrame) -> pd.DataFrame:
    ev = _onball_events(events).copy()
    if "episode_id" in ev.columns:
        ev = ev[ev["episode_id"] > 0].copy()
    ev = ev.sort_values(["episode_id", "frame_id"], kind="stable").reset_index(drop=True)
    ev["next_player"] = ev.groupby("episode_id")["player_id"].shift(-1)
    ev["next_team"] = ev["next_player"].apply(_infer_team_from_player)
    ev["next_type"] = ev.groupby("episode_id")["event_type_norm"].shift(-1)
    ev["next_x"] = ev.groupby("episode_id")["x"].shift(-1)
    ev["next_y"] = ev.groupby("episode_id")["y"].shift(-1)
    ev["next_ts"] = ev.groupby("episode_id")["timestamp"].shift(-1)
    ev["ts_sec"] = _to_seconds_series(ev["timestamp"])
    ev["next_ts_sec"] = _to_seconds_series(ev["next_ts"])
    ev["dt"] = ev["next_ts_sec"] - ev["ts_sec"]

    is_pass = ev["event_type_norm"].isin(["kick", "pass"])
    valid_next = ev["next_player"].notna() & (ev["next_team"] != "out") & (ev["next_type"] != "out")
    same_episode_window = ev["dt"].notna() & (ev["dt"] >= 0) & (ev["dt"] <= 10.0)
    pass_df = ev[is_pass & valid_next & same_episode_window].copy()

    pass_df["end_x_use"] = pass_df["end_x"].where(pass_df["end_x"].notna(), pass_df["next_x"])
    pass_df["end_y_use"] = pass_df["end_y"].where(pass_df["end_y"].notna(), pass_df["next_y"])
    pass_df = pass_df.dropna(subset=["x", "y", "end_x_use", "end_y_use"])
    pass_df = pass_df[
        (pass_df["x"] >= 0)
        & (pass_df["x"] <= config.PITCH_X)
        & (pass_df["y"] >= 0)
        & (pass_df["y"] <= config.PITCH_Y)
        & (pass_df["end_x_use"] >= 0)
        & (pass_df["end_x_use"] <= config.PITCH_X)
        & (pass_df["end_y_use"] >= 0)
        & (pass_df["end_y_use"] <= config.PITCH_Y)
    ]
    return pass_df


def _draw_pass_arrows(ax, pass_df: pd.DataFrame, title: str):
    mps.field(color="green", fig=ax.figure, ax=ax, show=False)
    colors = {"home": "#d62728", "away": "#1f77b4"}
    for team, color in colors.items():
        team_df = pass_df[pass_df["team"] == team]
        if team_df.empty:
            continue
        for _, row in team_df.iterrows():
            dx = float(row["end_x_use"] - row["x"])
            dy = float(row["end_y_use"] - row["y"])
            ax.arrow(
                float(row["x"]),
                float(row["y"]),
                dx,
                dy,
                width=0.02,
                head_width=0.7,
                head_length=1.0,
                color=color,
                alpha=0.28,
                length_includes_head=True,
                zorder=8500,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, fontweight="bold")


def _events_by_player(events: pd.DataFrame, player_id: str) -> pd.DataFrame:
    return events[events["player_id"].astype(str) == str(player_id)].copy()


def _pass_similarity_score(metrics: dict[str, float]) -> dict[str, float]:
    sim_time = float(np.exp(-metrics["mean_time_distance"] / 0.75))
    sim_start = float(np.exp(-metrics["mean_start_distance"] / 2.5))
    sim_end = float(np.exp(-metrics["mean_end_distance"] / 4.0))
    sim_len = float(np.exp(-metrics["mean_kick_length_abs_error"] / 3.5))
    sim_mape = float(np.exp(-metrics["mean_kick_length_mape"] / 0.8))
    sim_dir = float((metrics["mean_direction_cosine"] + 1.0) * 0.5)

    weights = {
        "time": 0.10,
        "start": 0.20,
        "end": 0.20,
        "length": 0.20,
        "mape": 0.10,
        "direction": 0.20,
    }
    score = (
        weights["time"] * sim_time
        + weights["start"] * sim_start
        + weights["end"] * sim_end
        + weights["length"] * sim_len
        + weights["mape"] * sim_mape
        + weights["direction"] * sim_dir
    )
    return {
        "score": float(score),
        "sim_time": sim_time,
        "sim_start": sim_start,
        "sim_end": sim_end,
        "sim_len": sim_len,
        "sim_mape": sim_mape,
        "sim_dir": sim_dir,
    }


def save_pass_arrows(events_gt: pd.DataFrame, events_pred: pd.DataFrame, save_fig):
    bins = (12, 8)
    pass_gt = _build_pass_table(events_gt)
    pass_pred = _build_pass_table(events_pred)
    key_players = _select_key_players(events_gt)

    for team in ["home", "away"]:
        gt_team_pass = pass_gt[pass_gt["team"] == team]
        pred_team_pass = pass_pred[pass_pred["team"] == team]

        start_metrics = _metrics_text(
            _hist2d_prob(gt_team_pass.rename(columns={"x": "sx", "y": "sy"}), bins=bins, x_col="sx", y_col="sy"),
            _hist2d_prob(
                pred_team_pass.rename(columns={"x": "sx", "y": "sy"}),
                bins=bins,
                x_col="sx",
                y_col="sy",
            ),
        )
        end_metrics = _metrics_text(
            _hist2d_prob(
                gt_team_pass.rename(columns={"end_x_use": "ex", "end_y_use": "ey"}),
                bins=bins,
                x_col="ex",
                y_col="ey",
            ),
            _hist2d_prob(
                pred_team_pass.rename(columns={"end_x_use": "ex", "end_y_use": "ey"}),
                bins=bins,
                x_col="ex",
                y_col="ey",
            ),
        )
        metrics_text = f"Start distribution: {start_metrics} | End distribution: {end_metrics}"

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        _draw_pass_arrows(ax, gt_team_pass, f"{team.upper()} Pass Arrows (True)")
        fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=10)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        save_fig(fig, f"pass_arrows_{team}_true.png")

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        _draw_pass_arrows(ax, pred_team_pass, f"{team.upper()} Pass Arrows (Pred)")
        fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=10)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        save_fig(fig, f"pass_arrows_{team}_pred.png")

    player_scores = []
    for pid in key_players:
        gt_player_events = _events_by_player(events_gt, pid)
        pred_player_events = _events_by_player(events_pred, pid)
        player_metric = metrics_possession.calc_pass_alignment_metrics(gt_player_events, pred_player_events)
        player_score = _pass_similarity_score(player_metric)
        player_scores.append((pid, player_score["score"]))

        metric_line = (
            f"Similarity score {player_score['score']:.3f} | "
            f"dt {player_metric['mean_time_distance']:.3f}s | "
            f"start {player_metric['mean_start_distance']:.2f}m | "
            f"end {player_metric['mean_end_distance']:.2f}m | "
            f"len {player_metric['mean_kick_length_abs_error']:.2f}m | "
            f"MAPE {player_metric['mean_kick_length_mape']:.3f} | "
            f"dir-cos {player_metric['mean_direction_cosine']:.3f}"
        )
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        _draw_pass_arrows(ax, pass_gt[pass_gt["player_id"] == pid], f"{pid} Passes (True)")
        fig.text(0.5, 0.02, metric_line, ha="center", fontsize=10)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        save_fig(fig, f"pass_arrows_player_{pid}_true.png")

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        _draw_pass_arrows(ax, pass_pred[pass_pred["player_id"] == pid], f"{pid} Passes (Pred)")
        fig.text(0.5, 0.02, metric_line, ha="center", fontsize=10)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        save_fig(fig, f"pass_arrows_player_{pid}_pred.png")

    if player_scores:
        labels = [pid for pid, _ in player_scores]
        values = [score for _, score in player_scores]
        fig, _ = visualization.plot_metric_bars(
            labels, values, title="Key Player Pass Similarity", ylabel="score (0-1)"
        )
        save_fig(fig, "pass_similarity_key_players.png")


def generate_pass_arrows_report(
    match_id: Optional[str] = None,
    output_dir: str = "application/pass_arrows",
    tracking: Optional[pd.DataFrame] = None,
    events_gt: Optional[pd.DataFrame] = None,
    events_pred: Optional[pd.DataFrame] = None,
):
    if tracking is None or events_gt is None or events_pred is None:
        if match_id is None:
            raise ValueError("match_id is required when dataframes are not provided")
        tracking, events_gt, events_pred = event_postprocessing.load_match_data(match_id)

    events_gt = event_postprocessing.prepare_events(events_gt, tracking)
    events_pred = event_postprocessing.prepare_events(events_pred, tracking)

    os.makedirs(output_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, "result")
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    def save_fig(fig, name: str):
        path = os.path.join(result_dir, name)
        fig.savefig(path, facecolor="white")
        plt.close(fig)

    save_pass_arrows(events_gt, events_pred, save_fig)


def main() -> None:
    generate_pass_arrows_report(match_id="J03WR9", output_dir="application/pass_arrows")


if __name__ == "__main__":
    main()
