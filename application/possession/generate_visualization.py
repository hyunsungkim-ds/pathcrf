import os
import shutil
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from application.possession import possession_metrics as metrics_possession
from datatools import event_postprocessing, visualization


def plot_true_pred_poss(
    values_true,
    values_pred,
    title: Optional[str] = None,
    x: Optional[list[int]] = None,
    grid: bool = False,
):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, values_true * 100, marker="o", color="tab:green", linewidth=2.2, label="True")
    ax.plot(x, values_pred * 100, marker="o", color="darkorange", linewidth=2.2, label="Pred")
    ax.set_xlabel("Time (min)", fontdict={"size": 18})
    ax.set_ylabel("Home Possession (%)", fontdict={"size": 18})
    ax.set_ylim(0, 100)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, axis="both", color="gray", linestyle="--", linewidth=1)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16)
    plt.show()
    return fig, ax


def generate_metric_report(
    match_id: Optional[str] = None,
    output_dir: str = "application/possession",
    tracking: Optional[pd.DataFrame] = None,
    true_events: Optional[pd.DataFrame] = None,
    pred_events: Optional[pd.DataFrame] = None,
):
    if tracking is None or true_events is None or pred_events is None:
        if match_id is None:
            raise ValueError("match_id is required when dataframes are not provided")
        tracking, true_events, pred_events = event_postprocessing.load_match_data(match_id)

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

    def save_series_compare(
        values_true,
        values_pred,
        title: Optional[str],
        filename: str,
        x: Optional[list[int]] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "home share",
        ylim: Optional[tuple[float, float]] = None,
        grid: bool = False,
    ):
        fig, ax = visualization.plot_series_compare(
            values_true,
            values_pred,
            labels=("True", "Pred"),
            x=x,
        )
        ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if grid:
            ax.grid(True, axis="both", linestyle=":", alpha=0.6)
        save_fig(fig, filename)

    # --- Possession frames (episode>0) ---
    frames_gt = metrics_possession.infer_team_poss(true_events, tracking)
    frames_pred = metrics_possession.infer_team_poss(pred_events, tracking)

    overall_acc = metrics_possession.calc_team_possession_accuracy(frames_gt, frames_pred)

    acc_by_min = metrics_possession.calc_possession_accuracy_by_minute(frames_gt, frames_pred)
    home_possession_by_abs_min = metrics_possession.home_poss_by_time(frames_gt, frames_pred, bin_minutes=1)
    home_possession_by_abs_5min = metrics_possession.home_poss_by_time(frames_gt, frames_pred, bin_minutes=5)

    # Overall metrics charts
    fig, ax = visualization.plot_bar_compare(
        overall_acc,
        1.0 - overall_acc,
        labels=("Acc", "Err"),
        title="Possession Accuracy (Overall)",
    )
    ax.set_ylabel("ratio")
    save_fig(fig, "possession_accuracy_overall.png")

    # Time series charts
    if not acc_by_min.empty:
        fig, ax = visualization.plot_timeline(acc_by_min["accuracy"].tolist(), label="accuracy")
        ax.set_xlabel("minute")
        ax.set_title("Possession Accuracy Timeline")
        save_fig(fig, "accuracy_timeline.png")

    if not home_possession_by_abs_min.empty:
        save_series_compare(
            home_possession_by_abs_min["gt_home_possession"].tolist(),
            home_possession_by_abs_min["pred_home_possession"].tolist(),
            title="Home Share Timeline",
            filename="home_share_timeline.png",
            x=home_possession_by_abs_min["minute"].tolist(),
            xlabel="minute",
        )

    if not home_possession_by_abs_5min.empty:
        save_series_compare(
            home_possession_by_abs_5min["gt_home_possession"].tolist(),
            home_possession_by_abs_5min["pred_home_possession"].tolist(),
            filename="home_share_timeline_5min.png",
            x=home_possession_by_abs_5min["minute"].tolist(),
            xlabel="minute (5-min bins)",
            title=None,
            ylim=(0.0, 1.0),
            grid=True,
        )

    # Zone accuracy / share error
    zone_grid = (6, 5)
    zone_acc_grid = metrics_possession.zone_accuracy_grid(frames_gt, frames_pred, grid=zone_grid)
    acc_labels = np.full((zone_grid[1], zone_grid[0]), "", dtype=object)
    for cy in range(zone_acc_grid.shape[0]):
        for cx in range(zone_acc_grid.shape[1]):
            val = zone_acc_grid[cy, cx]
            if np.isnan(val):
                continue
            acc_labels[cy, cx] = f"{val:.2f}"

    fig, _ = visualization.plot_zone_grid(
        zone_acc_grid,
        title="Zone Accuracy (Frame-level)",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Accuracy (0-1)",
        text_labels=acc_labels,
        text_kwargs={"fontsize": 9, "color": "black", "ha": "center", "va": "center"},
    )
    save_fig(fig, "zone_accuracy.png")

    zone_acc_timeline = metrics_possession.zone_accuracy_timeline(frames_gt, frames_pred, grid=zone_grid)
    if not zone_acc_timeline.empty:
        fig, _ = visualization.plot_zone_timeline_grid(
            zone_acc_timeline,
            grid=zone_grid,
            title="Zone Accuracy Timeline",
        )
        save_fig(fig, "zone_accuracy_timeline_grid.png")

    zone_share_df = metrics_possession.zone_home_share_error_grid(frames_gt, frames_pred, grid=zone_grid)
    gt_share_grid = np.full((zone_grid[1], zone_grid[0]), np.nan, dtype=float)
    pred_share_grid = np.full((zone_grid[1], zone_grid[0]), np.nan, dtype=float)
    diff_pp_grid = np.full((zone_grid[1], zone_grid[0]), np.nan, dtype=float)
    text_gt = np.full((zone_grid[1], zone_grid[0]), "", dtype=object)
    text_pred = np.full((zone_grid[1], zone_grid[0]), "", dtype=object)
    text_diff = np.full((zone_grid[1], zone_grid[0]), "", dtype=object)

    for _, row in zone_share_df.iterrows():
        cx = int(row["cell_x"])
        cy = int(row["cell_y"])
        gt_share = float(row["gt_home_possession"])
        pred_share = float(row["pred_home_possession"])
        gt_share_grid[cy, cx] = gt_share
        pred_share_grid[cy, cx] = pred_share
        diff_pp = (pred_share - gt_share) * 100.0
        diff_pp_grid[cy, cx] = diff_pp
        text_gt[cy, cx] = f"{int(round(gt_share * 100))}%"
        text_pred[cy, cx] = f"{int(round(pred_share * 100))}%"
        text_diff[cy, cx] = f"{diff_pp:+.1f}%"

    text_kwargs = {
        "fontsize": 8,
        "color": "black",
        "ha": "center",
        "va": "center",
        "fontweight": "bold",
    }
    fig, _ = visualization.plot_zone_grid(
        gt_share_grid,
        title="Home Possession (True)",
        cmap="RdYlBu_r",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Home Possession (True)",
        text_labels=text_gt,
        text_kwargs=text_kwargs,
    )
    save_fig(fig, "home_possession_by_zone_true.png")

    fig, _ = visualization.plot_zone_grid(
        pred_share_grid,
        title="Home Possession (Pred)",
        cmap="RdYlBu_r",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Home Possession (Pred)",
        text_labels=text_pred,
        text_kwargs=text_kwargs,
    )
    save_fig(fig, "home_possession_by_zone_pred.png")

    fig, _ = visualization.plot_zone_grid(
        diff_pp_grid,
        title="Pred - True (%)",
        cmap="RdBu_r",
        vmin=-20.0,
        vmax=20.0,
        cbar_label="Diff (Pred-True)",
        text_labels=text_diff,
        text_kwargs=text_kwargs,
    )
    save_fig(fig, "home_possession_by_zone_diff.png")

    # --- Kick alignment (NW) ---
    kick_align = metrics_possession.calc_pass_alignment_metrics(true_events, pred_events)

    kick_labels = [
        "time dist",
        "start dist",
        "end dist",
        "len abs",
        "len MAPE",
        "dir cos",
    ]
    kick_values = [
        kick_align["mean_time_distance"],
        kick_align["mean_start_distance"],
        kick_align["mean_end_distance"],
        kick_align["mean_kick_length_abs_error"],
        kick_align["mean_kick_length_mape"],
        kick_align["mean_direction_cosine"],
    ]
    fig, _ = visualization.plot_metric_bars(
        kick_labels,
        kick_values,
        title="Kick Alignment (NW, means)",
        ylabel="value",
    )
    save_fig(fig, "kick_alignment_summary.png")


if __name__ == "__main__":
    generate_metric_report(match_id="J03WR9", output_dir="application/possession")
