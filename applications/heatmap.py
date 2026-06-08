"""
Event heatmaps for the tutorial's downstream analysis.

Renders side-by-side KDE heatmaps of true vs. predicted event locations.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import axes
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from scipy.ndimage import gaussian_filter

import datatools.matplotsoccer as mps
from datatools import config


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


def _filter_inplay_events(events: pd.DataFrame, team: str = None) -> pd.DataFrame:
    events = events.copy()
    teams = ["home", "away"] if team is None else [team]
    events["team"] = events["player_id"].apply(_infer_team_from_player)
    events = events[events["team"].isin(teams)]
    events = events[events["event_type"].isin(["kick", "control"])].copy()
    events = events[(events["start_x"].between(0, config.PITCH_X)) & (events["start_y"].between(0, config.PITCH_Y))]
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


def _draw_heatmap(
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


def plot_true_pred_heatmaps(
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
        true_map = _kde_from_points(true_filtered, bins=bins, sigma=sigma, x_col="start_x", y_col="start_y")
        pred_map = _kde_from_points(pred_filtered, bins=bins, sigma=sigma, x_col="start_x", y_col="start_y")
        interpolation = "bicubic"
    else:
        true_map = _hist2d_prob(true_filtered, bins=bins, x_col="start_x", y_col="start_y")
        pred_map = _hist2d_prob(pred_filtered, bins=bins, x_col="start_x", y_col="start_y")
        interpolation = "nearest"

    vmax = max(float(true_map.max()), float(pred_map.max()), 1e-6) if vmax is None else float(vmax)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    im0 = _draw_heatmap(
        axes[0],
        true_map,
        title=None,
        cmap="turbo",
        vmin=0.0,
        vmax=vmax,
        interpolation=interpolation,
    )
    im1 = _draw_heatmap(
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
