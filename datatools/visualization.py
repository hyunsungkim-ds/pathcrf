from __future__ import annotations

from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datatools import config
from datatools.matplotsoccer import field

_STYLE_APPLIED = False


def apply_style():
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#1f2937",
            "axes.labelcolor": "#111827",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#e5e7eb",
            "grid.linestyle": ":",
            "grid.alpha": 0.6,
            "legend.frameon": False,
            "font.family": "DejaVu Sans",
        }
    )
    _STYLE_APPLIED = True


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(axis="both", which="both", length=0)


def draw_pitch(ax=None):
    apply_style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    field(fig=fig, ax=ax, color="green", show=False)
    return fig, ax


def plot_timeline(series, ax=None, label: str = "value"):
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure
    ax.plot(series, color="black", linewidth=2, label=label)
    ax.set_xlabel("time")
    ax.set_ylabel(label)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    _clean_axes(ax)
    return fig, ax


def plot_series_compare(
    series_gt: Sequence[float],
    series_pred: Sequence[float],
    ax=None,
    labels=("True", "Pred"),
    x: Sequence[float] | None = None,
):
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure
    if x is None:
        x_vals = np.arange(len(series_gt))
    else:
        x_vals = np.asarray(x, dtype=float)
    ax.plot(x_vals, series_gt, color="tab:green", linewidth=2.2, label=labels[0])
    ax.plot(x_vals, series_pred, color="darkorange", linewidth=2.2, label=labels[1])
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    _clean_axes(ax)
    return fig, ax


def plot_bar_compare(gt_value: float, pred_value: float, ax=None, labels=("True", "Pred"), title: str | None = None):
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure
    bars = ax.bar([0, 1], [gt_value, pred_value], color=["#2563eb", "#ef4444"])
    ax.set_xticks([0, 1], labels=labels)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha="center", va="bottom", fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    _clean_axes(ax)
    return fig, ax


def plot_metric_bars(
    labels: Sequence[str],
    values: Sequence[float],
    ax=None,
    title: str | None = None,
    ylabel: str | None = None,
):
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    else:
        fig = ax.figure

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color="#2563eb")
    ax.set_xticks(x, labels=labels, rotation=25, ha="right")
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    _clean_axes(ax)
    return fig, ax


def plot_zone_grid(
    home_share_grid: np.ndarray,
    ax=None,
    title: str | None = None,
    cmap: str = "RdYlBu_r",
    alpha: float = 0.65,
    show_colorbar: bool = True,
    cbar_label: str = "Home Possession (1=Home, 0=Away)",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_ticks: list[float] | None = None,
    cbar_ticklabels: list[str] | None = None,
    text_labels: np.ndarray | None = None,
    text_kwargs: dict | None = None,
):
    apply_style()
    fig, ax = draw_pitch(ax=ax)
    gx = home_share_grid.shape[1]
    gy = home_share_grid.shape[0]
    cell_w = config.PITCH_X / gx
    cell_h = config.PITCH_Y / gy
    norm = plt.Normalize(vmin, vmax)
    cmap_obj = plt.get_cmap(cmap)
    cmap_obj = cmap_obj.copy()
    cmap_obj.set_bad(color=(0, 0, 0, 0))

    for cy in range(gy):
        for cx in range(gx):
            val = float(home_share_grid[cy, cx])
            if np.isnan(val):
                continue
            color = cmap_obj(norm(val))
            rect = Rectangle(
                (cx * cell_w, cy * cell_h),
                cell_w,
                cell_h,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=alpha,
                zorder=4,
            )
            ax.add_patch(rect)

    if title:
        ax.set_title(title, color="black")

    if text_labels is not None:
        if text_kwargs is None:
            text_kwargs = {"fontsize": 8, "color": "black", "ha": "center", "va": "center"}
        for cy in range(gy):
            for cx in range(gx):
                label = text_labels[cy, cx]
                if label is None or (isinstance(label, float) and np.isnan(label)):
                    continue
                label_str = str(label)
                if label_str.strip() == "":
                    continue
                ax.text(
                    (cx + 0.5) * cell_w,
                    (cy + 0.5) * cell_h,
                    label_str,
                    **text_kwargs,
                )

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(cbar_label)
        if cbar_ticks is None:
            cbar_ticks = [vmin, (vmin + vmax) / 2.0, vmax]
        cbar.set_ticks(cbar_ticks)
        if cbar_ticklabels is None:
            if vmin == 0.0 and vmax == 1.0:
                cbar_ticklabels = ["Away", "Neutral", "Home"]
            else:
                cbar_ticklabels = [f"{t:.2f}" for t in cbar_ticks]
        cbar.set_ticklabels(cbar_ticklabels)
    return fig, ax


def plot_zone_timeline_grid(
    zone_df,
    grid: tuple[int, int] = (6, 4),
    title: str | None = None,
    x_label: str = "minute (absolute)",
    y_label: str = "accuracy",
    ylim: tuple[float, float] = (0.0, 1.0),
):
    apply_style()
    gx, gy = grid
    fig, axes = plt.subplots(gy, gx, figsize=(gx * 2.2, gy * 1.8), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for cy in range(gy):
        for cx in range(gx):
            ax = axes[gy - 1 - cy, cx]
            cell_df = zone_df[(zone_df["cell_x"] == cx) & (zone_df["cell_y"] == cy)]
            if not cell_df.empty:
                ax.plot(cell_df["minute"], cell_df["accuracy"], color="#111827", linewidth=1.0)
            ax.set_ylim(*ylim)
            ax.set_title(f"Z{cy * gx + cx + 1}", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.3)
            ax.tick_params(labelsize=7)
            _clean_axes(ax)

    fig.suptitle(title or "Zone Accuracy Timeline", y=0.98, fontsize=12, fontweight="bold")
    fig.supxlabel(x_label, fontsize=10)
    fig.supylabel(y_label, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


def plot_zone_share_triptych(
    gt_grid: np.ndarray,
    pred_grid: np.ndarray,
    diff_grid: np.ndarray,
    title: str | None = None,
    cmap: str = "RdYlBu_r",
    diff_cmap: str = "RdBu_r",
    vmin: float = 0.0,
    vmax: float = 1.0,
    diff_v: float = 0.2,
    text_gt: np.ndarray | None = None,
    text_pred: np.ndarray | None = None,
    text_diff: np.ndarray | None = None,
    metrics_text: str | None = None,
):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), gridspec_kw={"wspace": 0.1})
    ax_gt, ax_pred, ax_diff = axes

    def _resolve_cmap(cmap_like):
        if isinstance(cmap_like, str):
            return plt.get_cmap(cmap_like)
        return cmap_like

    def _clone_cmap(cmap_like):
        cmap = _resolve_cmap(cmap_like)
        colors = cmap(np.linspace(0, 1, 256))
        return mcolors.ListedColormap(colors)

    def _soften_cmap(cmap_like, strength: float = 0.85):
        cmap = _resolve_cmap(cmap_like)
        colors = cmap(np.linspace(0, 1, 256))
        colors[:, :3] = colors[:, :3] * strength + (1 - strength) * 1.0
        return mcolors.ListedColormap(colors)

    def _draw_grid(
        ax,
        grid,
        cmap_name,
        norm,
        text_labels=None,
        title_text: str | None = None,
    ):
        fig_inner, ax_inner = draw_pitch(ax=ax)
        ax_inner.set_aspect("equal", adjustable="box")
        gx = grid.shape[1]
        gy = grid.shape[0]
        cell_w = config.PITCH_X / gx
        cell_h = config.PITCH_Y / gy
        cmap_obj = _clone_cmap(cmap_name)
        cmap_obj.set_bad(color=(0, 0, 0, 0))

        for cy in range(gy):
            for cx in range(gx):
                val = float(grid[cy, cx])
                if np.isnan(val):
                    continue
                color = cmap_obj(norm(val))
                rect = Rectangle(
                    (cx * cell_w, cy * cell_h),
                    cell_w,
                    cell_h,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.8,
                    zorder=4,
                )
                ax_inner.add_patch(rect)

        if text_labels is not None:
            for cy in range(gy):
                for cx in range(gx):
                    label = text_labels[cy, cx]
                    if label is None or (isinstance(label, float) and np.isnan(label)):
                        continue
                    label_str = str(label)
                    if label_str.strip() == "":
                        continue
                    ax_inner.text(
                        (cx + 0.5) * cell_w,
                        (cy + 0.5) * cell_h,
                        label_str,
                        fontsize=8,
                        color="black",
                        fontweight="bold",
                        ha="center",
                        va="center",
                        zorder=5,
                    )

        if title_text:
            ax_inner.set_title(title_text, fontsize=10.5, fontweight="bold")
        return ax_inner

    norm_main = plt.Normalize(vmin, vmax)
    norm_diff = plt.Normalize(-diff_v, diff_v)

    diff_cmap_soft = _soften_cmap(diff_cmap, strength=0.7)
    _draw_grid(ax_gt, gt_grid, cmap, norm_main, text_labels=text_gt, title_text="Home Possession (True)")
    _draw_grid(ax_pred, pred_grid, cmap, norm_main, text_labels=text_pred, title_text="Home Possession (Pred)")
    _draw_grid(ax_diff, diff_grid, diff_cmap_soft, norm_diff, text_labels=text_diff, title_text="Pred - True (%)")

    # Individual colorbars next to each panel (uniform size/pad)
    def _add_cbar(target_ax, cmap_name, norm, label):
        sm = plt.cm.ScalarMappable(cmap=_resolve_cmap(cmap_name), norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(target_ax)
        cax = divider.append_axes("right", size="2.6%", pad=0.04)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(label, fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)
        return cbar

    cbar_gt = _add_cbar(ax_gt, cmap, norm_main, "Home Possession (True)")
    cbar_pred = _add_cbar(ax_pred, cmap, norm_main, "Home Possession (Pred)")
    cbar_diff = _add_cbar(ax_diff, diff_cmap_soft, norm_diff, "Diff (Pred-True)")
    diff_ticks = np.linspace(-diff_v, diff_v, 5)
    cbar_diff.set_ticks(diff_ticks)
    for tick_val, label in zip(cbar_diff.get_ticks(), cbar_diff.ax.get_yticklabels()):
        if abs(tick_val) < 1e-6:
            label.set_fontweight("bold")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.95)

    if metrics_text:
        fig.text(0.5, 0.04, metrics_text, ha="center", va="bottom", fontsize=9, color="#111827")

    fig.subplots_adjust(left=0.005, right=0.97, top=0.85, bottom=0.06, wspace=0.1)
    return fig, axes
