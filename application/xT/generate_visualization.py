import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datatools import config, utils, visualization


def _apply_style():
    visualization.apply_style()


def plot_bar_chart(df: pd.DataFrame, title: str, filename: str) -> None:
    _apply_style()
    stats = df.groupby(["player_id", "team"])["value"].sum().reset_index()
    stats = stats.sort_values("value", ascending=False).head(15)

    plt.figure(figsize=(10, 8), facecolor="white")
    colors = ["tab:red" if t == "home" else "tab:blue" for t in stats["team"]]

    plt.barh(stats["player_id"], stats["value"], color=colors)
    plt.title(title, fontsize=15, fontweight="bold", color="black")
    plt.xlabel("Total xT Contribution", color="black")
    plt.ylabel("Player", color="black")
    plt.xticks(color="black")
    plt.yticks(color="black")
    plt.grid(axis="x", linestyle=":", alpha=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, facecolor="white")
    plt.close()


def plot_scatter_correlation(
    df_act: pd.DataFrame,
    df_pred: pd.DataFrame,
    filename: str,
    title_suffix: str = "",
) -> None:
    _apply_style()
    act_stats = (
        df_act.groupby(["player_id", "team"])["value"].sum().reset_index().rename(columns={"value": "act_value"})
    )
    pred_stats = (
        df_pred.groupby(["player_id", "team"])["value"].sum().reset_index().rename(columns={"value": "pred_value"})
    )
    merged = pd.merge(act_stats, pred_stats, on=["player_id", "team"], how="inner")

    if merged.empty:
        return

    corr = merged["act_value"].corr(merged["pred_value"])
    mae = np.mean(np.abs(merged["act_value"] - merged["pred_value"]))

    plt.figure(figsize=(10, 10), facecolor="white")
    colors = ["tab:red" if t == "home" else "tab:blue" for t in merged["team"]]
    plt.scatter(merged["act_value"], merged["pred_value"], c=colors, s=150, alpha=0.8, edgecolors="black")

    for _, row in merged.iterrows():
        pid_parts = row["player_id"].split("_")
        label = pid_parts[1] if len(pid_parts) > 1 else row["player_id"]
        plt.text(
            row["act_value"],
            row["pred_value"],
            label,
            fontsize=9,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    all_vals = np.concatenate([merged["act_value"], merged["pred_value"]])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    buf = (max_val - min_val) * 0.1 if max_val > min_val else 1.0
    lims = [min_val - buf, max_val + buf]
    plt.plot(lims, lims, "k--", alpha=0.5, zorder=0)
    plt.xlim(lims)
    plt.ylim(lims)

    title_str = f"Total xT Correlation (Act vs Pred) {title_suffix}\nr={corr:.3f}, MAE={mae:.3f}"
    plt.title(title_str, fontsize=16, fontweight="bold", color="black")
    plt.xlabel("Actual xT (GT)", fontsize=12, color="black")
    plt.ylabel("Predicted xT (Pred)", fontsize=12, color="black")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, facecolor="white")
    plt.close()


def _dominance_series(df: pd.DataFrame, bin_size: int = 60) -> pd.DataFrame:
    data = df.copy()
    data = data[data["team"].isin(["home", "away"])]
    data["abs_seconds"] = utils.compute_abs_seconds(data)
    data = data[data["abs_seconds"].notna()].copy()
    data["bin"] = (data["abs_seconds"] // bin_size).astype(int)

    grouped = data.groupby(["bin", "team"])["value"].sum().unstack(fill_value=0.0)
    if "home" not in grouped.columns:
        grouped["home"] = 0.0
    if "away" not in grouped.columns:
        grouped["away"] = 0.0
    dominance = grouped["home"] - grouped["away"]
    out = dominance.reset_index().rename(columns={0: "dominance"})
    out["minute"] = out["bin"] * (bin_size / 60.0)
    return out[["minute", "dominance"]]


def plot_xt_dominance(gt_df: pd.DataFrame, pred_df: pd.DataFrame, filename: str, bin_size: int = 60):
    _apply_style()
    gt_dom = _dominance_series(gt_df, bin_size=bin_size)
    pred_dom = _dominance_series(pred_df, bin_size=bin_size)
    merged = pd.merge(gt_dom, pred_dom, on="minute", how="outer", suffixes=("_gt", "_pred")).fillna(0.0)
    x = merged["minute"].to_numpy()
    y_gt = merged["dominance_gt"].to_numpy()
    y_pred = merged["dominance_pred"].to_numpy()

    plt.figure(figsize=(12, 5), facecolor="white")
    ax = plt.gca()

    ax.fill_between(x, 0, y_gt, where=y_gt >= 0, color="tab:red", alpha=0.28)
    ax.fill_between(x, 0, y_gt, where=y_gt < 0, color="tab:blue", alpha=0.28)
    ax.plot(x, y_gt, color="black", linewidth=1.8, label="GT (solid)")

    ax.fill_between(x, 0, y_pred, where=y_pred >= 0, color="tab:red", alpha=0.15)
    ax.fill_between(x, 0, y_pred, where=y_pred < 0, color="tab:blue", alpha=0.15)
    ax.plot(x, y_pred, color="black", linewidth=1.6, linestyle="--", label="Pred (dashed)")

    ax.axhline(0, color="gray", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("minute (absolute)")
    ax.set_ylabel("xT dominance (Home - Away)")
    ax.set_title("xT Dominance Timeline (GT vs Pred)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, facecolor="white")
    plt.close()

    return merged


def plot_dominance_absdiff(merged: pd.DataFrame, filename: str) -> None:
    _apply_style()
    data = merged.copy()
    data["abs_diff"] = np.abs(data["dominance_pred"] - data["dominance_gt"])
    plt.figure(figsize=(12, 4), facecolor="white")
    plt.plot(data["minute"], data["abs_diff"], color="black", linewidth=1.6)
    plt.xlabel("minute (absolute)")
    plt.ylabel("|Pred - GT|")
    plt.title("xT Dominance Absolute Difference")
    plt.tight_layout()
    plt.savefig(filename, facecolor="white")
    plt.close()


def plot_dominance_distributions(gt_dom: np.ndarray, pred_dom: np.ndarray, filename: str) -> None:
    _apply_style()
    plt.figure(figsize=(10, 5), facecolor="white")
    plt.hist(gt_dom, bins=30, alpha=0.6, label="GT", color="tab:red")
    plt.hist(pred_dom, bins=30, alpha=0.6, label="Pred", color="tab:blue")
    plt.xlabel("xT dominance (Home - Away)")
    plt.ylabel("count")
    plt.title("Dominance Distribution (GT vs Pred)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, facecolor="white")
    plt.close()


def plot_team_accuracy_timeline(df: pd.DataFrame, filename: str) -> None:
    _apply_style()
    if df.empty:
        return
    plt.figure(figsize=(10, 4), facecolor="white")
    plt.plot(df["minute"], df["home_acc"], color="tab:red", label="Home accuracy")
    plt.plot(df["minute"], df["away_acc"], color="tab:blue", label="Away accuracy")
    plt.xlabel("minute (absolute)")
    plt.ylabel("accuracy")
    plt.title("Frame-level xT Dominance Accuracy (by team)")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, facecolor="white")
    plt.close()


def plot_xt_absdiff_timeline(df: pd.DataFrame, filename: str, title: str) -> None:
    _apply_style()
    if df.empty:
        return
    fig, _ = visualization.plot_series_compare(
        df["gt_home_xt"].tolist(),
        df["pred_home_xt"].tolist(),
        labels=("True", "Pred"),
        x=df["minute"].tolist(),
    )
    fig.axes[0].set_xlabel("minute (absolute)")
    fig.axes[0].set_ylabel("home xT")
    fig.axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(filename, facecolor="white")
    plt.close(fig)

    # Absolute difference line
    plt.figure(figsize=(10, 4), facecolor="white")
    plt.plot(df["minute"], df["abs_diff"], color="black", linewidth=1.6)
    plt.xlabel("minute (absolute)")
    plt.ylabel("|Pred - GT|")
    plt.title("Home xT Absolute Difference")
    plt.tight_layout()
    plt.savefig(filename.replace(".png", "_absdiff.png"), facecolor="white")
    plt.close()
