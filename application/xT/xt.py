import os
import sys

import numpy as np
import pandas as pd

# import seaborn as sns # Removed to prevent style interference

# Add current directory to path

sys.path.append(os.getcwd())


import datatools.config as config
from application.xT import generate_visualization as xt_viz
from datatools import event_postprocessing, utils


# --- xT Model Class ---
class ExpectedThreat:
    def __init__(self):
        # 16x12 Grid from Karun.in (Iteration 5)
        self.w = 16
        self.h = 12

        # Iteration 5 Matrix
        self.xt = np.array(
            [
                [
                    0.00296004,
                    0.00352297,
                    0.00330204,
                    0.00420215,
                    0.00439247,
                    0.00536515,
                    0.00684887,
                    0.00780055,
                    0.01043721,
                    0.01333563,
                    0.01765856,
                    0.02470358,
                    0.03662223,
                    0.04760151,
                    0.05975949,
                    0.06578558,
                ],
                [
                    0.00275691,
                    0.00494295,
                    0.00400234,
                    0.00421394,
                    0.00443753,
                    0.00539051,
                    0.00668854,
                    0.00847057,
                    0.01126381,
                    0.01468686,
                    0.02034976,
                    0.02947076,
                    0.04190818,
                    0.05604305,
                    0.07527312,
                    0.09197374,
                ],
                [
                    0.00268693,
                    0.00336729,
                    0.00335631,
                    0.00371109,
                    0.00415348,
                    0.00566635,
                    0.00707534,
                    0.00939567,
                    0.01155197,
                    0.01555290,
                    0.02240519,
                    0.03445009,
                    0.04561828,
                    0.06690585,
                    0.09416450,
                    0.10250460,
                ],
                [
                    0.00358969,
                    0.00434323,
                    0.00438610,
                    0.00381354,
                    0.00411750,
                    0.00566072,
                    0.00707434,
                    0.00895699,
                    0.01246695,
                    0.01580693,
                    0.02306735,
                    0.03410358,
                    0.04367388,
                    0.06758101,
                    0.08926881,
                    0.12343150,
                ],
                [
                    0.00478916,
                    0.00478556,
                    0.00578078,
                    0.00422990,
                    0.00431685,
                    0.00577808,
                    0.00723533,
                    0.01022452,
                    0.01328732,
                    0.01704136,
                    0.02466663,
                    0.03250499,
                    0.04797052,
                    0.06770993,
                    0.12583011,
                    0.15790748,
                ],
                [
                    0.00423290,
                    0.00501068,
                    0.00664310,
                    0.00394516,
                    0.00439798,
                    0.00557801,
                    0.00753300,
                    0.00923287,
                    0.01309449,
                    0.01713390,
                    0.02504395,
                    0.03339954,
                    0.05158937,
                    0.08876171,
                    0.17106001,
                    0.41276952,
                ],
                [
                    0.00438908,
                    0.00509963,
                    0.00688781,
                    0.00458198,
                    0.00452180,
                    0.00533587,
                    0.00769691,
                    0.00965982,
                    0.01237659,
                    0.01861790,
                    0.02491715,
                    0.03458397,
                    0.04204503,
                    0.11587611,
                    0.17575394,
                    0.37113671,
                ],
                [
                    0.00482951,
                    0.00558899,
                    0.00687475,
                    0.00454202,
                    0.00457075,
                    0.00556190,
                    0.00769927,
                    0.00948950,
                    0.01402657,
                    0.01816222,
                    0.02447586,
                    0.03670206,
                    0.04634167,
                    0.07745307,
                    0.12374932,
                    0.15781981,
                ],
                [
                    0.00430422,
                    0.00463247,
                    0.00453273,
                    0.00396030,
                    0.00438584,
                    0.00581539,
                    0.00784810,
                    0.00901551,
                    0.01275794,
                    0.01660788,
                    0.02445459,
                    0.03550369,
                    0.04684623,
                    0.06541014,
                    0.09610465,
                    0.12585320,
                ],
                [
                    0.00330149,
                    0.00395762,
                    0.00369135,
                    0.00364606,
                    0.00459375,
                    0.00590507,
                    0.00796759,
                    0.01011944,
                    0.01268789,
                    0.01592258,
                    0.02356959,
                    0.03562401,
                    0.04907667,
                    0.06758899,
                    0.09606561,
                    0.13603932,
                ],
                [
                    0.00268240,
                    0.00341955,
                    0.00402539,
                    0.00424809,
                    0.00472127,
                    0.00622247,
                    0.00800747,
                    0.00953849,
                    0.01198965,
                    0.01498599,
                    0.02174446,
                    0.03125558,
                    0.04410014,
                    0.06031013,
                    0.07677631,
                    0.09367269,
                ],
                [
                    0.00249304,
                    0.00287951,
                    0.00376659,
                    0.00428305,
                    0.00471227,
                    0.00574952,
                    0.00721403,
                    0.00832862,
                    0.01117202,
                    0.01436892,
                    0.01943166,
                    0.02617436,
                    0.03901571,
                    0.05122993,
                    0.06255972,
                    0.06791559,
                ],
            ]
        )

    def get_xt(self, x, y, width=105.0, height=68.0):
        """
        Get xT with Relative Coordinate Mapping.
        """
        # 1. Normalize (0~1)
        rel_x = max(0, min(x / width, 1.0 - 1e-9))
        rel_y = max(0, min(y / height, 1.0 - 1e-9))

        # 2. Map to Index
        c = int(rel_x * self.w)
        r = int(rel_y * self.h)

        return self.xt[r, c]

    def get_xt_flipped(self, x, y, width=105.0, height=68.0):
        # 180 degree rotation from opponent's perspective
        x_flip = width - x
        y_flip = height - y
        return self.get_xt(x_flip, y_flip, width, height)


# --- Calculation Logic ---


def _validate_episode_id(events: pd.DataFrame) -> None:
    if "episode_id" not in events.columns:
        raise ValueError("episode_id missing in events. xT requires episode_id.")
    if events["episode_id"].fillna(0).eq(0).all():
        raise ValueError("episode_id all 0. xT requires valid episode_id values.")


def _infer_period_directions_from_events(events: pd.DataFrame, tracking: pd.DataFrame) -> dict[tuple[int, str], str]:
    ev = event_postprocessing.prepare_events(events, tracking, map_episode_id=False)
    _validate_episode_id(ev)
    if "x" not in ev.columns:
        return {}

    def _to_sec(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, str):
            try:
                return utils.timestamp_to_seconds(val)
            except Exception:
                return np.nan
        return float(val)

    ev = ev.sort_values(["episode_id", "frame_id"], kind="stable").copy()
    next_pid = ev.groupby("episode_id")["player_id"].shift(-1)
    next_x = ev.groupby("episode_id")["x"].shift(-1)
    next_ts = ev.groupby("episode_id")["timestamp"].shift(-1)

    cur_ts = ev["timestamp"].apply(_to_sec)
    next_ts_sec = next_ts.apply(_to_sec)
    dt = next_ts_sec - cur_ts

    team = ev["player_id"].str.split("_").str[0]
    next_team = next_pid.str.split("_").str[0]
    same_team = next_pid.notna() & (team == next_team)
    valid = same_team & (dt <= 10)

    dx = next_x - ev["x"]

    ev = ev.assign(team=team, dx=dx, valid=valid)
    directions: dict[tuple[int, str], str] = {}
    for (pid, t), group in ev.groupby(["period_id", "team"]):
        sample = group[group["valid"]]["dx"].dropna()
        if sample.empty:
            continue
        med = float(sample.median())
        directions[(int(pid), t)] = "right" if med >= 0 else "left"
    return directions


def process_xt(events_in, tracking, is_pred=False, mode="penalty", period_directions=None):
    """
    mode: 'penalty' or 'no_penalty'
    """
    model = ExpectedThreat()
    events = events_in.copy()
    events = event_postprocessing.prepare_events(events, tracking, map_episode_id=False)
    _validate_episode_id(events)
    if "episode_id" in events.columns:
        events = events[events["episode_id"] > 0].copy()

    # Pre-calculate Team column for speed and grouping
    if "team" not in events.columns:
        events["team"] = events["player_id"].apply(lambda x: str(x).split("_")[0])

    results = []

    # Pre-calculate period start times for Relative Time
    period_start_times = {}
    for p in tracking["period_id"].unique():
        p_track = tracking[tracking["period_id"] == p]
        if not p_track.empty:
            period_start_times[p] = utils.series_to_seconds(p_track["timestamp"]).min()

    # Determine direction using event flow (fallback to previous if missing)
    process_xt.period_directions = {}
    if period_directions is None:
        period_directions = _infer_period_directions_from_events(events, tracking)

    unique_periods = tracking["period_id"].unique()
    for pid in unique_periods:
        for team in ["home", "away"]:
            process_xt.period_directions[(int(pid), team)] = period_directions.get((int(pid), team), "right")

    # Helper for Timestamp
    def get_timestamp(row):
        t_raw = row["timestamp"] if "timestamp" in row else 0
        if isinstance(t_raw, str):
            try:
                return utils.timestamp_to_seconds(t_raw)
            except Exception:
                return 0.0
        if pd.isna(t_raw):
            return 0.0
        return float(t_raw)

    # Stats Counters
    stats_same_team_total = 0
    stats_gap_broken = 0

    for i in range(len(events)):
        curr = events.iloc[i]

        # --- Define Start/End State ---
        team_str = curr["team"]
        if team_str not in ["home", "away"]:
            continue

        # Get Direction using PERIOD_ID
        ep_id = curr.get("episode_id", 0)  # Restore for result dict
        p_id = curr.get("period_id", 1)

        direction = process_xt.period_directions.get((p_id, team_str), "right")  # Default right

        if direction == "left":
            # Attacking Left (0). Need to Flip to match Grid (105).
            def normalize(x, y):
                return config.PITCH_X - x, config.PITCH_Y - y

        else:
            # Attacking Right (105). No Flip.
            def normalize(x, y):
                return x, y

        raw_start_x = curr.get("x", curr.get("start_x"))
        raw_start_y = curr.get("y", curr.get("start_y"))
        start_x, start_y = normalize(raw_start_x, raw_start_y)

        if i >= len(events) - 1:
            continue

        next_ev = events.iloc[i + 1]
        curr_team = team_str
        next_pid = str(next_ev["player_id"])
        next_team = next_pid.split("_")[0] if "_" in next_pid else "unknown"

        t_curr = get_timestamp(curr)
        t_next = get_timestamp(next_ev)
        dt = t_next - t_curr

        curr_episode = curr.get("episode_id", -1)
        next_episode = next_ev.get("episode_id", -1)
        if dt > 10.0 or curr_episode != next_episode:
            continue

        if curr_team == next_team:
            stats_same_team_total += 1
            if dt > 10.0:
                stats_gap_broken += 1

        if is_pred:
            raw_end_x = curr.get("end_x", next_ev.get("x", next_ev.get("start_x")))
            raw_end_y = curr.get("end_y", next_ev.get("y", next_ev.get("start_y")))
        else:
            raw_end_x = next_ev.get("x", next_ev.get("start_x"))
            raw_end_y = next_ev.get("y", next_ev.get("start_y"))

        end_x, end_y = normalize(raw_end_x, raw_end_y)
        is_success = curr_team == next_team

        # --- Calculation ---
        if start_x is None or start_y is None or end_x is None or end_y is None:
            continue
        out_of_play = False
        if not (0 <= start_x <= config.PITCH_X and 0 <= start_y <= config.PITCH_Y):
            start_val = 0.0
            out_of_play = True
        else:
            start_val = model.get_xt(start_x, start_y)
        if not (0 <= end_x <= config.PITCH_X and 0 <= end_y <= config.PITCH_Y):
            end_val = 0.0
            out_of_play = True
        else:
            end_val = model.get_xt(end_x, end_y)

        if out_of_play:
            xt_added = 0.0
            is_success = False
        elif is_success:
            # Standard xT Added
            xt_added = end_val - start_val
        else:
            # Failure Case
            if mode == "penalty":
                # User Formula: (Opponent End xT) - (My Start xT) [Implies Penalty context]
                # Standard xT Penalty: - (Opponent Threat) - (My Start Threat)
                # Opponent Perspective: Field is Flipped relative to My "End" position
                # My End X is in My Perspective.
                # Opponent Perspective X = 105 - My_End_X

                opp_x = config.PITCH_X - end_x
                opp_y = config.PITCH_Y - end_y

                opp_threat = model.get_xt(opp_x, opp_y)

                # Net Value = (Value After) - (Value Before)
                # Value After = - Opponent Threat (Negative utility for me)
                # Value Before = My Start Threat

                xt_added = -opp_threat - start_val
            else:
                # No-penalty turnover: lose only own threat (0 - start_xT), no opponent bonus.
                xt_added = -start_val

        # Calculate Relative Time
        p_id = curr["period_id"]
        t_abs = t_curr

        t_start = period_start_times.get(p_id, 0)
        rel_sec = max(0, t_abs - t_start)

        results.append(
            {
                "frame_id": curr["frame_id"],
                "timestamp": t_abs,
                "rel_seconds": rel_sec,
                "player_id": curr["player_id"],
                "team": team_str,
                "period_id": p_id,
                "episode_id": ep_id,
                "type": curr["event_type"],
                "start_x": raw_start_x,
                "start_y": raw_start_y,
                "end_x": raw_end_x,
                "end_y": raw_end_y,
                "xt_start": start_val,
                "xt_end": end_val if is_success else 0,
                "value": xt_added,
                "is_success": is_success,
            }
        )

    # Print Stats
    if stats_same_team_total > 0:
        ratio = (stats_gap_broken / stats_same_team_total) * 100
        lbl = "Predicted" if is_pred else "Actual"
        print(f"[{lbl}] Same-Team Transitions: {stats_same_team_total} | Gap > 10s: {stats_gap_broken} ({ratio:.2f}%)")

    return pd.DataFrame(results)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-9
    p = p.astype(float) + eps
    q = q.astype(float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def _frame_level_dominance_mae(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> float:
    def frame_dom(df: pd.DataFrame) -> pd.Series:
        g = df.groupby(["frame_id", "team"])["value"].sum().unstack(fill_value=0.0)
        if "home" not in g.columns:
            g["home"] = 0.0
        if "away" not in g.columns:
            g["away"] = 0.0
        return g["home"] - g["away"]

    dom_true = frame_dom(df_true)
    dom_pred = frame_dom(df_pred)
    merged = pd.concat({"gt": dom_true, "pred": dom_pred}, axis=1).fillna(0.0)
    return float(np.mean(np.abs(merged["pred"] - merged["gt"])))


def _frame_xt_by_team(df: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    team_sum = df.groupby(["frame_id", "team"])["value"].sum().unstack(fill_value=0.0)
    if "home" not in team_sum.columns:
        team_sum["home"] = 0.0
    if "away" not in team_sum.columns:
        team_sum["away"] = 0.0
    team_sum = team_sum.reset_index()

    meta_cols = ["frame_id", "period_id", "timestamp", "episode_id"]
    meta = tracking[meta_cols].copy()
    merged = meta.merge(team_sum, on="frame_id", how="left").fillna(0.0)
    return merged


def _dominance_labels(dom: pd.Series) -> pd.Series:
    return np.where(dom > 0, "home", np.where(dom < 0, "away", "draw"))


def _compute_team_accuracy(frame_gt: pd.DataFrame, frame_pred: pd.DataFrame) -> dict[str, float]:
    merged = (
        frame_gt[["frame_id", "home", "away"]]
        .merge(frame_pred[["frame_id", "home", "away"]], on="frame_id", suffixes=("_gt", "_pred"), how="outer")
        .fillna(0.0)
    )
    gt_dom = merged["home_gt"] - merged["away_gt"]
    pred_dom = merged["home_pred"] - merged["away_pred"]
    gt_label = _dominance_labels(gt_dom)
    pred_label = _dominance_labels(pred_dom)

    mask = gt_label != "draw"
    overall = float((gt_label[mask] == pred_label[mask]).mean()) if mask.any() else 0.0

    home_mask = gt_label == "home"
    away_mask = gt_label == "away"
    home_acc = float((pred_label[home_mask] == "home").mean()) if home_mask.any() else 0.0
    away_acc = float((pred_label[away_mask] == "away").mean()) if away_mask.any() else 0.0

    return {"overall": overall, "home": home_acc, "away": away_acc}


def _compute_team_accuracy_by_minute(frame_gt: pd.DataFrame, frame_pred: pd.DataFrame) -> pd.DataFrame:
    merged = (
        frame_gt[["frame_id", "period_id", "timestamp", "home", "away"]]
        .merge(frame_pred[["frame_id", "home", "away"]], on="frame_id", suffixes=("_gt", "_pred"), how="outer")
        .fillna(0.0)
    )
    merged["abs_seconds"] = utils.compute_abs_seconds(merged)
    merged = merged[merged["abs_seconds"].notna()].copy()
    merged["minute"] = (merged["abs_seconds"] // 60).astype(int)

    gt_dom = merged["home_gt"] - merged["away_gt"]
    pred_dom = merged["home_pred"] - merged["away_pred"]
    gt_label = _dominance_labels(gt_dom)
    pred_label = _dominance_labels(pred_dom)
    merged["gt_label"] = gt_label
    merged["pred_label"] = pred_label

    rows = []
    for minute, mdf in merged.groupby("minute"):
        home_mask = mdf["gt_label"] == "home"
        away_mask = mdf["gt_label"] == "away"
        home_acc = float((mdf.loc[home_mask, "pred_label"] == "home").mean()) if home_mask.any() else np.nan
        away_acc = float((mdf.loc[away_mask, "pred_label"] == "away").mean()) if away_mask.any() else np.nan
        rows.append({"minute": int(minute), "home_acc": home_acc, "away_acc": away_acc})
    return pd.DataFrame(rows).sort_values("minute")


def _compute_home_xt_absdiff_series(frame_gt: pd.DataFrame, frame_pred: pd.DataFrame) -> pd.DataFrame:
    merged = (
        frame_gt[["frame_id", "period_id", "timestamp", "home", "away"]]
        .merge(frame_pred[["frame_id", "home", "away"]], on="frame_id", suffixes=("_gt", "_pred"), how="outer")
        .fillna(0.0)
    )
    merged["abs_seconds"] = utils.compute_abs_seconds(merged)
    merged = merged[merged["abs_seconds"].notna()].copy()
    merged["minute"] = (merged["abs_seconds"] // 60).astype(int)

    rows = []
    for minute, mdf in merged.groupby("minute"):
        gt_home = mdf["home_gt"].sum()
        pred_home = mdf["home_pred"].sum()
        diff = abs(pred_home - gt_home)
        rows.append(
            {
                "minute": int(minute),
                "gt_home_xt": gt_home,
                "pred_home_xt": pred_home,
                "abs_diff": diff,
            }
        )

    return pd.DataFrame(rows).sort_values("minute")


def _compute_overall_home_xt_absdiff(frame_gt: pd.DataFrame, frame_pred: pd.DataFrame) -> dict[str, float]:
    gt_home = frame_gt["home"].sum()
    pred_home = frame_pred["home"].sum()
    diff = abs(pred_home - gt_home)
    return {"gt_home_xt": gt_home, "pred_home_xt": pred_home, "abs_diff": diff}


def _save_mode_outputs(
    res_act: pd.DataFrame,
    res_pred: pd.DataFrame,
    tracking: pd.DataFrame,
    out_dir: str,
    mode: str,
) -> None:
    if mode not in {"penalty", "nopenalty"}:
        raise ValueError(f"Unsupported mode: {mode}")

    is_penalty = mode == "penalty"
    mode_title = "penalty" if is_penalty else "no penalty"
    mode_suffix = "" if is_penalty else "_nopenalty"

    # Shared visualizations
    xt_viz.plot_bar_chart(res_act, "Actual xT", f"{out_dir}/xt_bar_act_{mode}.png")
    xt_viz.plot_bar_chart(res_pred, "Predicted xT", f"{out_dir}/xt_bar_pred_{mode}.png")
    xt_viz.plot_scatter_correlation(
        res_act,
        res_pred,
        f"{out_dir}/xt_scatter_act_vs_pred_{mode}.png",
        title_suffix=f"({mode_title})",
    )

    # Dominance visuals and metrics
    dom_path = f"{out_dir}/xt_dominance_timeline_gt_pred{mode_suffix}.png"
    merged_dom = xt_viz.plot_xt_dominance(res_act, res_pred, dom_path, bin_size=60)
    xt_viz.plot_dominance_absdiff(merged_dom, f"{out_dir}/xt_dominance_absdiff{mode_suffix}.png")
    xt_viz.plot_dominance_distributions(
        merged_dom["dominance_gt"].to_numpy(),
        merged_dom["dominance_pred"].to_numpy(),
        f"{out_dir}/xt_dominance_distribution{mode_suffix}.png",
    )

    dom_true = merged_dom["dominance_gt"].to_numpy()
    dom_pred = merged_dom["dominance_pred"].to_numpy()
    if dom_true.size and dom_pred.size:
        min_v = float(min(dom_true.min(), dom_pred.min()))
        max_v = float(max(dom_true.max(), dom_pred.max()))
        bins = np.linspace(min_v, max_v, 31) if max_v > min_v else np.linspace(min_v - 1, max_v + 1, 31)
        p, _ = np.histogram(dom_true, bins=bins)
        q, _ = np.histogram(dom_pred, bins=bins)
        jsd = _js_divergence(p, q)
        bin_mae = float(np.mean(np.abs(dom_pred - dom_true)))
    else:
        jsd = 0.0
        bin_mae = 0.0

    frame_mae = _frame_level_dominance_mae(res_act, res_pred)
    with open(f"{out_dir}/xt_dominance_metrics{mode_suffix}.txt", "w") as f:
        f.write(f"Mode: {mode_title}\n")
        f.write(f"Dominance JSD (bins): {jsd:.6f}\n")
        f.write(f"Dominance MAE (per bin): {bin_mae:.6f}\n")
        f.write(f"Dominance MAE (per frame): {frame_mae:.6f}\n")

    # Frame-level metrics and timelines
    frame_gt = _frame_xt_by_team(res_act, tracking)
    frame_pred = _frame_xt_by_team(res_pred, tracking)
    acc = _compute_team_accuracy(frame_gt, frame_pred)
    acc_timeline = _compute_team_accuracy_by_minute(frame_gt, frame_pred)
    xt_viz.plot_team_accuracy_timeline(acc_timeline, f"{out_dir}/xt_accuracy_timeline_by_team{mode_suffix}.png")

    overall_home_xt = _compute_overall_home_xt_absdiff(frame_gt, frame_pred)
    home_xt_series = _compute_home_xt_absdiff_series(frame_gt, frame_pred)
    xt_viz.plot_xt_absdiff_timeline(
        home_xt_series,
        f"{out_dir}/xt_home_xt_absdiff_timeline{mode_suffix}.png",
        title=f"Home xT Timeline ({mode_title})",
    )

    with open(f"{out_dir}/xt_frame_accuracy_metrics{mode_suffix}.txt", "w") as f:
        f.write(f"Mode: {mode_title}\n")
        f.write(f"Overall dominance accuracy: {acc['overall']:.4f}\n")
        f.write(f"Home dominance accuracy: {acc['home']:.4f}\n")
        f.write(f"Away dominance accuracy: {acc['away']:.4f}\n")
        f.write(f"GT home xT total: {overall_home_xt['gt_home_xt']:.4f}\n")
        f.write(f"Pred home xT total: {overall_home_xt['pred_home_xt']:.4f}\n")
        f.write(f"Home xT absolute diff: {overall_home_xt['abs_diff']:.4f}\n")


def main():
    try:
        match_id = "J03WR9"
        tracking, ev_act, ev_pred = event_postprocessing.load_match_data(match_id)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    out_dir = "application/xT/result"
    os.makedirs(out_dir, exist_ok=True)

    # Infer directions from GT events once (use for both GT/Pred)
    period_dirs = _infer_period_directions_from_events(ev_act, tracking)

    # --- 1. No Penalty Mode ---
    print("\n--- Running No Penalty Mode ---")
    res_act_nop = process_xt(ev_act, tracking, is_pred=False, mode="no_penalty", period_directions=period_dirs)
    res_pred_nop = process_xt(ev_pred, tracking, is_pred=True, mode="no_penalty", period_directions=period_dirs)
    _save_mode_outputs(res_act_nop, res_pred_nop, tracking, out_dir, mode="nopenalty")

    # --- 2. Penalty Mode (Main Analysis) ---
    print("\n--- Running Penalty Mode (Turnover Logic Enabled) ---")

    # 2.1 Actual + Pred
    res_act = process_xt(ev_act, tracking, is_pred=False, mode="penalty", period_directions=period_dirs)
    res_pred = process_xt(ev_pred, tracking, is_pred=True, mode="penalty", period_directions=period_dirs)
    _save_mode_outputs(res_act, res_pred, tracking, out_dir, mode="penalty")

    # 9. Shot Fairness Analysis (User Request)
    # print("\n--- Analyzing Shot Mechanics & Fairness ---")  # REMOVED: CSV not needed
    # analyze_shot_mechanics(res_act, res_pred)


def analyze_shot_mechanics(df_act, df_pred):
    """
    Analyzes how SHOT events are handled (excluding blocks), examining gaps.
    """
    # 1. Identify Shots in GT (Strictly 'shot', excluding 'block')
    mask_shot = df_act["type"].astype(str).str.contains("shot", case=False, na=False)
    mask_no_block = ~df_act["type"].astype(str).str.contains("block", case=False, na=False)

    shots_act = df_act[mask_shot & mask_no_block].copy()

    if shots_act.empty:
        print("No shot events found in Actual Data.")
        return

    print(f"\n[Actual Data] Found {len(shots_act)} Shot Events (excluding blocks).")

    # Pre-calculate Time Gaps & Next Event Type
    def build_next_info_map(df):
        df_s = df.sort_values("frame_id").reset_index(drop=True)
        # Shift -1 to get next event info
        df_s["next_ts"] = df_s["timestamp"].shift(-1)
        df_s["next_type"] = df_s["type"].shift(-1)

        df_s["gap"] = df_s["next_ts"] - df_s["timestamp"]

        # Return dataframe indexed by frame_id with relevant columns
        return df_s.set_index("frame_id")[["gap", "next_type"]]

    next_info_act = build_next_info_map(df_act)
    next_info_pred = build_next_info_map(df_pred)

    # 2. Match with Pred
    shots_act = shots_act.sort_values("frame_id")

    # IMPORTANT: Preserve Pred Frame ID for Gap Lookup
    df_pred_sorted = df_pred.sort_values("frame_id").copy()
    df_pred_sorted["pred_fid"] = df_pred_sorted["frame_id"]  # Backup key

    merged = pd.merge_asof(
        shots_act, df_pred_sorted, on="frame_id", suffixes=("_act", "_pred"), direction="nearest", tolerance=25
    )

    report_data = []

    print("\n[Shot Analysis Report - Matched Pred Events]")
    print(
        f"{'Frame':<8} | {'ActType':<10} | {'GapA':<6} | {'NextAct':<10} | {'PredType':<10} | {'GapP':<6} | {'NextPred':<10}"
    )
    print("-" * 90)

    for _, row in merged.iterrows():
        f_id = int(row["frame_id"])
        a_type = str(row["type_act"])
        a_val = row["value_act"]

        # Act Info
        if f_id in next_info_act.index:
            try:
                # Handle potential duplicate frame_ids by taking first
                res = next_info_act.loc[f_id]
                if isinstance(res, pd.DataFrame):
                    res = res.iloc[0]
                g_act = float(res["gap"]) if not pd.isna(res["gap"]) else 0.0
                nt_act = str(res["next_type"]) if not pd.isna(res["next_type"]) else "End"
            except:
                g_act = 0.0
                nt_act = "Error"
        else:
            g_act = 0.0
            nt_act = "Unknown"

        # Pred Info
        if pd.isna(row["type_pred"]):
            p_type = "NONE"
            p_val = 0.0
            p_fid = -1
            g_pred = 0.0
            nt_pred = "NONE"
        else:
            p_type = str(row["type_pred"])
            p_val = row["value_pred"]
            # Pred Frame ID
            p_fid = int(row["pred_fid"])

            if p_fid in next_info_pred.index:
                try:
                    res = next_info_pred.loc[p_fid]
                    if isinstance(res, pd.DataFrame):
                        res = res.iloc[0]
                    g_pred = float(res["gap"]) if not pd.isna(res["gap"]) else 0.0
                    nt_pred = str(res["next_type"]) if not pd.isna(res["next_type"]) else "End"
                except:
                    g_pred = 0.0
                    nt_pred = "Error"
            else:
                g_pred = 0.0
                nt_pred = "Unknown"

        print(
            f"{f_id:<8} | {a_type[:10]:<10} | {g_act:6.2f} | {nt_act[:10]:<10} | {p_type[:10]:<10} | {g_pred:6.2f} | {nt_pred[:10]:<10}"
        )

        report_data.append(
            {
                "frame_id": f_id,
                "act_type": a_type,
                "act_xt": a_val,
                "act_next_gap": g_act,
                "act_next_type": nt_act,
                "pred_type": p_type,
                "pred_xt": p_val,
                "pred_next_gap": g_pred,
                "pred_next_type": nt_pred,
            }
        )

    # Save Report
    out_csv = "application/xT/result/shot_fairness_report.csv"
    pd.DataFrame(report_data).to_csv(out_csv, index=False)
    print(f"\nSaved detailed shot comparison to: {out_csv}")


if __name__ == "__main__":
    main()
