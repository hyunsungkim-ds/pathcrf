import os
import sys

import numpy as np
import pandas as pd

# Add project root to path (assuming script is run from root or datatools)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# If running from root, current working directory is already in path
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from datatools import config

# --- Constants & Configuration ---
MATCH_NAME = "J03WR9"

# Pass Types
PASS_DIRECT = list(set(config.PASS_LIKE_OPEN) & {"pass"})
PASS_LIKE = [x for x in config.PASS_LIKE_OPEN if x not in (set(config.PASS_LIKE_OPEN) & {"pass", "keeper_punch"})]
PASS_SETPIECE = [x for x in config.SET_PIECE if x not in (set(config.SET_PIECE) & {"shot_penalty", "shot_freekick"})]
PASS_GK = list(set(config.PASS_LIKE_OPEN) & {"keeper_punch"}) + ["keeper_sweeper"]
PASS_SHOOTING = list(set(config.SET_PIECE) & {"shot_penalty", "shot_freekick"})

PASS_TYPES = PASS_DIRECT + PASS_LIKE + PASS_SETPIECE + PASS_GK + PASS_SHOOTING

# Control Types
CONTROL_DIRECT = list(set(config.INCOMING) & {"control"})
CONTROL_INCOMING = [x for x in config.INCOMING if x not in (set(config.INCOMING_GK) | {"control"})]
CONTROL_GK = [x for x in config.INCOMING_GK if x != "keeper_sweeper"]
CONTROL_TACKLE = list(set(config.MINOR) & {"tackle"})
CONTROL_DRIBBLE = list(set(config.MINOR) & {"take_on", "second_take_on"})
CONTROL_FAILED = ["bad_touch", "dispossessed"]

CONTROL_TYPES = CONTROL_DIRECT + CONTROL_INCOMING + CONTROL_GK + CONTROL_TACKLE + CONTROL_DRIBBLE + CONTROL_FAILED


def timestamp_to_seconds(ts):
    """Convert timestamp string (MM:SS) or float to seconds."""
    if isinstance(ts, str):
        try:
            parts = ts.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return 0.0
        except ValueError:
            return 0.0
    return float(ts) if ts is not None else 0.0


def get_compatible_types(pred_type):
    """Return list of GT types compatible with a given prediction type."""
    ptype = str(pred_type).lower()
    if ptype == "pass":
        return [t.lower() for t in PASS_TYPES]
    elif ptype == "control":
        return [t.lower() for t in CONTROL_TYPES]
    return []


def is_compatible(gt_type, pred_type):
    """Check if a GT event type is compatible with a prediction type."""
    compatible_list = get_compatible_types(pred_type)
    return str(gt_type).lower() in compatible_list


# --- Data Loading & Preprocessing ---


def load_and_preprocess_data(match_name=MATCH_NAME):
    """Load tracking and event data, apply preprocessing."""
    paths = {
        "tracking": f"data/sportec/tracking_processed/{match_name}.parquet",
        "act": f"data/output_events/event_rdp/{match_name}.parquet",
        "pred": f"data/output_events/event_rdp_pred/{match_name}.parquet",
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} data not found at: {path}")

    print("Loading data...")
    tracking = pd.read_parquet(paths["tracking"])
    events_act = pd.read_parquet(paths["act"])
    events_pred = pd.read_parquet(paths["pred"])

    # Preprocessing
    tracking = tracking[tracking["phase_id"] != 0]

    events_act = _map_episode_id(events_act, tracking)
    events_pred = _map_episode_id(events_pred, tracking)

    # Convert timestamps
    events_act["ts_sec"] = events_act["timestamp"].apply(timestamp_to_seconds)
    events_pred["ts_sec"] = events_pred["timestamp"].apply(timestamp_to_seconds)

    return events_act, events_pred


def _map_episode_id(events, tracking):
    """Map episode_id from tracking to events using frame_id."""
    if "episode_id" in events.columns and not events["episode_id"].fillna(0).eq(0).all():
        return events

    if "frame_id" not in tracking.columns:
        print("Error: tracking data missing 'frame_id'")
        return events

    # Drop NaNs
    if events["frame_id"].isnull().any():
        n_dropped = events["frame_id"].isnull().sum()
        print(f"Warning: Dropping {n_dropped} events with NaN frame_id")
        events = events.dropna(subset=["frame_id"])

    # Mapping
    ep_lookup = tracking.set_index("frame_id")["episode_id"].to_dict()
    events["episode_id"] = events["frame_id"].map(lambda x: ep_lookup.get(int(x), -1))

    # Drop unmapped
    missing = events["episode_id"] == -1
    if missing.any():
        print(f"Warning: Dropping {missing.sum()} events not found in tracking")
        events = events[~missing]

    return events


# --- Metrics Calculation ---


def _find_best_match(row, candidates, threshold, is_recall_mode):
    """Find the best matching candidate based on time difference."""
    best_match = None
    min_time_diff = float("inf")

    for _, cand in candidates.iterrows():
        # Determine GT and Pred based on mode
        evt_gt = row if is_recall_mode else cand
        evt_pred = cand if is_recall_mode else row

        # Check compatibility
        if not is_compatible(evt_gt["event_type"], evt_pred["event_type"]):
            continue

        # Check time difference
        diff = abs(cand["ts_sec"] - row["ts_sec"])
        if diff <= threshold and diff < min_time_diff:
            min_time_diff = diff
            best_match = cand

    return best_match


def match_events(source_df, target_df, threshold, is_recall_mode=True, type_stats=None):
    """
    Generic matching function for both Recall (GT->Pred) and Precision (Pred->GT).
    """
    matched_count = 0

    for idx, row in source_df.iterrows():
        # GT type extraction (for stats)
        gt_type = str(row["event_type"]).lower() if is_recall_mode else None

        if is_recall_mode and type_stats is not None:
            type_stats[gt_type]["total"] += 1
            type_stats[gt_type]["condition_passed"] += 1

        # Filter candidates by Player & Episode
        mask = (target_df["player_id"] == row["player_id"]) & (target_df["episode_id"] == row["episode_id"])
        candidates = target_df[mask]

        if candidates.empty:
            if is_recall_mode and type_stats is not None:
                type_stats[gt_type]["condition_failed"] += 0
            continue

        # Find best match using helper function
        best_match = _find_best_match(row, candidates, threshold, is_recall_mode)

        if best_match is not None:
            matched_count += 1

            # Stats updates
            if is_recall_mode and type_stats is not None:
                type_stats[gt_type]["matched"] += 1
                ptype = str(best_match["event_type"]).lower()
                if ptype == "pass":
                    type_stats[gt_type]["matched_to_pass"] += 1
                elif ptype == "control":
                    type_stats[gt_type]["matched_to_control"] += 1

            elif not is_recall_mode and type_stats is not None:
                # Precision mode stats update
                gt_type_match = str(best_match["event_type"]).lower()
                if gt_type_match in type_stats:
                    type_stats[gt_type_match]["pred_matched"] += 1

        # Count attempts (Precision mode only)
        if not is_recall_mode and type_stats is not None:
            for _, cand in candidates.iterrows():
                gt_cand_type = str(cand["event_type"]).lower()
                if is_compatible(cand["event_type"], row["event_type"]):
                    if gt_cand_type in type_stats:
                        type_stats[gt_cand_type]["pred_attempts"] += 1

    return matched_count


def calculate_metrics(events_act, events_pred, threshold, detailed=False):
    """Main metrics calculation wrapper."""

    # 1. Filter usable events
    pred_target = ["pass", "control"]
    gt_target = PASS_TYPES + CONTROL_TYPES
    gt_target_lower = [t.lower() for t in gt_target]

    pred_filtered = events_pred[events_pred["event_type"].astype(str).str.lower().isin(pred_target)].copy()
    gt_filtered = events_act[events_act["event_type"].astype(str).str.lower().isin(gt_target_lower)].copy()

    # 2. Prepare Stats Container
    type_stats = None
    if detailed:
        type_stats = {
            t: {
                "total": 0,
                "condition_passed": 0,
                "condition_failed": 0,
                "matched": 0,
                "matched_to_pass": 0,
                "matched_to_control": 0,
                "pred_attempts": 0,
                "pred_matched": 0,
            }
            for t in gt_filtered["event_type"].unique()
        }

    # 3. Calculate Recall (GT -> Pred)
    tp_gt = match_events(gt_filtered, pred_filtered, threshold, is_recall_mode=True, type_stats=type_stats)

    # 4. Calculate Precision (Pred -> GT)
    tp_pred = match_events(pred_filtered, gt_filtered, threshold, is_recall_mode=False, type_stats=type_stats)

    # 5. Compute Final Scores
    n_gt = len(gt_filtered)
    n_pred = len(pred_filtered)

    recall = tp_gt / n_gt if n_gt > 0 else 0.0
    precision = tp_pred / n_pred if n_pred > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_count": n_gt,
        "pred_count": n_pred,
        "type_stats": type_stats,
    }


# --- Reporting ---


def print_detailed_report(stats):
    """Print formatted detailed report table."""

    def _print_section(title, types_list):
        print(f"\n📊 {title}")
        print("-" * 100)
        print(f"{'GT Type':<15} {'Total':>6} {'Recall':>8} {'Matched':>7} {'Prec':>7} {'F1':>7}")
        print("-" * 100)

        target_types = [t.lower() for t in types_list]
        for gt_type in sorted(stats.keys()):
            if str(gt_type).lower() in target_types:
                s = stats[gt_type]
                if s["total"] == 0:
                    continue

                recall = s["matched"] / s["total"]
                # Precision: pred success for this type
                prec = s["pred_matched"] / s["pred_attempts"] if s["pred_attempts"] > 0 else 0.0
                f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0.0

                print(f"{gt_type:<15} {s['total']:>6} {recall:>7.1%} {s['matched']:>7} {prec:>7.1%} {f1:>7.1%}")

    _print_section("Pass Series (Pred=Pass)", PASS_TYPES)
    _print_section("Control Series (Pred=Control)", CONTROL_TYPES)


def main():
    try:
        events_act, events_pred = load_and_preprocess_data()

        # Simple Summary
        print("\n=== Event Metrics Summary ===")
        for th in [0.4, 0.6]:
            res = calculate_metrics(events_act, events_pred, th, detailed=False)
            print(f"Threshold {th}s: F1={res['f1']:.4f} (P={res['precision']:.4f}, R={res['recall']:.4f})")

        # Detailed Analysis
        print("\n" + "=" * 80)
        print("Detailed Analysis (Threshold 0.6s)")
        print("=" * 80)

        res_detailed = calculate_metrics(events_act, events_pred, 0.6, detailed=True)
        print_detailed_report(res_detailed["type_stats"])

        print("\nDone.")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
