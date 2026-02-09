"""
Generate GT+Pred COMBINED animations for FP (False Positive) shot cases.
Simple and focused: imports logic from datatools.shot_detection and analysis.analyze_shots
"""

import os
import sys
import pandas as pd
import numpy as np
import subprocess

sys.path.append(os.getcwd())

from datatools.animator import Animator
from datatools.shot_detection import (
    prepare_prediction_data, infer_team_goal_mapping, infer_goalkeepers,
    get_pass_event_mask, get_goal_out_bonus_mask, is_in_attacking_third,
    is_attacking_team_event, has_gk_intervention,
    calculate_shot_detection
)
from analysis.analyze_shots import get_fn_fp_frames # Reusable matching function from analyze_shots

GT_PATH = "data/output_events/event_rdp/J03WR9.parquet"
PRED_PATH = "data/output_events/event_rdp_pred/J03WR9.parquet"
TRACKING_PATH = "data/sportec/tracking_processed/J03WR9.parquet"
OUTPUT_DIR = "analysis/shot_animations"


def generate_combined_animation(gt_events, pred_events, tracking, target_frame, case_num, case_type="FP"):
    """
    Generate GT+Pred COMBINED animation for a single case (FP or FN).
    
    Args:
        gt_events: GT events dataframe
        pred_events: Pred events dataframe (with scores)
        tracking: Tracking dataframe
        target_frame: Target frame ID
        case_num: Case number for filename
        case_type: "FP" or "FN"
    """
    # Time window: -50 to +150 frames
    start_frame = target_frame - 50
    end_frame = target_frame + 150
    
    # Tracking slice
    track_slice = tracking[
        (tracking["frame_id"] >= start_frame) &
        (tracking["frame_id"] <= end_frame)
    ].copy().reset_index(drop=True)
    
    if track_slice.empty:
        print(f"  ⚠️ No tracking data for frame {target_frame}")
        return

    # Event slices
    gt_slice = gt_events[
        (gt_events["frame_id"] >= start_frame) &
        (gt_events["frame_id"] <= end_frame)
    ].copy().reset_index(drop=True)
    
    pred_slice = pred_events[
        (pred_events["frame_id"] >= start_frame) &
        (pred_events["frame_id"] <= end_frame)
    ].copy()
    pred_slice = pred_slice.rename(columns={"start_x": "x", "start_y": "y"})
    
    # Calculate durations for event persistence
    for df_slice in [gt_slice, pred_slice]:
        if not df_slice.empty and "frame_id" in df_slice.columns:
            df_slice.sort_values("frame_id", inplace=True)
            df_slice.reset_index(drop=True, inplace=True)
            next_frame = df_slice["frame_id"].shift(-1)
            df_slice["duration"] = (next_frame - df_slice["frame_id"]).fillna(50).astype(int)
            df_slice.loc[df_slice["duration"] < 25, "duration"] = 25
    
    # Highlight FP pred event (closest to fp_frame)
    if not pred_slice.empty:
        pred_slice["_dist"] = abs(pred_slice["frame_id"] - target_frame)
        closest_idx = pred_slice["_dist"].idxmin()
        closest_frame = pred_slice.loc[closest_idx, "frame_id"]
        
        mask = pred_slice["frame_id"] == closest_frame
        if mask.any():
            evt_type = pred_slice.loc[mask, "event_type"].iloc[0] if "event_type" in pred_slice.columns else "unknown"
            
            if evt_type == "control":
                pred_slice.loc[mask, "show_circle"] = True
                pred_slice.loc[mask, "circle_color"] = "orange"  # FP color
            else:
                pred_slice.loc[mask, "show_arrow"] = True
                pred_slice.loc[mask, "arrow_color"] = "orange"  # FP color
                if "next_x" in pred_slice.columns:
                    pred_slice.loc[mask, "end_x"] = pred_slice.loc[mask, "next_x"]
                    pred_slice.loc[mask, "end_y"] = pred_slice.loc[mask, "next_y"]
    
    # 1. Generate GT View
    anim_gt = Animator(
        track_dict={"main": track_slice},
        events=gt_slice,
        show_events=True,
        show_times=True,
        play_speed=1,
        anonymize=False
    )
    
    # Temporary filenames
    save_path_gt = os.path.join(OUTPUT_DIR, f"temp_{case_type}_{case_num}_GT.mp4")
    anim_gt.run(fps=25).save(save_path_gt, writer="ffmpeg")
    
    # 2. Generate Pred View (with predicted ball trajectory)
    track_slice_players = track_slice.drop(columns=[c for c in track_slice.columns if "ball" in c])
    
    # Use exact frames from track_slice to ensure index alignment
    pred_ball_df = track_slice[["frame_id"]].copy()
    
    pe_unique = pred_slice.groupby("frame_id")[["x", "y"]].first().reset_index()
    # print(f"  [Debug] Pred events count: {len(pe_unique)}")
    
    pred_ball_df = pred_ball_df.merge(pe_unique, on="frame_id", how="left")
    pred_ball_df = pred_ball_df.rename(columns={"x": "ball_x", "y": "ball_y"})
    pred_ball_df[["ball_x", "ball_y"]] = pred_ball_df[["ball_x", "ball_y"]].interpolate(limit=50)
    
    track_dict_pred = {
        "main": track_slice_players,
        "cyan": pred_ball_df
    }
    
    anim_pred = Animator(
        track_dict=track_dict_pred,
        events=pred_slice,
        show_events=True,
        show_times=True,
        play_speed=1,
        anonymize=False
    )
    
    save_path_pred = os.path.join(OUTPUT_DIR, f"temp_{case_type}_{case_num}_PRED.mp4")
    anim_pred.run(fps=25).save(save_path_pred, writer="ffmpeg")
    
    # 3. Combine Side-by-Side (GT left, Pred right)
    filename = f"{case_type}_case_{case_num:02d}_COMBINED_frame{target_frame}.mp4"
    save_path = os.path.join(OUTPUT_DIR, filename)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", save_path_gt,
        "-i", save_path_pred,
        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
        "-map", "[v]",
        save_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Cleanup temps
    if os.path.exists(save_path_gt): os.remove(save_path_gt)
    if os.path.exists(save_path_pred): os.remove(save_path_pred)


def main():
    # Load Data
    print(f"Loading GT: {GT_PATH}")
    gt_events = pd.read_parquet(GT_PATH)
    
    print(f"Loading Pred: {PRED_PATH}")
    pred_events = pd.read_parquet(PRED_PATH)
    
    print(f"Loading Tracking: {TRACKING_PATH}")
    tracking = pd.read_parquet(TRACKING_PATH)
    
    # --- Apply V5 Scoring Logic ---
    pred_events = prepare_prediction_data(pred_events)
    
    # Apply all V5 rules
    is_pass = get_pass_event_mask(pred_events)
    # Detect Goal Out bonus mask (Need to import get_goal_out_bonus_mask if used, or infer)
    
    is_in_attack_zone = is_in_attacking_third(pred_events)
    team_mapping = infer_team_goal_mapping(pred_events)
    is_attacking_team = is_attacking_team_event(pred_events, team_mapping)
    
    gk_map = infer_goalkeepers(pred_events)
    has_gk = has_gk_intervention(pred_events, gk_map)
    has_goal_out = get_goal_out_bonus_mask(pred_events)
    
    rule_results = {
        "is_pass_event": is_pass,
        "is_attacking_team": is_attacking_team,
        "is_in_attacking_zone": is_in_attack_zone,
        "has_gk_intervention": has_gk,
        "has_goal_out": has_goal_out
    }
    
    pred_events = calculate_shot_detection(pred_events, rule_results)
    
    # --- Window-Based Matching ---
    tp_frames, fn_frames, fp_frames = get_fn_fp_frames(gt_events, pred_events, window_frames=40)
    
    shot_types = ["shot", "shot_penalty", "shot_freekick"]
    total_gt = len(gt_events[gt_events["event_type"].isin(shot_types)])
    total_pred = len(tp_frames) + len(fp_frames)
    
    print(f"Total GT Shots: {total_gt}")
    print(f"Total Pred Shots: {total_pred}")
    print(f"TP: {len(tp_frames)}, FP: {len(fp_frames)}, FN: {len(fn_frames)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate FP animations
    print(f"\nGenerating animations for {len(fp_frames)} FP cases...")
    for i, fp_frame in enumerate(fp_frames, 1):
        print(f"[{i}/{len(fp_frames)}] Processing FP Frame {fp_frame}...")
        try:
            generate_combined_animation(gt_events, pred_events, tracking, fp_frame, i, "FP")
            print(f"  ✅ Saved: FP_case_{i:02d}_COMBINED_frame{fp_frame}.mp4")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Generate FN animations
    print(f"\nGenerating animations for {len(fn_frames)} FN cases...")
    for i, fn_frame in enumerate(fn_frames, 1):
        print(f"[{i}/{len(fn_frames)}] Processing FN Frame {fn_frame}...")
        try:
            generate_combined_animation(gt_events, pred_events, tracking, fn_frame, i, "FN")
            print(f"  ✅ Saved: FN_case_{i:02d}_COMBINED_frame{fn_frame}.mp4")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\n✅ All animations complete!")
    print(f"Output directory: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
