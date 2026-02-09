import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.getcwd())

from datatools import config, evaluate_events, shot_detection

# Use constants from shot_detection
CUTOFF_SCORE = shot_detection.CUTOFF_SCORE

# === MAIN ANALYSIS FUNCTIONS ===

def get_fn_fp_frames(gt_events, pred_events, window_frames=40):
    """
    Perform Window-Based Matching to identify FN and FP frames.
    
    Args:
        gt_events: Ground truth events dataframe
        pred_events: Predicted events dataframe (with is_pred_shot_union)
        window_frames: Matching window size (default: 40 frames = 1.6 seconds)
    
    Returns:
        tuple: (tp_frames, fn_frames, fp_frames)
    """
    shot_types = ["shot", "shot_penalty", "shot_freekick"]
    gt_shots = gt_events[gt_events["event_type"].isin(shot_types)].copy()
    pred_shots = pred_events[pred_events["is_pred_shot_union"]].copy()
    
    tp_frames = []
    fn_frames = []
    matched_pred_indices = set()
    
    # Calculate TP and FN
    for idx, gt_row in gt_shots.iterrows():
        gt_frame = gt_row["frame_id"]
        
        nearby_preds = pred_shots[
            (pred_shots["frame_id"] >= gt_frame - window_frames) &
            (pred_shots["frame_id"] <= gt_frame + window_frames)
        ]
        
        if nearby_preds.empty:
            fn_frames.append(gt_frame)
        else:
            tp_frames.append(gt_frame)
            matched_pred_indices.update(nearby_preds.index.tolist())
    
    # Calculate FP
    fp_frames = []
    for idx, pred_row in pred_shots.iterrows():
        if idx in matched_pred_indices:
            continue
        fp_frames.append(pred_row["frame_id"])
    
    return tp_frames, fn_frames, fp_frames


def analyze_shots(gt_path: str, pred_path: str):
    print(f"Loading GT: {gt_path}")
    print(f"Loading Pred: {pred_path}")
    
    gt_events = pd.read_parquet(gt_path)
    pred_events = pd.read_parquet(pred_path)
    
    # Baseline stats
    print("\n[Baseline] Running evaluate_events...")
    original_stats, _ = evaluate_events.evaluate_events(gt_events, pred_events)
    print("Original Stats (Note: Shot is likely missing or merged):")
    print(original_stats)
    
    # Prepare data & Detect Shots
    print("\n[Preprocessing] Running Shot Detection...")
    pred_events = shot_detection.detect_shots(pred_events)
    
    total_shots = pred_events["is_pred_shot_union"].sum()
    print(f"\n[Result] Total Pred Shots: {total_shots} (Mandatory & Score >= {CUTOFF_SCORE})")
    
    # === WINDOW-BASED MATCHING VALIDATION ===
    WINDOW_FRAMES = 40  # ±40 frames = ±1.6 seconds
    
    print("\n" + "="*80)
    print(f"VALIDATION RESULTS (Window ±{WINDOW_FRAMES} Frames / {WINDOW_FRAMES/25:.1f}s)")
    print("="*80)
    
    # === SCORE-PRIORITIZED MATCHING LOGIC ===
    # Find the BEST candidate (highest score) within the window for each GT shot.
    
    shot_types = ["shot", "shot_penalty", "shot_freekick"]
    gt_shots = gt_events[gt_events["event_type"].isin(shot_types)].copy()
    total_gt = len(gt_shots)
    
    # Pre-filter confident shots for FP calculation
    confident_pred_shots = pred_events[pred_events["is_pred_shot_union"]].copy()
    total_pred = len(confident_pred_shots)
    
    tp = 0
    fn_frames = []
    
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    for idx, gt_row in gt_shots.iterrows():
        gt_frame = gt_row["frame_id"]
        
        # Find ALL candidates within window
        all_candidates = pred_events[
            (pred_events["frame_id"] >= gt_frame - WINDOW_FRAMES) &
            (pred_events["frame_id"] <= gt_frame + WINDOW_FRAMES)
        ]
        
        # Valid candidates (Passing threshold)
        valid_candidates = all_candidates[all_candidates["is_pred_shot_union"]]
        
        if not valid_candidates.empty:
            # Pick the valid candidate with the HIGHEST TOTAL SCORE
            best_candidate = valid_candidates.loc[valid_candidates["score_total"].idxmax()]
            
            tp += 1
            matched_gt_indices.add(idx)
            matched_pred_indices.add(best_candidate.name) 
        else:
            # No valid candidates -> FN
            fn_frames.append(gt_frame)
    
    # Check Precision (FP)
    # FPs are confident shots that were NOT matched to any GT
    fp = 0
    fp_frames = []
    
    for idx, pred_row in confident_pred_shots.iterrows():
        # If this index was already matched to a GT shot, skip it
        if idx in matched_pred_indices:
            continue
            
        # Avoid Double Penalization:
        # Ignore checks if this Pred is within window of a GT (even if that GT matched a better Pred).
        # We only count it as FP if it is completely unmatched and far from any GT.
        
        is_near_any_gt = False
        for _, gt_row in gt_shots.iterrows():
             if abs(gt_row["frame_id"] - pred_row["frame_id"]) <= WINDOW_FRAMES:
                 is_near_any_gt = True
                 break
        
        if not is_near_any_gt:
            fp += 1
            fp_frames.append(pred_row["frame_id"])
    
    fn = total_gt - tp
    # Precision denominator is Total Preds (confident ones)
    recall = tp / total_gt * 100 if total_gt > 0 else 0.0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"\n[Final Results (Score-Prioritized Matching)]")
    print(f"GT Shots: {total_gt}")
    print(f"Pred Shots (Confident): {total_pred}")
    print(f"TP (GT Found): {tp} / {total_gt}")
    print(f"FN (GT Missed): {fn}")
    print(f"FP (Preds Unmatched): {fp}")
    print(f"Recall: {recall:.1f}%")
    print(f"Precision: {precision:.1f}%")
    print(f"F1: {f1:.2f}")
    
    print(f"\nFN Frames: {fn_frames}")
    
    # FP Summary
    if fp > 0:
        print(f"\n{'='*80}")
        print(f"FALSE POSITIVE SUMMARY")
        print(f"{'='*80}")
        print(f"Total FP: {fp}")
        print(f"FP Frames (First 10): {fp_frames[:10]}")
    
    print(f"\n{'='*80}")
    print("END OF ANALYSIS")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    GT_path = "data/output_events/event_rdp/J03WR9.parquet"
    Pred_path = "data/output_events/event_rdp_pred/J03WR9.parquet"
    
    if os.path.exists(GT_path) and os.path.exists(Pred_path):
        analyze_shots(GT_path, Pred_path)
    else:
        print("Data files not found. Please check paths in __main__.")
