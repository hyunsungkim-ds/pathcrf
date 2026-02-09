import pandas as pd
import numpy as np
from datatools import config

# ============================================================================
# CONSTANTS
# ============================================================================

# Field Dimensions
PITCH_X = config.PITCH_X
PITCH_Y = config.PITCH_Y

PITCH_CENTER_Y = PITCH_Y / 2.0

GOAL_LEFT = np.array([0.0, PITCH_CENTER_Y])
GOAL_RIGHT = np.array([PITCH_X, PITCH_CENTER_Y])

# Detection Thresholds
MAX_DISTANCE_TO_GOAL = 35.0
SIDELINE_EXCLUSION_MARGIN = 15.0
ATTACKING_ZONE_THRESHOLD = 70.0
GK_INTERVENTION_MAX_DISTANCE = 5.0

# Speed Thresholds
SPEED_THRESHOLD_NEAR = 12.0          # For distance < 18m
SPEED_THRESHOLD_FAR_BASE = 12.0      # Base for distance >= 18m
SPEED_THRESHOLD_FAR_SLOPE = 0.25     # Slope for linear increase
SPEED_DISTANCE_BOUNDARY = 18.0       # Distance boundary

# Model Parameters
CUTOFF_SCORE = 145.0
WEIGHT_SPEED = 1.0
WEIGHT_DIRECTION = 40.0
WEIGHT_POSITION = 70.0
WEIGHT_CROSS_PENALTY = 50.0

# Bonus Scores
BONUS_ATTACKING_TEAM = 25.0
BONUS_GK_INTERVENTION = 30.0
BONUS_GOAL_OUT = 25.0

BETA_LATERAL = 2.2           # Lateral Penalty (Ellipse Shape)
MAX_EFFECTIVE_DIST = 50.0    # Effective Distance Range

# === HELPER FUNCTIONS ===

def prepare_prediction_data(pred_events):
    pred_events = pred_events.sort_values(["period_id", "frame_id"]).reset_index(drop=True)
    
    # Next event info
    pred_events["next_x"] = pred_events["start_x"].shift(-1)
    pred_events["next_y"] = pred_events["start_y"].shift(-1)
    pred_events["next_player_id"] = pred_events["player_id"].shift(-1)
    
    # Team info from player_id (format: "home_28" or "away_15")
    pred_events["team_id"] = pred_events["player_id"].astype(str).apply(
        lambda x: x.split('_')[0] if '_' in x else np.nan
    )
    pred_events["next_team_id"] = pred_events["team_id"].shift(-1)
    
    # Handle period boundaries
    period_change = pred_events["period_id"] != pred_events["period_id"].shift(-1)
    pred_events.loc[period_change, ["next_x", "next_y", "next_player_id", "next_team_id"]] = np.nan
    
    # Adjust out event positions for goal line outs
    # Goal line out: X is goal line (0 or 105), Y should be current Y (not fixed 34.0)
    is_next_out_left = pred_events["next_player_id"] == "out_left"
    is_next_out_right = pred_events["next_player_id"] == "out_right"
    
    pred_events.loc[is_next_out_left, "next_x"] = 0.0
    pred_events.loc[is_next_out_left, "next_y"] = pred_events.loc[is_next_out_left, "start_y"]
    
    pred_events.loc[is_next_out_right, "next_x"] = 105.0
    pred_events.loc[is_next_out_right, "next_y"] = pred_events.loc[is_next_out_right, "start_y"]
    
    # Distances to goals
    curr_pos = pred_events[["start_x", "start_y"]].values
    dist_left = np.linalg.norm(curr_pos - GOAL_LEFT, axis=1)
    dist_right = np.linalg.norm(curr_pos - GOAL_RIGHT, axis=1)
    
    pred_events["dist_left"] = dist_left
    pred_events["dist_right"] = dist_right
    pred_events["dist_to_goal"] = np.minimum(dist_left, dist_right)
    
    # Calculate speed (m/s at 25 fps)
    dx = pred_events["next_x"] - pred_events["start_x"]
    dy = pred_events["next_y"] - pred_events["start_y"]
    distance = np.sqrt(dx**2 + dy**2)
    
    # Calculate frame difference to handle gaps
    next_frame = pred_events["frame_id"].shift(-1)
    frame_diff = next_frame - pred_events["frame_id"]
    
    speed = np.where(
        frame_diff > 0,
        distance / (frame_diff / 25.0),
        0
    )
    pred_events["speed_mps"] = speed
    
    return pred_events


def infer_team_goal_mapping(pred_events):
    """Infer which goal each team attacks per period."""
    mapping = {}
    for period in pred_events['period_id'].dropna().unique():
        period_events = pred_events[pred_events['period_id'] == period].copy()
        
        # Exclude out events and out team from direction inference
        valid_moves = period_events[
            (period_events['event_type'] != 'out') &
            (period_events['team_id'] != 'out')
        ].dropna(subset=['next_x', 'start_x', 'team_id']).copy()
        
        if len(valid_moves) == 0:
            continue
        
        valid_moves['dx'] = valid_moves['next_x'] - valid_moves['start_x']
        team_direction = valid_moves.groupby('team_id')['dx'].mean()
        
        mapping[period] = {team: (dx > 0) for team, dx in team_direction.items()}
    return mapping


def infer_goalkeepers(pred_events):
    mapping = infer_team_goal_mapping(pred_events)
    gk_map = {}
    
    valid = pred_events.dropna(subset=['period_id', 'team_id', 'player_id', 'start_x'])
    avg_pos = valid.groupby(['period_id', 'team_id', 'player_id'])['start_x'].mean().reset_index()
    
    for period in mapping:
        gk_map[period] = {}
        for team, attacks_right in mapping[period].items():
            team_players = avg_pos[(avg_pos['period_id'] == period) & (avg_pos['team_id'] == team)]
            if team_players.empty:
                continue
            
            if attacks_right:
                gk_id = team_players.loc[team_players['start_x'].idxmin(), 'player_id']
            else:
                gk_id = team_players.loc[team_players['start_x'].idxmax(), 'player_id']
            
            gk_map[period][team] = gk_id
            
    return gk_map


# === RULE FUNCTIONS ===

def get_pass_event_mask(pred_events):
    return pred_events["event_type"] == "pass"


def get_goal_out_bonus_mask(pred_events):
    rule_mask = pd.Series(False, index=pred_events.index)
    LOOKAHEAD_STEPS = 10
    MAX_FRAME_DIFF = 100
    
    for i in range(1, LOOKAHEAD_STEPS + 1):
        future_type = pred_events["event_type"].shift(-i)
        future_x = pred_events["start_x"].shift(-i)
        future_frame = pred_events["frame_id"].shift(-i)
        
        is_out = future_type == "out"
        is_goalline = (future_x <= 0) | (future_x >= 105)
        is_within_time = (future_frame - pred_events["frame_id"]) <= MAX_FRAME_DIFF
        
        match = is_out & is_goalline & is_within_time
        rule_mask |= match.fillna(False)
    
    return rule_mask


def is_in_attacking_third(pred_events):
    attacking_right = pred_events["dist_right"] < pred_events["dist_left"]
    
    in_right_attack_zone = pred_events["start_x"] > ATTACKING_ZONE_THRESHOLD
    in_left_attack_zone = pred_events["start_x"] < (105 - ATTACKING_ZONE_THRESHOLD)
    
    return (attacking_right & in_right_attack_zone) | (~attacking_right & in_left_attack_zone)


def is_attacking_team_event(pred_events, team_goal_mapping):
    dist_right = pred_events["dist_right"]
    dist_left = pred_events["dist_left"]
    
    def is_attacking(row):
        period = row['period_id']
        team = row['team_id']
        idx = row.name
        
        event_attacks_right = dist_right[idx] < dist_left[idx]
        
        if pd.isna(team) or pd.isna(period):
            return False
        if period not in team_goal_mapping:
            return False
        
        team_attacks_right = team_goal_mapping[period].get(team, None)
        if team_attacks_right is None:
            return False
        
        return team_attacks_right == event_attacks_right
    
    return pred_events.apply(is_attacking, axis=1)


def has_gk_intervention(pred_events, gk_map):
    curr_team = pred_events["team_id"]
    next_team = pred_events["next_team_id"]
    is_turnover = (curr_team != next_team) & pd.notna(curr_team) & pd.notna(next_team)
    
    def is_gk(row):
        p, t, pid = row['period_id'], row['next_team_id'], row['next_player_id']
        if pd.isna(p) or pd.isna(t) or pd.isna(pid):
            return False
        return pid == gk_map.get(p, {}).get(t)
    
    is_next_player_gk = pred_events.apply(is_gk, axis=1)
    
    next_pos = pred_events[["next_x", "next_y"]].values
    d_left = np.linalg.norm(next_pos - GOAL_LEFT, axis=1)
    d_right = np.linalg.norm(next_pos - GOAL_RIGHT, axis=1)
    d_goal = np.minimum(d_left, d_right)
    
    is_gk_valid = is_next_player_gk & (d_goal < GK_INTERVENTION_MAX_DISTANCE)
    
    # Exclude cases where next event is out
    next_event_type = pred_events["event_type"].shift(-1)
    is_next_out = next_event_type == "out"
    
    return is_turnover & is_gk_valid & (~is_next_out)


# === SCORING FUNCTIONS ===

def calculate_geometric_score(pred_events):
    """
    Calculates the Unified Geometric Score.
    Combines Direction (Similarity) and Position (Elliptical Centrality) into a single metric,
    applying penalties for specific edge cases (Lateral, Sideline, Deep-Wide).
    """
    # --- 1. Direction Component (Similarity) ---
    vec_pass_x = pred_events["next_x"] - pred_events["start_x"]
    vec_pass_y = pred_events["next_y"] - pred_events["start_y"]
    
    curr_pos = pred_events[["start_x", "start_y"]].values
    dist_left = np.linalg.norm(curr_pos - GOAL_LEFT, axis=1)
    dist_right = np.linalg.norm(curr_pos - GOAL_RIGHT, axis=1)
    
    left_mask = dist_left < dist_right
    target_goal = np.where(left_mask[:, None], GOAL_LEFT, GOAL_RIGHT)
    
    vec_goal_x = target_goal[:, 0] - pred_events["start_x"]
    vec_goal_y = target_goal[:, 1] - pred_events["start_y"]
    
    norm_pass = np.sqrt(vec_pass_x**2 + vec_pass_y**2)
    norm_goal = np.sqrt(vec_goal_x**2 + vec_goal_y**2)
    dot = (vec_pass_x * vec_goal_x) + (vec_pass_y * vec_goal_y)
    
    similarity = np.divide(dot, (norm_pass * norm_goal), out=np.zeros_like(dot), where=(norm_pass * norm_goal) > 0)
    base_dir_score = similarity * WEIGHT_DIRECTION

    # --- 2. Position Component (Elliptical) ---
    dx_pos = 105.0 - pred_events["start_x"]
    dy_pos = np.abs(34.0 - pred_events["start_y"])
    d_eff = np.sqrt(dx_pos**2 + (BETA_LATERAL * dy_pos)**2)
    base_pos_score = np.maximum(0, 1.0 - (d_eff / MAX_EFFECTIVE_DIST)) * WEIGHT_POSITION

    # --- 3. Penalties (Hard Rules) ---
    dx = vec_pass_x.fillna(0)
    dy = vec_pass_y.fillna(0)
    lat_ratio = np.abs(dy) / (np.abs(dx) + 1e-6)
    
    y_dist = np.abs(pred_events["start_y"] - PITCH_CENTER_Y)
    dist_to_goal = pred_events["dist_to_goal"]
    
    # A. Lateral Penalty (Strict)
    mask_lat = (lat_ratio > 2.0) & (y_dist > 12.0)
    
    # B. Sideline Penalty
    mask_sideline = (y_dist > 30.0)
    
    # C. Conditional Deep Wide Penalty
    mask_deep_wide = (y_dist > 12.0) & (dist_to_goal > 20.0) & (similarity < 0.9)
    
    penalty_mask = mask_lat | mask_sideline | mask_deep_wide
    penalty_score = penalty_mask.astype(float) * WEIGHT_CROSS_PENALTY
    
    # --- Final Geometric Score ---
    final_score = base_dir_score + base_pos_score - penalty_score
    
    return pd.Series(final_score, index=pred_events.index), pd.Series(similarity, index=pred_events.index)


def calculate_speed_score(pred_events):
    dist_to_goal = pred_events["dist_to_goal"]
    speed = pred_events["speed_mps"]
    
    thresholds = np.where(
        dist_to_goal < SPEED_DISTANCE_BOUNDARY,
        SPEED_THRESHOLD_NEAR,
        SPEED_THRESHOLD_FAR_BASE + (dist_to_goal - SPEED_DISTANCE_BOUNDARY) * SPEED_THRESHOLD_FAR_SLOPE
    )
    
    surplus = speed - thresholds
    return surplus.clip(upper=20.0)


def calculate_shot_detection(pred_events, rule_results):
    def get_result(key):
        return rule_results[key].astype(float) if key in rule_results else 0.0
    
    # Mandatory filters
    mask_mandatory = (
        rule_results["is_pass_event"] &
        rule_results["is_in_attacking_zone"] & 
        (pred_events["dist_to_goal"] < MAX_EFFECTIVE_DIST) # Safety check
    )
    
    # Bonuses
    bonus_gk = get_result("has_gk_intervention") * BONUS_GK_INTERVENTION
    bonus_out = get_result("has_goal_out") * BONUS_GOAL_OUT
    bonus_att_team = get_result("is_attacking_team") * BONUS_ATTACKING_TEAM
    
    # Scoring components
    score_geom, dir_similarity = calculate_geometric_score(pred_events)
    speed_surplus = calculate_speed_score(pred_events)
    
    score_speed = speed_surplus * WEIGHT_SPEED
    
    BASE_SCORE_VAL = 47.0 
    
    total_score = (
        BASE_SCORE_VAL + 
        score_speed + 
        score_geom + 
        bonus_gk + 
        bonus_out + 
        bonus_att_team
    )
    
    # Final Mask
    final_mask = (
        mask_mandatory & 
        (total_score >= CUTOFF_SCORE)
    )
    
    pred_events["score_total"] = total_score
    pred_events["score_speed"] = score_speed
    pred_events["score_geom"] = score_geom
    pred_events["is_pred_shot_union"] = final_mask
    
    return pred_events


def detect_shots(pred_events):
    """
    High-level API to detect shots.
    Applies all preprocessing and rules.
    Returns modified dataframe with 'is_pred_shot_union' and 'score_total' columns.
    """
    pred_events = prepare_prediction_data(pred_events.copy())
    
    is_pass = get_pass_event_mask(pred_events)
    has_goal_out = get_goal_out_bonus_mask(pred_events)
    is_in_attack_zone = is_in_attacking_third(pred_events)
    
    team_mapping = infer_team_goal_mapping(pred_events)
    is_attacking_team = is_attacking_team_event(pred_events, team_mapping)
    
    gk_map = infer_goalkeepers(pred_events)
    mask_gk_intervention = has_gk_intervention(pred_events, gk_map)
    
    rule_results = {
        "is_pass_event": is_pass,
        "is_attacking_team": is_attacking_team,
        "is_in_attacking_zone": is_in_attack_zone,
        "has_gk_intervention": mask_gk_intervention,
        "has_goal_out": has_goal_out
    }
    
    pred_events = calculate_shot_detection(pred_events, rule_results)
    return pred_events
