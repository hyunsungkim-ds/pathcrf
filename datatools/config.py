PITCH_X, PITCH_Y = 105.0, 68.0
RDP_FRAME_SCALE = 1e-6
RDP_MIN_ANGLE = 15.0
POSS_MAX_DIST = 5.0

TRACKING_COLS = [
    "period_id",
    "timestamp",
    "phase_id",
    "episode_id",
    "ball_state",
    "ball_owning_team_id",
    "player_id",
]
EVENT_COLS = [
    "period_id",
    "phase_id",
    "start_frame",
    "start_time",
    "end_frame",
    "end_time",
    "from",
    "to",
    "type",
    "subtype",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]
ALIGNED_COLS = [
    "frame_id",
    "period_id",
    "episode_id",
    "timestamp",
    "player_id",
    "spadl_type",
    "success",
    "score",
]
TOUCH_COLS = [
    "frame_id",
    "period_id",
    "episode_id",
    "timestamp",
    "player_id",
    "event_type",
    "start_x",
    "start_y",
]

PASS_LIKE_OPEN = ["pass", "cross", "shot", "clearance", "shot_block", "keeper_punch", "keeper_save", "keeper_sweeper"]
SET_PIECE_OOP = ["throw_in", "goalkick", "corner_short", "corner_crossed"]
SET_PIECE = SET_PIECE_OOP + ["freekick_short", "freekick_crossed", "shot_freekick", "shot_penalty"]
INCOMING = ["control", "interception", "ball_recovery", "keeper_claim", "keeper_pick_up"]
MINOR = ["tackle", "take_on", "second_take_on", "foul", "bad_touch", "dispossessed"]

OUTGOING = PASS_LIKE_OPEN + SET_PIECE + ["tackle", "bad_touch", "dispossessed", "kick"]
SIMPLE_TYPES = ["kick", "control", "out"]
