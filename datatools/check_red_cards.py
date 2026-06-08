"""Check for red cards (incomplete teams) in tracking data files."""

import os
import re
import sys

import pandas as pd


def check_red_card(tracking: pd.DataFrame, team_size: int = 11) -> bool:
    """
    Check if all phases have exactly team_size players on each team.
    If not, print when the player count changes.
    Returns True if all phases are complete (no red cards).
    """
    all_complete = True

    for phase in sorted(tracking["phase_id"].unique()):
        if phase == 0:
            continue
        pt = tracking[tracking["phase_id"] == phase].dropna(axis=1, how="all")
        home_players = sorted({c[:-2] for c in pt.columns if re.match(r"home_\d+_x", c)})
        away_players = sorted({c[:-2] for c in pt.columns if re.match(r"away_\d+_x", c)})
        n_home, n_away = len(home_players), len(away_players)

        if n_home != team_size or n_away != team_size:
            ts_start = pt["timestamp"].iloc[0]
            ts_end = pt["timestamp"].iloc[-1]
            print(f"  Phase {phase}: home={n_home}, away={n_away} "
                  f"({ts_start:.0f}s - {ts_end:.0f}s)")
            all_complete = False

    if all_complete:
        print("  All phases: 22 players")

    return all_complete


def check_directory(tracking_dir: str, team_size: int = 11):
    """Check all parquet files in a directory for red cards."""
    files = sorted(f for f in os.listdir(tracking_dir) if f.endswith(".parquet"))
    print(f"Checking {len(files)} files in {tracking_dir}\n")

    incomplete = []
    for fname in files:
        match_id = os.path.splitext(fname)[0]
        tracking = pd.read_parquet(os.path.join(tracking_dir, fname))
        print(f"[{match_id}]")
        if not check_red_card(tracking, team_size):
            incomplete.append(match_id)

    print(f"\n{len(incomplete)} match(es) with red cards: {incomplete}")


if __name__ == "__main__":
    tracking_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sportec/tracking_processed"
    check_directory(tracking_dir)
