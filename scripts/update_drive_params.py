"""
update_drive_params.py
======================
Recompute per-team drive-outcome distributions from actual season-to-date
play-by-play data, then write them into game_simulator.py's parameter block.

Run once per week after Monday Night Football finishes:

    python3 -I scripts/update_drive_params.py --week 11

What it updates in game_simulator.py
--------------------------------------
  BASE_DRIVE       — league-average drive outcomes (TD/FG/Punt/TO/MFG/Safety)
  BASE_PASS        — per-team pass rate by down
  DEF_EPA          — per-team defensive EPA adjustment
  OFF_EPA          — per-team offensive EPA adjustment (new field)

The script writes a dated parameter snapshot to:
    reports/drive_params_week<NN>.py

That snapshot is what game_simulator.py imports when --week is supplied.
If the snapshot already exists for the requested week, it is skipped
(predictions must not be altered post-kickoff — CLAUDE.md principle #1).

Install requirements (one time):
    pip install nfl-data-py pandas numpy

Usage:
    python3 -I scripts/update_drive_params.py --week 11 [--season 2026] [--dry-run]

Flags:
    --week      Current NFL week number (drives filter to weeks < this value)
    --season    Season year (default: 2026)
    --dry-run   Print the new parameters without writing any files
"""

import argparse
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

try:
    import nfl_data_py as nfl
except ImportError:
    print("ERROR: nfl-data-py not installed. Run: pip install nfl-data-py")
    sys.exit(1)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--week",    type=int, required=True,  help="Current week (data through week-1)")
parser.add_argument("--season",  type=int, default=2026,   help="Season year")
parser.add_argument("--dry-run", action="store_true",      help="Print params, don't write files")
args = parser.parse_args()

WEEK   = args.week
SEASON = args.season
OUT    = f"reports/drive_params_week{WEEK:02d}.py"

# Guard: never overwrite an existing snapshot (CLAUDE.md: predictions not edited)
if not args.dry_run and os.path.exists(OUT):
    print(f"✓ Snapshot already exists: {OUT}  (skipping — no post-kickoff edits)")
    sys.exit(0)

print(f"Loading {SEASON} play-by-play through Week {WEEK - 1}…")
pbp_all = nfl.import_pbp_data(years=[SEASON], downcast=True)
# Filter to completed weeks before the current one
pbp_all = pbp_all[pbp_all["week"] < WEEK].copy()
print(f"  {len(pbp_all):,} plays across {pbp_all['week'].nunique()} weeks\n")

if pbp_all.empty:
    print("No data found. Check --season value and nfl-data-py coverage.")
    sys.exit(1)

teams = sorted(pbp_all["posteam"].dropna().unique())
print(f"  Teams found: {len(teams)}")

# ── 1. League-average drive outcomes ─────────────────────────────────────────
# We approximate drive outcomes from play-by-play scoring events.
# A "drive" is identified by (game_id, drive).

drives = pbp_all.groupby(["game_id", "drive"]).agg(
    posteam=("posteam", "first"),
    td=("touchdown", "max"),
    fg_made=("field_goal_result", lambda x: (x == "made").any()),
    fg_missed=("field_goal_result", lambda x: (x == "no_good").any()),
    safety=("safety", "max"),
    fumble_lost=("fumble_lost", "max"),
    interception=("interception", "max"),
    result=("drive_end_transition", "first"),
).reset_index()

# Clean up: coerce bool columns
for col in ["td", "fg_made", "fg_missed", "safety", "fumble_lost", "interception"]:
    drives[col] = drives[col].fillna(0).astype(float)

n = len(drives)
if n == 0:
    print("No drives found — cannot compute parameters.")
    sys.exit(1)

td_rate      = drives["td"].mean()
fg_rate      = drives["fg_made"].mean()
to_rate      = (drives["fumble_lost"] + drives["interception"]).clip(upper=1).mean()
mfg_rate     = drives["fg_missed"].mean()
safety_rate  = drives["safety"].mean()
# Punt = whatever is left
punt_rate    = max(0.0, 1.0 - td_rate - fg_rate - to_rate - mfg_rate - safety_rate)

BASE_DRIVE = [
    round(td_rate,    4),
    round(fg_rate,    4),
    round(punt_rate,  4),
    round(to_rate,    4),
    round(mfg_rate,   4),
    round(safety_rate,4),
]

print(f"League-average drive outcomes (from {n:,} drives, {SEASON} Wks 1–{WEEK-1}):")
labels = ["TD", "FG", "Punt", "Turnover", "Missed FG", "Safety"]
for lbl, val in zip(labels, BASE_DRIVE):
    print(f"  {lbl:<12}: {val:.1%}")
print(f"  {'Sum':<12}: {sum(BASE_DRIVE):.1%}\n")

# ── 2. Per-team pass rate by down ─────────────────────────────────────────────
pbp_rp = pbp_all[pbp_all["play_type"].isin(["run", "pass"])].copy()

BASE_PASS = {}
for team in teams:
    t = pbp_rp[pbp_rp["posteam"] == team]
    team_pass = {}
    for down in [1, 2, 3, 4]:
        d = t[t["down"] == down]
        if len(d) >= 10:
            team_pass[down] = round(float((d["play_type"] == "pass").mean()), 3)
        else:
            # Fall back to league average for this down
            ld = pbp_rp[pbp_rp["down"] == down]
            team_pass[down] = round(float((ld["play_type"] == "pass").mean()), 3)
    BASE_PASS[team] = team_pass

# ── 3. Per-team EPA adjustments ───────────────────────────────────────────────
# OFF_EPA: mean EPA per play when the team has the ball
# DEF_EPA: mean EPA per play allowed when the team is on defense
#   (positive = worse defense; negative = better defense)
#   We store it as the OPPONENT's gain, so negative = good D for us.

OFF_EPA = {}
DEF_EPA = {}

for team in teams:
    off_plays = pbp_rp[pbp_rp["posteam"] == team]["epa"].dropna()
    def_plays = pbp_rp[pbp_rp["defteam"] == team]["epa"].dropna()

    OFF_EPA[team] = round(float(off_plays.mean()), 4) if len(off_plays) >= 20 else 0.0
    # DEF_EPA sign: same as game_simulator convention — positive = bad defense
    DEF_EPA[team] = round(float(def_plays.mean()), 4) if len(def_plays) >= 20 else 0.0

print("Top 5 offenses (EPA/play):")
for team, val in sorted(OFF_EPA.items(), key=lambda x: -x[1])[:5]:
    print(f"  {team}: {val:+.4f}")
print("\nTop 5 defenses (lowest EPA allowed):")
for team, val in sorted(DEF_EPA.items(), key=lambda x: x[1])[:5]:
    print(f"  {team}: {val:+.4f}")
print()

# ── 4. Write snapshot ─────────────────────────────────────────────────────────
snapshot = f'''"""
Drive parameter snapshot — {SEASON} Season, through Week {WEEK - 1}
Generated: {date.today()}  |  {n:,} drives, {len(pbp_all):,} plays

Import in game_simulator.py with --week {WEEK}:
    from reports.drive_params_week{WEEK:02d} import BASE_DRIVE, BASE_PASS, OFF_EPA, DEF_EPA
"""

# League-average drive outcomes [TD, FG, Punt, Turnover, MissedFG, Safety]
BASE_DRIVE = {BASE_DRIVE}

# Per-team pass rate by down  {{team: {{1: p1d, 2: p2d, 3: p3d, 4: p4d}}}}
BASE_PASS = {BASE_PASS!r}

# Per-team offensive EPA/play (higher = better offense)
OFF_EPA = {OFF_EPA!r}

# Per-team defensive EPA/play allowed (lower = better defense)
DEF_EPA = {DEF_EPA!r}
'''

if args.dry_run:
    print("── DRY RUN — parameters NOT written ──")
    print(snapshot[:2000])
else:
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w") as f:
        f.write(snapshot)
    print(f"✓ Wrote {OUT}")
    print(f"  Pass to game_simulator.py with:  python3 scripts/game_simulator.py HOME AWAY --week {WEEK}")

print("\nDone.")
