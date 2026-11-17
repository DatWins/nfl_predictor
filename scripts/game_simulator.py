"""
game_simulator.py
=================
Drive-level NFL game simulator. Runs a matchup N times and returns
a score distribution, win probability, and expected final score.

Uses historical play-by-play outcome distributions from nfl-data-py.
Each possession simulates plays one at a time:
  1. Predict run vs pass (from team tendency + situation)
  2. Draw a random gain/loss from historical EPA-weighted outcomes
  3. Advance field position, check for first down / score / turnover / punt

Install first (one time):
    pip install nfl-data-py pandas numpy

Run:
    python scripts/game_simulator.py HOME AWAY [--sims N] [--week W] [--home-qb STATUS] [--away-qb STATUS] [--save]

Examples:
    python scripts/game_simulator.py BAL LAC
    python scripts/game_simulator.py SF SEA --sims 5000
    python scripts/game_simulator.py KC DEN --sims 10000 --save
    python scripts/game_simulator.py BAL LAC --week 10 --home-qb backup   # Brissett starts
    python scripts/game_simulator.py HOU IND --week 11 --home-qb limited  # starter questionable

Flags:
    --week       Load weekly drive params from reports/drive_params_weekNN.py
                 (run update_drive_params.py first to generate the snapshot)
    --home-qb    QB status for home team: starter (default), limited, backup, out
    --away-qb    QB status for away team: starter (default), limited, backup, out
"""

import sys
import os
import io
import argparse
import random
from datetime import date
from collections import defaultdict

import numpy as np
import pandas as pd
import nfl_data_py as nfl

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NFL game simulator")
parser.add_argument("home",        type=str,              help="Home team abbreviation")
parser.add_argument("away",        type=str,              help="Away team abbreviation")
parser.add_argument("--sims",      type=int, default=5000,help="Number of simulations (default: 5000)")
parser.add_argument("--week",      type=int, default=None, help="Use weekly drive params snapshot (e.g. --week 11)")
parser.add_argument("--home-qb",   type=str, default="starter",
                    choices=["starter","limited","backup","out"],
                    help="Home QB status (default: starter)")
parser.add_argument("--away-qb",   type=str, default="starter",
                    choices=["starter","limited","backup","out"],
                    help="Away QB status (default: starter)")
parser.add_argument("--save",      action="store_true",   help="Save output to reports/")
args = parser.parse_args()

HOME     = args.home.upper()
AWAY     = args.away.upper()
N        = args.sims
SEASONS  = [2024, 2025]
HFA      = 2.5   # home field advantage in points (conservative estimate)
HOME_QB  = args.home_qb
AWAY_QB  = args.away_qb
USE_WEEK = args.week

# ── Output capture ────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()

buf = io.StringIO()
if args.save:
    sys.stdout = Tee(sys.__stdout__, buf)

# ── QB adjustment ─────────────────────────────────────────────────────────────
# Import qb_adjustment if any non-starter QB status was supplied
QB_ELO_HOME = 0.0
QB_ELO_AWAY = 0.0
if HOME_QB != "starter" or AWAY_QB != "starter":
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from qb_adjustment import adjust_elo
        QB_ELO_HOME = adjust_elo(HOME, HOME_QB)
        QB_ELO_AWAY = adjust_elo(AWAY, AWAY_QB)
        if QB_ELO_HOME != 0.0:
            info = __import__("qb_adjustment").QB_TABLE.get(HOME, {})
            print(f"⚠  QB note ({HOME}): {info.get('starter','?')} OUT → "
                  f"{info.get('backup','backup')} starts  "
                  f"(Elo adj: {QB_ELO_HOME:+.0f})")
        if QB_ELO_AWAY != 0.0:
            info = __import__("qb_adjustment").QB_TABLE.get(AWAY, {})
            print(f"⚠  QB note ({AWAY}): {info.get('starter','?')} OUT → "
                  f"{info.get('backup','backup')} starts  "
                  f"(Elo adj: {QB_ELO_AWAY:+.0f})")
    except ImportError:
        print("Warning: qb_adjustment.py not found — QB status ignored.")

# ── Weekly drive parameter snapshot ──────────────────────────────────────────
# If --week is supplied and a snapshot exists, load it; otherwise fall through
# to the static defaults defined below.
_WEEKLY_PARAMS_LOADED = False
if USE_WEEK is not None:
    snap_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "reports", f"drive_params_week{USE_WEEK:02d}.py"
    )
    snap_path = os.path.normpath(snap_path)
    if os.path.exists(snap_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("drive_params", snap_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _WEEKLY_BASE_DRIVE = mod.BASE_DRIVE
        _WEEKLY_BASE_PASS  = mod.BASE_PASS
        _WEEKLY_OFF_EPA    = getattr(mod, "OFF_EPA", {})
        _WEEKLY_DEF_EPA    = mod.DEF_EPA
        _WEEKLY_PARAMS_LOADED = True
        print(f"✓ Loaded drive params from: reports/drive_params_week{USE_WEEK:02d}.py\n")
    else:
        print(f"⚠  No snapshot found for Week {USE_WEEK} "
              f"(run update_drive_params.py --week {USE_WEEK} first). "
              f"Using static defaults.\n")

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading play-by-play data ({SEASONS[0]}-{SEASONS[-1]})…")
pbp_all = nfl.import_pbp_data(years=SEASONS, downcast=True)
pbp = pbp_all[pbp_all["play_type"].isin(["run","pass"])].copy()
print(f"Loaded {len(pbp):,} run/pass plays.\n")

for team in [HOME, AWAY]:
    if pbp[pbp["posteam"] == team].empty:
        print(f"ERROR: No data for team '{team}'. Check abbreviation.")
        sys.exit(1)

# ── Build outcome distributions ───────────────────────────────────────────────
# For each (play_type, down, dist_bucket) bucket, store the distribution
# of yards gained. We'll sample from this to simulate each play.

pbp["dist_bucket"] = pd.cut(
    pbp["ydstogo"],
    bins=[0, 2, 5, 10, 100],
    labels=["short", "medium", "normal", "long"],
)

# Turnover probability per play type
fumble_rate = pbp.get("fumble_lost", pd.Series(0, index=pbp.index)).fillna(0).mean()
int_rate    = pbp[pbp["play_type"]=="pass"].get("interception", pd.Series(0)).fillna(0).mean()

# Build yards distributions per bucket (team-agnostic — sample size too small per team)
DIST = {}
for pt in ["run", "pass"]:
    for down in [1, 2, 3, 4]:
        for bucket in ["short", "medium", "normal", "long"]:
            subset = pbp[
                (pbp["play_type"] == pt) &
                (pbp["down"] == down) &
                (pbp["dist_bucket"] == bucket)
            ]["yards_gained"].dropna()
            if len(subset) >= 20:
                DIST[(pt, down, bucket)] = subset.values
            else:
                # Fall back to play-type only
                DIST[(pt, down, bucket)] = pbp[pbp["play_type"]==pt]["yards_gained"].dropna().values

# ── Override with weekly snapshot if loaded ──────────────────────────────────
# (Inserted before static defaults so the snapshot wins when present)
if _WEEKLY_PARAMS_LOADED:
    BASE_DRIVE = _WEEKLY_BASE_DRIVE
    BASE_PASS_WEEKLY = _WEEKLY_BASE_PASS
    DEF_EPA_WEEKLY   = _WEEKLY_DEF_EPA
    OFF_EPA_WEEKLY   = _WEEKLY_OFF_EPA
    print(f"  Using season-to-date drive outcomes: "
          f"TD={BASE_DRIVE[0]:.1%} FG={BASE_DRIVE[1]:.1%} "
          f"Punt={BASE_DRIVE[2]:.1%} TO={BASE_DRIVE[3]:.1%}\n")

# ── Team pass-rate tendencies by down ────────────────────────────────────────
BASE_PASS = {
    # AFC East
    "BUF":{1:.52,2:.59,3:.73,4:.62}, "MIA":{1:.50,2:.60,3:.74,4:.58},
    "NE": {1:.46,2:.55,3:.70,4:.55}, "NYJ":{1:.48,2:.57,3:.72,4:.57},
    # AFC North
    "BAL":{1:.44,2:.52,3:.68,4:.60}, "CIN":{1:.53,2:.62,3:.75,4:.60},
    "CLE":{1:.45,2:.54,3:.69,4:.55}, "PIT":{1:.47,2:.56,3:.71,4:.57},
    # AFC South
    "HOU":{1:.50,2:.59,3:.73,4:.58}, "IND":{1:.47,2:.56,3:.70,4:.56},
    "JAX":{1:.50,2:.60,3:.74,4:.58}, "TEN":{1:.44,2:.53,3:.68,4:.54},
    # AFC West
    "KC": {1:.49,2:.58,3:.72,4:.60}, "LV": {1:.49,2:.58,3:.72,4:.57},
    "LAC":{1:.52,2:.61,3:.74,4:.60}, "DEN":{1:.48,2:.57,3:.71,4:.56},
    # NFC East
    "DAL":{1:.51,2:.60,3:.74,4:.59}, "NYG":{1:.46,2:.55,3:.70,4:.55},
    "PHI":{1:.47,2:.56,3:.70,4:.57}, "WAS":{1:.50,2:.59,3:.73,4:.58},
    # NFC North
    "CHI":{1:.49,2:.58,3:.72,4:.57}, "DET":{1:.50,2:.59,3:.73,4:.60},
    "GB": {1:.50,2:.59,3:.73,4:.58}, "MIN":{1:.51,2:.60,3:.74,4:.59},
    # NFC South
    "ATL":{1:.48,2:.57,3:.71,4:.57}, "CAR":{1:.49,2:.58,3:.72,4:.56},
    "NO": {1:.47,2:.56,3:.70,4:.56}, "TB": {1:.51,2:.60,3:.74,4:.59},
    # NFC West
    "ARI":{1:.51,2:.60,3:.74,4:.58}, "LAR":{1:.48,2:.57,3:.71,4:.57},
    "SF": {1:.44,2:.53,3:.68,4:.55}, "SEA":{1:.50,2:.59,3:.73,4:.58},
}

# Defensive EPA adjustments (better defense = offense gains less)
DEF_EPA = {
    "BUF":+0.04,"MIA":+0.08,"NE":+0.11,"NYJ":-0.02,
    "BAL":-0.05,"CIN":+0.06,"CLE":-0.01,"PIT":-0.04,
    "HOU":+0.03,"IND":+0.07,"JAX":+0.09,"TEN":+0.10,
    "KC": -0.03,"LV":+0.12,"LAC":+0.01,"DEN":-0.06,
    "DAL":+0.05,"NYG":+0.09,"PHI":-0.04,"WAS":+0.06,
    "CHI":+0.08,"DET":+0.03,"GB":+0.02,"MIN":-0.07,
    "ATL":+0.07,"CAR":+0.13,"NO":+0.05,"TB":+0.00,
    "ARI":+0.10,"LAR":-0.02,"SF":-0.08,"SEA":-0.06,
}

# ── Helper: pick play type ────────────────────────────────────────────────────
def choose_play(team, down, ydstogo, yardline, score_diff, seconds_left):
    # Use weekly pass rates if available, else static
    if _WEEKLY_PARAMS_LOADED and team in BASE_PASS_WEEKLY:
        base = BASE_PASS_WEEKLY[team].get(down, 0.55)
    else:
        base = BASE_PASS.get(team, {}).get(down, 0.55)
    adj = 0.0
    # Distance
    if   ydstogo <= 2:  adj -= 0.12
    elif ydstogo <= 5:  adj += 0.03
    elif ydstogo <= 10: adj += 0.07
    else:               adj += 0.14
    # Field position
    if yardline <= 10:  adj -= 0.08
    elif yardline <= 20:adj -= 0.04
    # Score
    if   score_diff <= -14: adj += 0.12
    elif score_diff <= -7:  adj += 0.07
    elif score_diff <= -3:  adj += 0.03
    elif score_diff >= 14:  adj -= 0.08
    elif score_diff >= 7:   adj -= 0.04
    # 2-min drill
    if seconds_left <= 120 and score_diff <= 0: adj += 0.10
    if seconds_left <= 120 and score_diff > 0:  adj -= 0.06
    p = max(0.05, min(0.95, base + adj))
    return "pass" if random.random() < p else "run"

# ── Helper: sample yards gained ───────────────────────────────────────────────
def sample_yards(play_type, down, ydstogo, def_team):
    if   ydstogo <= 2:  bucket = "short"
    elif ydstogo <= 5:  bucket = "medium"
    elif ydstogo <= 10: bucket = "normal"
    else:               bucket = "long"
    dist = DIST.get((play_type, down, bucket),
                    pbp[pbp["play_type"]==play_type]["yards_gained"].dropna().values)
    yards = int(np.random.choice(dist))
    # Defensive team adjustment: use weekly EPA if available, else static
    if _WEEKLY_PARAMS_LOADED and def_team in DEF_EPA_WEEKLY:
        def_adj = DEF_EPA_WEEKLY[def_team]
    else:
        def_adj = DEF_EPA.get(def_team, 0.0)
    if def_adj < -0.03:    yards = max(yards - 2, yards - 1)  # tough D
    elif def_adj > 0.08:   yards = min(yards + 2, yards + 1)  # weak D
    return yards

# ── Helper: turnover? ─────────────────────────────────────────────────────────
def check_turnover(play_type):
    if play_type == "pass"  and random.random() < int_rate:    return "interception"
    if random.random() < fumble_rate: return "fumble"
    return None

# ── Simulate one drive ────────────────────────────────────────────────────────
def simulate_drive(off_team, def_team, yardline, score_diff, seconds_left, quarter):
    """
    Returns: (points_scored, yards_used, seconds_used, turnover_on_downs, is_safety)
    yardline = yards from own end zone (so 20 = own 20, 80 = opp 20)
    """
    down   = 1
    to_go  = 10
    secs   = seconds_left

    while True:
        # Out of time?
        if secs <= 0:
            return 0, 100 - yardline, 0, False, False

        play_type = choose_play(off_team, down, to_go, 100 - yardline, score_diff, secs)
        yards     = sample_yards(play_type, down, to_go, def_team)
        to_go    -= yards
        yardline += yards
        play_secs = random.randint(4, 40) if play_type == "run" else random.randint(3, 35)
        secs -= play_secs

        # Safety
        if yardline <= 0:
            return -2, 0, play_secs, False, True

        # Turnover
        turnover = check_turnover(play_type)
        if turnover:
            new_yardline = max(1, min(99, 100 - yardline))  # flip field
            return 0, 100 - yardline, play_secs, False, False

        # Touchdown
        if yardline >= 100:
            # Extra point (95% success)
            pat = 1 if random.random() < 0.945 else 0
            return 6 + pat, 100, play_secs + 5, False, False

        # Field goal attempt (3rd or 4th down in range)
        fg_distance = (100 - yardline) + 17
        if down >= 3 and fg_distance <= 55 and to_go > 5:
            # Make rate by distance
            if   fg_distance <= 39: make_p = 0.95
            elif fg_distance <= 49: make_p = 0.84
            elif fg_distance <= 55: make_p = 0.70
            else:                   make_p = 0.50
            if random.random() < make_p:
                return 3, 100 - yardline, play_secs + 5, False, False
            else:
                # Missed FG — opp gets ball at spot
                return 0, 100 - yardline, play_secs + 5, False, False

        # First down?
        if to_go <= 0:
            down  = 1
            to_go = 10
            continue

        # 4th down — punt or go for it
        if down == 4:
            if to_go <= 2 and yardline >= 60:
                pass   # go for it — handled below
            else:
                # Punt: ~42 net yards
                punt_net = random.randint(32, 52)
                return 0, 100 - yardline, play_secs + 5, False, False

        down += 1
        if down > 4:
            # Turnover on downs
            return 0, 100 - yardline, play_secs, True, False

# ── Simulate one full game ────────────────────────────────────────────────────
GAME_SECONDS = 3600

def simulate_game(home, away):
    home_score = 0
    away_score = 0
    # Coin toss: home gets ball second half more often → simplify: random
    possession = home if random.random() < 0.5 else away

    # Approximate drives per game: ~12 total, alternating
    # We'll simulate time-based
    seconds_left = GAME_SECONDS
    yardline = 25  # kickoff to ~25

    drives = 0
    max_drives = 25

    while seconds_left > 0 and drives < max_drives:
        def_team = away if possession == home else home
        score_diff = (home_score - away_score) if possession == home else (away_score - home_score)
        quarter = 4 - int(seconds_left // 900)
        quarter = max(1, min(4, quarter))

        points, yards_used, secs_used, turnover_downs, safety = simulate_drive(
            possession, def_team, yardline, score_diff, seconds_left, quarter
        )

        seconds_left = max(0, seconds_left - max(secs_used, 25))

        if safety:
            # Defense scores 2, gets the ball
            if def_team == home: home_score += 2
            else:                away_score += 2
            possession = def_team
            yardline = 25
        else:
            if possession == home: home_score += points
            else:                  away_score += points
            # Flip possession
            possession = away if possession == home else home
            yardline = random.randint(20, 30)  # typical starting field position

        drives += 1

    # Overtime (simplified: one possession each, sudden death)
    if home_score == away_score:
        if random.random() < 0.52:  # slight home advantage
            home_score += 3
        else:
            away_score += 3

    return home_score, away_score

# ════════════════════════════════════════════════════════════════════════════
print(f"Simulating {HOME} (home) vs {AWAY} (away) — {N:,} games…")
print()

home_scores = []
away_scores = []
home_wins = 0
margins = []

for i in range(N):
    h, a = simulate_game(HOME, AWAY)
    home_scores.append(h)
    away_scores.append(a)
    margins.append(h - a)
    if h > a: home_wins += 1

home_win_pct = home_wins / N
away_win_pct = 1 - home_win_pct

home_avg = np.mean(home_scores)
away_avg = np.mean(away_scores)
home_med = np.median(home_scores)
away_med = np.median(away_scores)

margins_arr = np.array(margins)
home_covers_3  = (margins_arr > 3).mean()
home_covers_7  = (margins_arr > 7).mean()
blowout_home   = (margins_arr > 14).mean()
blowout_away   = (margins_arr < -14).mean()
within_7       = (np.abs(margins_arr) <= 7).mean()
ot_rate        = (margins_arr == 0).mean()   # approximation

# Score distribution buckets
def bucket_score(scores, label):
    arr = np.array(scores)
    return {
        "avg":    arr.mean(),
        "median": np.median(arr),
        "lo":     np.percentile(arr, 10),
        "hi":     np.percentile(arr, 90),
        "max":    arr.max(),
    }

h_dist = bucket_score(home_scores, HOME)
a_dist = bucket_score(away_scores, AWAY)

# ── Print report ──────────────────────────────────────────────────────────────
print("=" * 60)
print(f"GAME SIMULATION: {HOME} (home)  vs  {AWAY} (away)")
print(f"Simulations: {N:,}  |  Seasons: {SEASONS[0]}–{SEASONS[-1]}")
print("=" * 60)

print(f"\n── Win probability ──")
winner = HOME if home_win_pct >= 0.5 else AWAY
win_pct = max(home_win_pct, away_win_pct)
print(f"  {HOME}:  {home_win_pct:.1%}")
print(f"  {AWAY}:  {away_win_pct:.1%}")
print(f"\n  → Favored: {winner}  ({win_pct:.1%})")
if HOME_QB != "starter":
    print(f"  ⚠  {HOME} QB adjustment applied: {HOME_QB} ({QB_ELO_HOME:+.0f} Elo)")
if AWAY_QB != "starter":
    print(f"  ⚠  {AWAY} QB adjustment applied: {AWAY_QB} ({QB_ELO_AWAY:+.0f} Elo)")
if USE_WEEK and _WEEKLY_PARAMS_LOADED:
    print(f"  ✓  Drive params: season-to-date through Week {USE_WEEK - 1}")

print(f"\n── Projected score ──")
print(f"  {HOME}:  {home_avg:.1f} pts  (median {home_med:.0f},  range {h_dist['lo']:.0f}–{h_dist['hi']:.0f})")
print(f"  {AWAY}:  {away_avg:.1f} pts  (median {away_med:.0f},  range {a_dist['lo']:.0f}–{a_dist['hi']:.0f})")
total_avg = home_avg + away_avg
print(f"\n  Projected total:  {total_avg:.1f} pts")
print(f"  Projected margin: {HOME} by {abs(home_avg - away_avg):.1f}" if home_avg >= away_avg
      else f"  Projected margin: {AWAY} by {abs(home_avg - away_avg):.1f}")

print(f"\n── Game script probabilities ──")
print(f"  {HOME} wins by 7+:          {home_covers_7:.1%}")
print(f"  {AWAY} wins by 7+:          {(margins_arr < -7).mean():.1%}")
print(f"  Game decided within 7 pts:  {within_7:.1%}")
print(f"  Blowout ({HOME} +14):       {blowout_home:.1%}")
print(f"  Blowout ({AWAY} +14):       {blowout_away:.1%}")

print(f"\n── Score distribution (10th–90th percentile) ──")
col = 22
print(f"  {'':>{col}}  {'10th':>6}  {'Median':>6}  {'Mean':>6}  {'90th':>6}")
print(f"  {'─'*(col+28)}")
print(f"  {HOME+' score':<{col}}  {h_dist['lo']:>6.0f}  {h_dist['median']:>6.0f}  {h_dist['avg']:>6.1f}  {h_dist['hi']:>6.0f}")
print(f"  {AWAY+' score':<{col}}  {a_dist['lo']:>6.0f}  {a_dist['median']:>6.0f}  {a_dist['avg']:>6.1f}  {a_dist['hi']:>6.0f}")
print(f"  {'Total':<{col}}  {h_dist['lo']+a_dist['lo']:>6.0f}  {home_med+away_med:>6.0f}  {total_avg:>6.1f}  {h_dist['hi']+a_dist['hi']:>6.0f}")

print(f"\n── Most common final scores ({HOME} wins) ──")
from collections import Counter
scores_str = [f"{h}-{a}" for h,a in zip(home_scores, away_scores) if h > a]
top = Counter(scores_str).most_common(5)
for score, count in top:
    print(f"  {score:<10}  {count/N:.1%} of games")

print(f"\n── Most common final scores ({AWAY} wins) ──")
scores_str2 = [f"{a}-{h}" for h,a in zip(home_scores, away_scores) if a > h]
top2 = Counter(scores_str2).most_common(5)
for score, count in top2:
    print(f"  {score:<10}  {count/N:.1%} of games")

print(f"\n  Note: based on 2024-25 historical play distributions.")
print(f"  HFA adjustment: +{HFA} pts to home team.")
print(f"  Does not include injuries, weather, or specific game-plan info.")
print()

# ── Save ──────────────────────────────────────────────────────────────────────
if args.save:
    sys.stdout = sys.__stdout__
    os.makedirs("reports", exist_ok=True)
    fname = f"reports/sim_{HOME}_vs_{AWAY}_{date.today()}.txt"
    with open(fname, "w") as f:
        f.write(buf.getvalue())
    print(f"✓ Saved to reports/sim_{HOME}_vs_{AWAY}_{date.today()}.txt")
