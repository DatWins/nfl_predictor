"""
qb_adjustment.py
================
Apply a QB-starter Elo adjustment before computing win probabilities.

Usage (standalone):
    python3 -I scripts/qb_adjustment.py

Or import and call adjust_elo() from predict / weekly-update scripts:

    from scripts.qb_adjustment import adjust_elo
    home_elo_adj, away_elo_adj = adjust_elo(home_team, away_team,
                                             home_qb_status, away_qb_status)
    elo_diff = (home_elo + home_elo_adj) - (away_elo + away_elo_adj) + HFA

QB status values
----------------
  "starter"  — normal starter playing (default; no adjustment)
  "backup"   — starter is out, backup steps in
  "limited"  — starter is listed as questionable/limited (partial adjustment)
  "out"      — same as "backup" (alias)

Elo adjustments are relative to baseline (starter present = 0).
Values derived from published research on QB replacement effects:
  - FiveThirtyEight QB-adjusted Elo: ~-70 Elo for avg starter→backup swap
  - Ben Baldwin / nflscrapr analysis: ~1.5 ppg reduction = ~-50 Elo
  - We use -150 for a clear backup, -60 for limited starter.
  - Elite-to-backup (e.g. Lamar → Brissett) is at the high end: -150+.

The table below captures the CURRENT 2026 season QB situation.
Update each week when injuries are reported (Thursday injury report).

Format:
  TEAM: {
      "starter": "Name",         # current Week 1 starter
      "backup":  "Name",         # primary backup
      "elo_delta": N,            # Elo shift when backup starts (negative = worse)
  }

The elo_delta values are estimates. Where the backup is an experienced
starter (e.g. a former Pro Bowl QB), the delta is smaller.
"""

# ── 2026 QB Roster Table ─────────────────────────────────────────────────────
# elo_delta: points added to team Elo when BACKUP starts (almost always negative)
# Adjust these as the season progresses and backups prove themselves (or not).

QB_TABLE = {
    # AFC East
    "BUF": {"starter": "Josh Allen",          "backup": "Mike White",       "elo_delta": -150},
    "MIA": {"starter": "Tua Tagovailoa",       "backup": "Tyler Huntley",    "elo_delta": -120},
    "NE":  {"starter": "Drake Maye",           "backup": "Joe Milton",       "elo_delta":  -80},
    "NYJ": {"starter": "Aaron Rodgers",        "backup": "Tyrod Taylor",     "elo_delta": -100},
    # AFC North
    "BAL": {"starter": "Lamar Jackson",        "backup": "Jacoby Brissett",  "elo_delta": -150},
    "CIN": {"starter": "Joe Burrow",           "backup": "Jake Browning",    "elo_delta": -130},
    "CLE": {"starter": "Deshaun Watson",       "backup": "Dorian Thompson-Robinson", "elo_delta": -60},
    "PIT": {"starter": "Russell Wilson",       "backup": "Kyle Allen",       "elo_delta": -80},
    # AFC South
    "HOU": {"starter": "C.J. Stroud",          "backup": "Case Keenum",      "elo_delta": -120},
    "IND": {"starter": "Anthony Richardson",   "backup": "Joe Flacco",       "elo_delta": -50},
    "JAX": {"starter": "Trevor Lawrence",      "backup": "C.J. Beathard",    "elo_delta": -120},
    "TEN": {"starter": "Will Levis",           "backup": "Mason Rudolph",    "elo_delta": -80},
    # AFC West
    "KC":  {"starter": "Patrick Mahomes",      "backup": "Carson Wentz",     "elo_delta": -170},
    "LV":  {"starter": "Gardner Minshew",      "backup": "Aidan O'Connell",  "elo_delta":  -50},
    "LAC": {"starter": "Justin Herbert",       "backup": "Easton Stick",     "elo_delta": -130},
    "DEN": {"starter": "Bo Nix",               "backup": "Zach Wilson",      "elo_delta":  -70},
    # NFC East
    "DAL": {"starter": "Dak Prescott",         "backup": "Trey Lance",       "elo_delta": -120},
    "NYG": {"starter": "Daniel Jones",         "backup": "Tommy DeVito",     "elo_delta":  -50},
    "PHI": {"starter": "Jalen Hurts",          "backup": "Kenny Pickett",    "elo_delta": -130},
    "WAS": {"starter": "Jayden Daniels",       "backup": "Marcus Mariota",   "elo_delta": -100},
    # NFC North
    "CHI": {"starter": "Caleb Williams",       "backup": "Tyson Bagent",     "elo_delta": -100},
    "DET": {"starter": "Jared Goff",           "backup": "Hendon Hooker",    "elo_delta": -110},
    "GB":  {"starter": "Jordan Love",          "backup": "Sean Clifford",    "elo_delta": -120},
    "MIN": {"starter": "Sam Darnold",          "backup": "Nick Mullens",     "elo_delta":  -70},
    # NFC South
    "ATL": {"starter": "Kirk Cousins",         "backup": "Taylor Heinicke",  "elo_delta":  -60},
    "CAR": {"starter": "Bryce Young",          "backup": "Andy Dalton",      "elo_delta":  -30},
    "NO":  {"starter": "Derek Carr",           "backup": "Jake Haener",      "elo_delta":  -90},
    "TB":  {"starter": "Baker Mayfield",       "backup": "John Wolford",     "elo_delta":  -90},
    # NFC West
    "ARI": {"starter": "Kyler Murray",         "backup": "Clayton Tune",     "elo_delta": -110},
    "LAR": {"starter": "Matthew Stafford",     "backup": "Brett Rypien",     "elo_delta": -100},
    "SF":  {"starter": "Brock Purdy",          "backup": "Brandon Allen",    "elo_delta": -100},
    "SEA": {"starter": "Geno Smith",           "backup": "Drew Lock",        "elo_delta":  -70},
}

# Status → fraction of elo_delta applied
STATUS_WEIGHT = {
    "starter":  0.00,   # normal; no adjustment
    "limited":  0.40,   # questionable/limited — partial effect
    "backup":   1.00,   # starter is out
    "out":      1.00,   # alias for backup
}


def adjust_elo(team: str, qb_status: str = "starter") -> float:
    """
    Return the Elo adjustment for one team given their QB status this week.

    Parameters
    ----------
    team       : NFL team abbreviation (e.g. "BAL")
    qb_status  : one of "starter", "limited", "backup", "out"

    Returns
    -------
    float — Elo points to ADD to this team's rating (negative when backup starts)
    """
    qb_status = qb_status.lower().strip()
    if qb_status not in STATUS_WEIGHT:
        raise ValueError(f"Unknown QB status '{qb_status}'. Use: starter, limited, backup, out")

    info   = QB_TABLE.get(team.upper())
    if info is None:
        raise ValueError(f"Team '{team}' not found in QB_TABLE")

    weight = STATUS_WEIGHT[qb_status]
    return info["elo_delta"] * weight


def win_prob_with_qb(elo_home: float, elo_away: float,
                     home_qb: str = "starter", away_qb: str = "starter",
                     home_team: str = "", away_team: str = "",
                     hfa: float = 65.0, neutral: bool = False) -> float:
    """
    Compute win probability for the home team, applying QB adjustments.

    Parameters
    ----------
    elo_home, elo_away : pre-game Elo ratings
    home_qb, away_qb   : QB status ("starter"/"limited"/"backup"/"out")
    home_team, away_team : team abbreviations (required if status != "starter")
    hfa                : home-field advantage in Elo points (default 65)
    neutral            : True for neutral-site games (no HFA)

    Returns
    -------
    float — P(home team wins)
    """
    import math

    adj_home = adjust_elo(home_team, home_qb) if home_team else 0.0
    adj_away = adjust_elo(away_team, away_qb) if away_team else 0.0

    effective_hfa = 0.0 if neutral else hfa
    elo_diff = (elo_home + adj_home) - (elo_away + adj_away) + effective_hfa

    return 1.0 / (math.pow(10.0, -elo_diff / 400.0) + 1.0)


# ── Example / self-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import math

    HFA = 65.0

    print("QB Adjustment Module — self-test")
    print("=" * 55)

    # Reproduce the BAL-LAC Week 10 situation
    BAL_ELO = 1562.0  # post-Week 9 (before the game)
    LAC_ELO = 1518.0

    # Without adjustment (what we actually predicted)
    base_diff = (BAL_ELO - LAC_ELO) + HFA   # BAL is home
    base_prob = 1 / (10 ** (-base_diff / 400) + 1)
    print(f"\nBAL (home, Lamar starting) vs LAC:")
    print(f"  Elo diff (w/ HFA): {base_diff:+.1f}")
    print(f"  Win prob (BAL):    {base_prob:.1%}  ← what we predicted")

    # With Brissett starting
    adj = adjust_elo("BAL", "backup")
    adj_diff = (BAL_ELO + adj - LAC_ELO) + HFA
    adj_prob = 1 / (10 ** (-adj_diff / 400) + 1)
    print(f"\nBAL (home, Brissett starting) vs LAC:")
    print(f"  QB adjustment:     {adj:+.1f} Elo")
    print(f"  Elo diff (w/ HFA): {adj_diff:+.1f}")
    print(f"  Win prob (BAL):    {adj_prob:.1%}  ← with QB adjustment")
    print(f"  Actual result:     LAC won 23-20 (coin flip game)")

    # Week 11 examples
    print("\n── Week 11 QB status examples ──")
    games = [
        ("HOU", "starter", "IND", "starter", False),
        ("BUF", "starter", "MIA", "starter", False),
        ("DET", "starter", "TB",  "starter", False),
        ("KC",  "starter", "ARI", "starter", False),
    ]

    ELO = {  # post-Week 10 ratings
        "HOU":1531, "IND":1555, "BUF":1567, "MIA":1495,
        "DET":1513, "TB":1419,  "KC":1590,  "ARI":1469,
    }

    print(f"  {'Home':<6} {'Away':<6} {'Home QB':<10} {'Away QB':<10} {'Home Win%':>10}")
    print(f"  {'-'*50}")
    for home, hqb, away, aqb, neutral in games:
        p = win_prob_with_qb(ELO[home], ELO[away], hqb, aqb, home, away, neutral=neutral)
        print(f"  {home:<6} {away:<6} {hqb:<10} {aqb:<10} {p:>10.1%}")

    print("\nTo use in weekly picks:")
    print("  from scripts.qb_adjustment import win_prob_with_qb")
    print("  p = win_prob_with_qb(elo_home, elo_away,")
    print("                       home_qb='backup', away_qb='starter',")
    print("                       home_team='BAL', away_team='LAC')")
