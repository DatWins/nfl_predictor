"""
Backtest FiveThirtyEight's Elo model (faithful reimplementation).

Parameters sourced from fivethirtyeight/nfl-elo-game/forecast.py (MIT licence):
  K=20, HFA=65, REVERT=1/3, season-reversion target=1505,
  MOV multiplier = log(|pd|+1) * 2.2 / (elo_diff*0.001 + 2.2)
  Win prob = 1 / (10^(-elo_diff/400) + 1)

Usage:
  python3 -I scripts/backtest_elo.py <games.csv> <first_season> <last_season>

Input CSV columns (538 format):
  date, season, neutral, playoff, team1, team2,
  elo1_pre, elo2_pre (or elo1/elo2), elo_prob1,
  score1, score2, result1

The script computes my_prob1 for every game using the reimplemented Elo
model, then compares log loss / Brier / accuracy against:
  - 538's published elo_prob1
  - Constant home-rate baseline (home win rate from prior seasons)

Ties (result1 == 0.5) are excluded from evaluation per 538 convention.
Sample sizes are reported alongside every metric.
"""

import csv
import math
import sys
from collections import defaultdict

# ── 538 Elo parameters ───────────────────────────────────────────────────────
HFA    = 65.0
K      = 20.0
REVERT = 1 / 3.0
REVERT_TO = 1505.0

# Hard-coded reversions from 538's forecast.py for specific historical team-seasons
REVERSIONS = {
    'CBD1925': 1502.032, 'RAC1926': 1403.384, 'LOU1926': 1307.201,
    'CIB1927': 1362.919, 'MNN1929': 1306.702, 'BFF1929': 1331.943,
    'LAR1944': 1373.977, 'PHI1944': 1497.988, 'ARI1945': 1353.939,
    'PIT1945': 1353.939, 'CLE1999': 1300.0,
}

EPS = 1e-12   # clamp for log-loss


# ── Elo maths ─────────────────────────────────────────────────────────────────

def win_prob(elo_diff):
    """P(team1 wins) given signed Elo difference (HFA already included)."""
    return 1.0 / (math.pow(10.0, -elo_diff / 400.0) + 1.0)


def mov_mult(pd, elo_diff, result1):
    """Margin-of-victory K multiplier (538 formula)."""
    if result1 == 0.5:          # tie: denominator is 1.0
        denom = 2.2
    elif result1 == 1.0:
        denom = elo_diff * 0.001 + 2.2
    else:
        denom = -elo_diff * 0.001 + 2.2
    return math.log(max(pd, 1) + 1.0) * (2.2 / denom)


# ── Metrics ───────────────────────────────────────────────────────────────────

def log_loss(p, y):
    p = max(EPS, min(1 - EPS, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def brier(p, y):
    return (p - y) ** 2


# ── CSV loading ───────────────────────────────────────────────────────────────

def load(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))

    # Normalise column names: 538 historical file uses elo1/elo2;
    # nflverse-derived files might use elo1_pre/elo2_pre.
    for row in rows:
        for src, dst in [('elo1_pre', 'elo1'), ('elo2_pre', 'elo2')]:
            if src in row and dst not in row:
                row[dst] = row[src]

        def _int(v):   return int(v)   if v not in ('', None) else None
        def _float(v): return float(v) if v not in ('', None) else None

        row['season']   = _int(row['season'])
        row['neutral']  = _int(row['neutral'])
        row['playoff']  = _int(row['playoff'])
        row['score1']   = _int(row.get('score1', ''))
        row['score2']   = _int(row.get('score2', ''))
        row['elo1']     = _float(row.get('elo1', ''))
        row['elo2']     = _float(row.get('elo2', ''))
        row['elo_prob1']= _float(row.get('elo_prob1', ''))
        row['result1']  = _float(row.get('result1', ''))

    return rows


# ── Elo simulation ────────────────────────────────────────────────────────────

def simulate(games):
    """
    Walk through all games in chronological order, compute my_prob1 for each,
    and update Elo ratings in place.

    Seed ratings: for the first game each team appears in, use elo1/elo2 from
    the CSV row (538's published pre-game rating).  After that we maintain our
    own running ratings so the backtest is fully self-contained.
    """
    teams = {}   # name -> {'elo': float, 'season': int | None}

    def get_team(name, seed_elo):
        if name not in teams:
            teams[name] = {'elo': seed_elo if seed_elo else 1500.0, 'season': None}
        return teams[name]

    for game in games:
        t1 = get_team(game['team1'], game.get('elo1'))
        t2 = get_team(game['team2'], game.get('elo2'))

        # Season reversion
        for t, name in [(t1, game['team1']), (t2, game['team2'])]:
            if t['season'] is not None and game['season'] != t['season']:
                key = '%s%s' % (name, game['season'])
                if key in REVERSIONS:
                    t['elo'] = REVERSIONS[key]
                else:
                    t['elo'] = REVERT_TO * REVERT + t['elo'] * (1 - REVERT)
            t['season'] = game['season']

        # Win probability
        hfa = 0.0 if game['neutral'] == 1 else HFA
        elo_diff = t1['elo'] - t2['elo'] + hfa
        game['my_prob1'] = win_prob(elo_diff)

        # Update ratings if game was played
        if game['score1'] is not None and game['result1'] is not None:
            pd    = abs(game['score1'] - game['score2'])
            mult  = mov_mult(pd, elo_diff, game['result1'])
            shift = K * mult * (game['result1'] - game['my_prob1'])
            t1['elo'] += shift
            t2['elo'] -= shift


# ── Reporting ─────────────────────────────────────────────────────────────────

def metrics(games, prob_key):
    """Return (n, log_loss_avg, brier_avg, accuracy) for non-tie completed games."""
    played = [g for g in games
              if g['result1'] is not None
              and g['result1'] != 0.5
              and g.get(prob_key) is not None]
    if not played:
        return 0, None, None, None
    n  = len(played)
    ll = sum(log_loss(g[prob_key], g['result1']) for g in played) / n
    br = sum(brier  (g[prob_key], g['result1']) for g in played) / n
    ac = sum(1 for g in played if (g[prob_key] > 0.5) == (g['result1'] == 1.0)) / n
    return n, ll, br, ac


def calibration(games, prob_key, bins=10):
    """Return list of (bucket_mid, predicted_rate, observed_rate, n) tuples."""
    played = [g for g in games
              if g['result1'] is not None
              and g['result1'] != 0.5
              and g.get(prob_key) is not None]
    buckets = defaultdict(list)
    for g in played:
        b = int(g[prob_key] * bins)
        b = min(b, bins - 1)
        buckets[b].append(g)
    rows = []
    for b in range(bins):
        gs = buckets[b]
        if not gs:
            continue
        mid  = (b + 0.5) / bins
        pred = sum(g[prob_key] for g in gs) / len(gs)
        obs  = sum(g['result1'] for g in gs) / len(gs)
        rows.append((mid, pred, obs, len(gs)))
    return rows


def print_season_table(seasons_data):
    """seasons_data: list of dicts with season, n, my_*, elo_*, home_*"""
    hdr = ('Season', 'n', 'My LL', 'Elo LL', 'Home LL',
           'My Brier', 'Elo Brier', 'My Acc', 'Elo Acc')
    fmt = '%-8s %5s  %7s %7s %7s   %7s %8s   %6s %6s'
    print(fmt % hdr)
    print('-' * 80)
    for d in seasons_data:
        def f(v): return ('%.4f' % v) if v is not None else '   -  '
        def pct(v): return ('%5.1f%%' % (v*100)) if v is not None else '   -  '
        print(fmt % (
            d['season'], d['n'],
            f(d['my_ll']),  f(d['elo_ll']),  f(d['home_ll']),
            f(d['my_br']),  f(d['elo_br']),
            pct(d['my_ac']), pct(d['elo_ac']),
        ))


def main():
    if len(sys.argv) != 4:
        print('Usage: python3 -I scripts/backtest_elo.py <games.csv> <first_season> <last_season>')
        sys.exit(1)

    path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    all_games = load(path)

    print(f'\nLoaded {len(all_games):,} games from {path}')
    seasons_in_file = sorted({g['season'] for g in all_games})
    print(f'Seasons in file: {seasons_in_file[0]}–{seasons_in_file[-1]}\n')

    # Run the full Elo simulation across ALL games (so ratings are warm by
    # the time we reach the evaluation window).
    simulate(all_games)

    eval_games = [g for g in all_games if first <= g['season'] <= last]

    # Constant home-rate baseline: home win rate across all prior seasons
    prior = [g for g in all_games
             if g['season'] < first
             and g['result1'] is not None
             and g['result1'] != 0.5
             and g['neutral'] == 0]
    home_rate = (sum(g['result1'] for g in prior) / len(prior)) if prior else 0.579
    print(f'Constant home-rate baseline: {home_rate:.4f}  (from {len(prior):,} non-neutral prior games)\n')

    # ── Season-by-season table ────────────────────────────────────────────────
    seasons_data = []
    for s in range(first, last + 1):
        sg = [g for g in eval_games if g['season'] == s]
        # Assign home_prob1 for non-neutral games; neutral get 0.5
        for g in sg:
            g['home_prob1'] = 0.5 if g['neutral'] == 1 else home_rate

        n_my,  ll_my,  br_my,  ac_my  = metrics(sg, 'my_prob1')
        n_elo, ll_elo, br_elo, ac_elo = metrics(sg, 'elo_prob1')
        n_hm,  ll_hm,  _,      _      = metrics(sg, 'home_prob1')

        seasons_data.append(dict(
            season=s, n=n_my,
            my_ll=ll_my,  elo_ll=ll_elo,  home_ll=ll_hm,
            my_br=br_my,  elo_br=br_elo,
            my_ac=ac_my,  elo_ac=ac_elo,
        ))

    print_season_table(seasons_data)

    # ── Pooled summary ────────────────────────────────────────────────────────
    print('\n── Pooled %d–%d ──' % (first, last))
    for g in eval_games:
        g['home_prob1'] = 0.5 if g['neutral'] == 1 else home_rate

    for label, key in [('My reimpl.', 'my_prob1'),
                        ('538 Elo   ', 'elo_prob1'),
                        ('Home base.', 'home_prob1')]:
        n, ll, br, ac = metrics(eval_games, key)
        if ll is not None:
            print(f'  {label}: log-loss {ll:.4f}  Brier {br:.4f}  acc {ac*100:.1f}%  (n={n:,})')

    # ── Calibration (my model, pooled) ────────────────────────────────────────
    print('\n── Calibration: my reimpl. (pooled) ──')
    print('  Bucket   Predicted  Observed    n')
    for mid, pred, obs, n in calibration(eval_games, 'my_prob1'):
        flag = '  ← over' if pred - obs > 0.05 else ('  ← under' if obs - pred > 0.05 else '')
        print(f'  {mid:.2f}      {pred:.3f}     {obs:.3f}   {n:5d}{flag}')

    print()


if __name__ == '__main__':
    main()
