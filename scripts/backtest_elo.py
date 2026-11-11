"""Backtest the FiveThirtyEight NFL Elo win probabilities from nfl_games.csv.

Usage: python3 -I backtest_elo.py <games.csv> <first_season> <last_season>

Scores each game's pre-game elo_prob1 (probability team1 wins) against the
result. Ties (result1 == 0.5) count as half a win. Also reports two baselines:
  - "home": a constant probability equal to the league-wide home win rate
    over the training seasons (all seasons before the test season)
  - "coin": always 0.5
Assumes team1 is the home team when neutral == 0.
"""
import csv
import math
import sys
from collections import defaultdict

EPS = 1e-12


def load(path):
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            yield {
                "season": int(r["season"]),
                "neutral": int(r["neutral"]),
                "playoff": int(r["playoff"]),
                "p": float(r["elo_prob1"]),
                "y": float(r["result1"]),
            }


def log_loss(p, y):
    p = min(max(p, EPS), 1 - EPS)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def brier(p, y):
    return (p - y) ** 2


def summarize(games, label):
    n = len(games)
    if n == 0:
        return None
    ll = sum(log_loss(g["p"], g["y"]) for g in games) / n
    br = sum(brier(g["p"], g["y"]) for g in games) / n
    acc = sum(
        1.0 if (g["p"] > 0.5 and g["y"] == 1) or (g["p"] < 0.5 and g["y"] == 0)
        else 0.5 if g["p"] == 0.5 or g["y"] == 0.5 else 0.0
        for g in games
    ) / n
    return n, ll, br, acc


def calibration(games, bins=10):
    buckets = defaultdict(lambda: [0, 0.0, 0.0])  # count, sum_p, sum_y
    for g in games:
        b = min(int(g["p"] * bins), bins - 1)
        buckets[b][0] += 1
        buckets[b][1] += g["p"]
        buckets[b][2] += g["y"]
    lines = []
    for b in sorted(buckets):
        c, sp, sy = buckets[b]
        lines.append(
            f"  {b/bins:.1f}-{(b+1)/bins:.1f}: n={c:5d}  "
            f"mean_pred={sp/c:.3f}  observed={sy/c:.3f}"
        )
    return "\n".join(lines)


def main():
    path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    games = list(load(path))

    # League-wide home win rate by season, used for the "home" baseline.
    home_wins = defaultdict(float)
    home_n = defaultdict(int)
    for g in games:
        if g["neutral"] == 0:
            home_wins[g["season"]] += g["y"]
            home_n[g["season"]] += 1

    print(f"{'season':>6} {'n':>5} {'elo_LL':>8} {'elo_Brier':>10} "
          f"{'elo_acc':>8} {'home_LL':>8} {'coin_LL':>8}")
    all_test = []
    for s in range(first, last + 1):
        test = [g for g in games if g["season"] == s]
        train_seasons = [t for t in home_n if t < s]
        if not test or not train_seasons:
            continue
        hw = sum(home_wins[t] for t in train_seasons)
        hn = sum(home_n[t] for t in train_seasons)
        base_p = hw / hn
        n, ll, br, acc = summarize(test, "elo")
        home_ll = sum(log_loss(base_p, g["y"]) for g in test) / n
        coin_ll = log_loss(0.5, 1) * 0.5 + log_loss(0.5, 0) * 0.5
        print(f"{s:>6} {n:>5} {ll:>8.4f} {br:>10.4f} {acc:>8.3f} "
              f"{home_ll:>8.4f} {coin_ll:>8.4f}")
        all_test.extend(test)

    if all_test:
        n, ll, br, acc = summarize(all_test, "all")
        print(f"\nPooled {first}-{last}: n={n} elo_LL={ll:.4f} "
              f"elo_Brier={br:.4f} elo_acc={acc:.3f}")
        print("\nCalibration (all test games, pre-game Elo probability):")
        print(calibration(all_test))


if __name__ == "__main__":
    main()
