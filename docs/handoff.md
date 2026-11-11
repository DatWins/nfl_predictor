# Handoff

## What exists

- `scripts/backtest_elo.py`: scores FiveThirtyEight's Elo win probabilities against results, season by season, with a constant home-rate baseline and a calibration table. Usage: `python3 -I scripts/backtest_elo.py <games.csv> <first_season> <last_season>`.
- `reports/`: backtest outputs. Current: 2010–2020.
- `predictions/`: empty. Weekly records start once a model is chosen.

## Input file

`nfl_games.csv` from the FiveThirtyEight `nfl-elo-game` repository, data folder. Columns: date, season, neutral, playoff, team1, team2, elo1, elo2, elo_prob1, score1, score2, result1. The file in use ends with the 2020 season. Ties have result1 = 0.5.

Assumption to confirm: team1 is the home team when neutral = 0.

## Open items

1. Get a version of the games file that includes 2021 onward, then rerun 2021–2025.
2. Add nflverse closing spreads and compare Elo probabilities to the market on the same games.
3. Read `forecast.py` in the FiveThirtyEight repo to document the exact Elo update rules.
4. Write a download script that records source URL and SHA-256 checksum.
5. Decide when to make the repository public. The methodology is meant to be public, but raw data should be fetched by each user, not redistributed.

## Known issues

- Backtest scores use pre-game probabilities from the source file, not recomputed Elo, so they inherit whatever parameters 538 used.
- Ties are scored as half a win in accuracy.
