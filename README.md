# NFL Predictor

A reproducible NFL game prediction project, built as a data analysis exercise. Every prediction is written down before kickoff, and every method can be backtested by anyone with the same data.

## Goals

1. Build transparent, well-calibrated win probabilities for NFL games.
2. Compare each model against simple baselines and published models.
3. Publish the full method, so others can rerun the backtests and check the results.

## Current status

- Baseline: FiveThirtyEight's Elo win probabilities, backtested on 2010–2020 (2,939 games). Log loss 0.629 against 0.690 for a constant home-rate baseline. See `reports/`.
- Not yet tested: 2021 onward, and any comparison with closing betting lines.

## Method

1. **Data.** Historical games and Elo probabilities from FiveThirtyEight's `nfl-elo-game` repository. Scores, schedules, and closing lines from the nflverse project. Source URLs and checksums are recorded by each download script.
2. **Models.** Each model is a function from pre-game information to a home win probability. Model parameters are versioned in `models/`.
3. **Evaluation.** Walk-forward backtests: each season is predicted using only earlier games. Metrics are log loss, Brier score, accuracy, and calibration tables. Every model is compared with a constant home-rate baseline.
4. **Predictions.** Each week's predictions are committed before the first kickoff and are never edited afterward. Results are graded against final scores, and later against closing lines.

## Reproducing the backtest

```bash
python3 -I scripts/backtest_elo.py <path-to-nfl_games.csv> 2010 2020
```

The input is the historical games file from the FiveThirtyEight repository. Data is not committed here; see `data/README.md`.

## Limitations

- Elo parameters were chosen by FiveThirtyEight using this same historical data, so backtest scores are slightly optimistic.
- NFL seasons are short (17 games per team), so a single season tells you little. Compare across many seasons.
- No injury, roster, or weather information is used yet.

## Layout

- `scripts/`: data download, backtest, prediction, and grading scripts
- `models/`: model definitions and parameter files
- `predictions/`: weekly prediction records, committed before kickoff
- `reports/`: backtest results and calibration tables
- `data/`: data documentation (raw files are not committed)
- `docs/handoff.md`: how to rerun and extend the pipeline
- `skills/`: reusable routines for weekly updates
