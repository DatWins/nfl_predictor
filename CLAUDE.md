# CLAUDE.md

Guidance for working in this repository.

## Principles

- Predictions are written before kickoff and never edited afterward. If a prediction is wrong, record it as wrong.
- Every model is compared to a constant home-rate baseline. Report that comparison with every result.
- Use only information available before kickoff. Check every feature for leakage.
- Report sample sizes next to every metric. Do not present one season as evidence.
- Keep raw data out of git. Scripts must download data and record its source and checksum.

## Commands

- Backtest: `python3 -I scripts/backtest_elo.py <games.csv> <first> <last>`

## Where things go

- New model: `models/<name>.py` plus a parameter file, and a backtest entry in `reports/`.
- Weekly predictions: `predictions/<season>-week<NN>.csv`, committed before the first kickoff.
- Methodology changes: update `README.md` and `docs/handoff.md` in the same commit.
