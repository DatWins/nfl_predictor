---
name: nfl-weekly-update
description: Weekly routine for the nfl_predictor repo: refresh data, write predictions before kickoff, grade last week's games, and update the track record.
---

# Weekly update

1. Refresh the games and schedule files. Record the source URL and checksum.
2. Confirm the season and week, and that no game in this week has started.
3. Run the model and write `predictions/<season>-week<NN>.csv` with the home team, away team, home win probability, and model version. Commit before the first kickoff.
4. After the week's games finish, grade the prior week's predictions against final scores. Do not edit the prediction file. Write grades to a separate file.
5. Update `reports/` with the running record: log loss, Brier score, accuracy, and the home-rate baseline, with sample size.
6. Note any data problems in `docs/handoff.md`.
