"""
Download nflverse schedule data for 2021 onward and produce a games CSV
in the same format as the 538 historical file, so backtest_elo.py can
consume it directly.

Requires:  pip install nflreadpy

Run this on a machine with internet access, then commit
  data/checksums.txt
but NOT data/raw/ (it is in .gitignore).

Usage:
  python3 scripts/fetch_schedules.py [--first 2021] [--last 2025]

Output files:
  data/raw/schedules_<first>_<last>.csv   raw nflverse data
  data/nfl_games_<first>_<last>.csv       converted, ready for backtest
  data/checksums.txt                      SHA-256 of raw file (appended)
"""

import argparse
import csv
import hashlib
import os
import sys
from datetime import datetime

NFLVERSE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/sched_{season}.csv"

TEAM_MAP = {
    # nflverse abbr -> 538 abbr where they differ
    'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BUF': 'BUF',
    'CAR': 'CAR', 'CHI': 'CHI', 'CIN': 'CIN', 'CLE': 'CLE',
    'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GB':  'GB',
    'HOU': 'HOU', 'IND': 'IND', 'JAX': 'JAX', 'KC':  'KC',
    'LA':  'LAR', 'LAC': 'LAC', 'LV':  'LV',  'MIA': 'MIA',
    'MIN': 'MIN', 'NE':  'NE',  'NO':  'NO',  'NYG': 'NYG',
    'NYJ': 'NYJ', 'PHI': 'PHI', 'PIT': 'PIT', 'SEA': 'SEA',
    'SF':  'SF',  'TB':  'TB',  'TEN': 'TEN', 'WAS': 'WAS',
    'JAC': 'JAX',  # nflverse alternate
    'OAK': 'OAK',  # pre-2020 (should not appear in 2021+ but map anyway)
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def download(first, last):
    try:
        import nflreadpy as nfl
    except ImportError:
        sys.exit('nflreadpy is not installed. Run: pip install nflreadpy')

    os.makedirs('data/raw', exist_ok=True)

    rows = []
    for season in range(first, last + 1):
        print(f'  Fetching {season}...', end=' ', flush=True)
        df = nfl.load_schedules(seasons=[season])
        # nflreadpy returns a Polars dataframe — use .filter() not boolean indexing
        if hasattr(df, 'filter'):
            df = df.filter(df['season'] == season)
            n = len(df)
        else:
            # fallback if pandas
            df = df[df['season'] == season]
            n = len(df)
        rows.append(df)
        print(f'{n} games')

    # Convert to pandas for CSV writing (works from both polars and pandas input)
    combined_rows = []
    for df in rows:
        if hasattr(df, 'to_pandas'):
            combined_rows.append(df.to_pandas())
        else:
            combined_rows.append(df)

    import pandas as pd
    combined = pd.concat(combined_rows, ignore_index=True)
    raw_path = f'data/raw/schedules_{first}_{last}.csv'
    combined.to_csv(raw_path, index=False)
    return combined, raw_path

def convert(df, out_path):
    """Convert nflverse schedule dataframe to 538-compatible CSV."""
    out_rows = []
    for _, row in df.iterrows():
        # nflverse uses home_team / away_team; 538 uses team1 (home) / team2 (away)
        home = TEAM_MAP.get(str(row.get('home_team', '')), str(row.get('home_team', '')))
        away = TEAM_MAP.get(str(row.get('away_team', '')), str(row.get('away_team', '')))

        # Scores (NaN for unplayed games)
        score1 = str(int(row['home_score'])) if str(row.get('home_score', '')) not in ('', 'nan', 'None') else ''
        score2 = str(int(row['away_score'])) if str(row.get('away_score', '')) not in ('', 'nan', 'None') else ''

        # Result for team1 (home)
        if score1 and score2:
            s1, s2 = int(score1), int(score2)
            if s1 > s2:   result1 = '1'
            elif s1 < s2: result1 = '0'
            else:         result1 = '0.5'
        else:
            result1 = ''

        # Neutral site: nflverse has location column; 'neutral' or stadium city checks
        neutral = '1' if str(row.get('location', '')).lower() == 'neutral' else '0'

        # Playoff flag
        game_type = str(row.get('game_type', 'REG'))
        playoff = '0' if game_type == 'REG' else '1'

        out_rows.append({
            'date':      str(row.get('gameday', '')),
            'season':    str(int(row['season'])),
            'neutral':   neutral,
            'playoff':   playoff,
            'team1':     home,
            'team2':     away,
            'elo1':      '',   # filled in by backtest script using running ratings
            'elo2':      '',
            'elo_prob1': '',   # no published 538 Elo for 2021+
            'score1':    score1,
            'score2':    score2,
            'result1':   result1,
        })

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f'  Wrote {len(out_rows)} rows → {out_path}')

def record_checksum(raw_path, checksum):
    os.makedirs('data', exist_ok=True)
    stamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    line = f'{checksum}  {raw_path}  fetched={stamp}  source=nflverse\n'
    with open('data/checksums.txt', 'a') as f:
        f.write(line)
    print(f'  Checksum recorded in data/checksums.txt')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--first', type=int, default=2021)
    p.add_argument('--last',  type=int, default=2025)
    args = p.parse_args()

    print(f'Downloading nflverse schedules {args.first}–{args.last}...')
    df, raw_path = download(args.first, args.last)

    checksum = sha256(raw_path)
    print(f'  SHA-256: {checksum}')
    record_checksum(raw_path, checksum)

    out_path = f'data/nfl_games_{args.first}_{args.last}.csv'
    print(f'Converting to 538 format → {out_path}')
    convert(df, out_path)

    print(f'\nTo backtest:')
    print(f'  python3 -I scripts/backtest_elo.py data/nfl_games_{args.first}_{args.last}.csv {args.first} {args.last}')
    print('\nNote: elo_prob1 column will be empty (no published 538 Elo for 2021+).')
    print('The backtest compares my_prob1 (our reimplementation) vs home-rate baseline only.')

if __name__ == '__main__':
    main()
