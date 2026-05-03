"""
generate_data.py — reads zidane_ratings_final.csv and writes JSON files for the web frontend.
Run after zidane.py. Outputs to docs/data/.
"""

import pandas as pd
import numpy as np
import json
import os
import re

os.makedirs('docs/data/teams', exist_ok=True)

print("Reading ratings...")
df = pd.read_csv('zidane_ratings_final.csv')
df['date'] = pd.to_datetime(df['date']).dt.date
df['last_match_date'] = pd.to_datetime(df['last_match_date'], errors='coerce').dt.date

def clean(val):
    if pd.isna(val):
        return ''
    return str(val)

def slug(name):
    return re.sub(r'[^\w]', '_', name).strip('_')

# ── 1. Current standings ─────────────────────────────────────────────────────
print("Writing current_standings.json...")
latest_id = int(df['ranking_id'].max())
latest = df[df['ranking_id'] == latest_id].sort_values('rank').copy()
latest_date = str(latest['date'].iloc[0])

standings_data = {
    'updated': latest_date,
    'teams': [
        {
            'rank':             int(r['rank']),
            'team':             r['team'],
            'league':           clean(r['league']),
            'rating':           round(float(r['rating']), 3),
            'games_played':     int(r['games_played']),
            'last_match':       clean(r['last_match']),
            'last_match_date':  clean(r['last_match_date']),
            'domestic_finish':  clean(r['domestic_finish']),
            'cl_finish':        clean(r['cl_finish']),
            'el_finish':        clean(r['el_finish']),
        }
        for _, r in latest.iterrows()
    ]
}

with open('docs/data/current_standings.json', 'w') as f:
    json.dump(standings_data, f, separators=(',', ':'))

# ── 2. GOAT table ─────────────────────────────────────────────────────────────
print("Writing goat_teams.json...")
eos = df[df['is_end_of_season'] == 1].copy()
eos = eos.sort_values('rating', ascending=False).head(50).reset_index(drop=True)

goat_data = [
    {
        'rank':            i + 1,
        'team':            r['team'],
        'season':          r['season'],
        'league':          clean(r['league']),
        'rating':          round(float(r['rating']), 3),
        'domestic_finish': clean(r['domestic_finish']),
        'cl_finish':       clean(r['cl_finish']),
        'el_finish':       clean(r['el_finish']),
    }
    for i, (_, r) in enumerate(eos.iterrows())
]

with open('docs/data/goat_teams.json', 'w') as f:
    json.dump(goat_data, f, separators=(',', ':'))

# ── 3. Per-team JSON files (game days only) ───────────────────────────────────
print("Writing per-team JSON files...")
game_days = df[df['is_game_day'] == 1].copy()
game_days = game_days.sort_values(['team', 'season', 'date'])

all_teams = sorted(df['team'].unique())
teams_index = []

for team in all_teams:
    tdf = game_days[game_days['team'] == team]
    if len(tdf) == 0:
        continue

    league = clean(df[df['team'] == team]['league'].iloc[-1])
    team_slug = slug(team)
    teams_index.append({'name': team, 'league': league, 'slug': team_slug})

    seasons = {}
    for season, sdf in tdf.groupby('season'):
        seasons[season] = [
            {
                'date':            str(r['date']),
                'rating':          round(float(r['rating']), 3),
                'rank':            int(r['rank']),
                'last_match':      clean(r['last_match']),
                'domestic_finish': clean(r['domestic_finish']),
                'cl_finish':       clean(r['cl_finish']),
                'el_finish':       clean(r['el_finish']),
            }
            for _, r in sdf.sort_values('date').iterrows()
        ]

    with open(f'docs/data/teams/{team_slug}.json', 'w') as f:
        json.dump({'team': team, 'league': league, 'seasons': seasons},
                  f, separators=(',', ':'))

teams_index.sort(key=lambda x: (x['league'], x['name']))
with open('docs/data/teams_index.json', 'w') as f:
    json.dump(teams_index, f, separators=(',', ':'))

print(f"Done. {len(teams_index)} teams, {len(standings_data['teams'])} in current standings.")
print(f"Standings date: {latest_date}")
