"""
generate_data.py — reads zidane_ratings_final.csv and writes JSON files for the web frontend.
Run after zidane.py. Outputs to docs/data/.
"""

import pandas as pd
import numpy as np
import json
import os
import re
from datetime import date

os.makedirs('docs/data/teams', exist_ok=True)

print("Reading ratings...")
df = pd.read_csv('zidane_ratings_final.csv')
df['date'] = pd.to_datetime(df['date']).dt.date
df['last_match_date'] = pd.to_datetime(df['last_match_date'], errors='coerce').dt.date

def season_is_complete(season_str):
    """Season YYYY-YY is complete once today is past July 31 of its end year."""
    end_year = int('20' + season_str[-2:])
    return date.today() > date(end_year, 7, 31)

# Fix finish labels for any in-progress seasons still in the cached CSV.
# zidane.py now handles this correctly on a fresh run; this patch covers
# the window between the code change and the next cron execution.
for season in df['season'].unique():
    if not season_is_complete(season):
        df.loc[(df['season'] == season) & (df['domestic_finish'] == 'Champion'), 'domestic_finish'] = '1st'
        df.loc[(df['season'] == season) & (df['domestic_finish'] == 'Runner-Up'), 'domestic_finish'] = '2nd'
        df.loc[df['season'] == season, 'cl_finish'] = ''
        df.loc[df['season'] == season, 'el_finish'] = ''

def clean(val):
    if pd.isna(val):
        return ''
    return str(val)

def slug(name):
    return re.sub(r'[^\w]', '_', name).strip('_')

def date_to_season(d):
    if d.month >= 8:
        return f"{d.year}-{str(d.year+1)[-2:]}"
    return f"{d.year-1}-{str(d.year)[-2:]}"

# ── Season records ────────────────────────────────────────────────────────────
# Cumulative W-D-L per team per season, computed from raw game data.
# Shootout winner counts as a W (same logic as CL/EL knockouts).
print("Computing season records...")
DOMESTIC_LEAGUES = {'EPL', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'}

games_raw = pd.read_csv('all_club_games.csv', parse_dates=['date'])
games_raw['home_score'] = pd.to_numeric(games_raw['home_score'], errors='coerce')
games_raw['away_score'] = pd.to_numeric(games_raw['away_score'], errors='coerce')
games_raw = games_raw.dropna(subset=['home_score', 'away_score']).copy()
games_raw['season'] = games_raw['date'].apply(date_to_season)

# Build season overrides for CL/EL games whose comp_season differs from the
# date-computed season (e.g. the 2020 COVID bubble: Aug 2020 CL games → 2019-20).
_euro = games_raw[
    games_raw['competition'].isin({'Champions League', 'Europa League'}) &
    games_raw['comp_season'].notna() &
    (games_raw['comp_season'].str.strip() != '')
].copy()
_season_override = {}  # (team, date_str) -> correct season
for _, _row in _euro.iterrows():
    _ds = str(_row['date'].date())
    _cs = _row['comp_season'].strip()
    if date_to_season(_row['date'].date()) != _cs:
        for _t in [_row['home_team'], _row['away_team']]:
            _season_override[(_t, _ds)] = _cs

if _season_override:
    df['season'] = df.apply(
        lambda r: _season_override.get((r['team'], str(r['date'])), r['season']),
        axis=1
    )

# League games only for the record column
games_dom = games_raw[games_raw['competition'].isin(DOMESTIC_LEAGUES)].copy()

def result_for(row, side):
    gf = row['home_score'] if side == 'home' else row['away_score']
    ga = row['away_score'] if side == 'home' else row['home_score']
    if gf > ga:
        return 'W'
    if ga > gf:
        return 'L'
    return 'D'

home_g = games_dom[['date', 'season', 'home_team']].copy()
home_g['team']   = games_dom['home_team']
home_g['result'] = games_dom.apply(lambda r: result_for(r, 'home'), axis=1)

away_g = games_dom[['date', 'season', 'away_team']].copy()
away_g['team']   = games_dom['away_team']
away_g['result'] = games_dom.apply(lambda r: result_for(r, 'away'), axis=1)

all_g = pd.concat([
    home_g[['date', 'season', 'team', 'result']],
    away_g[['date', 'season', 'team', 'result']],
]).sort_values(['team', 'season', 'date']).reset_index(drop=True)

all_g['is_w'] = (all_g['result'] == 'W').astype(int)
all_g['is_d'] = (all_g['result'] == 'D').astype(int)
all_g['is_l'] = (all_g['result'] == 'L').astype(int)

all_g['cum_w'] = all_g.groupby(['team', 'season'])['is_w'].cumsum()
all_g['cum_d'] = all_g.groupby(['team', 'season'])['is_d'].cumsum()
all_g['cum_l'] = all_g.groupby(['team', 'season'])['is_l'].cumsum()
all_g['record'] = (all_g['cum_w'].astype(int).astype(str) + '-' +
                   all_g['cum_d'].astype(int).astype(str) + '-' +
                   all_g['cum_l'].astype(int).astype(str))

# For current standings: latest record per team in the current season
current_season_label = date_to_season(pd.Timestamp.today())
cur_records = (
    all_g[all_g['season'] == current_season_label]
    .sort_values('date')
    .groupby('team')['record']
    .last()
    .to_dict()
)

# Running record lookup: (team, date_str) -> record after last game that date
running_lookup = (
    all_g.sort_values(['team', 'season', 'date'])
    .groupby(['team', 'date'])['record']
    .last()
    .reset_index()
)
running_lookup['date_str'] = running_lookup['date'].astype(str)
record_by_team_date = {
    (row['team'], row['date_str']): row['record']
    for _, row in running_lookup.iterrows()
}

# ── 1. Current standings ─────────────────────────────────────────────────────
print("Writing current_standings.json...")
latest_id = int(df['ranking_id'].max())
latest = df[df['ranking_id'] == latest_id].sort_values('rank').copy()
latest_date = str(latest['date'].iloc[0])

standings_data = {
    'updated': latest_date,
    'teams': [
        {
            'rank':            int(r['rank']),
            'team':            r['team'],
            'league':          clean(r['league']),
            'rating':          round(float(r['rating']), 3),
            'record':          cur_records.get(r['team'], '—'),
            'games_played':    int(r['games_played']),
            'last_match':      clean(r['last_match']),
            'last_match_date': clean(r['last_match_date']),
            'domestic_finish': clean(r['domestic_finish']),
            'cl_finish':       clean(r['cl_finish']),
            'el_finish':       clean(r['el_finish']),
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
                'record':          record_by_team_date.get((team, str(r['date'])), '—'),
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

# ── 4. Season standings files ─────────────────────────────────────────────────
print("Writing season standings files...")
os.makedirs('docs/data/seasons', exist_ok=True)

# Build notable date labels: date_str -> combined label string
notable = {}

def add_label(date_str, label):
    notable[date_str] = f"{notable[date_str]} · {label}" if date_str in notable else label

# End of each domestic league per season
for (comp, gseason), grp in games_dom.groupby(['competition', 'season']):
    if not season_is_complete(gseason):
        continue
    add_label(str(grp['date'].max().date()), f'End of {comp}')

# CL/EL finals
for comp, lbl in [('Champions League', 'CL Final'), ('Europa League', 'EL Final')]:
    euro_df = games_raw[
        (games_raw['competition'] == comp) &
        games_raw['comp_season'].notna() &
        (games_raw['comp_season'].str.strip() != '')
    ]
    for comp_season, grp in euro_df.groupby('comp_season'):
        if not season_is_complete(comp_season):
            continue
        add_label(str(grp['date'].max().date()), lbl)

# Group all ranking snapshots by snapshot date's season (pure date-based, no override)
snap_df = df.copy()
snap_df['snapshot_season'] = snap_df['date'].apply(date_to_season)

all_seasons = sorted(snap_df['snapshot_season'].unique())

for season in all_seasons:
    sdf = snap_df[snap_df['snapshot_season'] == season]
    snapshots = []
    for ranking_id, rdf in sdf.groupby('ranking_id'):
        rdf = rdf.sort_values('rank')
        snap_date = str(rdf['date'].iloc[0])
        teams_snap = [
            {
                'rank':            int(r['rank']),
                'team':            r['team'],
                'league':          clean(r['league']),
                'rating':          round(float(r['rating']), 3),
                'record':          record_by_team_date.get((r['team'], snap_date), '—'),
                'last_match':      clean(r['last_match']),
                'last_match_date': clean(r['last_match_date']),
                'domestic_finish': clean(r['domestic_finish']),
                'cl_finish':       clean(r['cl_finish']),
                'el_finish':       clean(r['el_finish']),
            }
            for _, r in rdf.iterrows()
        ]
        snapshots.append({'date': snap_date, 'label': notable.get(snap_date), 'teams': teams_snap})

    snapshots.sort(key=lambda x: x['date'])
    with open(f'docs/data/seasons/{season}.json', 'w') as f:
        json.dump({'season': season, 'snapshots': snapshots}, f, separators=(',', ':'))

with open('docs/data/seasons_index.json', 'w') as f:
    json.dump(list(reversed(all_seasons)), f, separators=(',', ':'))

print(f"Done. {len(teams_index)} teams, {len(standings_data['teams'])} in current standings.")
print(f"Wrote {len(all_seasons)} season files. Standings date: {latest_date}")
