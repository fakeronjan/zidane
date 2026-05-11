"""
generate_data.py — reads zidane_ratings_final.csv and writes JSON files for the web frontend.
Run after zidane.py. Outputs to docs/data/.
"""

import pandas as pd
import numpy as np
import json
import os
import re
from bisect import bisect_right
from datetime import date, datetime, timezone

os.makedirs('docs/data/teams', exist_ok=True)

print("Reading ratings...")
df = pd.read_csv('zidane_ratings_final.csv')
df['date'] = pd.to_datetime(df['date']).dt.date
df['last_match_date'] = pd.to_datetime(df['last_match_date'], errors='coerce').dt.date

def season_is_complete(season_str):
    """Season YYYY-YY is complete once today is past July 31 of its end year.
    Derives end_year from the 4-digit start year so "1992-93" parses correctly."""
    start_year = int(season_str[:4])
    end_year   = start_year + 1
    return date.today() > date(end_year, 7, 31)

# Fix finish labels for any in-progress seasons still in the cached CSV.
# zidane.py now handles this correctly on a fresh run; this patch covers
# the window between the code change and the next cron execution.
# Backwards-compat: an older cached CSV may not have domestic_cup_finish.
if 'domestic_cup_finish' not in df.columns:
    df['domestic_cup_finish'] = ''

for season in df['season'].unique():
    if not season_is_complete(season):
        df.loc[(df['season'] == season) & (df['domestic_finish'] == 'Champion'), 'domestic_finish'] = '1st'
        df.loc[(df['season'] == season) & (df['domestic_finish'] == 'Runner-Up'), 'domestic_finish'] = '2nd'
        df.loc[df['season'] == season, 'cl_finish'] = ''
        df.loc[df['season'] == season, 'el_finish'] = ''
        df.loc[df['season'] == season, 'domestic_cup_finish'] = ''

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

# COVID patch: Serie A finished Aug 1-2, 2020. No other domestic league played in August 2020,
# so any domestic game in Aug 2020 belongs to the 2019-20 season, not 2020-21.
all_g.loc[
    (all_g['date'].dt.year == 2020) & (all_g['date'].dt.month == 8),
    'season',
] = '2019-20'

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

# Per-team, per-season sorted record history.
# Keyed by (team, season); each value is a pair of parallel lists (dates, records)
# sorted by date so bisect_right gives O(log n) forward-fill for any snapshot date.
_rec_sorted = (
    all_g.groupby(['team', 'season', 'date'])['record']
    .last()
    .reset_index()
    .sort_values(['team', 'season', 'date'])
)
_rec_sorted['date_str'] = pd.to_datetime(_rec_sorted['date']).dt.strftime('%Y-%m-%d')
_rec_hist = {}
for key, grp in _rec_sorted.groupby(['team', 'season']):
    _rec_hist[key] = (list(grp['date_str']), list(grp['record']))

def record_as_of(team, season, snap_date_str):
    """Domestic W-D-L for team in season as of snap_date_str (forward-filled)."""
    entry = _rec_hist.get((team, season))
    if not entry:
        return '0-0-0'
    dates, recs = entry
    idx = bisect_right(dates, snap_date_str) - 1
    return recs[idx] if idx >= 0 else '0-0-0'

# Exact-date lookup for current standings (current season only)
record_by_team_date = {
    (team, dates[i]): recs[i]
    for (team, _season), (dates, recs) in _rec_hist.items()
    for i in range(len(dates))
}

# ZIDANE league rank: position within domestic league on each ranking snapshot
df['lg_rank'] = (
    df.groupby(['ranking_id', 'league'])['rank']
    .rank(method='min')
    .astype(int)
)

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
            'record':          cur_records.get(r['team'], '0-0-0'),
            'games_played':    int(r['games_played']),
            'last_match':      clean(r['last_match']),
            'last_match_date': clean(r['last_match_date']),
            'domestic_finish': clean(r['domestic_finish']),
            'cl_finish':       clean(r['cl_finish']),
            'el_finish':       clean(r['el_finish']),
            'domestic_cup_finish': clean(r.get('domestic_cup_finish', '')),
        }
        for _, r in latest.iterrows()
    ]
}

with open('docs/data/current_standings.json', 'w') as f:
    json.dump(standings_data, f, separators=(',', ':'))

# ── 2. GOAT table ─────────────────────────────────────────────────────────────
# Eligibility: must win the domestic league OR the Champions League this
# season. Domestic cup and Europa League do NOT qualify on their own. This
# filters out cluster-inflation entries — pre-2004 in particular still has
# years where one Big-5 league had a coordinated strong CL run and several
# of its mid-table teams' ratings inflated together. A "championship gate"
# keeps only the seasons that produced a real trophy lift.
#
# Warm-up exclusion (1992-93 → 1994-95) still applies — the rolling Massey
# window isn't fully populated in those seasons.
GOAT_WARMUP_SEASONS = {'1992-93', '1993-94', '1994-95'}
# CL/EL data via openfootball doesn't reliably cover pre-2011-12. Without
# UCL games in the rolling window, pre-2011 domestic-only ratings can't
# be calibrated against European elites — a Bordeaux 1998-99 type season
# (won Ligue 1, never played UCL elite teams in our data) ends up looking
# artificially strong vs Pep's 2008-09 Barcelona (whose UCL dominance
# isn't in our data). Restrict GOAT to the calibrated window.
GOAT_FIRST_SEASON = '2011-12'
print("Writing goat_teams.json...")
eos = df[df['is_end_of_season'] == 1].copy()
eos = eos[~eos['season'].isin(GOAT_WARMUP_SEASONS)]
eos = eos[eos['season'] >= GOAT_FIRST_SEASON]
eos = eos[(eos['domestic_finish'] == 'Champion') | (eos['cl_finish'] == 'Champion')]
eos = eos.sort_values('rating', ascending=False).head(50).reset_index(drop=True)

# End-of-season domestic record per (team, season) — used by GOAT and the
# Champions table. Built once here so both can share the lookup.
final_record_lookup = {
    (row['team'], row['season']): row['record']
    for _, row in (
        all_g.sort_values('date')
        .groupby(['team', 'season'])
        .last()
        .reset_index()[['team', 'season', 'record']]
        .iterrows()
    )
}

goat_data = [
    {
        'rank':            i + 1,
        'team':            r['team'],
        'season':          r['season'],
        'league':          clean(r['league']),
        'rating':          round(float(r['rating']), 3),
        'record':          final_record_lookup.get((r['team'], r['season']), '0-0-0'),
        'domestic_finish': clean(r['domestic_finish']),
        'cl_finish':       clean(r['cl_finish']),
        'el_finish':       clean(r['el_finish']),
        'domestic_cup_finish': clean(r.get('domestic_cup_finish', '')),
    }
    for i, (_, r) in enumerate(eos.iterrows())
]

with open('docs/data/goat_teams.json', 'w') as f:
    json.dump(goat_data, f, separators=(',', ':'))

# ── 3. Per-team JSON files (game days only) ───────────────────────────────────
print("Writing per-team JSON files...")
game_days = df[(df['is_game_day'] == 1) | (df['is_end_of_season'] == 1)].copy()
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
                'date':              str(r['date']),
                'rating':            round(float(r['rating']), 3),
                'rank':              int(r['rank']),
                'lg_rank':           int(r['lg_rank']),
                'is_end_of_season':  int(r['is_end_of_season']),
                'record':            record_as_of(team, season, str(r['date'])),
                'last_match':        clean(r['last_match']),
                'domestic_finish':   clean(r['domestic_finish']),
                'cl_finish':         clean(r['cl_finish']),
                'el_finish':         clean(r['el_finish']),
                'domestic_cup_finish': clean(r.get('domestic_cup_finish', '')),
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

# Group all ranking snapshots by snapshot date's season
snap_df = df.copy()
snap_df['snapshot_season'] = snap_df['date'].apply(date_to_season)

# COVID patch: every snapshot in August 2020 belongs to 2019-20.
# That month was the CL/EL bubble (Quarters → Final, Aug 12-23) plus the
# final Serie A matchday (Aug 1-2). No 2020-21 domestic season played that month.
snap_df.loc[
    snap_df['date'].apply(lambda d: d.year == 2020 and d.month == 8),
    'snapshot_season',
] = '2019-20'

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
                'lg_rank':         int(r['lg_rank']),
                'team':            r['team'],
                'league':          clean(r['league']),
                'rating':          round(float(r['rating']), 3),
                'record':          record_as_of(r['team'], season, snap_date),
                'last_match':      clean(r['last_match']),
                'last_match_date': clean(r['last_match_date']),
                'domestic_finish': clean(r['domestic_finish']),
                'cl_finish':       clean(r['cl_finish']),
                'el_finish':       clean(r['el_finish']),
            'domestic_cup_finish': clean(r.get('domestic_cup_finish', '')),
            }
            for _, r in rdf.iterrows()
        ]
        snapshots.append({'date': snap_date, 'label': notable.get(snap_date), 'teams': teams_snap})

    snapshots.sort(key=lambda x: x['date'])
    with open(f'docs/data/seasons/{season}.json', 'w') as f:
        json.dump({'season': season, 'snapshots': snapshots}, f, separators=(',', ':'))

seasons_meta = {
    'seasons':    list(reversed(all_seasons)),
    'first_date': str(games_raw['date'].min().date()),
    'last_date':  str(games_raw['date'].max().date()),
    'generated_at': datetime.now(timezone.utc).isoformat(),
}
with open('docs/data/seasons_index.json', 'w') as f:
    json.dump(seasons_meta, f, separators=(',', ':'))

# ── 5. Champions table ────────────────────────────────────────────────────────
print("Writing champions.json...")

eos = df[df['is_end_of_season'] == 1].copy()

# final_record_lookup is defined alongside the GOAT block above and reused here.

# Domestic champion detection from raw game points.
# zidane.py merges domestic_finish before patching the season on CL-bubble EOS rows
# (Aug 2020 → 2019-20), so the domestic_finish column in the CSV is wrong for 2019-20.
# Computing from game standings is immune to that ordering bug.
#
# Serie A 2019-20 also extended into August 2020 (final matchday Aug 1-2), so those
# 10 games fall in '2020-21' by date_to_season. Override them back to '2019-20'.
games_dom_pts = games_dom.copy()
_sa_aug2020 = (
    (games_dom_pts['competition'] == 'Serie A') &
    (games_dom_pts['date'].dt.year == 2020) &
    (games_dom_pts['date'].dt.month == 8)
)
games_dom_pts.loc[_sa_aug2020, 'season'] = '2019-20'

_dpts_h = games_dom_pts[['season', 'competition', 'home_team', 'home_score', 'away_score']].copy()
_dpts_h['team'] = _dpts_h['home_team']
_dpts_h['pts']  = np.where(_dpts_h['home_score'] > _dpts_h['away_score'], 3,
                  np.where(_dpts_h['home_score'] == _dpts_h['away_score'], 1, 0))
_dpts_h['gd']   = _dpts_h['home_score'] - _dpts_h['away_score']

_dpts_a = games_dom_pts[['season', 'competition', 'away_team', 'home_score', 'away_score']].copy()
_dpts_a['team'] = _dpts_a['away_team']
_dpts_a['pts']  = np.where(_dpts_a['away_score'] > _dpts_a['home_score'], 3,
                  np.where(_dpts_a['home_score'] == _dpts_a['away_score'], 1, 0))
_dpts_a['gd']   = _dpts_a['away_score'] - _dpts_a['home_score']

dom_final_pts = (
    pd.concat([_dpts_h[['competition', 'season', 'team', 'pts', 'gd']],
               _dpts_a[['competition', 'season', 'team', 'pts', 'gd']]])
    .groupby(['competition', 'season', 'team'])[['pts', 'gd']].sum().reset_index()
)

def _dom_entry(team_name, finish_label, season_str):
    """Build a champion-table team dict from EOS data, overriding domestic_finish."""
    team_eos = eos[(eos['team'] == team_name) & (eos['season'] == season_str)]
    if team_eos.empty:
        return {'team': team_name, 'league': '', 'rating': None, 'rank': None, 'lg_rank': None,
                'record': final_record_lookup.get((team_name, season_str), '0-0-0'),
                'domestic_finish': finish_label, 'cl_finish': '', 'el_finish': '', 'domestic_cup_finish': ''}
    r = team_eos.iloc[0]
    return {
        'team':            team_name,
        'league':          clean(r['league']),
        'rating':          round(float(r['rating']), 3),
        'rank':            int(r['rank']),
        'lg_rank':         int(r['lg_rank']),
        'record':          final_record_lookup.get((team_name, season_str), '0-0-0'),
        'domestic_finish': finish_label,
        'cl_finish':       clean(r['cl_finish']),
        'el_finish':       clean(r['el_finish']),
        'domestic_cup_finish': clean(r.get('domestic_cup_finish', '')),
    }

def euro_team_dict(team, season):
    team_eos = eos[(eos['team'] == team) & (eos['season'] == season)]
    if team_eos.empty:
        return {'team': team, 'league': '', 'rating': None, 'rank': None, 'lg_rank': None, 'record': '0-0-0',
                'domestic_finish': '', 'cl_finish': '', 'el_finish': '', 'domestic_cup_finish': ''}
    row = team_eos.iloc[0]
    return {
        'team':             team,
        'league':           clean(row['league']),
        'rating':           round(float(row['rating']), 3),
        'rank':             int(row['rank']),
        'lg_rank':          int(row['lg_rank']),
        'record':           final_record_lookup.get((team, season), '0-0-0'),
        'domestic_finish':  clean(row['domestic_finish']),
        'cl_finish':        clean(row['cl_finish']),
        'el_finish':        clean(row['el_finish']),
        'domestic_cup_finish': clean(row.get('domestic_cup_finish', '')),
    }

champions = {}

# Domestic leagues — rank by final points from game data
for league in sorted(DOMESTIC_LEAGUES):
    entries = []
    league_pts = dom_final_pts[dom_final_pts['competition'] == league]
    for season in sorted(league_pts['season'].unique(), reverse=True):
        if not season_is_complete(season):
            continue
        s_pts = league_pts[league_pts['season'] == season].sort_values(['pts', 'gd'], ascending=False).reset_index(drop=True)
        if len(s_pts) < 2:
            continue
        champ_name = s_pts.iloc[0]['team']
        ru_name    = s_pts.iloc[1]['team']
        entries.append({
            'season':    season,
            'champion':  _dom_entry(champ_name, 'Champion',  season),
            'runner_up': _dom_entry(ru_name,    'Runner-Up', season),
        })
    champions[league] = entries

# European cups
for comp, key in [('Champions League', 'Champions League'), ('Europa League', 'Europa League')]:
    euro_games = games_raw[
        (games_raw['competition'] == comp) &
        games_raw['comp_season'].notna() &
        (games_raw['comp_season'].str.strip() != '')
    ].copy()
    entries = []
    for comp_season in sorted(euro_games['comp_season'].str.strip().unique(), reverse=True):
        if not season_is_complete(comp_season):
            continue
        sdf      = euro_games[euro_games['comp_season'].str.strip() == comp_season]
        final    = sdf[sdf['date'] == sdf['date'].max()].iloc[-1]
        home, away = final['home_team'], final['away_team']
        hs, as_  = int(final['home_score']), int(final['away_score'])
        if hs > as_:
            champion, runner_up = home, away
        elif as_ > hs:
            champion, runner_up = away, home
        else:
            sw = final['shootout_winner']
            if pd.notna(sw) and str(sw).strip():
                champion  = str(sw).strip()
                runner_up = away if champion == home else home
            else:
                continue
        final_score = f"{hs}-{as_}"
        if hs == as_:
            final_score += " (pen)"
        entries.append({
            'season':      comp_season,
            'final_score': final_score,
            'champion':    euro_team_dict(champion,   comp_season),
            'runner_up':   euro_team_dict(runner_up,  comp_season),
        })
    champions[key] = entries

with open('docs/data/champions.json', 'w') as f:
    json.dump(champions, f, separators=(',', ':'))

print(f"Done. {len(teams_index)} teams, {len(standings_data['teams'])} in current standings.")
print(f"Wrote {len(all_seasons)} season files. Standings date: {latest_date}")
