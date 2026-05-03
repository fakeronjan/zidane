# ============================================================
# ZIDANE - European Club Soccer Power Rankings
# Based on MESSI / LOGAN / OLANDIS architecture
# ============================================================

import pandas as pd
import numpy as np
import requests
import io
import re
from datetime import date
import warnings
warnings.filterwarnings('ignore')
import rankit
from rankit.Table import Table
from rankit.Ranker import MasseyRanker

# ============================================================
# PARAMETERS
# ============================================================

window_game_days = 200   # rolling window in game days (not calendar days)
margin_cap      = 4      # max goal margin fed into Massey
shootout_margin = 0.5    # margin assigned to a penalty shootout win
home_field_adv  = 0.5    # same as MESSI / OLANDIS
min_games       = 15     # minimum games in rolling window to appear in final output

# ============================================================
# DATA SOURCE CONFIGURATION
# ============================================================

CHAMPIONS_LEAGUE_BASE = 'https://raw.githubusercontent.com/openfootball/champions-league/master'

# Domestic leagues via football-data.co.uk (free CSVs, no key required)
# URL pattern: {FDCO_BASE}/{YYZZ}/{code}.csv  e.g. 1112/E0.csv for EPL 2011-12
FDCO_BASE = 'https://www.football-data.co.uk/mmz4281'
FDCO_LEAGUE_CODES = {
    'EPL':        'E0',
    'La Liga':    'SP1',
    'Bundesliga': 'D1',
    'Serie A':    'I1',
    'Ligue 1':    'F1',
}

# All 5 leagues + CL present from 2011-12 onward
FIRST_SEASON_YEAR = 2011

# Dynamically compute the current season
_today = date.today()
_cur_start = _today.year if _today.month >= 8 else _today.year - 1

def make_season(start_year):
    return f"{start_year}-{str(start_year + 1)[-2:]}"

ALL_SEASONS = [make_season(y) for y in range(FIRST_SEASON_YEAR, _cur_start + 1)]

# ============================================================
# TEAM NAME NORMALIZATION
# ============================================================
# The EL .txt files use names like "Manchester United (ENG)" or "Man United".
# The domestic JSON files use names like "Manchester United FC".
# This dict maps every known variant → one canonical name.
# Populate after first data pull by inspecting mismatches.

TEAM_NAME_MAP = {
    # ====================================================================
    # ENGLAND — football-data.co.uk uses short names; canonical = full FC
    # ====================================================================
    'Arsenal':                          'Arsenal FC',
    'Aston Villa':                      'Aston Villa FC',
    'Blackburn':                        'Blackburn Rovers FC',
    'Bolton':                           'Bolton Wanderers FC',
    'Bournemouth':                      'AFC Bournemouth',
    'Brentford':                        'Brentford FC',
    'Brighton':                         'Brighton & Hove Albion FC',
    'Brighton & Hove Albion':           'Brighton & Hove Albion FC',
    'Burnley':                          'Burnley FC',
    'Cardiff':                          'Cardiff City FC',
    'Chelsea':                          'Chelsea FC',
    'Crystal Palace':                   'Crystal Palace FC',
    'Everton':                          'Everton FC',
    'Fulham':                           'Fulham FC',
    'Hull':                             'Hull City AFC',
    'Leeds':                            'Leeds United FC',
    'Leicester':                        'Leicester City FC',
    'Leicester City':                   'Leicester City FC',
    'Liverpool':                        'Liverpool FC',
    'Luton':                            'Luton Town FC',
    'Man City':                         'Manchester City FC',
    'Manchester City':                  'Manchester City FC',
    'Man United':                       'Manchester United FC',
    'Manchester United':                'Manchester United FC',
    'Middlesbrough':                    'Middlesbrough FC',
    'Newcastle':                        'Newcastle United FC',
    'Newcastle United':                 'Newcastle United FC',
    'Norwich':                          'Norwich City FC',
    'Norwich City':                     'Norwich City FC',
    "Nott'm Forest":                    'Nottingham Forest FC',
    'QPR':                              'Queens Park Rangers FC',
    'Reading':                          'Reading FC',
    'Sheffield United':                 'Sheffield United FC',
    'Southampton':                      'Southampton FC',
    'Stoke':                            'Stoke City FC',
    'Sunderland':                       'Sunderland AFC',
    'Swansea':                          'Swansea City AFC',
    'Tottenham':                        'Tottenham Hotspur FC',
    'Tottenham Hotspur':                'Tottenham Hotspur FC',
    'Watford':                          'Watford FC',
    'West Brom':                        'West Bromwich Albion FC',
    'West Bromwich Albion':             'West Bromwich Albion FC',
    'West Ham':                         'West Ham United FC',
    'West Ham United':                  'West Ham United FC',
    'Wigan':                            'Wigan Athletic FC',
    'Wolves':                           'Wolverhampton Wanderers FC',
    'Wolverhampton Wanderers':          'Wolverhampton Wanderers FC',

    # ====================================================================
    # SPAIN — fdco uses English abbreviations; canonical = full Spanish name
    # ====================================================================
    'Alaves':                           'Deportivo Alavés',
    'CD Alavés':                        'Deportivo Alavés',
    'Almeria':                          'UD Almería',
    'Ath Bilbao':                       'Athletic Club de Bilbao',
    'Athletic Club':                    'Athletic Club de Bilbao',
    'Ath Madrid':                       'Club Atlético de Madrid',
    'Atletico Madrid':                  'Club Atlético de Madrid',
    'Atlético Madrid':                  'Club Atlético de Madrid',
    'Atletico de Madrid':               'Club Atlético de Madrid',
    'Barcelona':                        'FC Barcelona',
    'Betis':                            'Real Betis Balompié',
    'Real Betis':                       'Real Betis Balompié',
    'Cadiz':                            'Cádiz CF',
    'Celta':                            'RC Celta de Vigo',
    'RC Celta':                         'RC Celta de Vigo',
    'Celta Vigo':                       'RC Celta de Vigo',
    'Eibar':                            'SD Eibar',
    'Elche':                            'Elche CF',
    'Espanol':                          'RCD Espanyol de Barcelona',
    'Espanyol':                         'RCD Espanyol de Barcelona',
    'Espanyol Barcelona':               'RCD Espanyol de Barcelona',
    'Getafe':                           'Getafe CF',
    'Girona':                           'Girona FC',
    'Granada':                          'Granada CF',
    'Las Palmas':                       'UD Las Palmas',
    'Leganes':                          'CD Leganés',
    'Levante':                          'Levante UD',
    'Malaga':                           'Málaga CF',
    'Mallorca':                         'RCD Mallorca',
    'Osasuna':                          'CA Osasuna',
    'Real Madrid':                      'Real Madrid CF',
    'Santander':                        'Racing de Santander',
    'Sevilla':                          'Sevilla FC',
    'Sociedad':                         'Real Sociedad de Fútbol',
    'Real Sociedad':                    'Real Sociedad de Fútbol',
    'Sp Gijon':                         'Sporting de Gijón',
    'Valencia':                         'Valencia CF',
    'Valladolid':                       'Real Valladolid CF',
    'Real Valladolid':                  'Real Valladolid CF',
    'Vallecano':                        'Rayo Vallecano de Madrid',
    'Rayo Vallecano':                   'Rayo Vallecano de Madrid',
    'Villarreal':                       'Villarreal CF',
    'Zaragoza':                         'Real Zaragoza',

    # ====================================================================
    # GERMANY — fdco uses German short names
    # ====================================================================
    'Augsburg':                         'FC Augsburg',
    'Bayern Munich':                    'FC Bayern München',
    'Bayern München':                   'FC Bayern München',
    'Bayer Leverkusen':                 'Bayer 04 Leverkusen',
    'Leverkusen':                       'Bayer 04 Leverkusen',
    'Bochum':                           'VfL Bochum 1848',
    'Braunschweig':                     'Eintracht Braunschweig',
    'Darmstadt':                        'SV Darmstadt 98',
    'Dortmund':                         'Borussia Dortmund',
    'Borussia Dortmund':                'Borussia Dortmund',
    'Ein Frankfurt':                    'Eintracht Frankfurt',
    'Eintracht Frankfurt':              'Eintracht Frankfurt',
    'FC Koln':                          '1. FC Köln',
    'Fortuna Dusseldorf':               'Fortuna Düsseldorf',
    'Freiburg':                         'SC Freiburg',
    'Hamburg':                          'Hamburger SV',
    'Hannover':                         'Hannover 96',
    'Heidenheim':                       '1. FC Heidenheim 1846',
    'Hertha':                           'Hertha BSC',
    'Hoffenheim':                       'TSG 1899 Hoffenheim',
    '1899 Hoffenheim':                  'TSG 1899 Hoffenheim',
    'Kaiserslautern':                   '1. FC Kaiserslautern',
    "M'gladbach":                       'Borussia Mönchengladbach',
    'Bor. Mönchengladbach':             'Borussia Mönchengladbach',
    'Borussia Mönchengladbach':         'Borussia Mönchengladbach',
    'Mainz':                            '1. FSV Mainz 05',
    'Nurnberg':                         '1. FC Nürnberg',
    'Paderborn':                        'SC Paderborn 07',
    'RB Leipzig':                       'RB Leipzig',
    'Schalke 04':                       'FC Schalke 04',
    'Stuttgart':                        'VfB Stuttgart',
    'Union Berlin':                     '1. FC Union Berlin',
    'Werder Bremen':                    'SV Werder Bremen',
    'Wolfsburg':                        'VfL Wolfsburg',

    # ====================================================================
    # ITALY — fdco uses short/common names
    # ====================================================================
    'Atalanta':                         'Atalanta BC',
    'Bologna':                          'Bologna FC 1909',
    'Bologna FC':                       'Bologna FC 1909',
    'Brescia':                          'Brescia Calcio',
    'Cagliari':                         'Cagliari Calcio',
    'Catania':                          'Calcio Catania',
    'Cesena':                           'AC Cesena',
    'Chievo':                           'AC ChievoVerona',
    'Empoli':                           'Empoli FC',
    'Fiorentina':                       'ACF Fiorentina',
    'Frosinone':                        'Frosinone Calcio',
    'Genoa':                            'Genoa CFC',
    'Hellas Verona':                    'Hellas Verona FC',
    'Verona':                           'Hellas Verona FC',
    'Inter':                            'FC Internazionale Milano',
    'Inter Milan':                      'FC Internazionale Milano',
    'Internazionale':                   'FC Internazionale Milano',
    'Juventus':                         'Juventus FC',
    'Lazio':                            'SS Lazio',
    'Lazio Roma':                       'SS Lazio',
    'Lecce':                            'US Lecce',
    'Livorno':                          'AS Livorno Calcio',
    'Milan':                            'AC Milan',
    'Monza':                            'AC Monza',
    'Napoli':                           'SSC Napoli',
    'Novara':                           'Novara Calcio',
    'Palermo':                          'US Città di Palermo',
    'Parma':                            'Parma Calcio 1913',
    'Parma FC':                         'Parma Calcio 1913',
    'Roma':                             'AS Roma',
    'Salernitana':                      'US Salernitana 1919',
    'Sampdoria':                        'UC Sampdoria',
    'Sassuolo':                         'US Sassuolo Calcio',
    'Sassuolo Calcio':                  'US Sassuolo Calcio',
    'Siena':                            'AC Siena',
    'Spal':                             'SPAL',
    'Torino':                           'Torino FC',
    'Udinese':                          'Udinese Calcio',

    # ====================================================================
    # FRANCE — fdco uses short names
    # ====================================================================
    'Ajaccio':                          'AC Ajaccio',
    'Amiens':                           'Amiens SC',
    'Angers':                           'Angers SCO',
    'Auxerre':                          'AJ Auxerre',
    'Bastia':                           'SC Bastia',
    'Bordeaux':                         'FC Girondins de Bordeaux',
    'Brest':                            'Stade Brestois 29',
    'Caen':                             'Stade Malherbe Caen',
    'Clermont':                         'Clermont Foot 63',
    'Dijon':                            'Dijon FCO',
    'Evian Thonon Gaillard':            'Evian Thonon Gaillard FC',
    'Guingamp':                         'En Avant de Guingamp',
    'Le Havre':                         'Le Havre AC',
    'Lens':                             'Racing Club de Lens',
    'RC Lens':                          'Racing Club de Lens',
    'Lille':                            'Lille OSC',
    'Lorient':                          'FC Lorient',
    'Lyon':                             'Olympique Lyonnais',
    'Olympique Lyon':                   'Olympique Lyonnais',
    'Marseille':                        'Olympique de Marseille',
    'Olympique Marseille':              'Olympique de Marseille',
    'Metz':                             'FC Metz',
    'Monaco':                           'AS Monaco FC',
    'AS Monaco':                        'AS Monaco FC',
    'Montpellier':                      'Montpellier HSC',
    'Nancy':                            'AS Nancy-Lorraine',
    'Nantes':                           'FC Nantes',
    'Nice':                             'OGC Nice',
    'Nimes':                            'Nîmes Olympique',
    'Paris SG':                         'Paris Saint-Germain FC',
    'Paris Saint-Germain':              'Paris Saint-Germain FC',
    'PSG':                              'Paris Saint-Germain FC',
    'Reims':                            'Stade de Reims',
    'Rennes':                           'Stade Rennais FC 1901',
    'Stade Rennais':                    'Stade Rennais FC 1901',
    'Sochaux':                          'FC Sochaux-Montbéliard',
    'St Etienne':                       'AS Saint-Étienne',
    'Strasbourg':                       'RC Strasbourg Alsace',
    'RC Strasbourg':                    'RC Strasbourg Alsace',
    'Toulouse':                         'Toulouse FC',
    'Valenciennes':                     'Valenciennes FC',

    # ====================================================================
    # EUROPEAN (non-top-5, appear in CL/EL — cross-league calibration)
    # ====================================================================
    'Crvena Zvezda':                    'FK Crvena Zvezda',
    'Dinamo Zagreb':                    'GNK Dinamo Zagreb',
    'Feyenoord':                        'Feyenoord Rotterdam',
    'Galatasaray':                      'Galatasaray SK',
    'PAE Olympiakos SFP':               'Olympiakos Piraeus',
    'PSV':                              'PSV Eindhoven',
    'Qarabağ Ağdam FK':                 'Qarabağ FK',
    'RB Salzburg':                      'FC Red Bull Salzburg',
    'Shakhtar Donetsk':                 'FK Shakhtar Donetsk',
    'SL Benfica':                       'Sport Lisboa e Benfica',
    'Slavia Praha':                     'SK Slavia Praha',
    'Sporting Braga':                   'Sporting Clube de Braga',
    'Sporting CP':                      'Sporting Clube de Portugal',
    'Sturm Graz':                       'SK Sturm Graz',
    'Union Saint-Gilloise':             'Royale Union Saint-Gilloise',
}

def normalize_team(raw_name):
    # Strip country code: "Arsenal FC (ENG)" → "Arsenal FC"
    name = re.sub(r'\s*\([A-Z]{2,3}\)\s*$', '', raw_name).strip()
    return TEAM_NAME_MAP.get(name, name)

# ============================================================
# STEP 1 - LOAD DOMESTIC LEAGUE JSON
# ============================================================
# Schema: { "name": "...", "matches": [ { "round", "date", "time",
#           "team1", "team2", "score": { "ht": [h,a], "ft": [h,a] } } ] }

def load_domestic_fdco(season):
    # football-data.co.uk season code: "2011-12" → "1112"
    season_code = f"{season[2:4]}{season[5:7]}"
    rows = []
    for league, code in FDCO_LEAGUE_CODES.items():
        url = f'{FDCO_BASE}/{season_code}/{code}.csv'
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                print(f"  Warning: {league} {season} not available ({r.status_code})")
                continue
            df_csv = pd.read_csv(
                io.StringIO(r.text),
                usecols=lambda c: c in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'],
                dtype={'FTHG': 'Int64', 'FTAG': 'Int64'},
            )
            df_csv = df_csv.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
            for _, m in df_csv.iterrows():
                try:
                    parsed_date = pd.to_datetime(m['Date'], dayfirst=True)
                except Exception:
                    continue
                rows.append({
                    'date':            parsed_date.strftime('%Y-%m-%d'),
                    'home_team':       normalize_team(str(m['HomeTeam']).strip()),
                    'away_team':       normalize_team(str(m['AwayTeam']).strip()),
                    'home_score':      int(m['FTHG']),
                    'away_score':      int(m['FTAG']),
                    'competition':     league,
                    'neutral':         False,
                    'shootout_winner': None,
                })
        except Exception as e:
            print(f"  Warning: could not load {league} {season}: {e}")
    return rows

# ============================================================
# STEP 2 - PARSE EUROPEAN COMPETITION TXT (CL + EL)
# ============================================================
# Used for both Champions League (cl.txt) and Europa League (el.txt).
# Both files live in openfootball/champions-league and share the same format.
#
#   Section headers  →  lines starting with =, #, »  (skip)
#   Date lines       →  "  Thu Sep/21 2023" or "  Thu Oct/5" (year implicit after first)
#   Match lines      →  "    21.00  Team A (XX)  v  Team B (XX)  2-1 (1-0)"
#   AET              →  "...  2-1 a.e.t. (1-1, 0-0)"        → use 2-1 as ft score
#   Penalties        →  "...  4-1 pen. 1-1 a.e.t."          → match score=1-1, winner by pen
#
# Two-legged knockout ties: each leg is a separate line, treated as
# an independent game. Aggregate score is irrelevant for Massey.

_MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,  'May': 5,  'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# Date line: optional day-of-week, then Mon/DD, optional year
_RE_DATE = re.compile(
    r'^\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Za-z]+)/(\d{1,2})(?:\s+(\d{4}))?'
)

# Penalty match: "pen_h-pen_a pen. match_h-match_a a.e.t."
_RE_PEN = re.compile(
    r'^\s+(?:\d{1,2}\.\d{2}\s+)?(.+?)\s+v\s+(.+?)\s+'
    r'(\d+)-(\d+)\s+pen\.\s+(\d+)-(\d+)\s+a\.e\.t\.'
)

# AET (no penalties): score includes extra time, no shootout needed
_RE_AET = re.compile(
    r'^\s+(?:\d{1,2}\.\d{2}\s+)?(.+?)\s+v\s+(.+?)\s+'
    r'(\d+)-(\d+)\s+a\.e\.t\.'
)

# Regular match (with or without halftime score in parentheses)
_RE_MATCH = re.compile(
    r'^\s+(?:\d{1,2}\.\d{2}\s+)?(.+?)\s+v\s+(.+?)\s+'
    r'(\d+)-(\d+)'
    r'(?:\s+\([\d\-,\s]+\))?'
    r'\s*$'
)

def parse_european_txt(season, competition, filename):
    url = f'{CHAMPIONS_LEAGUE_BASE}/{season}/{filename}'
    rows = []
    try:
        text = requests.get(url, timeout=10).text
    except Exception as e:
        print(f"  Warning: could not load {competition} {season}: {e}")
        return rows

    current_date = None
    current_year = None

    for line in text.splitlines():
        # Skip headers and blank lines
        if not line.strip() or line.strip().startswith(('=', '#', '»')):
            continue

        # Date line
        dm = _RE_DATE.match(line)
        if dm:
            month_str, day_str, year_str = dm.group(1), dm.group(2), dm.group(3)
            if year_str:
                current_year = int(year_str)
            if current_year and month_str in _MONTH_MAP:
                current_date = f"{current_year}-{_MONTH_MAP[month_str]:02d}-{int(day_str):02d}"
            continue

        if current_date is None:
            continue

        # Penalty match
        pm = _RE_PEN.match(line)
        if pm:
            team1, team2 = normalize_team(pm.group(1)), normalize_team(pm.group(2))
            pen_h, pen_a   = int(pm.group(3)), int(pm.group(4))
            match_h, match_a = int(pm.group(5)), int(pm.group(6))
            winner = team1 if pen_h > pen_a else team2
            rows.append({
                'date':            current_date,
                'home_team':       team1,
                'away_team':       team2,
                'home_score':      match_h,
                'away_score':      match_a,
                'competition':     competition,
                'comp_season':     season,
                'neutral':         False,
                'shootout_winner': winner,
            })
            continue

        # AET match (no penalties)
        am = _RE_AET.match(line)
        if am:
            rows.append({
                'date':            current_date,
                'home_team':       normalize_team(am.group(1)),
                'away_team':       normalize_team(am.group(2)),
                'home_score':      int(am.group(3)),
                'away_score':      int(am.group(4)),
                'competition':     competition,
                'comp_season':     season,
                'neutral':         False,
                'shootout_winner': None,
            })
            continue

        # Regular match
        mm = _RE_MATCH.match(line)
        if mm:
            rows.append({
                'date':            current_date,
                'home_team':       normalize_team(mm.group(1)),
                'away_team':       normalize_team(mm.group(2)),
                'home_score':      int(mm.group(3)),
                'away_score':      int(mm.group(4)),
                'competition':     competition,
                'comp_season':     season,
                'neutral':         False,
                'shootout_winner': None,
            })

    return rows

# ============================================================
# STEP 4 - BUILD MASTER GAME FILE
# ============================================================

print("Loading all game data...")

all_rows = []
for season in ALL_SEASONS:
    print(f"  Loading {season}...")
    all_rows.extend(load_domestic_fdco(season))
    all_rows.extend(parse_european_txt(season, 'Champions League', 'cl.txt'))
    all_rows.extend(parse_european_txt(season, 'Europa League',    'el.txt'))

df = pd.DataFrame(all_rows)
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Warn on any games with unexpected nulls
null_scores = df[df['home_score'].isna() | df['away_score'].isna()]
if len(null_scores):
    print(f"  WARNING: {len(null_scores)} rows with missing scores — check loaders")

df.to_csv('all_club_games.csv', index=False)
print(f"Master game file saved: {len(df)} rows")

# ============================================================
# STEP 5 - HANDLE SHOOTOUTS / PENALTIES
# ============================================================
# Same logic as MESSI: when AET score is level and shootout_winner
# is set, override margin with +/- 0.5 for the winner.

df['raw_margin_home'] = df['home_score'] - df['away_score']

shootout_mask = (
    df['shootout_winner'].notna() &
    (df['raw_margin_home'] == 0)
)

df['margin_home'] = df['raw_margin_home'].astype(float)
df['margin_away'] = -df['margin_home']

df.loc[shootout_mask & (df['shootout_winner'] == df['home_team']), 'margin_home'] =  shootout_margin
df.loc[shootout_mask & (df['shootout_winner'] == df['home_team']), 'margin_away'] = -shootout_margin
df.loc[shootout_mask & (df['shootout_winner'] == df['away_team']), 'margin_home'] = -shootout_margin
df.loc[shootout_mask & (df['shootout_winner'] == df['away_team']), 'margin_away'] =  shootout_margin

# ============================================================
# STEP 6 - MARGIN CAP
# ============================================================

df['margin_home'] = df['margin_home'].clip(-margin_cap, margin_cap)
df['margin_away'] = -df['margin_home']

# ============================================================
# STEP 7 - HOME FIELD ADJUSTMENT
# ============================================================

df['hfa'] = np.where(df['neutral'] == True, 0, home_field_adv)
df['adj_margin_home'] = df['margin_home'] - df['hfa']
df['adj_margin_away'] = -df['adj_margin_home']

# ============================================================
# STEP 8 - WIN FLAGS AND SEASON LABELS
# ============================================================

df['home_win'] = np.where(
    df['margin_home'] > 0, 1,
    np.where(df['margin_home'] == 0, 0.5, 0)
)
df['away_win'] = 1 - df['home_win']

def date_to_season(d):
    if d.month >= 8:
        return f"{d.year}-{str(d.year + 1)[-2:]}"
    return f"{d.year - 1}-{str(d.year)[-2:]}"

df['season'] = df['date'].apply(date_to_season)

# ============================================================
# STEP 9 - DATE IDs FOR ROLLING WINDOW
# ============================================================

df['grouped_date_id'] = df.groupby('date').ngroup() + 1

# ============================================================
# BUILD LAST MATCH STRINGS
# ============================================================
# Format: "W vs. Real Madrid CF 2-1 (Champions League)"
#         "L @ Liverpool FC 0-3 (EPL)"

df['home_score_int'] = pd.to_numeric(df['home_score'], errors='coerce')
df['away_score_int'] = pd.to_numeric(df['away_score'], errors='coerce')
df = df.dropna(subset=['home_score_int', 'away_score_int']).copy()
df['home_score_int'] = df['home_score_int'].astype(int)
df['away_score_int'] = df['away_score_int'].astype(int)

df['home_result_flag'] = np.where(df['home_win'] == 1, 'W', np.where(df['home_win'] == 0.5, 'D', 'L'))
df['away_result_flag'] = np.where(df['away_win'] == 1, 'W', np.where(df['away_win'] == 0.5, 'D', 'L'))

df['home_last_match'] = (
    df['home_result_flag'] + ' vs. ' +
    df['away_team'] + ' ' +
    df['home_score_int'].map(str) + '-' + df['away_score_int'].map(str) +
    ' (' + df['competition'] + ')'
)
df['away_last_match'] = (
    df['away_result_flag'] + ' @ ' +
    df['home_team'] + ' ' +
    df['away_score_int'].map(str) + '-' + df['home_score_int'].map(str) +
    ' (' + df['competition'] + ')'
)

lastmatch_home = df[['date', 'home_team', 'home_last_match']].copy()
lastmatch_home.columns = ['date', 'name', 'last_match']
lastmatch_away = df[['date', 'away_team', 'away_last_match']].copy()
lastmatch_away.columns = ['date', 'name', 'last_match']
lastmatch_df = pd.concat([lastmatch_home, lastmatch_away]).reset_index(drop=True)
lastmatch_df['date'] = pd.to_datetime(lastmatch_df['date']).dt.date

# ============================================================
# STEP 10 - ROLLING MASSEY RATINGS (ZIDANE RATINGS)
# ============================================================
# One snapshot per game-day, rolling 200-game-day window,
# linear recency weighting by game days (not calendar days).
# Breaks and international windows do not count against recency.
# No tournament weights — all club games are competitive by definition.
# Incremental: skips date IDs already in zidane_ratings.csv.

print("Starting ZIDANE rating calculations...")

max_date_id = int(df['grouped_date_id'].max())

try:
    zidane_df = pd.read_csv('zidane_ratings.csv')
    max_ranked = int(zidane_df['ranking_id'].max())
    min_ranked = int(zidane_df['ranking_id'].min())
    print(f"Existing ratings found. Ranked IDs: {min_ranked} to {max_ranked}")
except FileNotFoundError:
    zidane_df = pd.DataFrame(columns=['ranking_id', 'ranking_date', 'season', 'name', 'rating', 'rank'])
    max_ranked = -1
    min_ranked = -1
    print("No existing ratings — running full history from scratch.")

last_printed_ym = None

for i in range(1, max_date_id + 1):

    if min_ranked <= i <= max_ranked:
        continue

    current_date = df.loc[df['grouped_date_id'] == i, 'date'].max()

    working_df = df.loc[
        (df['grouped_date_id'] >= i - window_game_days + 1) &
        (df['grouped_date_id'] <= i)
    ].copy()

    if len(working_df) < 10:
        continue

    working_df['game_days_ago'] = i - working_df['grouped_date_id']
    working_df['date_weight']   = 1 - (working_df['game_days_ago'] / window_game_days)

    # No tournament weight — single weight factor
    working_df['weighted_margin_home'] = working_df['adj_margin_home'] * working_df['date_weight']
    working_df['weighted_margin_away'] = -working_df['weighted_margin_home']

    # Drop zero-weighted rows to avoid Massey solver issues
    working_df = working_df[working_df['weighted_margin_home'] != 0]
    if len(working_df) < 10:
        continue

    current_ym = current_date.strftime('%Y-%m')
    if current_ym != last_printed_ym:
        pct = round(100 * i / max_date_id)
        print(f"  Ratings: {current_date.strftime('%B %Y')} ({pct}% complete)")
        last_printed_ym = current_ym

    try:
        soccer_table = Table(
            working_df,
            ['home_team', 'away_team', 'weighted_margin_home', 'weighted_margin_away']
        )
        zidane_massey = MasseyRanker(soccer_table)
        ranked = zidane_massey.rank()

        if ranked['rating'].isna().any() or np.isinf(ranked['rating']).any():
            continue

        ranked['ranking_id']   = i
        ranked['ranking_date'] = current_date.date()
        ranked['season']       = date_to_season(current_date)

        home_gp = working_df.groupby('home_team').size().reset_index(name='gp_home')
        away_gp = working_df.groupby('away_team').size().reset_index(name='gp_away')
        home_gp.columns = ['name', 'gp_home']
        away_gp.columns = ['name', 'gp_away']
        gp = pd.merge(home_gp, away_gp, on='name', how='outer').fillna(0)
        gp['games_played'] = (gp['gp_home'] + gp['gp_away']).astype(int)
        ranked = pd.merge(ranked, gp[['name', 'games_played']], on='name', how='left')
        ranked['games_played'] = ranked['games_played'].fillna(0).astype(int)

        zidane_df = pd.concat([zidane_df, ranked], axis=0, sort=False).reset_index(drop=True)

    except Exception as e:
        print(f"  Skipping date ID {i} due to solver error: {e}")
        continue

zidane_df.sort_values(['ranking_id', 'name'], inplace=True)
zidane_df.drop_duplicates(keep='first', inplace=True)
zidane_df['ranking_date'] = pd.to_datetime(zidane_df['ranking_date']).dt.date
zidane_df.to_csv('zidane_ratings.csv', index=False)
print("zidane_ratings.csv saved!")

# ============================================================
# STEP 10b - COMPUTE STANDINGS BY COMPETITION
# ============================================================
# One row per team × season × competition: GP/W/D/L/GF/GA/GD/Pts.
# Covers all five domestic leagues, Champions League, and Europa League.

print("Computing standings...")

# Build a team-perspective view (one row per team per game)
home_view = df[['season', 'competition', 'home_team', 'away_team',
                'home_score_int', 'away_score_int', 'home_win']].copy()
home_view.columns = ['season', 'competition', 'team', 'opponent', 'gf', 'ga', 'result']

away_view = df[['season', 'competition', 'away_team', 'home_team',
                'away_score_int', 'home_score_int', 'away_win']].copy()
away_view.columns = ['season', 'competition', 'team', 'opponent', 'gf', 'ga', 'result']

team_view = pd.concat([home_view, away_view], ignore_index=True)
team_view['w'] = (team_view['result'] == 1  ).astype(int)
team_view['d'] = (team_view['result'] == 0.5).astype(int)
team_view['l'] = (team_view['result'] == 0  ).astype(int)

standings_df = (
    team_view
    .groupby(['season', 'competition', 'team'])
    .agg(gp=('gf', 'count'), w=('w', 'sum'), d=('d', 'sum'),
         l=('l', 'sum'), gf=('gf', 'sum'), ga=('ga', 'sum'))
    .reset_index()
)
standings_df['gd']  = standings_df['gf'] - standings_df['ga']
standings_df['pts'] = standings_df['w'] * 3 + standings_df['d']
standings_df = standings_df.sort_values(
    ['season', 'competition', 'pts', 'gd', 'gf'],
    ascending=[True, True, False, False, False]
)

standings_df.to_csv('zidane_standings.csv', index=False)
print(f"zidane_standings.csv saved! ({len(standings_df)} rows)")

# ============================================================
# STEP 10c - DETECT COMPETITION FINISHES
# ============================================================

print("Detecting competition finishes...")

def season_is_complete(season_str):
    """A season YYYY-YY is complete once today is past July 31 of the end year.
    This clears the CL final (late May/June) before marking anything done."""
    end_year = int('20' + season_str[-2:])
    return date.today() > date(end_year, 7, 31)

domestic_records = []
cl_records       = []
el_records       = []

# --- Domestic leader / champion ---
# Complete seasons: label top 2 as Champion / Runner-Up.
# In-progress seasons: label as 1st / 2nd (no trophy awarded yet).
dom_std = standings_df[standings_df['competition'].isin(FDCO_LEAGUE_CODES.keys())].copy()
for (season, competition), group in dom_std.groupby(['season', 'competition']):
    group = group.sort_values(['pts', 'gd', 'gf'], ascending=False)
    complete = season_is_complete(season)
    labels = ['Champion', 'Runner-Up'] if complete else ['1st', '2nd']
    for i, label in enumerate(labels):
        if i < len(group):
            domestic_records.append({
                'season': season, 'team': group.iloc[i]['team'],
                'domestic_finish': label,
            })

# --- CL / EL champion / runner-up ---
# Group by comp_season (the source file folder, e.g. "2019-20") not the
# computed date season. This fixes the 2019-20 Lisbon/Frankfurt bubble:
# those finals were played in August 2020, which date_to_season labels
# "2020-21", but they belong to the 2019-20 competition season.
# Skip any comp_season that is not yet complete — no champion until the
# final has been played.
for competition, records, col in [
    ('Champions League', cl_records, 'cl_finish'),
    ('Europa League',    el_records, 'el_finish'),
]:
    comp_df = df[df['competition'] == competition].dropna(subset=['comp_season']).copy()
    for comp_season, sdf in comp_df.groupby('comp_season'):
        if not season_is_complete(comp_season):
            continue  # final not yet played

        last_date = sdf['date'].max()
        final = sdf[sdf['date'] == last_date].iloc[-1]
        home, away, margin = final['home_team'], final['away_team'], final['margin_home']

        if margin > 0:
            champion, runner_up = home, away
        elif margin < 0:
            champion, runner_up = away, home
        else:
            sw = final.get('shootout_winner')
            if pd.notna(sw):
                champion  = sw
                runner_up = away if sw == home else home
            else:
                continue  # final result unknown (CL JSON lacks pen data)

        records.append({'season': comp_season, 'team': champion,  col: 'Champion'})
        records.append({'season': comp_season, 'team': runner_up, col: 'Runner-Up'})

domestic_finish_df = pd.DataFrame(domestic_records) if domestic_records else pd.DataFrame(columns=['season', 'team', 'domestic_finish'])
cl_finish_df       = pd.DataFrame(cl_records)       if cl_records       else pd.DataFrame(columns=['season', 'team', 'cl_finish'])
el_finish_df       = pd.DataFrame(el_records)       if el_records       else pd.DataFrame(columns=['season', 'team', 'el_finish'])

# Validation print — spot check before merging
print(f"\n  {'Season':<10} {'Domestic Champions'}")
print(f"  {'-'*55}")
for _, row in domestic_finish_df[domestic_finish_df['domestic_finish'] == 'Champion'].sort_values('season').iterrows():
    print(f"  {row['season']:<10} {row['team']}")

all_comp_seasons = sorted(set(cl_finish_df['season'].tolist() + el_finish_df['season'].tolist()))
print(f"\n  {'Season':<10} {'CL Champion':<35} {'EL Champion'}")
print(f"  {'-'*75}")
for season in all_comp_seasons:
    cl_row = cl_finish_df[(cl_finish_df['season'] == season) & (cl_finish_df['cl_finish'] == 'Champion')]
    el_row = el_finish_df[(el_finish_df['season'] == season) & (el_finish_df['el_finish'] == 'Champion')]
    cl_name = cl_row['team'].values[0] if len(cl_row) else '(in progress)'
    el_name = el_row['team'].values[0] if len(el_row) else '(no data)'
    print(f"  {season:<10} {cl_name:<35} {el_name}")

# ============================================================
# STEP 11 - MERGE INTO FINAL OUTPUT
# ============================================================

print("\nBuilding final output file...")

final_df = zidane_df.copy()
final_df.rename(columns={'ranking_date': 'date'}, inplace=True)
final_df['date'] = pd.to_datetime(final_df['date'])

# Most recent snapshot flag
latest_id = final_df['ranking_id'].max()
final_df['most_recent'] = np.where(final_df['ranking_id'] == latest_id, 1, 0)

# League membership: the domestic competition a team most recently appeared in.
# Uses both home and away appearances for robustness.
dom_games = df[df['competition'].isin(FDCO_LEAGUE_CODES.keys())].copy()
league_home = dom_games[['date', 'home_team', 'competition']].rename(columns={'home_team': 'name'})
league_away = dom_games[['date', 'away_team', 'competition']].rename(columns={'away_team': 'name'})
league_lookup = (
    pd.concat([league_home, league_away])
    .sort_values('date')
    .groupby('name')['competition']
    .last()
    .reset_index()
    .rename(columns={'competition': 'league'})
)
final_df = pd.merge(final_df, league_lookup, on='name', how='left')
final_df['league'] = final_df['league'].fillna('European/Other')

# Last match string via merge_asof (carries forward most recent result)
lastmatch_df_sorted = lastmatch_df.copy()
lastmatch_df_sorted['date'] = pd.to_datetime(lastmatch_df_sorted['date'])
lastmatch_df_sorted = lastmatch_df_sorted.sort_values('date')

final_df = final_df.sort_values('date')
final_df = pd.merge_asof(
    final_df,
    lastmatch_df_sorted.rename(columns={'date': 'match_date'}),
    left_on='date',
    right_on='match_date',
    by='name',
    direction='backward'
)
final_df['last_match']      = final_df['last_match'].fillna('No match yet')
final_df['last_match_date'] = final_df['match_date'].dt.date
final_df.drop(columns=['match_date'], inplace=True)
final_df['date'] = final_df['date'].dt.date

final_df['is_game_day'] = np.where(final_df['date'] == final_df['last_match_date'], 1, 0)

final_df.rename(columns={'name': 'team'}, inplace=True)

# Merge finish flags
final_df = pd.merge(final_df, domestic_finish_df, on=['season', 'team'], how='left')
final_df = pd.merge(final_df, cl_finish_df,       on=['season', 'team'], how='left')
final_df = pd.merge(final_df, el_finish_df,        on=['season', 'team'], how='left')
final_df['domestic_finish'] = final_df['domestic_finish'].fillna('')
final_df['cl_finish']       = final_df['cl_finish'].fillna('')
final_df['el_finish']       = final_df['el_finish'].fillna('')

# is_cl_final_day — snapshot falls on the CL final date for that comp season.
# Uses the actual game date (not computed season) to handle the 2019-20 bubble.
cl_final_date_set = set(
    df[df['competition'] == 'Champions League']
    .dropna(subset=['comp_season'])
    .groupby('comp_season')['date']
    .max()
    .dt.date
    .values
)
final_df['is_cl_final_day'] = np.where(final_df['date'].isin(cl_final_date_set), 1, 0)

# is_domestic_final_day — snapshot falls on the last day of that team's
# own domestic league for that season. Per-league, per-season.
dom_final_dates = (
    dom_games.groupby(['competition', 'season'])['date']
    .max()
    .dt.date
    .reset_index()
    .rename(columns={'competition': 'league', 'date': 'dom_final_date'})
)
final_df = pd.merge(final_df, dom_final_dates, on=['league', 'season'], how='left')
final_df['is_domestic_final_day'] = np.where(
    final_df['date'] == final_df['dom_final_date'], 1, 0
)
final_df.drop(columns=['dom_final_date'], inplace=True)

# is_end_of_season — fires on the true end of each sport-season: the latest date
# across domestic leagues AND CL/EL finals, where CL/EL is matched by comp_season
# (not date_to_season). This correctly places the 2019-20 COVID bubble CL final
# (August 23, 2020) in the "2019-20" season rather than "2020-21".
#
# We also patch the `season` label on any rows whose EOS date falls outside the
# date_to_season boundary (i.e. the August 2020 bubble), so that the season
# column remains consistent with the EOS flag.
dom_season_max = dom_games.groupby('season')['date'].max().rename('dom_end')
euro_games_df  = df[df['competition'].isin(['Champions League', 'Europa League'])].dropna(subset=['comp_season'])
euro_season_max = euro_games_df.groupby('comp_season')['date'].max().rename('euro_end')
season_bounds = (
    pd.merge(dom_season_max, euro_season_max, left_index=True, right_index=True, how='outer')
    .assign(true_eos=lambda x: x[['dom_end', 'euro_end']].max(axis=1))
)
season_bounds.index.name = 'sport_season'
season_bounds = season_bounds.reset_index()

# Build date → sport_season mapping (one EOS date per sport-season)
eos_map = (
    season_bounds.dropna(subset=['true_eos'])
    .assign(eos_date=lambda x: pd.to_datetime(x['true_eos']).dt.date)
    [['sport_season', 'eos_date']]
    .set_index('eos_date')['sport_season']
)

# Patch `season` on rows where the EOS date falls in a different date_to_season year
# (e.g. August 23, 2020 has date_to_season="2020-21" but belongs to sport-season "2019-20")
final_df['date'] = pd.to_datetime(final_df['date']).dt.date
eos_date_to_sport_season = eos_map.to_dict()
season_patch = {d: s for d, s in eos_date_to_sport_season.items() if date_to_season(pd.Timestamp(d)) != s}
for patch_date, correct_season in season_patch.items():
    final_df.loc[final_df['date'] == patch_date, 'season'] = correct_season

eos_date_set = set(eos_map.index.values)
final_df['is_end_of_season'] = np.where(final_df['date'].isin(eos_date_set), 1, 0)

# Final column order
final_df = final_df[[
    'ranking_id', 'date', 'season', 'team', 'league',
    'rating', 'rank',
    'games_played',
    'domestic_finish', 'cl_finish', 'el_finish',
    'is_cl_final_day', 'is_domestic_final_day', 'is_end_of_season',
    'last_match_date', 'last_match', 'is_game_day',
    'most_recent',
]]

final_df.sort_values(['ranking_id', 'rank'], inplace=True)
final_df.drop_duplicates(keep='first', inplace=True)

# Minimum games filter — drops relegated clubs with sparse history
# and non-top-5 clubs appearing only via European competition
final_df = final_df[final_df['games_played'] >= min_games]

# Renumber ranks contiguously within each snapshot after filtering
final_df['rank'] = (
    final_df.groupby('ranking_id')['rating']
    .rank(ascending=False, method='min')
    .astype(int)
)

final_df.to_csv('zidane_ratings_final.csv', index=False)
print("zidane_ratings_final.csv saved!")
print(f"\nTotal rows in final output: {len(final_df)}")

print(f"\nMost recent ZIDANE ratings (top 20):")
print(final_df[final_df['most_recent'] == 1][
    ['rank', 'team', 'league', 'rating', 'games_played', 'last_match_date', 'last_match']
].head(20).to_string(index=False))

# ============================================================
# SEASON-END TOP 5 REPORT
# ============================================================
# Last snapshot of each season — good for spot-checking responsiveness

print(f"\n{'='*65}")
print("ZIDANE — Top 5 at end of each season")
print(f"{'='*65}")

season_end_ids = final_df.groupby('season')['ranking_id'].max()

for season, snap_id in season_end_ids.items():
    snap = final_df[final_df['ranking_id'] == snap_id].copy()
    snap_date = snap['date'].iloc[0]
    print(f"\n  {season}  (snapshot: {snap_date})")
    print(f"  {'Rank':<6} {'Team':<35} {'League':<12} {'Rating':>8}")
    print(f"  {'-'*63}")
    for _, row in snap.head(5).iterrows():
        print(f"  {int(row['rank']):<6} {row['team']:<35} {row['league']:<12} {row['rating']:>8.3f}")
