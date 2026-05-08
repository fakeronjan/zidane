# ============================================================
# ZIDANE - European Club Soccer Power Rankings
# Based on MESSI / LOGAN / OLANDIS architecture
# ============================================================

import pandas as pd
import numpy as np
# rankit==0.2 uses deprecated numpy aliases (np.int, np.float, np.bool) removed in numpy 1.24+.
# Restore them before rankit import so the Massey solver works.
if not hasattr(np, 'int'):   np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'):  np.bool = bool
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

# Historical backfill source: jalapic/engsoccerdata raw CSVs.
# Used for 1992-93 through 2010-11 (before openfootball/CL coverage and to
# extend domestic coverage further back than football-data.co.uk's 1994-95).
# Schema: Date, Season, home, visitor, FT, hgoal, vgoal, tier, ...
# "Season=1992" denotes the 1992-93 campaign (autumn-spring).
ENGSOCCERDATA_BASE = 'https://raw.githubusercontent.com/jalapic/engsoccerdata/master/data-raw'
ENGSOCCERDATA_FILES = {
    'EPL':        'england.csv',
    'La Liga':    'spain.csv',
    'Bundesliga': 'germany.csv',
    'Serie A':    'italy.csv',
    'Ligue 1':    'france.csv',
}

# Switchover: pre-LAST_LEGACY_SEASON_YEAR uses engsoccerdata, post uses
# football-data.co.uk + openfootball (the existing pipeline).
LAST_LEGACY_SEASON_YEAR = 2010   # 2010-11 is the last legacy season
FIRST_SEASON_YEAR       = 1992   # 1992-93: EPL rebrand + CL rebrand

# Domestic cup data sources (Phase 3). Each cup uses the most reliable source
# for each era. Wikipedia gap-fill for the historical pre-openfootball window
# is loaded from a separate static CSV (cups_historical.csv) — see
# scrape_domestic_cups.py.
OPENFOOTBALL_COUNTRY_BASE = 'https://raw.githubusercontent.com/openfootball'

# (cup_name, country_repo, filename, first_openfootball_season)
# first_openfootball_season is the earliest year openfootball *actually*
# has cup.txt for that country — anything earlier is loaded from
# cups_historical.csv (Wikipedia scrape) instead.
DOMESTIC_CUPS = [
    ('FA Cup',          'england',     'facup.txt',     2018),  # engsoccerdata covers pre-2018
    ('DFB-Pokal',       'deutschland', 'cup.txt',       2018),  # openfootball cup.txt only 2018-19+
    ('Copa del Rey',    'espana',      'cup.txt',       2020),  # openfootball cup.txt only 2020-21+
    ('Coppa Italia',    'italy',       'cup.txt',       2020),  # openfootball cup.txt only 2020-21+
    # Coupe de France: no openfootball data — pure Wikipedia scrape, see Phase 3b
]

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

    # ====================================================================
    # HISTORICAL — engsoccerdata name variants (1992-93 → 2010-11)
    # Map any engsoccerdata variant to the same canonical used post-2011 so
    # teams that span both eras don't appear as duplicates in standings/output.
    # Defunct top-flight teams get a canonical name in the prevailing local
    # convention.
    # ====================================================================

    # England (engsoccerdata uses common names without "FC")
    'Barnsley':                         'Barnsley FC',
    'Birmingham City':                  'Birmingham City FC',
    'Blackburn Rovers':                 'Blackburn Rovers FC',
    'Blackpool':                        'Blackpool FC',
    'Bolton Wanderers':                 'Bolton Wanderers FC',
    'Bradford City':                    'Bradford City AFC',
    'Charlton Athletic':                'Charlton Athletic FC',
    'Coventry City':                    'Coventry City FC',
    'Derby County':                     'Derby County FC',
    'Hull City':                        'Hull City AFC',
    'Ipswich Town':                     'Ipswich Town FC',
    'Leeds United':                     'Leeds United FC',
    'Nottingham Forest':                'Nottingham Forest FC',
    'Oldham Athletic':                  'Oldham Athletic AFC',
    'Portsmouth':                       'Portsmouth FC',
    'Queens Park Rangers':              'Queens Park Rangers FC',
    'Sheffield Wednesday':              'Sheffield Wednesday FC',
    'Stoke City':                       'Stoke City FC',
    'Swindon Town':                     'Swindon Town FC',
    'Wigan Athletic':                   'Wigan Athletic FC',
    'Wimbledon':                        'Wimbledon FC',

    # Spain (engsoccerdata mixes accented and English-stripped variants)
    'Athletic Bilbao':                  'Athletic Club de Bilbao',
    'Cadiz CF':                         'Cádiz CF',
    'CD Logrones':                      'CD Logroñés',
    'CD Numancia':                      'CD Numancia',
    'CF Extremadura':                   'CF Extremadura',
    'CP Merida':                        'CP Mérida',
    'Deportivo La Coruna':              'Deportivo de La Coruña',
    'Gimnastic':                        'Gimnàstic de Tarragona',
    'Hercules CF':                      'Hércules CF',
    'Malaga CF':                        'Málaga CF',
    'Racing Santander':                 'Racing de Santander',
    'SD Compostela':                    'SD Compostela',
    'Sporting Gijon':                   'Sporting de Gijón',
    'UD Almeria':                       'UD Almería',
    'UD Salamanca':                     'UD Salamanca',
    'UE Lleida':                        'UE Lleida',
    'Xerez CD':                         'Xerez CD',

    # Germany (engsoccerdata has a unique style with "1." prefix and no umlauts)
    '1. FC Koln':                       '1. FC Köln',
    '1. FC Nurnberg':                   '1. FC Nürnberg',
    '1. FC Saarbrucken':                '1. FC Saarbrücken',
    'Bayer 05 Uerdingen':               'KFC Uerdingen 05',
    'Bayern Munchen':                   'FC Bayern München',
    'Bor. Monchengladbach':             'Borussia Mönchengladbach',
    'TSV 1860 Munchen':                 'TSV 1860 München',
    'VfL Bochum':                       'VfL Bochum 1848',

    # Italy (engsoccerdata sometimes prefixes; sometimes uses old names)
    'Bologna FC':                       'Bologna FC 1909',
    'Chievo Verona':                    'AC ChievoVerona',
    'Inter':                            'FC Internazionale Milano',
    'Lazio Roma':                       'SS Lazio',
    'Parma AC':                         'Parma Calcio 1913',
    'US Palermo':                       'US Città di Palermo',

    # France
    'AS Monaco':                        'AS Monaco FC',
    'AS Nancy':                         'AS Nancy-Lorraine',
    'AS Saint-Etienne':                 'AS Saint-Étienne',
    'ATAC Troyes':                      'ESTAC Troyes',
    'EA Guingamp':                      'En Avant de Guingamp',
    'FC Sochaux':                       'FC Sochaux-Montbéliard',
    'Girondins Bordeaux':               'FC Girondins de Bordeaux',
    'Nimes Olympique':                  'Nîmes Olympique',
    'SM Caen':                          'Stade Malherbe Caen',
    'Stade Brest':                      'Stade Brestois 29',
    'Stade Rennes':                     'Stade Rennais FC 1901',

    # Champions League — engsoccerdata variants
    'AFC Ajax':                         'AFC Ajax',
    'Bayern Munich':                    'FC Bayern München',
    'Crvena Zvezda':                    'FK Crvena Zvezda',
    'CSKA Moskva':                      'PFC CSKA Moskva',
    'Dinamo Bucuresti':                 'FC Dinamo București',
    'Dinamo Kiev':                      'FC Dynamo Kyiv',
    'Dinamo Moskva':                    'FC Dynamo Moscow',
    'Dynamo Kyiv':                      'FC Dynamo Kyiv',
    'FC Porto':                         'FC Porto',
    'Grasshoppers Zurich':              'Grasshopper Club Zürich',
    'Kobenhavn':                        'FC København',
    'Lazio':                            'SS Lazio',
    'Lokomotiv Moskva':                 'FC Lokomotiv Moscow',
    'Olympiacos':                       'Olympiakos Piraeus',
    'Panathinaikos':                    'Panathinaikos FC',
    'Partizan Belgrade':                'FK Partizan',
    'Rangers':                          'Rangers FC',
    'Rapid Bucuresti':                  'FC Rapid București',
    'Rapid Wien':                       'SK Rapid Wien',
    'Schalke 04':                       'FC Schalke 04',
    'Sevilla':                          'Sevilla FC',
    'Spartak Moskva':                   'FC Spartak Moscow',
    'Steaua Bucuresti':                 'FCSB',
    'Twente':                           'FC Twente',
    'Valencia CF':                      'Valencia CF',
    'Villarreal':                       'Villarreal CF',
    'Vitoria Guimaraes':                'Vitória SC',
    'Werder Bremen':                    'SV Werder Bremen',
    'Zenit St. Petersburg':             'FC Zenit Saint Petersburg',
    'sc Heerenveen':                    'sc Heerenveen',

    # ====================================================================
    # WIKIPEDIA — UEFA Cup / Europa League name variants (Phase 2 backfill)
    # Wikipedia generally uses common short names ("Ajax", "Benfica") rather
    # than the full club name. These map to the same canonical we use
    # everywhere else so teams aren't double-counted.
    # ====================================================================

    # England (Wikipedia variants)
    'Aberdeen':                         'Aberdeen FC',
    'Birmingham City':                  'Birmingham City FC',
    'Charlton Athletic':                'Charlton Athletic FC',
    'Dundalk':                          'Dundalk FC',
    'Fulham':                           'Fulham FC',
    'Heart of Midlothian':              'Heart of Midlothian FC',
    'Hibernian':                        'Hibernian FC',
    'Sligo Rovers':                     'Sligo Rovers FC',

    # Spain
    'Atlético Madrid':                  'Club Atlético de Madrid',
    'Deportivo La Coruña':              'Deportivo de La Coruña',
    'Espanyol':                         'RCD Espanyol de Barcelona',
    'Málaga':                           'Málaga CF',
    'Mallorca':                         'RCD Mallorca',
    'Real Betis':                       'Real Betis Balompié',
    'Real Sociedad':                    'Real Sociedad de Fútbol',

    # Germany
    'TSG Hoffenheim':                   'TSG 1899 Hoffenheim',
    'Mainz 05':                         '1. FSV Mainz 05',

    # Italy
    'Inter Milan':                      'FC Internazionale Milano',
    'Milan':                            'AC Milan',

    # France
    'Bordeaux':                         'FC Girondins de Bordeaux',
    'Saint-Étienne':                    'AS Saint-Étienne',

    # Netherlands
    'AZ':                               'AZ Alkmaar',
    'ADO Den Haag':                     'ADO Den Haag',
    'Ajax':                             'AFC Ajax',
    'Heerenveen':                       'sc Heerenveen',
    'Roda JC':                          'Roda JC',
    'Twente':                           'FC Twente',
    'Vitesse':                          'Vitesse Arnhem',
    'Willem II':                        'Willem II',

    # Belgium / Switzerland / Austria / Portugal
    'Anderlecht':                       'RSC Anderlecht',
    'Antwerp':                          'Royal Antwerp FC',
    'Brugge':                           'Club Brugge',
    'Genk':                             'KRC Genk',
    'Gent':                             'KAA Gent',
    'Standard Liège':                   'Standard Liège',
    'Beveren':                          'KSK Beveren',
    'Lokeren':                          'KSC Lokeren',
    'Mechelen':                         'KV Mechelen',
    'Mouscron':                         'Excelsior Mouscron',
    'Westerlo':                         'KVC Westerlo',
    'Charleroi':                        'Sporting Charleroi',
    'Zulte Waregem':                    'SV Zulte Waregem',
    'Lierse':                           'Lierse SK',
    'Young Boys':                       'BSC Young Boys',
    'Basel':                            'FC Basel',
    'Sion':                             'FC Sion',
    'Zürich':                           'FC Zürich',
    'Servette':                         'Servette FC',
    'St. Gallen':                       'FC St. Gallen',
    'Thun':                             'FC Thun',
    'Luzern':                           'FC Luzern',
    'Grasshopper':                      'Grasshopper Club Zürich',
    'Sturm Graz':                       'SK Sturm Graz',
    'Austria Wien':                     'Austria Wien',
    'Rapid Wien':                       'SK Rapid Wien',
    'Red Bull Salzburg':                'FC Red Bull Salzburg',
    'Salzburg':                         'FC Red Bull Salzburg',
    'LASK':                             'LASK',
    'Mattersburg':                      'SV Mattersburg',
    'Ried':                             'SV Ried',
    'Pasching':                         'SV Pasching',
    'Wolfsberger AC':                   'Wolfsberger AC',
    'Benfica':                          'Sport Lisboa e Benfica',
    'Porto':                            'FC Porto',
    'Sporting':                         'Sporting Clube de Portugal',
    'Sporting CP':                      'Sporting Clube de Portugal',
    'Braga':                            'Sporting Clube de Braga',
    'Vitória de Guimarães':             'Vitória SC',
    'Vitória SC':                       'Vitória SC',
    'Marítimo':                         'CS Marítimo',
    'Nacional':                         'CD Nacional',
    'Boavista':                         'Boavista FC',
    'Belenenses':                       'CF Os Belenenses',
    'Paços de Ferreira':                'FC Paços de Ferreira',
    'Académica':                        'Académica de Coimbra',
    'Penafiel':                         'FC Penafiel',
    'Estrela da Amadora':               'CF Estrela da Amadora',
    'Beira-Mar':                        'SC Beira-Mar',
    'Leiria':                           'União de Leiria',

    # Greece
    'AEK Athens':                       'AEK Athens',
    'Aris':                             'Aris Thessaloniki',
    'Panathinaikos':                    'Panathinaikos FC',
    'PAOK':                             'PAOK',
    'Olympiakos':                       'Olympiakos Piraeus',
    'Panionios':                        'Panionios',
    'Atromitos':                        'Atromitos',
    'Asteras Tripolis':                 'Asteras Tripoli',
    'Skoda Xanthi':                     'Xanthi FC',
    'Xanthi':                           'Xanthi FC',
    'OFI':                              'OFI Crete',
    'Iraklis':                          'Iraklis Thessaloniki',
    'Levadiakos':                       'Levadiakos',
    'Kerkyra':                          'PAE Kerkyra',
    'Veria':                            'PAE Veria',
    'Larissa':                          'AEL Larissa',
    'AEL Larissa':                      'AEL Larissa',
    'Volos':                            'Volos NFC',
    'Apollon Smyrnis':                  'Apollon Smyrnis',

    # Turkey
    'Fenerbahçe':                       'Fenerbahce',
    'Galatasaray':                      'Galatasaray SK',
    'Beşiktaş':                         'Besiktas',
    'Trabzonspor':                      'Trabzonspor',
    'Konyaspor':                        'Konyaspor',
    'İstanbul Başakşehir':              'Istanbul Başakşehir',
    'Bursaspor':                        'Bursaspor',
    'Gaziantepspor':                    'Gaziantepspor',
    'Sivasspor':                        'Sivasspor',
    'Akhisarspor':                      'Akhisarspor',
    'Eskişehirspor':                    'Eskişehirspor',
    'Gençlerbirliği':                   'Gençlerbirliği',
    'Kayserispor':                      'Kayserispor',
    'Antalyaspor':                      'Antalyaspor',
    'Alanyaspor':                       'Alanyaspor',

    # Russia / Ukraine / Eastern Europe
    'CSKA Moscow':                      'PFC CSKA Moskva',
    'Spartak Moscow':                   'FC Spartak Moscow',
    'Lokomotiv Moscow':                 'FC Lokomotiv Moscow',
    'Dynamo Moscow':                    'FC Dynamo Moscow',
    'Zenit Saint Petersburg':           'FC Zenit Saint Petersburg',
    'Rubin Kazan':                      'FC Rubin Kazan',
    'Krasnodar':                        'FC Krasnodar',
    'Anzhi Makhachkala':                'FC Anzhi Makhachkala',
    'Rostov':                           'FC Rostov',
    'Terek Grozny':                     'FC Akhmat Grozny',
    'Akhmat Grozny':                    'FC Akhmat Grozny',
    'FC Moscow':                        'FC Moscow',
    'Amkar Perm':                       'Amkar Perm',
    'Tom Tomsk':                        'FC Tom Tomsk',
    'Saturn':                           'FC Saturn Ramenskoye',
    'Kuban Krasnodar':                  'Kuban Krasnodar',
    'Spartak Nalchik':                  'Spartak Nalchik',
    'Krylia Sovetov':                   'Krylia Sovetov Samara',
    'Krylia Sovetov Samara':            'Krylia Sovetov Samara',
    'Dynamo Kyiv':                      'FC Dynamo Kyiv',
    'Shakhtar Donetsk':                 'FK Shakhtar Donetsk',
    'Metalist Kharkiv':                 'Metalist Kharkiv',
    'Dnipro Dnipropetrovsk':            'FC Dnipro Dnipropetrovsk',
    'Dnipro':                           'FC Dnipro Dnipropetrovsk',
    'Chornomorets Odesa':               'Chornomorets Odessa',
    'Karpaty Lviv':                     'Karpaty Lviv',
    'Vorskla Poltava':                  'Vorskla Poltava',
    'Zorya Luhansk':                    'Zorya Lugansk',
    'Olimpik Donetsk':                  'Olimpik Donetsk',
    'Oleksandriya':                     'FC Oleksandriya',
    'Mariupol':                         'FC Mariupol',
    'Tavriya Simferopol':               'Tavriya Simferopol',
    'Metalurh Donetsk':                 'Metalurh Donetsk',
    'Metalurh Zaporizhzhia':            'Metalurh Zaporizhzhia',

    # Czech Republic / Slovakia
    'Sparta Prague':                    'AC Sparta Praha',
    'Slavia Prague':                    'SK Slavia Praha',
    'Viktoria Plzeň':                   'FC Viktoria Plzeň',
    'Plzeň':                            'FC Viktoria Plzeň',
    'Mladá Boleslav':                   'FK Mladá Boleslav',
    'Jablonec':                         'FK Jablonec',
    'Liberec':                          'FC Slovan Liberec',
    'Slovan Liberec':                   'FC Slovan Liberec',
    'Teplice':                          'FK Teplice',
    'Sigma Olomouc':                    'SK Sigma Olomouc',
    'Olomouc':                          'SK Sigma Olomouc',
    'Slovan Bratislava':                'ŠK Slovan Bratislava',
    'Trnava':                           'Spartak Trnava',
    'Spartak Trnava':                   'Spartak Trnava',
    'Žilina':                           'MŠK Žilina',
    'MŠK Žilina':                       'MŠK Žilina',
    'Ružomberok':                       'MFK Ružomberok',

    # Scandinavia
    'Copenhagen':                       'FC København',
    'Brøndby':                          'Brøndby IF',
    'Midtjylland':                      'FC Midtjylland',
    'Nordsjælland':                     'FC Nordsjælland',
    'Aalborg':                          'AaB',
    'AaB':                              'AaB',
    'Esbjerg':                          'Esbjerg fB',
    'OB':                               'Odense BK',
    'Lyngby':                           'Lyngby BK',
    'Randers':                          'Randers FC',
    'Sønderjyske':                      'SønderjyskE',
    'Silkeborg':                        'Silkeborg IF',
    'Horsens':                          'AC Horsens',
    'Rosenborg':                        'Rosenborg BK',
    'Molde':                            'Molde FK',
    'Brann':                            'SK Brann',
    'Vålerenga':                        'Vålerenga IF',
    'Lillestrøm':                       'Lillestrøm SK',
    'Stabæk':                           'Stabæk',
    'Tromsø':                           'Tromsø IL',
    'Viking':                           'Viking FK',
    'Bodø/Glimt':                       'FK Bodø/Glimt',
    'Strømsgodset':                     'Strømsgodset',
    'Odd':                              'Odds BK',
    'Odds BK':                          'Odds BK',
    'Aalesund':                         'Aalesunds FK',
    'Sarpsborg 08':                     'Sarpsborg 08 FF',
    'Häcken':                           'BK Häcken',
    'BK Häcken':                        'BK Häcken',
    'AIK':                              'AIK',
    'IFK Göteborg':                     'IFK Goteborg',
    'IFK Norrköping':                   'IFK Norrköping',
    'Malmö FF':                         'Malmo FF',
    'Djurgården':                       'Djurgardens IF',
    'Djurgårdens IF':                   'Djurgardens IF',
    'Hammarby':                         'Hammarby',
    'Helsingborg':                      'Helsingborgs IF',
    'Helsingborgs IF':                  'Helsingborgs IF',
    'Kalmar FF':                        'Kalmar FF',
    'Halmstads BK':                     'Halmstads BK',
    'Östers IF':                        'Östers IF',
    'Elfsborg':                         'IF Elfsborg',
    'IF Elfsborg':                      'IF Elfsborg',
    'GAIS':                             'GAIS',
    'Gefle':                            'Gefle IF',
    'AGF':                              'AGF',
    'HJK':                              'HJK Helsinki',
    'TPS':                              'TPS Turku',
    'Tampere United':                   'Tampere United',
    'Inter Turku':                      'Inter Turku',
    'KuPS':                             'KuPS Kuopio',
    'Honka':                            'FC Honka',
    'FC Lahti':                         'FC Lahti',
    'Haka':                             'Haka Valkeakoski',
    'MyPa':                             'MYPA',
    'MYPA':                             'MYPA',

    # Balkans
    'Red Star Belgrade':                'FK Crvena Zvezda',
    'Crvena zvezda':                    'FK Crvena Zvezda',
    'Partizan':                         'FK Partizan',
    'Vojvodina':                        'FK Vojvodina',
    'Čukarički':                        'FK Čukarički',
    'Spartak Subotica':                 'Spartak Subotica',
    'OFK Beograd':                      'OFK Beograd',
    'Sarajevo':                         'FK Sarajevo',
    'Željezničar':                      'Zeljeznicar Sarajevo',
    'Široki Brijeg':                    'NK Siroki Brijeg',
    'Zrinjski Mostar':                  'HSK Zrinjski Mostar',
    'Borac Banja Luka':                 'Borac Banja Luka',
    'Dinamo Zagreb':                    'GNK Dinamo Zagreb',
    'Hajduk Split':                     'Hajduk Split',
    'Rijeka':                           'HNK Rijeka',
    'Lokomotiva Zagreb':                'Lokomotiva Zagreb',
    'Slaven Belupo':                    'NK Slaven Belupo',
    'Osijek':                           'NK Osijek',
    'Maribor':                          'NK Maribor',
    'Olimpija Ljubljana':               'NK Olimpija Ljubljana',
    'Domžale':                          'NK Domžale',
    'Celje':                            'NK Celje',
    'Koper':                            'FC Koper',
    'Mura':                             'NŠ Mura',

    # Romania / Bulgaria / Hungary / Poland
    'Steaua București':                 'FCSB',
    'FCSB':                             'FCSB',
    'Dinamo București':                 'FC Dinamo București',
    'Rapid București':                  'FC Rapid București',
    'Astra Giurgiu':                    'Astra Giurgiu',
    'CFR Cluj':                         'CFR Cluj',
    'Vaslui':                           'FC Vaslui',
    'Universitatea Craiova':            'Universitatea Craiova',
    'Politehnica Timișoara':            'FC Politehnica Timisoara',
    'Levski Sofia':                     'Levski Sofia',
    'CSKA Sofia':                       'CSKA Sofia',
    'Slavia Sofia':                     'PFC Slavia Sofia',
    'Lokomotiv Plovdiv':                'PFC Lokomotiv Plovdiv',
    'Lokomotiv Sofia':                  'Lokomotiv Sofia',
    'Litex Lovech':                     'PFC Litex Lovech',
    'Ludogorets Razgrad':               'PFC Ludogorets Razgrad',
    'Ludogorets':                       'PFC Ludogorets Razgrad',
    'Botev Plovdiv':                    'Botev Plovdiv',
    'Cherno More':                      'Cherno More Varna',
    'Beroe Stara Zagora':               'Beroe Stara Zagora',
    'Slavia Sofia':                     'PFC Slavia Sofia',
    'Debrecen':                         'Debreceni VSC',
    'Debreceni VSC':                    'Debreceni VSC',
    'Ferencváros':                      'Ferencvaros',
    'Videoton':                         'Videoton FC',
    'Honvéd':                           'Budapest Honvéd',
    'Újpest':                           'Újpest FC',
    'Diósgyőr':                         'Diósgyőri VTK',
    'MTK Budapest':                     'MTK',
    'Wisła Kraków':                     'Wisla Krakow',
    'Lech Poznań':                      'Lech Poznan',
    'Legia Warsaw':                     'Legia Warsaw',
    'Polonia Warsaw':                   'Polonia Warszawa',
    'Polonia Warszawa':                 'Polonia Warszawa',
    'Śląsk Wrocław':                    'Śląsk Wrocław',
    'Lechia Gdańsk':                    'Lechia Gdańsk',
    'Pogoń Szczecin':                   'Pogoń Szczecin',
    'Cracovia':                         'KS Cracovia',
    'Ruch Chorzów':                     'Ruch Chorzów',
    'Górnik Zabrze':                    'Górnik Zabrze',
    'Zagłębie Lubin':                   'Zaglebie Lubin',
    'Bełchatów':                        'GKS Bełchatów',

    # Cyprus / Israel / Other
    'APOEL':                            'APOEL Nikosia',
    'Anorthosis':                       'Anorthosis Famagusta',
    'AEK Larnaca':                      'AEK Larnaca',
    'Apollon Limassol':                 'Apollon Limassol',
    'AEL Limassol':                     'AEL Limassol',
    'Omonia':                           'Omonia Nicosia',
    'Omonia Nicosia':                   'Omonia Nicosia',
    'Maccabi Tel Aviv':                 'Maccabi Tel Aviv',
    'Maccabi Haifa':                    'Maccabi Haifa',
    'Hapoel Tel Aviv':                  'Hapoel Tel Aviv',
    'Hapoel Be\'er Sheva':              'Hapoel Be\'er Sheva',
    'Bnei Sakhnin':                     'Bnei Sakhnin',
    'Beitar Jerusalem':                 'Beitar Jerusalem',
    'Hapoel Haifa':                     'Hapoel Haifa',

    # Various
    'Vaduz':                            'FC Vaduz',
    'F91 Dudelange':                    'F91 Dudelange',
    'Differdange 03':                   'FC Differdange 03',
    'Jeunesse Esch':                    'Jeunesse Esch',
    'Fola Esch':                        'CS Fola Esch',
    'Drita':                            'KF Drita',
    'Tirana':                           'KF Tirana',
    'Skënderbeu':                       'KF Skënderbeu',
    'Vllaznia':                         'FK Vllaznia',
    'Partizani':                        'FK Partizani Tirana',
    'Teuta':                            'KS Teuta',
    'Kukësi':                           'FK Kukësi',
    'Laçi':                             'KF Laçi',
    'Sileks':                           'FK Sileks',
    'Vardar':                           'Vardar Skopje',
    'Sloga Jugomagnat':                 'Sloga Jugomagnat Skopje',
    'Pelister':                         'FK Pelister',
    'Rabotnički':                       'Rabotnicki',
    'Shkëndija':                        'KF Shkëndija',
    'Renova':                           'Renova',
    'Alashkert':                        'FC Alashkert',
    'Ararat-Armenia':                   'FC Ararat-Armenia',
    'Pyunik':                           'Pyunik',
    'Mika':                             'FC Mika',
    'Banants':                          'FC Banants',
    'Ulisses':                          'Ulisses FC',
    'Shirak':                           'Shirak SC',
    'Astana':                           'FK Astana',
    'Kairat':                           'Kairat',
    'Aktobe':                           'Aktobe',
    'Tobol':                            'FC Tobol',
    'Ordabasy':                         'FC Ordabasy',
    'Atyrau':                           'FC Atyrau',
    'Irtysh':                           'Irtysh Pavlodar',
    'Shakhter Karagandy':               'FC Shakhter Karagandy',
    'Qarabağ':                          'Qarabağ FK',
    'Gabala':                           'FK Gabala',
    'Inter Baku':                       'Inter Baku PIK',
    'Khazar Lankaran':                  'Khazar Lankaran',
    'Olimpik Baku':                     'Olimpik Baku',
    'AZAL Baku':                        'AZAL Baku',
    'Baku':                             'Baku',
    'Neftchi Baku':                     'Neftchi PFC Baku',
    'Dinamo Tbilisi':                   'Dinamo Tbilisi',
    'Torpedo Kutaisi':                  'Torpedo Kutaisi',
    'Sioni Bolnisi':                    'Sioni Bolnisi',
    'WIT Georgia':                      'WIT Georgia',
    'Locomotive Tbilisi':               'Locomotive Tbilisi',
    'Ameri Tbilisi':                    'Ameri Tbilisi',
    'Saburtalo Tbilisi':                'Saburtalo Tbilisi',
    'BATE Borisov':                     'BATE Borisov',
    'Dinamo Minsk':                     'Dinamo Minsk',
    'Shakhtyor Soligorsk':              'Shakhtyor Soligorsk',
    'Gomel':                            'Gomel',
    'Belshina Bobruisk':                'Belshina Bobruisk',
    'Naftan Novopolotsk':               'Naftan Novopolotsk',
    'Slavia Mozyr':                     'FC Slavia Mozyr',
    'MTZ-RIPO Minsk':                   'MTZ-RIPO Minsk',
    'BFC Daugava':                      'BFC Daugava',
    'Skonto':                           'Skonto',
    'Ventspils':                        'Ventspils',
    'Liepājas Metalurgs':               'Liepajas Metalurgs',
    'Liepāja':                          'FK Liepāja',
    'RFS':                              'FK RFS',
    'Riga':                             'Riga FC',
    'FBK Kaunas':                       'FBK Kaunas',
    'Ekranas':                          'FK Ekranas',
    'Atlantas':                         'Atlantas',
    'Sūduva':                           'FK Sūduva',
    'Kauno Žalgiris':                   'Kauno Žalgiris',
    'Žalgiris':                         'FK Žalgiris',
    'Sheriff Tiraspol':                 'Sheriff Tiraspol',
    'Zimbru Chișinău':                  'Zimbru Chisinau',
    'Tiraspol':                         'FC Tiraspol',
    'Dacia Chișinău':                   'Dacia Chișinău',
    'Milsami':                          'Milsami Orhei',
    'Petrocub':                         'CS Petrocub',
    'Rapid Ghidighici':                 'Rapid Ghidighici',
    'Sant Julià':                       'UE Sant Julia',
    'Santa Coloma':                     'FC Santa Coloma',
    'UE Santa Coloma':                  'UE Santa Coloma',
    'Lusitanos':                        'FC Lusitanos',
    'Inter Club d\'Escaldes':           'Inter Club d\'Escaldes',
    'Engordany':                        'CE Engordany',
    'KÍ':                               'KI Klaksvik',
    'KI Klaksvik':                      'KI Klaksvik',
    'KÍ Klaksvík':                      'KI Klaksvik',
    'B36':                              'B36 Torshavn',
    'B68':                              'B68 Toftir',
    'EB/Streymur':                      'EB / Streymur',
    'NSÍ Runavík':                      'NSI',
    'NSÍ':                              'NSI',
    'HB':                               'HB Torshavn',
    'HB Torshavn':                      'HB Torshavn',
    'Víkingur':                         'Vikingur Reykjavik',
    'Víkingur Reykjavík':               'Vikingur Reykjavik',
    'KR':                               'KR Reykjavik',
    'Valur':                            'Valur Reykjavik',
    'FH':                               'FH',
    'IBV':                              'IB Vestmannaeyja',
    'ÍBV':                              'IB Vestmannaeyja',
    'Stjarnan':                         'Stjarnan',
    'Breiðablik':                       'Breiðablik',
    'IA':                               'IA Akranes',
    'IA Akranes':                       'IA Akranes',
    'Linfield':                         'Linfield',
    'Glentoran':                        'Glentoran',
    'Cliftonville':                     'Cliftonville',
    'Crusaders':                        'Crusaders',
    'Coleraine':                        'Coleraine',
    'Portadown':                        'Portadown',
    'Larne':                            'Larne',
    'Ballymena United':                 'Ballymena United',
    'Cork City':                        'Cork City',
    'Shamrock Rovers':                  'Shamrock Rovers',
    'Bohemians':                        'Bohemians',
    'Bohemian':                         'Bohemians',
    'Derry City':                       'Derry City',
    'St Patrick\'s Athletic':           'St Patrick\'s Athletic',
    'Drogheda United':                  'Drogheda United',
    'Shelbourne':                       'Shelbourne',
    'TNS':                              'The New Saints',
    'The New Saints':                   'The New Saints',
    'Bala Town':                        'Bala Town',
    'Connah\'s Quay Nomads':            'Connah\'s Quay Nomads',
    'Bangor City':                      'Bangor City',
    'Llanelli':                         'Llanelli',
    'Rhyl':                             'Rhyl',
    'Marsaxlokk':                       'Marsaxlokk',
    'Birkirkara':                       'Birkirkara',
    'Hibernians':                       'Hibernians',
    'Floriana':                         'Floriana',
    'Sliema Wanderers':                 'Sliema Wanderers',
    'Valletta':                         'Valletta',
    'Balzan':                           'Balzan',
    'Gżira United':                     'Gżira United',
    'Tre Fiori':                        'Tre Fiori',
    'Tre Penne':                        'SP Tre Penne',
    'La Fiorita':                       'La Fiorita',
    'Folgore':                          'SS Folgore/Falciano',
    'Murata':                           'Murata',
    'Domagnano':                        'SS Domagnano',
    'Pennarossa':                       'SP Pennarossa',
    'Cosmos':                           'SS Cosmos',
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
# STEP 1b - LOAD HISTORICAL DOMESTIC + CL FROM ENGSOCCERDATA
# ============================================================
# jalapic/engsoccerdata provides clean CSVs for the 5 domestic leagues
# (1888-2024 for England, similar coverage for others) and Champions League /
# European Cup (1955-2017). We use it for the 1992-93 -> 2010-11 backfill
# window only — the existing fdco + openfootball pipeline takes over from
# 2011-12 onward.
#
# Season convention: row Season=N denotes the N -> N+1 campaign.
# Bundesliga file mixes tier 1 + tier 2; we filter to tier=1 only.
# The "FT" column has form "H-A" (e.g., "2-1"); hgoal/vgoal already split.

# Module-level cache so each CSV downloads once across all seasons.
_engsoccerdata_cache = {}

def _load_engsoccerdata_csv(filename):
    if filename in _engsoccerdata_cache:
        return _engsoccerdata_cache[filename]
    url = f'{ENGSOCCERDATA_BASE}/{filename}'
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df_csv = pd.read_csv(io.StringIO(r.text), low_memory=False)
        _engsoccerdata_cache[filename] = df_csv
        return df_csv
    except Exception as e:
        print(f"  Warning: could not load engsoccerdata {filename}: {e}")
        _engsoccerdata_cache[filename] = pd.DataFrame()
        return _engsoccerdata_cache[filename]


def load_domestic_engsoccerdata(season):
    """Load all 5 domestic leagues for the given season string (e.g., '1992-93')
    from engsoccerdata CSVs. Returns dict-list matching load_domestic_fdco schema."""
    season_year = int(season[:4])
    rows = []
    for league, filename in ENGSOCCERDATA_FILES.items():
        df_csv = _load_engsoccerdata_csv(filename)
        if df_csv.empty:
            continue
        sdf = df_csv[df_csv['Season'] == season_year].copy()
        if 'tier' in sdf.columns:
            sdf = sdf[sdf['tier'] == 1]
        if sdf.empty:
            print(f"  Warning: engsoccerdata {league} {season} has no rows")
            continue
        sdf = sdf.dropna(subset=['Date', 'home', 'visitor', 'hgoal', 'vgoal'])
        for _, m in sdf.iterrows():
            try:
                parsed_date = pd.to_datetime(m['Date'])
            except Exception:
                continue
            rows.append({
                'date':            parsed_date.strftime('%Y-%m-%d'),
                'home_team':       normalize_team(str(m['home']).strip()),
                'away_team':       normalize_team(str(m['visitor']).strip()),
                'home_score':      int(m['hgoal']),
                'away_score':      int(m['vgoal']),
                'competition':     league,
                'neutral':         False,
                'shootout_winner': None,
            })
    return rows


def load_champs_engsoccerdata(season):
    """Load Champions League / European Cup for the given season from
    engsoccerdata's champs.csv. Sets comp_season so the CL champion logic
    in step 10c works the same way as the openfootball-derived rows."""
    season_year = int(season[:4])
    df_csv = _load_engsoccerdata_csv('champs.csv')
    if df_csv.empty:
        return []
    sdf = df_csv[df_csv['Season'] == season_year].copy()
    if sdf.empty:
        return []
    sdf = sdf.dropna(subset=['Date', 'home', 'visitor', 'hgoal', 'vgoal'])
    rows = []
    for _, m in sdf.iterrows():
        try:
            parsed_date = pd.to_datetime(m['Date'])
        except Exception:
            continue
        # pens column: e.g. "5-4" if a shootout was played, NaN otherwise.
        # When present, the AET score is in hgoal/vgoal and we encode the
        # shootout winner so step 5 can override the margin to ±0.5.
        shootout_winner = None
        pens_val = m.get('pens')
        if pd.notna(pens_val) and isinstance(pens_val, str) and '-' in pens_val:
            try:
                ph, pa = (int(x) for x in pens_val.split('-'))
                home_norm = normalize_team(str(m['home']).strip())
                away_norm = normalize_team(str(m['visitor']).strip())
                shootout_winner = home_norm if ph > pa else away_norm
            except Exception:
                shootout_winner = None
        rows.append({
            'date':            parsed_date.strftime('%Y-%m-%d'),
            'home_team':       normalize_team(str(m['home']).strip()),
            'away_team':       normalize_team(str(m['visitor']).strip()),
            'home_score':      int(m['hgoal']),
            'away_score':      int(m['vgoal']),
            'competition':     'Champions League',
            'comp_season':     season,
            'neutral':         False,
            'shootout_winner': shootout_winner,
        })
    return rows


def load_facup_engsoccerdata(season):
    """Load FA Cup matches for the given season from engsoccerdata facup.csv.
    Coverage: 1871-2018 (138 seasons). Used for FA Cup history up to 2017-18;
    openfootball/england facup.txt takes over from 2018-19 onward.
    Schema: Date, Season, home, visitor, FT, hgoal, vgoal, round, tie, aet, pen, pens, hp, vp, ..."""
    season_year = int(season[:4])
    df_csv = _load_engsoccerdata_csv('facup.csv')
    if df_csv.empty:
        return []
    sdf = df_csv[df_csv['Season'] == season_year].copy()
    if sdf.empty:
        return []
    sdf = sdf.dropna(subset=['Date', 'home', 'visitor', 'hgoal', 'vgoal'])
    rows = []
    for _, m in sdf.iterrows():
        try:
            parsed_date = pd.to_datetime(m['Date'])
        except Exception:
            continue
        # facup.csv stores penalty shootout info in 'pens' column as "h-a" string
        # (e.g., "5-4") with hp/vp as integer counts. Use pens when hgoal == vgoal.
        shootout_winner = None
        if int(m['hgoal']) == int(m['vgoal']):
            pens_val = m.get('pens')
            if pd.notna(pens_val) and isinstance(pens_val, str) and '-' in pens_val:
                try:
                    ph, pa = (int(x) for x in pens_val.split('-'))
                    home_norm = normalize_team(str(m['home']).strip())
                    away_norm = normalize_team(str(m['visitor']).strip())
                    shootout_winner = home_norm if ph > pa else away_norm
                except Exception:
                    pass
        rows.append({
            'date':            parsed_date.strftime('%Y-%m-%d'),
            'home_team':       normalize_team(str(m['home']).strip()),
            'away_team':       normalize_team(str(m['visitor']).strip()),
            'home_score':      int(m['hgoal']),
            'away_score':      int(m['vgoal']),
            'competition':     'FA Cup',
            'comp_season':     season,
            'neutral':         False,
            'shootout_winner': shootout_winner,
        })
    return rows


# ============================================================
# STEP 1d - LOAD HISTORICAL DOMESTIC CUPS
# ============================================================
# Phase 3b backfill: scrape_domestic_cups.py produces a static CSV of
# DFB-Pokal / Copa del Rey / Coppa Italia matches for the historical era
# before openfootball took over. Coverage is uneven by year — older
# Wikipedia pages use a non-fevent format we can't fully parse, but the
# finals are always captured (which is what champion / treble detection
# needs). Where we have full rounds, those add rating signal too.

_cups_historical_df = None

def _load_cups_historical():
    global _cups_historical_df
    if _cups_historical_df is not None:
        return _cups_historical_df
    try:
        _cups_historical_df = pd.read_csv('cups_historical.csv')
    except FileNotFoundError:
        print("  Warning: cups_historical.csv not found — skipping cup backfill")
        _cups_historical_df = pd.DataFrame()
    return _cups_historical_df


def load_cup_historical(season, cup):
    df_csv = _load_cups_historical()
    if df_csv.empty:
        return []
    sdf = df_csv[(df_csv['comp_season'] == season) & (df_csv['cup'] == cup)].copy()
    if sdf.empty:
        return []
    sdf = sdf.dropna(subset=['date', 'home', 'visitor', 'hgoal', 'vgoal'])
    rows = []
    for _, m in sdf.iterrows():
        try:
            parsed_date = pd.to_datetime(m['date'])
        except Exception:
            continue
        sw_raw = m.get('shootout_winner')
        if pd.notna(sw_raw) and isinstance(sw_raw, str) and sw_raw.strip():
            shootout_winner = normalize_team(sw_raw.strip())
        else:
            shootout_winner = None
        rows.append({
            'date':            parsed_date.strftime('%Y-%m-%d'),
            'home_team':       normalize_team(str(m['home']).strip()),
            'away_team':       normalize_team(str(m['visitor']).strip()),
            'home_score':      int(m['hgoal']),
            'away_score':      int(m['vgoal']),
            'competition':     cup,
            'comp_season':     season,
            'neutral':         False,
            'shootout_winner': shootout_winner,
        })
    return rows


# ============================================================
# STEP 1c - LOAD HISTORICAL UEFA CUP / EUROPA LEAGUE
# ============================================================
# Phase 2 backfill: scrape_el_uefacup.py produces a static CSV of UEFA Cup
# and Europa League matches from 2004-05 through 2019-20 (sourced from
# Wikipedia season sub-pages). The 2020-21+ era is already covered by
# openfootball/champions-league el.txt files via parse_european_txt.
#
# The CSV schema mirrors champs.csv: date, home, visitor, FT, hgoal, vgoal,
# comp_season, shootout_winner. We label everything as 'Europa League' even
# for the pre-2009 UEFA Cup era — they're a single continuous competition
# lineage and the front-end only knows EL/CL/Conference League.

_uefacup_historical_df = None

def _load_uefacup_historical():
    global _uefacup_historical_df
    if _uefacup_historical_df is not None:
        return _uefacup_historical_df
    try:
        _uefacup_historical_df = pd.read_csv('uefacup_historical.csv')
    except FileNotFoundError:
        print("  Warning: uefacup_historical.csv not found — skipping UEFA Cup backfill")
        _uefacup_historical_df = pd.DataFrame()
    return _uefacup_historical_df


def load_uefacup_historical(season):
    """Return UEFA Cup / Europa League matches for the given season string
    (e.g., '2005-06') as dict-list matching the schema used elsewhere."""
    df_csv = _load_uefacup_historical()
    if df_csv.empty:
        return []
    sdf = df_csv[df_csv['comp_season'] == season].copy()
    if sdf.empty:
        return []
    sdf = sdf.dropna(subset=['date', 'home', 'visitor', 'hgoal', 'vgoal'])
    rows = []
    for _, m in sdf.iterrows():
        try:
            parsed_date = pd.to_datetime(m['date'])
        except Exception:
            continue
        sw_raw = m.get('shootout_winner')
        if pd.notna(sw_raw) and isinstance(sw_raw, str) and sw_raw.strip():
            shootout_winner = normalize_team(sw_raw.strip())
        else:
            shootout_winner = None
        rows.append({
            'date':            parsed_date.strftime('%Y-%m-%d'),
            'home_team':       normalize_team(str(m['home']).strip()),
            'away_team':       normalize_team(str(m['visitor']).strip()),
            'home_score':      int(m['hgoal']),
            'away_score':      int(m['vgoal']),
            'competition':     'Europa League',
            'comp_season':     season,
            'neutral':         False,
            'shootout_winner': shootout_winner,
        })
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

# Date line: optional day-of-week, then Mon/DD, optional year.
# Some openfootball country-cup files wrap the date in square brackets like
# "[Tue Sep/22]" — the regex tolerates either form.
_RE_DATE = re.compile(
    r'^\s*\[?\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Za-z]+)/(\d{1,2})(?:\s+(\d{4}))?\s*\]?'
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

# Some openfootball country-cup files (notably Coppa Italia 2020-21 → 2023-24)
# use an older format that omits the " v " separator and writes the score
# inline between the home and away team:
#   "  20.30  Como 1907                3-4 pen. 2-2 a.e.t. (2-2, 1-0)  US Catanzaro"
#   "  18.30  Ternana Calcio           4-3 pen. 1-1 a.e.t. (1-1, 0-1)  US Avellino"
#   "  19.00  AC Perugia Calcio        1-0 (0-0)  FC Südtirol"
# The split between team A and the score (and between the trailing
# parenthetical and team B) is two-or-more spaces.
_RE_OLDCUP_PEN = re.compile(
    r'^\s+(?:\d{1,2}\.\d{2}\s+)?(.+?)\s{2,}'
    r'(\d+)-(\d+)\s+pen\.\s+(\d+)-(\d+)\s+a\.e\.t\.'
    r'(?:\s+\([\d\-,\s]+\))?'
    r'\s{2,}(.+?)\s*$'
)
_RE_OLDCUP_AET = re.compile(
    r'^\s+(?:\d{1,2}\.\d{2}\s+)?(.+?)\s{2,}'
    r'(\d+)-(\d+)\s+a\.e\.t\.'
    r'(?:\s+\([\d\-,\s]+\))?'
    r'\s{2,}(.+?)\s*$'
)
_RE_OLDCUP_MATCH = re.compile(
    r'^\s+(?:\d{1,2}\.\d{2}\s+)?(.+?)\s{2,}'
    r'(\d+)-(\d+)'
    r'(?:\s+\([\d\-,\s]+\))?'
    r'\s{2,}(.+?)\s*$'
)

def parse_european_txt(season, competition, filename, repo_base=None):
    """Parse a football.TXT file from openfootball. By default reads from
    the champions-league repo; pass repo_base to read from a country repo
    (e.g., openfootball/deutschland for DFB-Pokal cup.txt files).

    Some country-cup files omit the year on the date line (older Coppa Italia
    format uses just "[Tue Sep/22]"). When that happens we seed current_year
    from the season string and flip it forward when we see a month wrap
    (Dec → Jan)."""
    base = repo_base if repo_base else CHAMPIONS_LEAGUE_BASE
    url = f'{base}/{season}/{filename}'
    rows = []
    try:
        text = requests.get(url, timeout=10).text
    except Exception as e:
        print(f"  Warning: could not load {competition} {season}: {e}")
        return rows

    current_date = None
    current_year = None
    last_month = None

    # Seed current_year from the season string so files that never list a
    # year (e.g., "[Tue Sep/22]") can still produce dates. The season starts
    # in autumn of the start year and ends in spring of the end year.
    try:
        current_year = int(season[:4])
    except Exception:
        pass

    for line in text.splitlines():
        # Skip headers and blank lines
        if not line.strip() or line.strip().startswith(('=', '#', '»')):
            continue

        # Date line
        dm = _RE_DATE.match(line)
        if dm:
            month_str, day_str, year_str = dm.group(1), dm.group(2), dm.group(3)
            month_num = _MONTH_MAP.get(month_str)
            if year_str:
                # Explicit year on the line — trust it, don't apply month-wrap.
                current_year = int(year_str)
            elif month_num and current_year and last_month is not None:
                # No explicit year. Detect month wrap (Dec -> Jan) and bump.
                if month_num < last_month and (last_month - month_num) >= 6:
                    current_year += 1
            if month_num and current_year:
                current_date = f"{current_year}-{month_num:02d}-{int(day_str):02d}"
                last_month = month_num
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
            continue

        # ---- Old country-cup format fallbacks (Coppa Italia 2020-23 etc.) ----
        ocp = _RE_OLDCUP_PEN.match(line)
        if ocp:
            team1 = normalize_team(ocp.group(1))
            team2 = normalize_team(ocp.group(6))
            pen_h, pen_a   = int(ocp.group(2)), int(ocp.group(3))
            match_h, match_a = int(ocp.group(4)), int(ocp.group(5))
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

        oca = _RE_OLDCUP_AET.match(line)
        if oca:
            rows.append({
                'date':            current_date,
                'home_team':       normalize_team(oca.group(1)),
                'away_team':       normalize_team(oca.group(4)),
                'home_score':      int(oca.group(2)),
                'away_score':      int(oca.group(3)),
                'competition':     competition,
                'comp_season':     season,
                'neutral':         False,
                'shootout_winner': None,
            })
            continue

        omm = _RE_OLDCUP_MATCH.match(line)
        if omm:
            rows.append({
                'date':            current_date,
                'home_team':       normalize_team(omm.group(1)),
                'away_team':       normalize_team(omm.group(4)),
                'home_score':      int(omm.group(2)),
                'away_score':      int(omm.group(3)),
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
    season_year = int(season[:4])
    if season_year <= LAST_LEGACY_SEASON_YEAR:
        # 1992-93 -> 2010-11: engsoccerdata for both domestic and Champions League.
        all_rows.extend(load_domestic_engsoccerdata(season))
        all_rows.extend(load_champs_engsoccerdata(season))
    else:
        # 2011-12 onward: original pipeline.
        all_rows.extend(load_domestic_fdco(season))
        all_rows.extend(parse_european_txt(season, 'Champions League', 'cl.txt'))
        all_rows.extend(parse_european_txt(season, 'Europa League',    'el.txt'))
        # Conference League launched 2021-22.
        if season_year >= 2021:
            all_rows.extend(parse_european_txt(season, 'Conference League', 'conf.txt'))
    # UEFA Cup / Europa League historical backfill: 2004-05 -> 2019-20.
    # openfootball coverage of EL only starts at 2020-21, so we layer in the
    # Wikipedia-scraped CSV for the gap. See scrape_el_uefacup.py.
    if 2004 <= season_year <= 2019:
        all_rows.extend(load_uefacup_historical(season))
    # Domestic cups (Phase 3a): engsoccerdata covers FA Cup pre-2018,
    # openfootball country repos cover the modern era for FA Cup, DFB-Pokal,
    # Copa del Rey, and Coppa Italia.
    if season_year <= 2017:
        all_rows.extend(load_facup_engsoccerdata(season))
    for cup_name, country_repo, txt_file, first_of in DOMESTIC_CUPS:
        if season_year >= first_of:
            repo_base = f'{OPENFOOTBALL_COUNTRY_BASE}/{country_repo}/master'
            all_rows.extend(parse_european_txt(season, cup_name, txt_file, repo_base=repo_base))
    # Domestic cups (Phase 3b): Wikipedia historical backfill for the gap
    # before openfootball coverage starts. cups_historical.csv only contains
    # rows for the relevant pre-openfootball years per cup, so we can call
    # for any season — empty-result calls are cheap (single dataframe filter).
    for cup_name, _, _, first_of in DOMESTIC_CUPS:
        if cup_name == 'FA Cup':
            continue  # FA Cup history is fully covered by engsoccerdata
        if season_year < first_of:
            all_rows.extend(load_cup_historical(season, cup_name))

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
# Incremental: skips date IDs already in zidane_ratings.csv.gz.
# Stored gzipped because the uncompressed file exceeds GitHub's 100MB
# per-file limit once the historical backfill is included.

print("Starting ZIDANE rating calculations...")

max_date_id = int(df['grouped_date_id'].max())

try:
    zidane_df = pd.read_csv('zidane_ratings.csv.gz')
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
zidane_df.to_csv('zidane_ratings.csv.gz', index=False, compression='gzip')
print("zidane_ratings.csv.gz saved!")

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
    This clears the CL final (late May/June) before marking anything done.
    Derives end_year from the 4-digit start year so historical seasons like
    "1992-93" don't get parsed as 2092-93."""
    start_year = int(season_str[:4])
    end_year   = start_year + 1
    return date.today() > date(end_year, 7, 31)

domestic_records = []
cl_records       = []
el_records       = []
cup_records      = []   # Phase 3: domestic cup champion / runner-up

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

# --- Domestic cup champion / runner-up (Phase 3) ---
# Handles 2-leg finals via aggregate: find the last match of comp_season,
# look at all matches between those same two teams in the trailing 21 days,
# sum the goals. Higher aggregate wins. Ties resolve via shootout_winner
# on the second leg, then by away-goals rule (latest leg's away goals).
DOMESTIC_CUP_NAMES = ['FA Cup', 'DFB-Pokal', 'Copa del Rey', 'Coppa Italia']
for cup in DOMESTIC_CUP_NAMES:
    cup_df = df[df['competition'] == cup].dropna(subset=['comp_season']).copy()
    for comp_season, sdf in cup_df.groupby('comp_season'):
        if not season_is_complete(comp_season):
            continue
        # Older Wikipedia cup pages occasionally include "see also" entries
        # from the next season (Oct of year+1). Filter to dates within the
        # cup season's natural range before identifying the final.
        season_start = pd.Timestamp(int(comp_season[:4]), 7, 1)
        season_end   = pd.Timestamp(int(comp_season[:4]) + 1, 7, 1)
        sdf = sdf[(sdf['date'] >= season_start) & (sdf['date'] < season_end)]
        if sdf.empty:
            continue
        last_date = sdf['date'].max()
        last_match = sdf[sdf['date'] == last_date].iloc[-1]
        team_a = last_match['home_team']
        team_b = last_match['away_team']
        # Pull all matches between these two teams in the final 21 days
        from datetime import timedelta
        window_start = last_date - pd.Timedelta(days=21)
        ties = sdf[
            (sdf['date'] >= window_start) &
            (((sdf['home_team'] == team_a) & (sdf['away_team'] == team_b)) |
             ((sdf['home_team'] == team_b) & (sdf['away_team'] == team_a)))
        ]
        # Sum goals from team_a perspective
        a_goals = 0
        b_goals = 0
        a_away_goals = 0
        b_away_goals = 0
        for _, m in ties.iterrows():
            if m['home_team'] == team_a:
                a_goals += m['home_score']; b_goals += m['away_score']
                b_away_goals += m['away_score']
            else:
                a_goals += m['away_score']; b_goals += m['home_score']
                a_away_goals += m['away_score']
        if a_goals > b_goals:
            champion, runner_up = team_a, team_b
        elif b_goals > a_goals:
            champion, runner_up = team_b, team_a
        else:
            sw = last_match.get('shootout_winner')
            if pd.notna(sw) and sw:
                champion = sw
                runner_up = team_b if sw == team_a else team_a
            elif a_away_goals > b_away_goals:
                champion, runner_up = team_a, team_b
            elif b_away_goals > a_away_goals:
                champion, runner_up = team_b, team_a
            else:
                continue  # truly indeterminate
        cup_records.append({'season': comp_season, 'team': champion,  'cup_finish': 'Champion',  'cup': cup})
        cup_records.append({'season': comp_season, 'team': runner_up, 'cup_finish': 'Runner-Up', 'cup': cup})

domestic_finish_df = pd.DataFrame(domestic_records) if domestic_records else pd.DataFrame(columns=['season', 'team', 'domestic_finish'])
cl_finish_df       = pd.DataFrame(cl_records)       if cl_records       else pd.DataFrame(columns=['season', 'team', 'cl_finish'])
el_finish_df       = pd.DataFrame(el_records)       if el_records       else pd.DataFrame(columns=['season', 'team', 'el_finish'])
cup_finish_df      = pd.DataFrame(cup_records)      if cup_records      else pd.DataFrame(columns=['season', 'team', 'cup_finish', 'cup'])
# For merging into ratings: collapse to a single domestic_cup_finish column
# (a team only ever wins one domestic cup per season since each team belongs
# to one league/country, so the per-team-per-season collapse is safe).
if not cup_finish_df.empty:
    cup_simple_df = cup_finish_df.rename(columns={'cup_finish': 'domestic_cup_finish'})[['season', 'team', 'domestic_cup_finish']]
else:
    cup_simple_df = pd.DataFrame(columns=['season', 'team', 'domestic_cup_finish'])

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

# Validation print: domestic cup champions per season
if not cup_finish_df.empty:
    print(f"\n  {'Season':<10} {'FA Cup':<26} {'DFB-Pokal':<26} {'Copa del Rey':<28} {'Coppa Italia'}")
    print(f"  {'-'*112}")
    cup_seasons = sorted(cup_finish_df['season'].unique())
    for season in cup_seasons:
        names = []
        for cup in DOMESTIC_CUP_NAMES:
            r = cup_finish_df[(cup_finish_df['season']==season) & (cup_finish_df['cup']==cup) & (cup_finish_df['cup_finish']=='Champion')]
            names.append(r['team'].values[0] if len(r) else '—')
        print(f"  {season:<10} {names[0]:<26} {names[1]:<26} {names[2]:<28} {names[3]}")

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

# Finish-flag merges happen AFTER the season-patch step below — otherwise
# the COVID-bubble Aug-2020 CL final row keeps its date_to_season "2020-21"
# label at merge time and gets no flags, even though it later gets patched
# back to "2019-20". See "season_patch" block below.

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
euro_games_df  = df[df['competition'].isin(['Champions League', 'Europa League', 'Conference League'])].dropna(subset=['comp_season'])
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

# Merge finish flags (post-season-patch — see comment above merge sites that
# used to live before this block). Doing it here ensures the bubble-final
# rows get their correct season's flags applied.
final_df = pd.merge(final_df, domestic_finish_df, on=['season', 'team'], how='left')
final_df = pd.merge(final_df, cl_finish_df,       on=['season', 'team'], how='left')
final_df = pd.merge(final_df, el_finish_df,       on=['season', 'team'], how='left')
final_df = pd.merge(final_df, cup_simple_df,      on=['season', 'team'], how='left')
final_df['domestic_finish']     = final_df['domestic_finish'].fillna('')
final_df['cl_finish']           = final_df['cl_finish'].fillna('')
final_df['el_finish']           = final_df['el_finish'].fillna('')
final_df['domestic_cup_finish'] = final_df['domestic_cup_finish'].fillna('')

# Final column order
final_df = final_df[[
    'ranking_id', 'date', 'season', 'team', 'league',
    'rating', 'rank',
    'games_played',
    'domestic_finish', 'cl_finish', 'el_finish', 'domestic_cup_finish',
    'is_cl_final_day', 'is_domestic_final_day', 'is_end_of_season',
    'last_match_date', 'last_match', 'is_game_day',
    'most_recent',
]]

final_df.sort_values(['ranking_id', 'rank'], inplace=True)
final_df.drop_duplicates(keep='first', inplace=True)

# Minimum games filter — drops relegated clubs with sparse history
# and non-top-5 clubs appearing only via European competition
final_df = final_df[final_df['games_played'] >= min_games]

# Re-center each snapshot's ratings around the Big-5 mean.
# Massey ratings are 0-centered across the FULL game-day network, but the
# network includes ~200 non-Big-5 teams from CL/EL qualifying rounds that
# usually lose to Big-5 opponents and absorb the negative side of the
# distribution. After min_games drops most of them from display, the
# remaining Big-5 + a few European-ladder teams sit on a heavily +shifted
# scale (mean ~+1.5, almost no negatives).
#
# Re-anchoring to the Big-5 mean makes the displayed ratings interpretable:
# 0 ≈ average Big-5 team in the rolling window. Non-Big-5 teams that
# survived min_games (deep CL/EL runs) get shifted with the same offset
# so their relative position is preserved. Ratings are NOT modified for
# anyone in zidane_ratings.csv.gz — only the final output / display.
BIG5_LEAGUES = set(FDCO_LEAGUE_CODES.keys())
big5_shift = (
    final_df[final_df['league'].isin(BIG5_LEAGUES)]
    .groupby('ranking_id')['rating']
    .mean()
    .rename('big5_mean')
)
final_df = final_df.merge(big5_shift, left_on='ranking_id', right_index=True, how='left')
final_df['rating'] = final_df['rating'] - final_df['big5_mean'].fillna(0)
final_df.drop(columns=['big5_mean'], inplace=True)

# Renumber ranks contiguously within each snapshot after filtering.
# (Order is unchanged by the constant per-snapshot shift, but rebuild
# anyway so ranks remain integer-clean.)
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
