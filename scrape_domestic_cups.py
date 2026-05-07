# ============================================================
# scrape_domestic_cups.py
# ============================================================
# One-time backfill of domestic cup match data for the historical era
# where openfootball doesn't yet cover us:
#   - DFB-Pokal       1992-93 → 2009-10  (18 seasons)
#   - Copa del Rey    1992-93 → 2011-12  (20 seasons)
#   - Coppa Italia    1992-93 → 2012-13  (21 seasons)
#
# Coupe de France is intentionally skipped — no famous treble candidates
# from French clubs in this period (PSG never won the Champions League),
# and the early-round amateur matches add little rating signal at high
# scraping cost.
#
# Wikipedia uses two match formats interchangeably:
#   1. <table class="fevent">   — modern style. Date may be in a paired
#      <span class="dtstart">. If absent (older pages), walk back through
#      preceding text to find a "DD Month YYYY" string near a round heading.
#   2. <table class="vevent">   — alternate style (e.g., DFB-Pokal). Each
#      match row has the date as plain text in the first TD.
#
# Output: cups_historical.csv with the same schema used elsewhere.

import re
import sys
import time
import csv
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup

UA = 'zidane-data-collection/1.0 (https://github.com/fakeronjan/zidane; one-time domestic cup backfill)'
WIKI_BASE = 'https://en.wikipedia.org'

# Per-cup config: (cup_name, slug_suffix, year_range)
# slug_suffix is everything after the season prefix in the Wikipedia URL.
# Year ranges cover everything before openfootball's reliable cup.txt
# coverage starts (which only kicks in for the latest 4-7 seasons).
CUPS = [
    ('DFB-Pokal',    '_DFB-Pokal',    range(1992, 2018)),  # 1992-93 → 2017-18 (gap before 2018-19 openfootball)
    ('Copa del Rey', '_Copa_del_Rey', range(1992, 2020)),  # 1992-93 → 2019-20 (gap before 2020-21 openfootball)
    ('Coppa Italia', '_Coppa_Italia', range(1992, 2020)),  # 1992-93 → 2019-20 (gap before 2020-21 openfootball)
]

# Months for date parsing
MONTHS = {m.lower(): i for i, m in enumerate(
    ['January','February','March','April','May','June',
     'July','August','September','October','November','December'], 1)}

# "23 August 2008" or "23rd August 2008"
RE_DATE_TEXT = re.compile(
    r'(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
    re.IGNORECASE
)

# Score regex (handles en-dash and hyphen)
RE_SCORE = re.compile(r'(\d+)\s*[–\-]\s*(\d+)')

# Penalty score after "Penalties" keyword (modern format) or "(X-Y p)" suffix (older)
RE_PEN_AFTER_KEYWORD = re.compile(r'Penalties.*?(\d+)\s*[–\-]\s*(\d+)', re.IGNORECASE | re.DOTALL)
RE_PEN_SUFFIX        = re.compile(r'(?:^|\s|\()(\d+)\s*[–\-]\s*(\d+)\s*(?:p|pen|pens)\b', re.IGNORECASE)


def fetch(url):
    req = Request(url, headers={'User-Agent': UA})
    with urlopen(req, timeout=30) as r:
        return r.read().decode('utf-8', errors='replace')


def make_url(cup_slug_suffix, start_year):
    # Wikipedia mostly uses "1992–93" (en-dash + 2-digit) but occasionally
    # "1999–2000" (4-digit) for the millennium-crossing season. Caller tries
    # both in scrape_cup_season.
    season = f'{start_year}%E2%80%93{str(start_year + 1)[-2:]}'
    return f'{WIKI_BASE}/wiki/{season}{cup_slug_suffix}'

def make_url_4digit(cup_slug_suffix, start_year):
    season = f'{start_year}%E2%80%93{start_year + 1}'
    return f'{WIKI_BASE}/wiki/{season}{cup_slug_suffix}'


def parse_date_text(text):
    """Return ISO date string for the first 'DD Month YYYY' match, else None."""
    m = RE_DATE_TEXT.search(text)
    if not m:
        return None
    day, month_name, year = m.group(1), m.group(2), m.group(3)
    month = MONTHS.get(month_name.lower())
    if not month:
        return None
    return f'{int(year):04d}-{month:02d}-{int(day):02d}'


def find_date_above(elem):
    """Walk backwards through preceding elements to find the nearest 'DD Month YYYY' string."""
    n = elem
    for _ in range(40):
        n = n.find_previous(['p', 'h2', 'h3', 'h4', 'b', 'i', 'div', 'span', 'td'])
        if not n:
            return None
        d = parse_date_text(n.get_text(' ', strip=True))
        if d:
            return d
    return None


def _team_from_th(th):
    """Extract team name from an fevent home/away cell, skipping flag links."""
    name_span = th.find(itemprop='name') or th
    for a in name_span.find_all('a'):
        if a.find_parent(class_='flagicon'):
            continue
        return a.get_text(strip=True)
    return name_span.get_text(' ', strip=True)


def parse_fevent_match(fe, season):
    """Return a dict for one fevent table (modern format), or None if unparseable."""
    home_th = fe.find('th', class_='fhome')
    away_th = fe.find('th', class_='faway')
    score_th = fe.find('th', class_='fscore')
    if not (home_th and away_th and score_th):
        return None

    home = _team_from_th(home_th)
    away = _team_from_th(away_th)

    score_text = score_th.get_text(' ', strip=True)
    sm = RE_SCORE.search(score_text)
    if not sm:
        return None
    hg, ag = int(sm.group(1)), int(sm.group(2))

    # Date: prefer paired dtstart, otherwise walk back through document
    dts = fe.find_previous(class_='dtstart')
    # Validate the dtstart hasn't been already consumed by a different fevent above
    if dts:
        # If there's another fevent between this one and the dtstart, it isn't ours
        prev_fe = fe.find_previous('table', class_='fevent')
        if prev_fe and prev_fe.find_previous(class_='dtstart') == dts:
            dts = None
    date = dts.get_text(strip=True) if dts else find_date_above(fe)
    if not date:
        return None

    # Penalty detection — only meaningful on tied final scores
    shootout_winner = None
    if hg == ag:
        full_text = fe.get_text(' ', strip=True)
        pen = RE_PEN_AFTER_KEYWORD.search(full_text) or RE_PEN_SUFFIX.search(full_text)
        if pen:
            ph, pa = int(pen.group(1)), int(pen.group(2))
            if ph != pa:
                shootout_winner = home if ph > pa else away

    return {
        'date':            date,
        'home':            home,
        'visitor':         away,
        'FT':              f'{hg}-{ag}',
        'hgoal':           hg,
        'vgoal':           ag,
        'comp_season':     season,
        'shootout_winner': shootout_winner,
    }


def parse_vevent_match(ve, season):
    """Return a dict for one vevent table (DFB-Pokal style), or None if unparseable."""
    rows = ve.find_all('tr', recursive=True)
    if not rows:
        return None
    first = rows[0]
    tds = first.find_all('td', recursive=False)
    if len(tds) < 4:
        return None
    # td[0]: date. td[1]: home team. td[2]: score. td[3]: away team.
    date = parse_date_text(tds[0].get_text(' ', strip=True))
    if not date:
        return None

    def _team(td):
        a = td.find('a')
        if a and not a.find_parent(class_='flagicon'):
            return a.get_text(strip=True)
        return td.get_text(' ', strip=True)

    home = _team(tds[1])
    away = _team(tds[3])
    sm = RE_SCORE.search(tds[2].get_text(' ', strip=True))
    if not sm:
        return None
    hg, ag = int(sm.group(1)), int(sm.group(2))

    # Penalty detection in the full vevent text
    shootout_winner = None
    if hg == ag:
        full_text = ve.get_text(' ', strip=True)
        pen = RE_PEN_AFTER_KEYWORD.search(full_text) or RE_PEN_SUFFIX.search(full_text)
        if pen:
            ph, pa = int(pen.group(1)), int(pen.group(2))
            if ph != pa:
                shootout_winner = home if ph > pa else away

    return {
        'date':            date,
        'home':            home,
        'visitor':         away,
        'FT':              f'{hg}-{ag}',
        'hgoal':           hg,
        'vgoal':           ag,
        'comp_season':     season,
        'shootout_winner': shootout_winner,
    }


def parse_page(html, season):
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    for fe in soup.find_all('table', class_='fevent'):
        m = parse_fevent_match(fe, season)
        if m:
            rows.append(m)
    for ve in soup.find_all('table', class_='vevent'):
        m = parse_vevent_match(ve, season)
        if m:
            rows.append(m)
    # Dedupe by (date, home, visitor)
    seen, unique = set(), []
    for r in rows:
        key = (r['date'], r['home'], r['visitor'])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def scrape_cup_season(cup_name, slug_suffix, start_year):
    season = f'{start_year}-{str(start_year + 1)[-2:]}'
    # Try both the 2-digit and 4-digit URL forms (millennium-crossing seasons
    # like "1999–2000" use the 4-digit form).
    last_err = None
    html = None
    for url in (make_url(slug_suffix, start_year), make_url_4digit(slug_suffix, start_year)):
        try:
            html = fetch(url)
            break
        except (HTTPError, URLError) as e:
            last_err = e
    if html is None:
        print(f'    ! could not fetch {cup_name} {season}: {last_err}')
        return []
    rows = parse_page(html, season)
    print(f'  {cup_name} {season}: {len(rows)} matches')
    return rows


def main():
    out_path = 'cups_historical.csv'
    fieldnames = ['date', 'home', 'visitor', 'FT', 'hgoal', 'vgoal', 'comp_season', 'cup', 'shootout_winner']
    all_rows = []
    for cup_name, slug_suffix, years in CUPS:
        print(f'\n=== {cup_name} ===')
        for y in years:
            rows = scrape_cup_season(cup_name, slug_suffix, y)
            for r in rows:
                r['cup'] = cup_name
            all_rows.extend(rows)
            time.sleep(1.0)  # polite delay
    all_rows.sort(key=lambda r: (r['cup'], r['date'], r['home']))
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f'\nSaved {len(all_rows)} matches to {out_path}')


if __name__ == '__main__':
    main()
