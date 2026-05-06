# ============================================================
# scrape_el_uefacup.py
# ============================================================
# One-time backfill of UEFA Cup / Europa League match data for
# 2004-05 through 2019-20 from Wikipedia. Outputs a CSV that
# zidane.py reads at runtime — no scraping happens on the daily
# cron, just a static lookup.
#
# Why 2004-05+: that's when Wikipedia transitioned to per-round
# sub-pages with structured fevent + dtstart pairs. Earlier
# seasons use a single page with no per-match dates.
# Why 2019-20 cutoff: openfootball/champions-league el.txt
# coverage starts at 2020-21, where the existing pipeline takes over.
#
# Each season has one main page plus several sub-pages
# (qualifying, first round / play-off, group stage, knockout).
# We discover sub-pages dynamically from the main page rather
# than hard-coding URLs since the structure changed at the
# 2009-10 UEFA Cup -> Europa League rebrand.

import re
import sys
import time
import csv
from urllib.parse import unquote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup

UA = 'zidane-data-collection/1.0 (https://github.com/fakeronjan/zidane; one-time UEFA Cup historical backfill)'
WIKI_BASE = 'https://en.wikipedia.org'

# Season -> main page slug. Wikipedia uses "UEFA Cup" through 2008-09 and
# "UEFA Europa League" from 2009-10 onward.
SEASONS = {}
for y in range(2004, 2009):
    SEASONS[f'{y}-{str(y + 1)[-2:]}'] = f'/wiki/{y}%E2%80%93{str(y + 1)[-2:]}_UEFA_Cup'
for y in range(2009, 2020):
    SEASONS[f'{y}-{str(y + 1)[-2:]}'] = f'/wiki/{y}%E2%80%93{str(y + 1)[-2:]}_UEFA_Europa_League'

# Sub-page filename fragments we want to follow from the main page.
# Anything matching these in the URL path is a round we want to scrape.
SUBPAGE_FRAGMENTS = (
    'qualifying',
    'first_round', 'second_round', 'third_round',
    'play_off_round', 'play-off_round',
    'group_stage', 'group_phase',
    'knockout_stage', 'knockout_phase',
)

# Skip Wikipedia metadata sub-pages.
SUBPAGE_BLACKLIST = ('Special:', 'Talk:', 'Help:')


def fetch(url):
    req = Request(url, headers={'User-Agent': UA})
    with urlopen(req, timeout=30) as r:
        return r.read().decode('utf-8', errors='replace')


def discover_subpages(main_html, season_url):
    """From the main season page, find all sub-page URLs we want to scrape."""
    soup = BeautifulSoup(main_html, 'html.parser')
    # Must be a wiki link that contains the same season (UEFA Cup or Europa League)
    # and one of the round fragments.
    season_basename = season_url.split('/wiki/')[1].split('#')[0]  # e.g. 2004%E2%80%9305_UEFA_Cup
    season_decoded = unquote(season_basename).lower()
    base_token = season_decoded  # full base — sub-pages prefix with this
    found = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        if any(b in href for b in SUBPAGE_BLACKLIST):
            continue
        if not href.startswith('/wiki/'):
            continue
        path = href.split('#')[0]
        path_decoded = unquote(path).lower()
        # Must start with the same season base
        if not path_decoded.startswith('/wiki/' + base_token):
            continue
        # Must contain one of our round fragments
        if not any(frag in path_decoded for frag in SUBPAGE_FRAGMENTS):
            continue
        found.add(path)
    return sorted(found)


# Score regex: handles "2–1", "2-1", with either en-dash or hyphen.
RE_SCORE = re.compile(r'(\d+)\s*[–\-]\s*(\d+)')

# Penalty score follows the "Penalties" keyword somewhere in the fevent text,
# e.g. "Penalties | Riera | Pandiani | ... | 1–3 | Kanouté | Dani Alves | ...".
# Wikipedia's old "(X-Y p)" suffix style also exists in some pages — handle both.
RE_PEN_AFTER_KEYWORD = re.compile(r'Penalties.*?(\d+)\s*[–\-]\s*(\d+)', re.IGNORECASE | re.DOTALL)
RE_PEN_SUFFIX        = re.compile(r'(?:^|\s|\()(\d+)\s*[–\-]\s*(\d+)\s*(?:p|pen|pens)\b', re.IGNORECASE)


def parse_match(fe, dtstart_iso, season):
    """Pull a single match dict from one fevent table + paired dtstart string."""
    # Home/away team names come from the named TH cells.
    home_th = fe.find('th', class_='fhome')
    away_th = fe.find('th', class_='faway')
    if not home_th or not away_th:
        return None

    # Strip flag icons / nested decoration; keep only the team link text.
    # The flag inside the cell wraps a link to the country's FA, so the first
    # <a> we'd find is the FA link, not the team link. The real team link is
    # inside <span itemprop="name"> and NOT inside a flagicon span.
    def _team(th):
        name_span = th.find(itemprop='name')
        scope = name_span if name_span else th
        for a in scope.find_all('a'):
            if a.find_parent(class_='flagicon'):
                continue
            return a.get_text(strip=True)
        # No team link — fall back to text content of the name span,
        # which strips out the flag image alt text reasonably well.
        text = scope.get_text(' ', strip=True)
        return text

    home = _team(home_th)
    away = _team(away_th)

    # Score from the fscore cell (could be "2–1" or "2–1 (a.e.t.)" depending on year)
    score_th = fe.find('th', class_='fscore')
    if not score_th:
        return None
    score_text = score_th.get_text(' ', strip=True)
    m = RE_SCORE.search(score_text)
    if not m:
        return None
    home_goals = int(m.group(1))
    away_goals = int(m.group(2))

    # Detect penalties. Only meaningful when the FT/AET score is tied —
    # otherwise the X-Y "1–3" embedded in player lists could be misread as
    # a penalty score.
    shootout_winner = None
    if home_goals == away_goals:
        full_text = fe.get_text(' ', strip=True)
        pen = RE_PEN_AFTER_KEYWORD.search(full_text) or RE_PEN_SUFFIX.search(full_text)
        if pen:
            ph, pa = int(pen.group(1)), int(pen.group(2))
            if ph != pa:
                shootout_winner = home if ph > pa else away

    return {
        'date':            dtstart_iso,
        'home':            home,
        'visitor':         away,
        'FT':              f'{home_goals}-{away_goals}',
        'hgoal':           home_goals,
        'vgoal':           away_goals,
        'comp_season':     season,
        'shootout_winner': shootout_winner,
    }


def parse_page(html, season):
    """Return list of match dicts from a single page (main or sub-page).
    Pairs each fevent table with the dtstart span that immediately precedes it.
    """
    soup = BeautifulSoup(html, 'html.parser')
    # Walk the document in order and pair dtstarts with fevents.
    # Pattern observed: dtstart appears just before its associated fevent.
    nodes = soup.find_all(['table', 'span'])
    rows = []
    pending_date = None
    for n in nodes:
        cls = n.get('class') or []
        if 'dtstart' in cls:
            pending_date = n.get_text(strip=True)  # already in YYYY-MM-DD form
        elif n.name == 'table' and 'fevent' in cls:
            if pending_date:
                m = parse_match(n, pending_date, season)
                if m:
                    rows.append(m)
            # don't reset pending_date — Wikipedia sometimes lists
            # multiple matches under one dtstart
    return rows


def scrape_season(season, main_path):
    main_url = WIKI_BASE + main_path
    print(f'  {season}: fetching main page...')
    main_html = fetch(main_url)
    matches = parse_page(main_html, season)
    sub_paths = discover_subpages(main_html, main_path)
    print(f'    {len(sub_paths)} sub-pages, {len(matches)} matches on main')
    for sub_path in sub_paths:
        time.sleep(1.0)  # be polite to Wikipedia
        try:
            sub_html = fetch(WIKI_BASE + sub_path)
        except (HTTPError, URLError) as e:
            print(f'    WARNING: failed to fetch {sub_path}: {e}')
            continue
        sub_matches = parse_page(sub_html, season)
        print(f'    {sub_path.split("/wiki/")[-1]}: {len(sub_matches)} matches')
        matches.extend(sub_matches)
    # Dedupe by (date, home, visitor) in case main page final overlaps with sub-page
    seen = set()
    unique = []
    for m in matches:
        key = (m['date'], m['home'], m['visitor'])
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    return unique


def main():
    out_path = 'uefacup_historical.csv'
    fieldnames = ['date', 'home', 'visitor', 'FT', 'hgoal', 'vgoal', 'comp_season', 'shootout_winner']
    all_rows = []
    for season, main_path in SEASONS.items():
        rows = scrape_season(season, main_path)
        all_rows.extend(rows)
        time.sleep(1.0)
    # Sort by date for human readability
    all_rows.sort(key=lambda r: (r['date'], r['home']))
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f'\nSaved {len(all_rows)} matches across {len(SEASONS)} seasons to {out_path}')


if __name__ == '__main__':
    main()
