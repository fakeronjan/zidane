"""League season labels from the fixture calendar, not the date.

The date rule (August or later = next season) misfiles league games played in
late July (Ligue 1 1999-2001 and Bundesliga 2001 kicked off July 28-31) and
the COVID-delayed 2019-20 finishes in July and August 2020. Here each league's
season starts after its longest gap between matches that ends between July 1
and October 31 (Serie A 2000-01 began October 1); every league match belongs
to the latest season start on or before its date.
"""
import pandas as pd

BIG5 = ('EPL', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1')


def _label(year):
    return f"{year}-{str(year + 1)[-2:]}"


def season_starts(dates):
    """Season start dates for one league, from its sorted match dates."""
    d = pd.Series(sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()))
    gaps = pd.DataFrame({'prev': d.shift(1), 'next': d}).dropna()
    gaps['len'] = (gaps['next'] - gaps['prev']).dt.days
    starts = []
    for year in sorted(d.dt.year.unique()):
        lo, hi = pd.Timestamp(year, 7, 1), pd.Timestamp(year, 10, 31)
        cand = gaps[(gaps['next'] >= lo) & (gaps['next'] <= hi)]
        if len(cand):
            starts.append(cand.loc[cand['len'].idxmax(), 'next'])
    return starts


def league_season_labels(games):
    """Season label for each Big-5 league match in `games` (date, competition);
    None for every other competition."""
    out = pd.Series(None, index=games.index, dtype=object)
    dates = pd.to_datetime(games['date'])
    for comp in BIG5:
        mask = games['competition'] == comp
        if not mask.any():
            continue
        starts = season_starts(dates[mask])
        for idx, dt in dates[mask].items():
            begun = [s for s in starts if s <= dt.normalize()]
            if begun:
                out[idx] = _label(begun[-1].year)
            else:  # before the first detected start: the date rule
                out[idx] = _label(dt.year if dt.month >= 8 else dt.year - 1)
    return out
