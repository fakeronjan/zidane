"""Big-5 league table rules: points for a win by era, and how ties are broken.

Points for a win went from 2 to 3 in the Premier League in 1981, Serie A and
Ligue 1 in 1994-95, and La Liga and the Bundesliga in 1995-96. La Liga and
Serie A break ties on head-to-head (points, then goal difference among the
tied clubs) before overall goal difference; the Premier League, Bundesliga,
and Ligue 1 use overall goal difference, then goals scored. (La Liga 2006-07:
Real Madrid and Barcelona finished level on 76, and Real Madrid won the title
on head-to-head despite Barcelona's better goal difference.)
"""
import pandas as pd

THREE_POINTS_FROM = {'EPL': 1981, 'Serie A': 1994, 'Ligue 1': 1994,
                     'La Liga': 1995, 'Bundesliga': 1995}
HEAD_TO_HEAD_FIRST = {'La Liga', 'Serie A'}


def points_per_win(league, season):
    """2 or 3 for a league season like '1993-94'."""
    return 3 if int(str(season)[:4]) >= THREE_POINTS_FROM.get(league, 0) else 2


def _rows(games, win):
    """Per-team points, goal difference, and goals scored from match rows."""
    t = {}
    for h, a, hs, as_ in zip(games['home_team'], games['away_team'],
                             games['home_score'], games['away_score']):
        for team, gf, ga in ((h, hs, as_), (a, as_, hs)):
            r = t.setdefault(team, [0, 0, 0])
            r[0] += win if gf > ga else (1 if gf == ga else 0)
            r[1] += gf - ga
            r[2] += gf
    return t


def ordered_table(games, league, season):
    """One league season's table, best first: a DataFrame of team, pts, gd,
    gf, in the league's own tiebreak order. `games` holds that season's
    matches (home_team, away_team, home_score, away_score)."""
    win = points_per_win(league, season)
    table = _rows(games, win)
    by_pts = {}
    for team, (pts, gd, gf) in table.items():
        by_pts.setdefault(pts, []).append(team)
    order = []
    for pts in sorted(by_pts, reverse=True):
        tied = by_pts[pts]
        if len(tied) > 1 and league in HEAD_TO_HEAD_FIRST:
            mini = games[games['home_team'].isin(tied) & games['away_team'].isin(tied)]
            h2h = _rows(mini, win)
            tied.sort(key=lambda t: (-h2h.get(t, [0, 0, 0])[0], -h2h.get(t, [0, 0, 0])[1],
                                     -table[t][1], -table[t][2], t))
        else:
            tied.sort(key=lambda t: (-table[t][1], -table[t][2], t))
        order.extend(tied)
    return pd.DataFrame([(t, *table[t]) for t in order], columns=['team', 'pts', 'gd', 'gf'])
