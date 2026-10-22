"""Big-5 league title and relegation odds by Monte Carlo.

At each matchday of a league season, the rest of that season is simulated
N_SIMS times from the ratings as of that day:

    home goals = X + Z,  away goals = Y + Z
    X ~ Poisson(mu + K*(O_home - D_away) + HFA/2 - DRAW_LAMBDA)
    Y ~ Poisson(mu + K*(O_away - D_home) - HFA/2 - DRAW_LAMBDA)
    Z ~ Poisson(DRAW_LAMBDA)   (shared goals; lifts the draw rate to the real one)

mu is the league's goals per team per match over the prior year, K shrinks
ZIDANE's ratings (which are overconfident as match forecasts), and HFA is
home advantage for that era. A promoted club's rating is blended toward the
typical promoted club's until it has played a few games. Each team's
strength also drifts by a random amount per simulation that shrinks as the
season runs out. Tables use the league's own points and tiebreakers
(zidane_rules). Relegation counts direct places fully and a relegation
playoff place at that league's historical rate of losing the playoff.

Fit on ZIDANE's 1993-2026 league matches (fit notes in each constant).
"""
import numpy as np
import pandas as pd

from zidane_rules import points_per_win, HEAD_TO_HEAD_FIRST
from zidane_seasons import BIG5, league_season_labels

N_SIMS = 10_000
K = 0.6                    # rating scale (title/relegation calibration, below)
DRAW_LAMBDA = 0.06         # shared-goal rate; reproduces the 25.2% draw rate (2010-26)
PROMOTED_PRIOR = -0.183    # mean season-end O and D of promoted clubs
PROMOTED_G0 = 8            # promoted club's own rating weight = gp / (gp + G0)
# Strength drift SD = DRIFT_SD0 * (share of season left)**DRIFT_K. K and
# DRIFT_SD0 chosen on all 165 finished league seasons (1993-94 to 2025-26) at
# pre-season and 25/50/75% of matchdays: k 0.6 / sd 0.15 is within 0.01 of the
# best title log loss at every checkpoint (1.855 / 1.203 / 0.795 / 0.474) and
# close to the best relegation Brier (0.131 / 0.094 / 0.072 / 0.054); each
# metric's own winner was worse on the other.
DRIFT_SD0, DRIFT_K = 0.15, 0.5


def hfa_for(date):
    """Home advantage in goals, by era (measured on ZIDANE's league matches)."""
    d = pd.Timestamp(date)
    if d < pd.Timestamp('2004-07-01'):
        return 0.475
    if d < pd.Timestamp('2015-07-01'):
        return 0.375
    if d < pd.Timestamp('2020-03-15'):
        return 0.35
    if d < pd.Timestamp('2021-07-01'):
        return 0.125        # empty stadiums
    return 0.275


def relegation_rule(league, season):
    """(direct relegation positions, playoff positions, P(playoff club goes
    down)) for a league season. Positions are 1-based from the top."""
    y = int(season[:4])
    if league == 'EPL':
        if y <= 1993:
            return [20, 21, 22], [], 0.0
        if y == 1994:
            return [19, 20, 21, 22], [], 0.0
        return [18, 19, 20], [], 0.0
    if league == 'La Liga':
        if y <= 1994 or y in (1997, 1998):
            return [19, 20], [17, 18], 0.5          # promoción
        if y == 1995:
            return [20, 21, 22], [], 0.0
        if y == 1996:
            return [19, 20, 21, 22], [18], 0.5
        return [18, 19, 20], [], 0.0
    if league == 'Serie A':
        if y <= 2003:
            return [15, 16, 17, 18], [], 0.0
        if y == 2004:
            return [19, 20], [18], 0.5
        return [18, 19, 20], [], 0.0
    if league == 'Bundesliga':
        if y <= 2008:
            return [16, 17, 18], [], 0.0
        return [17, 18], [16], 0.2                  # 16th lost 3 of 17 playoffs
    if league == 'Ligue 1':
        if y == 1996:
            return [17, 18, 19, 20], [], 0.0
        if 1997 <= y <= 2000:
            return [16, 17, 18], [], 0.0
        if y == 2001:
            return [17, 18], [], 0.0
        if 2017 <= y <= 2021:
            return [19, 20], [18], 0.4              # barrage
        if y == 2022:
            return [17, 18, 19, 20], [], 0.0
        if y >= 2023:
            return [17, 18], [16], 0.4
        return [18, 19, 20], [], 0.0
    raise ValueError(league)


class SeasonSim:
    """One league season: its clubs, played matches, and remaining fixtures."""

    def __init__(self, league, season, games, ratings, complete):
        self.league, self.season = league, season
        self.games = games.sort_values('date').reset_index(drop=True)
        self.teams = sorted(set(self.games['home_team']) | set(self.games['away_team']))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.ratings = ratings
        self.complete = complete
        self.win = points_per_win(league, season)
        played = set(zip(self.games['home_team'], self.games['away_team']))
        # Remaining fixtures: a finished season's are its later matches; the
        # current season's are every home-and-away pairing not yet played.
        self.unplayed = [] if complete else [(h, a) for h in self.teams for a in self.teams
                                              if h != a and (h, a) not in played]

    def _strength(self, date):
        r = self.ratings[self.ratings['date'] <= date]
        last = r.groupby('team').tail(1).set_index('team')
        gp = self.games[self.games['date'] <= date]
        n = pd.concat([gp['home_team'], gp['away_team']]).value_counts()
        O, D = np.zeros(len(self.teams)), np.zeros(len(self.teams))
        for t, i in self.idx.items():
            o = last['rating_o'].get(t, np.nan); d = last['rating_d'].get(t, np.nan)
            if self.promoted.get(t):
                w = n.get(t, 0) / (n.get(t, 0) + PROMOTED_G0)
                o = PROMOTED_PRIOR if np.isnan(o) else w * o + (1 - w) * PROMOTED_PRIOR
                d = PROMOTED_PRIOR if np.isnan(d) else w * d + (1 - w) * PROMOTED_PRIOR
            O[i] = PROMOTED_PRIOR if np.isnan(o) else o
            D[i] = PROMOTED_PRIOR if np.isnan(d) else d
        return O, D

    def odds_at(self, date, mu, rng):
        """{team: (title prob, relegation prob)} after the matches of `date`."""
        date = pd.Timestamp(date)
        done = self.games[self.games['date'] <= date]
        if self.complete:
            later = self.games[self.games['date'] > date]
            fixtures = list(zip(later['home_team'], later['away_team']))
        else:
            fixtures = self.unplayed
        n, S = len(self.teams), N_SIMS
        pts = np.zeros(n); gd = np.zeros(n); gf = np.zeros(n)
        h2h_p = np.zeros((n, n)); h2h_g = np.zeros((n, n))
        for h, a, hs, as_ in zip(done['home_team'], done['away_team'], done['home_score'], done['away_score']):
            i, j = self.idx[h], self.idx[a]
            ph = self.win if hs > as_ else (1 if hs == as_ else 0)
            pa = self.win if as_ > hs else (1 if hs == as_ else 0)
            pts[i] += ph; pts[j] += pa; gd[i] += hs - as_; gd[j] += as_ - hs; gf[i] += hs; gf[j] += as_
            h2h_p[i, j] += ph; h2h_p[j, i] += pa; h2h_g[i, j] += hs - as_; h2h_g[j, i] += as_ - hs
        PTS = np.tile(pts, (S, 1)); GD = np.tile(gd, (S, 1)); GF = np.tile(gf, (S, 1))
        use_h2h = self.league in HEAD_TO_HEAD_FIRST
        if use_h2h:
            HP = np.tile(h2h_p, (S, 1, 1)); HG = np.tile(h2h_g, (S, 1, 1))
        if fixtures:
            O, D = self._strength(date)
            total = 2 * len(self.teams) - 2
            played_frac = len(done) / max(1, len(done) + len(fixtures))
            sd = DRIFT_SD0 * (1 - played_frac) ** DRIFT_K
            E = rng.normal(0, sd, size=(S, n)) if sd > 0 else np.zeros((S, n))
            hi = np.array([self.idx[h] for h, _ in fixtures]); ai = np.array([self.idx[a] for _, a in fixtures])
            hfa = hfa_for(date)
            lam_h = mu + K * (O[hi] - D[ai] + E[:, hi] - E[:, ai]) + hfa / 2 - DRAW_LAMBDA
            lam_a = mu + K * (O[ai] - D[hi] + E[:, ai] - E[:, hi]) - hfa / 2 - DRAW_LAMBDA
            z = rng.poisson(DRAW_LAMBDA, size=lam_h.shape)
            hg = rng.poisson(np.clip(lam_h, 0.02, None)) + z
            ag = rng.poisson(np.clip(lam_a, 0.02, None)) + z
            ph = np.where(hg > ag, self.win, np.where(hg == ag, 1, 0))
            pa = np.where(ag > hg, self.win, np.where(hg == ag, 1, 0))
            for m in range(len(fixtures)):
                i, j = hi[m], ai[m]
                PTS[:, i] += ph[:, m]; PTS[:, j] += pa[:, m]
                GD[:, i] += hg[:, m] - ag[:, m]; GD[:, j] += ag[:, m] - hg[:, m]
                GF[:, i] += hg[:, m]; GF[:, j] += ag[:, m]
                if use_h2h:
                    HP[:, i, j] += ph[:, m]; HP[:, j, i] += pa[:, m]
                    HG[:, i, j] += hg[:, m] - ag[:, m]; HG[:, j, i] += ag[:, m] - hg[:, m]
        # Sort key: points, then head-to-head among clubs level on points
        # (La Liga, Serie A), then goal difference, then goals scored.
        key = PTS * 1e12 + (GD + 500) * 1e3 + GF
        if use_h2h:
            tied = PTS[:, :, None] == PTS[:, None, :]
            mp = (HP * tied).sum(axis=2); mg = (HG * tied).sum(axis=2)
            key = PTS * 1e12 + mp * 1e9 + (mg + 500) * 1e6 + (GD + 500) * 1e3 + GF
        order = np.argsort(-key, axis=1, kind='stable')
        pos = np.empty_like(order); pos[np.arange(S)[:, None], order] = np.arange(1, n + 1)
        direct, playoff, p_down = relegation_rule(self.league, self.season)
        title = (pos == 1).mean(axis=0)
        releg = np.isin(pos, direct).mean(axis=0) + p_down * np.isin(pos, playoff).mean(axis=0)
        return {t: (float(title[i]), float(releg[i])) for t, i in self.idx.items()}


def league_games(games):
    """Big-5 league matches with their league season label."""
    L = games[games['competition'].isin(BIG5)].dropna(subset=['home_score', 'away_score']).copy()
    L['date'] = pd.to_datetime(L['date'])
    L['league_season'] = league_season_labels(L)
    return L


def league_mu(L, league, date):
    """Goals per team per match in `league` over the year before `date`."""
    w = L[(L['competition'] == league) & (L['date'] <= date) & (L['date'] > date - pd.Timedelta(days=365))]
    if len(w) < 50:
        w = L[(L['competition'] == league) & (L['date'] <= date)].tail(380)
    return float((w['home_score'].sum() + w['away_score'].sum()) / (2 * max(1, len(w))))


def season_odds(L, ratings, league, season, current_season, seed=0):
    """DataFrame of date, team, title, relegation for every matchday of one
    league season (plus the day before it starts)."""
    S = L[(L['competition'] == league) & (L['league_season'] == season)]
    if S.empty:
        return pd.DataFrame(columns=['date', 'league', 'season', 'team', 'title', 'releg'])
    seasons = sorted(L['league_season'].unique())
    i = seasons.index(season)
    prev = set()
    if i > 0:
        P = L[(L['competition'] == league) & (L['league_season'] == seasons[i - 1])]
        prev = set(P['home_team']) | set(P['away_team'])
    sim = SeasonSim(league, season, S, ratings, complete=(season != current_season))
    sim.promoted = {t: (t not in prev and i > 0) for t in sim.teams}
    rng = np.random.default_rng(seed)
    dates = [S['date'].min() - pd.Timedelta(days=1)] + sorted(S['date'].dt.normalize().unique())
    out = []
    for d in dates:
        mu = league_mu(L, league, pd.Timestamp(d))
        for t, (ti, re) in sim.odds_at(d, mu, rng).items():
            out.append((pd.Timestamp(d), league, season, t, ti, re))
    return pd.DataFrame(out, columns=['date', 'league', 'season', 'team', 'title', 'releg'])


# ── Cache: finished seasons are computed once ───────────────────────────────
import hashlib as _hashlib
import multiprocessing as _mp
import os as _os
import pickle as _pickle

_JOB = {}


def _season_inputs(L, ratings, league, season):
    S = L[(L['competition'] == league) & (L['league_season'] == season)]
    teams = set(S['home_team']) | set(S['away_team'])
    lo_d, hi_d = S['date'].min() - pd.Timedelta(days=400), S['date'].max()
    r = ratings[ratings['team'].isin(teams) & (ratings['date'] >= lo_d) & (ratings['date'] <= hi_d)]
    return S, r


def _fingerprint(L, ratings, league, season, current_season):
    h = _hashlib.sha256(open(_os.path.abspath(__file__), 'rb').read())
    for mod in ('zidane_rules.py', 'zidane_seasons.py'):
        h.update(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), mod), 'rb').read())
    S, r = _season_inputs(L, ratings, league, season)
    # The prior season's clubs decide who counts as promoted; the year before
    # each date sets mu.
    lg = L[(L['competition'] == league) & (L['date'] > S['date'].min() - pd.Timedelta(days=400)) &
           (L['date'] <= S['date'].max())]
    h.update(lg[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'league_season']]
             .sort_values(['date', 'home_team']).to_csv(index=False).encode())
    r = r[['date', 'team', 'rating_o', 'rating_d']].sort_values(['date', 'team']).copy()
    r[['rating_o', 'rating_d']] = r[['rating_o', 'rating_d']].round(3)
    h.update(r.to_csv(index=False).encode())
    h.update(repr((season == current_season)).encode())
    return h.hexdigest()


def _one(key):
    league, season = key
    j = _JOB
    S, r = _season_inputs(j['L'], j['ratings'], league, season)
    L = j['L'][(j['L']['competition'] == league)]
    return key, season_odds(L, r, league, season, j['current'])


def compute_cached(games, ratings, current_season, cache_dir='league_odds_cache',
                   workers=None, log=print):
    """Title and relegation odds for every Big-5 league season (DataFrame of
    date, league, season, team, title, releg), reusing cached seasons whose
    inputs haven't changed."""
    _os.makedirs(cache_dir, exist_ok=True)
    L = league_games(games)
    ratings = ratings[['date', 'team', 'rating_o', 'rating_d']].copy()
    ratings['date'] = pd.to_datetime(ratings['date'])
    ratings = ratings.drop_duplicates(['date', 'team']).sort_values('date')
    keys = sorted({(c, s) for c, s in zip(L['competition'], L['league_season'])})
    results, todo, sigs = {}, [], {}
    for k in keys:
        sigs[k] = _fingerprint(L, ratings, k[0], k[1], current_season)
        path = _os.path.join(cache_dir, f"{k[0].replace(' ', '_')}_{k[1]}.pkl")
        if _os.path.exists(path):
            try:
                sig, payload = _pickle.load(open(path, 'rb'))
                if sig == sigs[k]:
                    results[k] = payload
                    continue
            except Exception:
                pass
        todo.append(k)
    log(f"  league odds: {len(results)} league-seasons from cache, computing {len(todo)}")
    if todo:
        _JOB.update(L=L, ratings=ratings, current=current_season)
        with _mp.get_context('fork').Pool(workers or _os.cpu_count()) as pool:
            for k, payload in pool.imap_unordered(_one, todo):
                results[k] = payload
                _pickle.dump((sigs[k], payload),
                             open(_os.path.join(cache_dir, f"{k[0].replace(' ', '_')}_{k[1]}.pkl"), 'wb'))
    return pd.concat([results[k] for k in keys], ignore_index=True)
