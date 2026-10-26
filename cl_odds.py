"""Champions League winner odds by Monte Carlo, 2003-04 onward.

At each Champions League matchday the rest of the competition is simulated
N_SIMS times with the same match model as league_odds.py (Poisson goals from
attack and defense ratings, home advantage, shared-goal draw term, strength
drift). Formats:

  2003-04 to 2023-24: eight groups of four (head-to-head tiebreakers); group
      winners meet runners-up in the round of 16 (not the same group or the
      same country); open draws from the quarterfinals; two-legged ties with
      away goals through 2020-21, then extra time and penalties; a one-match
      final. 2019-20 finished in Lisbon with one-match quarterfinals and
      semifinals once that was announced (June 17, 2020).
  2024-25 on: one 36-club league phase; places 1-8 go to the round of 16,
      9-24 play a two-legged playoff round, and a seeded bracket follows.

A draw that hasn't happened yet is simulated; once it has, the real pairings
are used (finished seasons: from the day after the previous round ends;
current season: once the fixtures are published in cl_schedule.csv). Ties
already decided are fixed, and a tie with one leg played continues from that
score.

Big-5 clubs use their ZIDANE ratings. ZIDANE's ratings cover Big-5 clubs only,
so every other club gets an estimate from its European results over the prior
two years against rated opponents, shrunk toward a typical non-Big-5 club
(NONBIG5_PRIOR, NONBIG5_WEIGHT; fit on 2005-26 CL matches).
"""
import numpy as np
import pandas as pd

from league_odds import K, DRAW_LAMBDA, DRIFT_SD0, DRIFT_K, hfa_for, N_SIMS
from zidane_seasons import BIG5

FIRST_SEASON = 2003
NONBIG5_PRIOR = -0.5        # O and D of a typical non-Big-5 CL club
NONBIG5_WEIGHT = 15         # prior counts as this many matches of evidence
NONBIG5_WINDOW_DAYS = 730
MU = 1.35                   # goals per team per CL match
ET_FACTOR = 1 / 3           # extra time = a third of a match
AWAY_GOALS_LAST = 2020      # away goals rule through 2020-21
LISBON_FROM = pd.Timestamp('2020-06-17')
EURO = ('Champions League', 'Europa League', 'Conference League')


# ── Structure from the match data ───────────────────────────────────────────
class Season:
    """One CL season: phase games, groups or league table, knockout ties."""

    def __init__(self, cl, year):
        self.y = year
        x = cl.sort_values('date')
        per_day = x.groupby(x['date'].dt.normalize()).size()
        days = per_day[(per_day.index >= pd.Timestamp(year, 9, 1)) & (per_day >= 6)].index
        self.phase_start = days.min() if len(days) else pd.Timestamp(year, 9, 15)
        self.league_phase = year >= 2024
        self.phase_end_cut = pd.Timestamp(year + 1, 2, 1) if self.league_phase else pd.Timestamp(year + 1, 1, 1)
        self.phase = x[(x['date'] >= self.phase_start) & (x['date'] < self.phase_end_cut)]
        self.ko = x[x['date'] >= self.phase_end_cut]
        self.teams = sorted(set(self.phase['home_team']) | set(self.phase['away_team']))
        self.groups = self._groups() if not self.league_phase else None

    def _groups(self):
        parent = {t: t for t in self.teams}

        def find(t):
            while parent[t] != t:
                parent[t] = parent[parent[t]]
                t = parent[t]
            return t
        for h, a in zip(self.phase['home_team'], self.phase['away_team']):
            parent[find(h)] = find(a)
        g = {}
        for t in self.teams:
            g.setdefault(find(t), []).append(t)
        return sorted(g.values(), key=lambda v: sorted(v))

    def ko_rounds(self):
        """Knockout ties in order: list of rounds, each a list of
        (team_a, team_b, [legs as (date, home, away, hs, as, shootout)])."""
        k = self.ko.copy()
        k['pair'] = [tuple(sorted(p)) for p in zip(k['home_team'], k['away_team'])]
        first = k.groupby('pair')['date'].min().sort_values()
        rounds, sizes = [], ([8, 8, 4, 2, 1] if self.league_phase else [8, 4, 2, 1])
        pairs = list(first.index)
        for n in sizes:
            chunk, pairs = pairs[:n], pairs[n:]
            rounds.append([(a, b, k[k['pair'] == (a, b)].sort_values('date')[
                ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'shootout_winner']]
                .itertuples(index=False, name=None)) for a, b in chunk])
        return [[(a, b, list(legs)) for a, b, legs in r] for r in rounds]


# ── Ratings ─────────────────────────────────────────────────────────────────
class Strength:
    """Attack/defense for any club on any date."""

    def __init__(self, ratings, euro_games, big5_teams):
        self.r = ratings
        self.big5 = big5_teams
        self.e = euro_games
        self._by_team = {t: g for t, g in ratings.groupby('team')}
        self._memo = {}

    def big5_od(self, team, date):
        g = self._by_team.get(team)
        if g is None:
            return None
        g = g[g['date'] <= date]
        if g.empty:
            return None
        return float(g['rating_o'].iloc[-1]), float(g['rating_d'].iloc[-1])

    def od(self, team, date):
        key = (team, pd.Timestamp(date).normalize())
        if key in self._memo:
            return self._memo[key]
        v = self.big5_od(team, date) if team in self.big5 else None
        if v is None:
            v = self._estimate(team, pd.Timestamp(date))
        self._memo[key] = v
        return v

    def _estimate(self, team, date):
        """European results vs rated opponents in the prior two years, shrunk
        toward NONBIG5_PRIOR."""
        e = self.e
        w = e[(e['date'] <= date) & (e['date'] > date - pd.Timedelta(days=NONBIG5_WINDOW_DAYS)) &
              ((e['home_team'] == team) | (e['away_team'] == team))]
        so, sd, n = 0.0, 0.0, 0
        for d, h, a, hs, as_ in zip(w['date'], w['home_team'], w['away_team'], w['home_score'], w['away_score']):
            opp, home = (a, True) if h == team else (h, False)
            if opp not in self.big5:
                continue
            ov = self.big5_od(opp, d - pd.Timedelta(days=1))
            if ov is None:
                continue
            hfa = hfa_for(d)
            gf, ga = (hs, as_) if home else (as_, hs)
            edge = hfa / 2 if home else -hfa / 2
            so += (gf - MU - edge) / K + ov[1]
            sd += -(ga - MU + edge) / K + ov[0]
            n += 1
        return ((so + NONBIG5_WEIGHT * NONBIG5_PRIOR) / (n + NONBIG5_WEIGHT),
                (sd + NONBIG5_WEIGHT * NONBIG5_PRIOR) / (n + NONBIG5_WEIGHT))


# ── Simulation ──────────────────────────────────────────────────────────────
class Sim:
    def __init__(self, season, strength, date, fixtures, rng, frac_left):
        self.s, self.date, self.rng = season, pd.Timestamp(date), rng
        self.teams = sorted(set(season.teams) | {t for f in fixtures for t in f[:2]})
        self.idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        O = np.zeros(n); D = np.zeros(n)
        for t, i in self.idx.items():
            O[i], D[i] = strength.od(t, self.date)
        sd = DRIFT_SD0 * frac_left ** DRIFT_K
        E = rng.normal(0, sd, size=(N_SIMS, n)) if sd > 0 else np.zeros((N_SIMS, n))
        self.O, self.D, self.E = O, D, E
        self.hfa = hfa_for(self.date)
        self.cap = None       # Daily Matchups: {(home, away): (home goals, away goals)} per sim

    def lam(self, att, dfn, edge):
        """Poisson mean per sim for att (index array, per sim) vs dfn."""
        S = np.arange(N_SIMS)
        x = MU + K * (self.O[att] - self.D[dfn] + self.E[S, att] - self.E[S, dfn]) + edge - DRAW_LAMBDA
        return np.clip(x, 0.02, None)

    def match(self, h, a, neutral=False, et=False):
        """Goals for home h and away a (index arrays per sim)."""
        e = 0.0 if neutral else self.hfa / 2
        f = ET_FACTOR if et else 1.0
        z = self.rng.poisson(DRAW_LAMBDA * f, N_SIMS)
        gh, ga = self.rng.poisson(self.lam(h, a, e) * f) + z, self.rng.poisson(self.lam(a, h, -e) * f) + z
        if self.cap is not None and not et and np.all(h == h[0]) and np.all(a == a[0]):
            self.cap.setdefault((self.teams[h[0]], self.teams[a[0]]), (gh.astype(np.int16), ga.astype(np.int16)))
        return gh, ga

    def tie(self, a, b, legs, single=False, final=False):
        """Winner per sim of a tie between a (hosts the second leg) and b.
        `legs`: real legs already played (rows), matched to the pair."""
        S = N_SIMS
        y = self.s.y
        if single or final:
            if legs:
                return self._winner_from_legs(a, b, legs)
            g_a, g_b = self.match(a, b, neutral=True)
            lv = g_a == g_b
            ea, eb = self.match(a, b, neutral=True, et=True)
            g_a = g_a + np.where(lv, ea, 0); g_b = g_b + np.where(lv, eb, 0)
            pens = self.rng.random(S) < 0.5
            return np.where(g_a > g_b, a, np.where(g_b > g_a, b, np.where(pens, a, b)))
        if len(legs) >= 2:
            return self._winner_from_legs(a, b, legs)
        # Leg 1 at b's ground, leg 2 at a's.
        if len(legs) == 1:
            _, h, aw, hs, as_, _so = legs[0]          # h is b, the leg-1 host
            b1, a1 = np.full(S, hs), np.full(S, as_)
        else:
            b1, a1 = self.match(b, a)
        a2, b2 = self.match(a, b)
        A, B = a1 + a2, b1 + b2
        win = np.where(A > B, a, np.where(B > A, b, -1))
        if y <= AWAY_GOALS_LAST:                     # a's away goals: leg 1; b's: leg 2
            win = np.where(win >= 0, win, np.where(a1 > b2, a, np.where(b2 > a1, b, -1)))
        lv = win < 0
        ea, eb = self.match(a, b, et=True)
        w2 = np.where(ea > eb, a, np.where(eb > ea, b, -1))
        if y <= AWAY_GOALS_LAST:                     # level in extra time, b scored away
            w2 = np.where((w2 < 0) & (eb > 0), b, w2)
        pens = self.rng.random(S) < 0.5
        w2 = np.where(w2 >= 0, w2, np.where(pens, a, b))
        return np.where(lv, w2, win)

    def _winner_from_legs(self, a, b, legs):
        ta, tb = self.teams[a[0]], self.teams[b[0]]
        ga = sum((hs if h == ta else as_) for _, h, aw, hs, as_, so in legs)
        gb = sum((hs if h == tb else as_) for _, h, aw, hs, as_, so in legs)
        so = legs[-1][5]
        if ga != gb:
            w = ta if ga > gb else tb
        elif self.s.y <= AWAY_GOALS_LAST and len(legs) == 2:
            aa = sum(as_ for _, h, aw, hs, as_, so_ in legs if aw == ta)
            ab = sum(as_ for _, h, aw, hs, as_, so_ in legs if aw == tb)
            w = ta if aa > ab else tb if ab > aa else (so if isinstance(so, str) else ta)
        else:
            w = so if isinstance(so, str) else ta
        return np.full(N_SIMS, self.idx[w])


def _table(sim, teams, played, remaining, h2h):
    """Simulated standings for a set of clubs: (S, n) ranks (1 = top) and the
    club index order. played: real results; remaining: fixtures to simulate."""
    S, n = N_SIMS, len(teams)
    li = {t: i for i, t in enumerate(teams)}
    P = np.zeros((S, n)); GD = np.zeros((S, n)); GF = np.zeros((S, n)); W = np.zeros((S, n))
    HP = np.zeros((S, n, n)) if h2h else None
    HG = np.zeros((S, n, n)) if h2h else None
    HF = np.zeros((S, n, n)) if h2h else None

    def add(i, j, gi, gj):
        P[:, i] += np.where(gi > gj, 3, np.where(gi == gj, 1, 0))
        P[:, j] += np.where(gj > gi, 3, np.where(gi == gj, 1, 0))
        GD[:, i] += gi - gj; GD[:, j] += gj - gi; GF[:, i] += gi; GF[:, j] += gj
        W[:, i] += gi > gj; W[:, j] += gj > gi
        if h2h:
            HP[:, i, j] += np.where(gi > gj, 3, np.where(gi == gj, 1, 0))
            HP[:, j, i] += np.where(gj > gi, 3, np.where(gi == gj, 1, 0))
            HG[:, i, j] += gi - gj; HG[:, j, i] += gj - gi
            HF[:, j, i] += gj           # away goals in head-to-head (j away)
    for h, a, hs, as_ in played:
        add(li[h], li[a], np.full(S, hs), np.full(S, as_))
    for h, a in remaining:
        gh, ga = sim.match(np.full(S, sim.idx[h]), np.full(S, sim.idx[a]))
        add(li[h], li[a], gh, ga)
    if h2h:
        tied = P[:, :, None] == P[:, None, :]
        mp = (HP * tied).sum(2); mg = (HG * tied).sum(2); mf = (HF * tied).sum(2)
        key = P * 1e12 + mp * 1e9 + (mg + 500) * 1e6 + mf * 1e4 + (GD + 500) * 10 + GF / 100
    else:
        key = P * 1e12 + (GD + 500) * 1e7 + GF * 1e4 + W * 10
    key = key + sim.rng.random((S, n)) * 1e-3
    order = np.argsort(-key, axis=1)          # order[s, k] = local index at place k+1
    return np.array([sim.idx[t] for t in teams])[order]


def _draw_r16(sim, winners, runners, group_of):
    """Round-of-16 pairings per sim: winners[s, g] vs runners[s, g'] with
    g != g' and not the same Big-5 country. Returns (S, 8) runner index for
    each winner slot."""
    S = N_SIMS
    country = {}
    for t, i in sim.idx.items():
        country[i] = sim.country.get(t)
    out = np.empty((S, 8), dtype=int)
    for s in range(S):
        w = winners[s]; r = runners[s]
        for _ in range(200):
            perm = sim.rng.permutation(8)
            ok = all(group_of[w[k]] != group_of[r[perm[k]]] and
                     (country[w[k]] is None or country[w[k]] != country[r[perm[k]]])
                     for k in range(8))
            if ok:
                break
        out[s] = r[perm]
    return out


def odds_at(season, strength, date, schedule, country, capture=None):
    """{team: P(win the CL)} after the matches of `date`. capture: an
    optional dict filled with per-simulation results (Daily Matchups); draws
    no extra random numbers, so the odds are unchanged."""
    date = pd.Timestamp(date)
    s = season
    rng = np.random.default_rng(int(date.strftime('%Y%m%d')))
    played_phase = s.phase[s.phase['date'] <= date]
    fut = s.phase[s.phase['date'] > date]
    # Remaining phase fixtures: the matches after `date`, plus (current
    # season) the scheduled ones not played yet. Before 2026-10-26 a
    # current-season date took only the scheduled fixtures, leaving out the
    # matches played between it and today.
    remaining = list(zip(fut['home_team'], fut['away_team']))
    sched = schedule[schedule['date'] > date] if schedule is not None else None
    if sched is not None and len(sched):
        ph_sched = sched[sched['stage'].isin(['GROUP_STAGE', 'LEAGUE_STAGE'])]
        remaining += list(zip(ph_sched['home_team'], ph_sched['away_team']))
    rounds = s.ko_rounds()
    ko_played = s.ko[s.ko['date'] <= date]
    total = len(s.phase) + len(s.ko) + len(remaining) + 1
    frac_left = max(0.0, 1 - (len(played_phase) + len(ko_played)) / max(1, total))
    fixtures = remaining
    sim = Sim(s, strength, date, fixtures, rng, frac_left)
    sim.country = country
    if capture is not None:
        sim.cap = capture.setdefault('matches', {})
    S = N_SIMS
    played = list(zip(played_phase['home_team'], played_phase['away_team'],
                      played_phase['home_score'], played_phase['away_score']))
    single_from = None
    if s.y == 2019 and date >= LISBON_FROM:
        single_from = 1                       # QF onward one match
    phase_over = len(remaining) == 0 and date >= s.phase['date'].max()

    def real_pairs(r_i):
        """Real pairings for knockout round r_i if its draw is known by `date`."""
        if r_i >= len(rounds) or not rounds[r_i]:
            return None
        prev_end = s.phase['date'].max() if r_i == 0 else max(
            leg[0] for a, b, legs in rounds[r_i - 1] for leg in legs) if rounds[r_i - 1] else None
        if prev_end is None or date <= prev_end:
            return None
        return [(a, b, [leg for leg in legs if leg[0] <= date]) for a, b, legs in rounds[r_i]]

    if not s.league_phase:
        # Groups
        group_of, W_, R_ = {}, [], []
        for gi, g in enumerate(s.groups):
            gp = [m for m in played if m[0] in g]
            gr = [m for m in remaining if m[0] in g]
            ordr = _table(sim, g, gp, gr, h2h=True)
            for t in g:
                group_of[sim.idx[t]] = gi
            W_.append(ordr[:, 0]); R_.append(ordr[:, 1])
        winners = np.stack(W_, 1); runners = np.stack(R_, 1)
        rp = real_pairs(0)
        if rp:
            ties = [(np.full(S, sim.idx[a]), np.full(S, sim.idx[b]), legs) for a, b, legs in rp]
        else:
            opp = _draw_r16(sim, winners, runners, group_of)
            ties = [(winners[:, k], opp[:, k], []) for k in range(8)]
        cur = [_host_second(sim, a, b, legs) for a, b, legs in ties]
        nxt = [sim.tie(a, b, legs) for a, b, legs in cur]
        rounds_left = [1, 2, 3]
    else:
        ordr = _table(sim, s.teams, played, remaining, h2h=False)
        rp = real_pairs(0)
        if rp:
            ties = [(np.full(S, sim.idx[a]), np.full(S, sim.idx[b]), legs) for a, b, legs in rp]
            po = _order_by_seed(sim, [((a, b), sim.tie(*_host_second(sim, a, b, legs)))
                                      for a, b, legs in ties], ordr)
        else:
            po = []
            for hi_pair, lo_pair in (((8, 9), (22, 23)), ((10, 11), (20, 21)),
                                     ((12, 13), (18, 19)), ((14, 15), (16, 17))):
                flip = sim.rng.random(S) < 0.5
                for k in (0, 1):
                    a = ordr[:, hi_pair[k]]
                    b = np.where(flip, ordr[:, lo_pair[1 - k]], ordr[:, lo_pair[k]])
                    po.append(sim.tie(a, b, []))
        # Round of 16: seeds 1-8 against playoff winners (1/2 vs the 15-18
        # ties, 3/4 vs 13/14-19/20, 5/6 vs 11/12-21/22, 7/8 vs 9/10-23/24).
        seeds = ordr[:, :8]
        group_for_seed = {0: 6, 1: 7, 2: 4, 3: 5, 4: 2, 5: 3, 6: 0, 7: 1}
        rp1 = real_pairs(1)
        if rp1:
            ties = [(np.full(S, sim.idx[a]), np.full(S, sim.idx[b]), legs) for a, b, legs in rp1]
            cur = [_host_second(sim, a, b, legs) for a, b, legs in ties]
        else:
            cur = [(seeds[:, k], po[group_for_seed[k]], []) for k in range(8)]
        nxt = [sim.tie(a, b, legs) for a, b, legs in cur]
        # Bracket: 1v8, 4v5, 2v7, 3v6 sides.
        nxt = [nxt[i] for i in (0, 7, 3, 4, 1, 6, 2, 5)] if not rp1 else nxt
        rounds_left = [2, 3, 4]
    # After real round-of-16 pairings in the league-phase era the bracket
    # isn't in the data, so later rounds are drawn openly.
    open_draw = (not s.league_phase) or bool(real_pairs(1))
    champs = _play_out(sim, s, nxt, rounds_left, real_pairs, single_from, open_draw=open_draw)
    if capture is not None:
        capture.update(teams=sim.teams, champ=champs, E=sim.E, O=sim.O, D=sim.D, hfa=sim.hfa)
    p = np.bincount(champs, minlength=len(sim.teams)) / S
    return {t: float(p[i]) for t, i in sim.idx.items() if t in set(s.teams)}


def _host_second(sim, a, b, legs):
    """(club hosting leg 2, other club, legs) for a real or simulated tie."""
    if legs:
        first_home = legs[0][1]
        ia, ib = a[0], b[0]
        host2 = ib if sim.teams[ia] == first_home else ia
        other = ia if host2 == ib else ib
        return (np.full(N_SIMS, host2), np.full(N_SIMS, other), legs)
    return (a, b, legs)


def _order_by_seed(sim, ties, ordr):
    """Real playoff-round winners ordered by the league-phase place of the
    higher-placed club in each tie (9th first), to line up with the bracket."""
    place = {int(c): k for k, c in enumerate(ordr[0])}     # league phase is final by then
    best = [min(place[int(pa[0])], place[int(pb[0])]) for (pa, pb), _w in ties]
    return [w for _, w in sorted(zip(best, [w for _p, w in ties]), key=lambda z: z[0])]


def _play_out(sim, s, alive, rounds_left, real_pairs, single_from, open_draw):
    """Play the remaining knockout rounds from per-slot winners `alive`."""
    S = N_SIMS
    final_round = rounds_left[-1]
    for r_i in rounds_left:
        rp = real_pairs(r_i)
        final = r_i == final_round
        single = single_from is not None and r_i >= (single_from if not s.league_phase else 99)
        if rp:
            nxt = []
            for a, b, legs in rp:
                A, B, L = _host_second(sim, np.full(S, sim.idx[a]), np.full(S, sim.idx[b]), legs)
                nxt.append(sim.tie(A, B, L, single=single, final=final))
            alive = nxt
            continue
        slots = np.stack(alive, 1)
        if open_draw:
            perm = np.argsort(sim.rng.random(slots.shape), axis=1)
            slots = np.take_along_axis(slots, perm, 1)
        alive = [sim.tie(slots[:, 2 * k], slots[:, 2 * k + 1], [], single=single, final=final)
                 for k in range(slots.shape[1] // 2)]
    return alive[0]


def season_odds(cl_games, euro_games, ratings, big5_teams, country, year, schedule=None):
    """DataFrame of date, team, cl_win for one season."""
    x = cl_games[cl_games['comp_season'] == f"{year}-{str(year + 1)[-2:]}"]
    s = Season(x, year)
    strength = Strength(ratings, euro_games, big5_teams)
    days = sorted(set(pd.to_datetime(x[x['date'] >= s.phase_start]['date']).dt.normalize()))
    out = []
    for d in [s.phase_start - pd.Timedelta(days=1)] + days:
        for t, p in odds_at(s, strength, d, schedule, country).items():
            out.append((d, f"{year}-{str(year + 1)[-2:]}", t, p))
    return pd.DataFrame(out, columns=['date', 'season', 'team', 'cl_win'])


# ── Cache: finished seasons are computed once ───────────────────────────────
import hashlib as _hashlib
import multiprocessing as _mp
import os as _os
import pickle as _pickle

_JOB = {}


def _inputs(cl, euro, ratings, year):
    lab = f"{year}-{str(year + 1)[-2:]}"
    x = cl[cl['comp_season'] == lab]
    lo_d, hi_d = x['date'].min() - pd.Timedelta(days=NONBIG5_WINDOW_DAYS + 30), x['date'].max()
    e = euro[(euro['date'] >= lo_d) & (euro['date'] <= hi_d)]
    teams = set(e['home_team']) | set(e['away_team'])
    r = ratings[ratings['team'].isin(teams) & (ratings['date'] >= lo_d) & (ratings['date'] <= hi_d)]
    return x, e, r


def _fingerprint(cl, euro, ratings, year, schedule):
    here = _os.path.dirname(_os.path.abspath(__file__))
    h = _hashlib.sha256()
    for mod in ('cl_odds.py', 'league_odds.py', 'zidane_seasons.py'):
        h.update(open(_os.path.join(here, mod), 'rb').read())
    x, e, r = _inputs(cl, euro, ratings, year)
    cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'shootout_winner', 'competition']
    h.update(e[cols].sort_values(['date', 'home_team']).to_csv(index=False).encode())
    r = r[['date', 'team', 'rating_o', 'rating_d']].sort_values(['date', 'team']).copy()
    r[['rating_o', 'rating_d']] = r[['rating_o', 'rating_d']].round(3)
    h.update(r.to_csv(index=False).encode())
    if schedule is not None:
        h.update(schedule.sort_values(['date', 'home_team']).to_csv(index=False).encode())
    return h.hexdigest()


def _one(year):
    j = _JOB
    return year, season_odds(j['cl'], j['euro'], j['ratings'], j['big5'], j['country'], year,
                             schedule=j['schedule'] if year == j['current'] else None)


def compute_cached(games, ratings, schedule=None, cache_dir='cl_odds_cache', workers=None, log=print):
    """CL winner odds for every season from FIRST_SEASON (date, season, team,
    cl_win), reusing cached seasons whose inputs haven't changed."""
    _os.makedirs(cache_dir, exist_ok=True)
    g = games.dropna(subset=['home_score', 'away_score']).copy()
    g['date'] = pd.to_datetime(g['date'])
    ratings = ratings[['date', 'team', 'rating_o', 'rating_d']].copy()
    ratings['date'] = pd.to_datetime(ratings['date'])
    ratings = ratings.drop_duplicates(['date', 'team']).sort_values('date')
    L = g[g['competition'].isin(BIG5)]
    big5 = set(L['home_team']) | set(L['away_team'])
    country = {t: lg for lg, x in L.groupby('competition') for t in set(x['home_team']) | set(x['away_team'])}
    cl = g[g['competition'] == 'Champions League']
    euro = g[g['competition'].isin(EURO)]
    years = sorted({int(s[:4]) for s in cl['comp_season'].dropna().unique() if int(s[:4]) >= FIRST_SEASON})
    current = max(years)
    if schedule is not None and len(schedule):
        schedule = schedule.copy()
        schedule['date'] = pd.to_datetime(schedule['date'])
        schedule = schedule[schedule['comp_season'] == f"{current}-{str(current + 1)[-2:]}"]
    results, todo, sigs = {}, [], {}
    for y in years:
        sigs[y] = _fingerprint(cl, euro, ratings, y, schedule if y == current else None)
        path = _os.path.join(cache_dir, f'{y}.pkl')
        if _os.path.exists(path):
            try:
                sig, payload = _pickle.load(open(path, 'rb'))
                if sig == sigs[y]:
                    results[y] = payload
                    continue
            except Exception:
                pass
        todo.append(y)
    log(f"  CL odds: {len(results)} seasons from cache, computing {len(todo)}")
    if todo:
        _JOB.update(cl=cl, euro=euro, ratings=ratings, big5=big5, country=country,
                    schedule=schedule, current=current)
        with _mp.get_context('fork').Pool(workers or _os.cpu_count()) as pool:
            for y, payload in pool.imap_unordered(_one, todo):
                results[y] = payload
                _pickle.dump((sigs[y], payload), open(_os.path.join(cache_dir, f'{y}.pkl'), 'wb'))
    return pd.concat([results[y] for y in years], ignore_index=True)
