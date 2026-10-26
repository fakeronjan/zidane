"""Daily Matchups: every Big-5 league match of a day, previewed from the
ratings going into it - home / draw / away probabilities (the pick is the
most likely of the three), ZIDANE's goal line and O/U, and the stakes (each
club's title, relegation and, 2024-25 on, Champions League qualification
odds with a win, a draw, and a loss). After the match: the final score and
whether ZIDANE's pick was right. The European port of DUNCAN's Daily
Matchups (duncan/daily_matchups.py).

Champions League matches (2003-04 on) come from the CL sim (cl_odds.py),
run from the previous CL match day exactly as the published CL odds are:
the match's 90-minute goals give its probabilities, line and O/U, and the
stakes are each club's odds of winning the CL with a win, a draw, and a
loss (split by the simulated result).

Everything comes from the league sim (league_odds.py), run exactly as the
published odds are (same dates, same seed), so a match's stakes start from
the odds the Standings show for the day before. Its Poisson goals give each
match's three probabilities, the line (mean goal margin, to the nearest
0.25; Pick'em when it names a different favorite than the win
probabilities) and the O/U (mean total goals, to the nearest 0.5). 1992-93
on, like the league odds.
"""
import hashlib
import multiprocessing as _mp
import os
import pickle

import numpy as np
import pandas as pd

import cl_odds
import league_odds
from zidane_rules import points_per_win

N_SIMS = league_odds.N_SIMS


def _record(w, d, l):
    return f"{w}-{d}-{l}"


def _per_point(y, pts, ctrl, sign):
    """(mean of y, change in y per extra league point) for outcome y (per-sim
    bool or float) and the club's final points in each sim, holding both
    clubs' drift fixed (DUNCAN's value of one more win). sign: +1 for odds
    that rise with points (title, CL places), -1 for relegation."""
    y = y.astype(float)
    m = float(y.mean())
    if np.ptp(pts) == 0:
        return m, 0.0
    X = np.column_stack([np.ones(len(y)), pts] + ([ctrl] if ctrl is not None else []))
    b = float(np.linalg.lstsq(X, y, rcond=None)[0][1])
    return m, (max(b, 0.0) if sign > 0 else min(b, 0.0))


def _wdl(m, b, pw, pd_, win_pts):
    """Mean m split into (win, draw, loss) for +b per point."""
    clip = lambda v: min(max(v, 0.0), 1.0)
    ept = win_pts * pw + pd_
    return clip(m + b * (win_pts - ept)), clip(m + b * (1 - ept)), clip(m - b * ept)


def _line(margin, fav_home):
    line = round(float(margin) * 4) / 4
    if line != 0 and (line > 0) != fav_home:
        return 0.0
    return line


def build_league_season(league, season, L, schedule, ratings, current, kickoffs, log=print):
    """One league season's days. L: league_odds.league_games output for this
    league (played matches, with league_season); schedule: unplayed matches
    of the current season (date, home_team, away_team, kickoff) or None;
    ratings: (date, team, rating, rank, rating_o, rating_d)."""
    S = L[L['league_season'] == season]
    if S.empty:
        return []
    seasons = sorted(L['league_season'].unique())
    i = seasons.index(season)
    prev = set()
    if i > 0:
        P = L[L['league_season'] == seasons[i - 1]]
        prev = set(P['home_team']) | set(P['away_team'])
    is_cur = season == current
    sim = league_odds.SeasonSim(league, season, S, ratings[['date', 'team', 'rating_o', 'rating_d']],
                                complete=not is_cur)
    sim.promoted = {t: (t not in prev and i > 0) for t in sim.teams}
    win_pts = points_per_win(league, season)
    rng = np.random.default_rng(0)                    # league_odds.season_odds' seed and date order
    dates = [S['date'].min() - pd.Timedelta(days=1)] + sorted(S['date'].dt.normalize().unique())
    played = S.assign(day=S['date'].dt.normalize())
    upcoming = None
    if is_cur and schedule is not None and len(schedule):
        upcoming = schedule[schedule['home_team'].isin(sim.idx) & schedule['away_team'].isin(sim.idx)]
        upcoming = upcoming.assign(day=pd.to_datetime(upcoming['date']).dt.normalize())
        upcoming = upcoming[upcoming['day'] > S['date'].max().normalize()]
    rsnap = ratings.sort_values('date')

    def snap_for(d):
        r = rsnap[rsnap['date'] <= d]
        return r.groupby('team').tail(1).set_index('team')

    days_out = []
    for k, d in enumerate(dates):
        nxt = dates[k + 1] if k + 1 < len(dates) else None
        if nxt is not None:
            day_rows = played[played['day'] == nxt]
        elif upcoming is not None and len(upcoming):
            nxt = upcoming['day'].min()                 # live season: the next match day
            day_rows = upcoming[upcoming['day'] == nxt].assign(home_score=np.nan, away_score=np.nan)
        else:
            day_rows = None
        mu = league_odds.league_mu(L, league, pd.Timestamp(d))
        cap = {} if day_rows is not None and len(day_rows) else None
        sim.odds_at(d, mu, rng, capture=cap)
        if cap is None or 'hg' not in cap:
            continue
        idx = {t: j for j, t in enumerate(cap['teams'])}
        pos = cap['pos']
        title = pos == 1
        releg = np.isin(pos, cap['direct']) + cap['p_down'] * np.isin(pos, cap['playoff'])
        qual = (pos <= cap['places']) if cap['places'] else None
        fx_index = {}
        for j, (h, a) in enumerate(cap['fixtures']):
            fx_index.setdefault((h, a), j)
        rt = snap_for(pd.Timestamp(d))
        done = S[S['date'] <= d]
        rec = {}
        for x in done.itertuples(index=False):
            for t, us, them in ((x.home_team, x.home_score, x.away_score), (x.away_team, x.away_score, x.home_score)):
                w, dd, l = rec.get(t, (0, 0, 0))
                rec[t] = (w + (us > them), dd + (us == them), l + (us < them))
        value = {}

        def per_point(t, opp):
            if t not in value:
                it = idx[t]
                ctrl = cap['E'][:, [it, idx[opp]]] if cap['E'] is not None else None
                pts = cap['PTS'][:, it]
                value[t] = (_per_point(title[:, it], pts, ctrl, 1), _per_point(releg[:, it], pts, ctrl, -1),
                            None if qual is None else _per_point(qual[:, it], pts, ctrl, 1))
            return value[t]

        games_out = []
        for x in day_rows.itertuples(index=False):
            h, a = x.home_team, x.away_team
            j = fx_index.get((h, a))
            if j is None:
                continue
            hg, ag = cap['hg'][:, j], cap['ag'][:, j]
            p_h, p_d, p_a = float((hg > ag).mean()), float((hg == ag).mean()), float((ag > hg).mean())
            stakes = {}
            for t, opp, pw in ((h, a, p_h), (a, h, p_a)):
                (ti_m, ti_b), (re_m, re_b), q = per_point(t, opp)
                ti = _wdl(ti_m, ti_b, pw, p_d, win_pts)
                re_ = _wdl(re_m, re_b, pw, p_d, win_pts)
                s = {'title_win': ti[0], 'title_draw': ti[1], 'title_loss': ti[2],
                     'releg_win': re_[0], 'releg_draw': re_[1], 'releg_loss': re_[2]}
                if q is not None:
                    qq = _wdl(q[0], q[1], pw, p_d, win_pts)
                    s.update(qual_win=qq[0], qual_draw=qq[1], qual_loss=qq[2])
                stakes['home' if t == h else 'away'] = s
            blank = pd.Series({'rating': 0.0, 'rank': np.nan})
            rh = rt.loc[h] if h in rt.index else blank
            ra = rt.loc[a] if a in rt.index else blank
            game = {
                'home': h, 'away': a, 'league': league,
                'home_record': _record(*rec.get(h, (0, 0, 0))), 'away_record': _record(*rec.get(a, (0, 0, 0))),
                'home_rank': None if pd.isna(rh['rank']) else int(rh['rank']),
                'away_rank': None if pd.isna(ra['rank']) else int(ra['rank']),
                'p_home': round(p_h, 4), 'p_draw': round(p_d, 4), 'p_away': round(p_a, 4),
                'line': _line(float((hg - ag).mean()), p_h >= p_a),
                'total': round(float((hg + ag).mean()) * 2) / 2,
                'kickoff': kickoffs.get((pd.Timestamp(x.date).normalize(), h, a)),
                'stakes': {sd: {kk: round(vv, 4) for kk, vv in v.items()} for sd, v in stakes.items()},
                'result': None if pd.isna(x.home_score) else {'home': int(x.home_score), 'away': int(x.away_score)},
                '_q': (min(rh['rating'], ra['rating']), max(rh['rating'], ra['rating'])),
                '_frac': len(done) / max(1, len(done) + len(cap['fixtures'])),
            }
            games_out.append(game)
        if games_out:
            days_out.append({'date': pd.Timestamp(nxt).strftime('%Y-%m-%d'), 'games': games_out})
    return days_out


def build_cl_season(year, cl, euro, ratings, big5, country, schedule, kickoffs):
    """One CL season's days (cl_odds inputs, as cl_odds.compute_cached
    prepares them). schedule: the current season's unplayed fixtures or None."""
    lab = f"{year}-{str(year + 1)[-2:]}"
    x = cl[cl['comp_season'] == lab]
    if x.empty:
        return []
    s = cl_odds.Season(x, year)
    strength = cl_odds.Strength(ratings, euro, big5)
    played_days = sorted(set(pd.to_datetime(x[x['date'] >= s.phase_start]['date']).dt.normalize()))
    days = [s.phase_start - pd.Timedelta(days=1)] + played_days
    rows = x.assign(day=x['date'].dt.normalize())
    upcoming = None
    if schedule is not None and len(schedule):
        upcoming = schedule.assign(day=pd.to_datetime(schedule['date']).dt.normalize())
        upcoming = upcoming[upcoming['day'] > (played_days[-1] if played_days else s.phase_start)]
    rsnap = ratings.sort_values('date')
    out = []
    for k, d in enumerate(days):
        if k + 1 < len(days):
            nxt = days[k + 1]
            day_rows = rows[rows['day'] == nxt]
        elif upcoming is not None and len(upcoming):
            nxt = upcoming['day'].min()
            day_rows = upcoming[upcoming['day'] == nxt].assign(home_score=np.nan, away_score=np.nan)
        else:
            break
        cap = {}
        cl_odds.odds_at(s, strength, d, schedule, country, capture=cap)
        idx = {t: j for j, t in enumerate(cap['teams'])}
        champ = cap['champ']
        r = rsnap[rsnap['date'] <= d].groupby('team').tail(1).set_index('team')
        done = s.phase[s.phase['date'] <= d]
        rec = {}
        for m in done.itertuples(index=False):
            for t, us, them in ((m.home_team, m.home_score, m.away_score), (m.away_team, m.away_score, m.home_score)):
                w, dd, l = rec.get(t, (0, 0, 0))
                rec[t] = (w + (us > them), dd + (us == them), l + (us < them))
        games_out = []
        for m in day_rows.itertuples(index=False):
            h, a = m.home_team, m.away_team
            got = cap['matches'].get((h, a))
            flip = False
            if got is None and (a, h) in cap['matches']:
                got, flip = cap['matches'][(a, h)], True     # neutral final, listed the other way round
            if got is None or h not in idx or a not in idx:
                continue
            hg, ag = (got[1], got[0]) if flip else got
            p_h, p_d, p_a = float((hg > ag).mean()), float((hg == ag).mean()), float((ag > hg).mean())
            stakes = {}
            for t, won, drew in ((h, hg > ag, hg == ag), (a, ag > hg, hg == ag)):
                y = champ == idx[t]
                part = lambda msk: float(y[msk].mean()) if msk.any() else None
                stakes['home' if t == h else 'away'] = {
                    'cl_win': part(won), 'cl_draw': part(drew), 'cl_loss': part(~won & ~drew)}
            blank = pd.Series({'rating': np.nan, 'rank': np.nan})
            rh = r.loc[h] if h in r.index else blank
            ra = r.loc[a] if a in r.index else blank
            qh = strength.od(h, d); qa = strength.od(a, d)
            game = {
                'home': h, 'away': a, 'league': 'Champions League',
                'home_record': _record(*rec.get(h, (0, 0, 0))), 'away_record': _record(*rec.get(a, (0, 0, 0))),
                'home_rank': None if pd.isna(rh['rank']) else int(rh['rank']),
                'away_rank': None if pd.isna(ra['rank']) else int(ra['rank']),
                'p_home': round(p_h, 4), 'p_draw': round(p_d, 4), 'p_away': round(p_a, 4),
                'line': _line(float((hg - ag).mean()), p_h >= p_a),
                'total': round(float((hg + ag).mean()) * 2) / 2,
                'kickoff': getattr(m, 'kickoff', None) if isinstance(getattr(m, 'kickoff', None), str)
                           else kickoffs.get((pd.Timestamp(m.date).normalize(), h, a)),
                'stakes': {sd: {kk: (None if vv is None else round(vv, 4)) for kk, vv in v.items()}
                           for sd, v in stakes.items()},
                'result': None if pd.isna(m.home_score) else {'home': int(m.home_score), 'away': int(m.away_score)},
                # Quality from the CL sim's own strengths (O + D), which cover
                # non-Big-5 clubs too.
                '_q': (min(qh[0] + qh[1], qa[0] + qa[1]), max(qh[0] + qh[1], qa[0] + qa[1])),
                '_frac': 1.0,
            }
            if not pd.isna(m.home_score) and isinstance(getattr(m, 'shootout_winner', None), str):
                game['result']['pens'] = 'home' if m.shootout_winner == h else 'away'
            games_out.append(game)
        if games_out:
            out.append({'date': pd.Timestamp(nxt).strftime('%Y-%m-%d'), 'games': games_out})
    return out


# ── Juice: Quality x Stakes ──────────────────────────────────────────────────
# As DUNCAN, after DILLON: Quality = 35% the worse club's rating, 35% the
# better club's, 30% closeness (1 - |P(home) - P(away)|); Stakes = both clubs'
# relegation and CL-qualification swings (win vs loss) + K x their title
# swing, K ramping 4 -> 8 over the league season. Each is ranked 0-100 against
# completed seasons and Juice = sqrt(Quality x Stakes).
TITLE_K_START, TITLE_K_END = 4.0, 8.0


def _pct(ref_sorted, x):
    lo = np.searchsorted(ref_sorted, x, side='left')
    hi = np.searchsorted(ref_sorted, x, side='right')
    return ((lo + hi + 1) / 2) / len(ref_sorted)


def add_juice(seasons, pool_seasons):
    """seasons: {season label: days}. Adds quality / stakes_score / juice."""
    rows = []
    for season, days in seasons.items():
        for d in days:
            for g in d['games']:
                qmin, qmax = g.pop('_q')
                frac = g.pop('_frac')
                k = TITLE_K_START + (TITLE_K_END - TITLE_K_START) * min(frac, 1.0)
                close = 1 - abs(g['p_home'] - g['p_away'])
                st = list(g['stakes'].values())
                if 'cl_win' in st[0]:
                    stake = sum(TITLE_K_END * ((s['cl_win'] or 0) - (s['cl_loss'] or 0)) for s in st)
                else:
                    stake = sum(k * (s['title_win'] - s['title_loss']) + (s['releg_loss'] - s['releg_win']) +
                                (s.get('qual_win', 0) - s.get('qual_loss', 0)) for s in st)
                rows.append((g, season in pool_seasons, qmin, qmax, close, stake))
    if not rows:
        return
    df = pd.DataFrame([r[1:] for r in rows], columns=['pool', 'qmin', 'qmax', 'close', 'stake'])
    ref = df[df['pool']] if df['pool'].any() else df
    parts = {c: np.sort(ref[c].to_numpy()) for c in ('qmin', 'qmax', 'close')}
    q_raw = sum(w * _pct(parts[c], df[c].to_numpy()) for c, w in (('qmin', .35), ('qmax', .35), ('close', .3)))
    mask = df['pool'].to_numpy() if df['pool'].any() else np.ones(len(df), bool)
    q = _pct(np.sort(q_raw[mask]), q_raw) * 100
    st_ref = np.sort(ref['stake'].to_numpy())
    for (g, *_), qq, ss in zip(rows, q, df['stake']):
        s_pct = min(_pct(st_ref, ss) * 100, 100)
        g['quality'] = int(round(min(qq, 100)))
        g['stakes_score'] = int(round(s_pct))
        g['juice'] = int(round(np.sqrt(min(qq, 100) * s_pct)))


# ── Cache: one entry per league season ───────────────────────────────────────
CACHE_DIR = 'matchups_cache'
_HERE = os.path.dirname(os.path.abspath(__file__))
_JOB = {}


def _fingerprint(key, J):
    league, season = key
    h = hashlib.sha256()
    for f in ('daily_matchups.py', 'league_odds.py', 'zidane_rules.py', 'zidane_seasons.py'):
        h.update(open(os.path.join(_HERE, f), 'rb').read())
    L = J['L'][J['L']['competition'] == league]
    S = L[L['league_season'] == season]
    lg = L[(L['date'] > S['date'].min() - pd.Timedelta(days=400)) & (L['date'] <= S['date'].max())]
    h.update(lg[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'league_season']]
             .sort_values(['date', 'home_team']).to_csv(index=False).encode())
    teams = set(S['home_team']) | set(S['away_team'])
    r = J['ratings'][J['ratings']['team'].isin(teams) & (J['ratings']['date'] >= S['date'].min() - pd.Timedelta(days=400)) &
                     (J['ratings']['date'] <= S['date'].max())]
    r = r.sort_values(['date', 'team']).copy()
    for c in ('rating', 'rating_o', 'rating_d'):
        r[c] = r[c].round(3)
    h.update(r.to_csv(index=False).encode())
    if season == J['current'] and J['schedule'] is not None:
        h.update(J['schedule'][J['schedule']['competition'] == league].sort_values(['date', 'home_team'])
                 .to_csv(index=False).encode())
    h.update(repr(sorted((str(k), v) for k, v in J['kickoffs'].items()
                         if k[1] in teams and S['date'].min() <= k[0] <= S['date'].max() + pd.Timedelta(days=60))).encode())
    h.update(repr(season == J['current']).encode())
    return h.hexdigest()


def _one(key):
    league, season = key
    J = _JOB
    L = J['L'][J['L']['competition'] == league]
    S = L[L['league_season'] == season]
    teams = set(S['home_team']) | set(S['away_team'])
    r = J['ratings'][J['ratings']['team'].isin(teams) & (J['ratings']['date'] >= S['date'].min() - pd.Timedelta(days=400))]
    sch = None if J['schedule'] is None else J['schedule'][J['schedule']['competition'] == league]
    return key, build_league_season(league, season, L, sch, r, J['current'], J['kickoffs'], log=lambda *_: None)


def build_cached(games, ratings, schedule, kickoffs, workers=None, log=print):
    """{(league, season label): days} for every Big-5 league season."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    L = league_odds.league_games(games)
    current = L['league_season'].max()
    ratings = ratings[['date', 'team', 'rating', 'rank', 'rating_o', 'rating_d']].copy()
    ratings['date'] = pd.to_datetime(ratings['date'])
    ratings = ratings.drop_duplicates(['date', 'team']).sort_values('date')
    _JOB.update(L=L, ratings=ratings, current=current, schedule=schedule, kickoffs=kickoffs)
    keys = sorted({(c, s) for c, s in zip(L['competition'], L['league_season'])})
    out, todo, sigs = {}, [], {}
    for k in keys:
        sigs[k] = _fingerprint(k, _JOB)
        path = os.path.join(CACHE_DIR, f"{k[0].replace(' ', '_')}_{k[1]}.pkl")
        if os.path.exists(path):
            try:
                sig, days = pickle.load(open(path, 'rb'))
                if sig == sigs[k]:
                    out[k] = days
                    continue
            except Exception:
                pass
        todo.append(k)
    log(f'  daily matchups: {len(keys) - len(todo)} league seasons from cache, computing {len(todo)}')
    if todo:
        with _mp.get_context('fork').Pool(workers or os.cpu_count()) as pool:
            for k, days in pool.imap_unordered(_one, sorted(todo, key=lambda k: k[1], reverse=True)):
                out[k] = days
                pickle.dump((sigs[k], days), open(os.path.join(CACHE_DIR, f"{k[0].replace(' ', '_')}_{k[1]}.pkl"), 'wb'))
    return out, current


def _cl_one(year):
    J = _JOB
    return year, build_cl_season(year, J['cl'], J['euro'], J['cl_ratings'], J['big5'], J['country'],
                                 J['cl_schedule'] if year == J['cl_current'] else None, J['kickoffs'])


def _cl_fingerprint(year, J):
    h = hashlib.sha256()
    for f in ('daily_matchups.py', 'cl_odds.py', 'league_odds.py', 'zidane_seasons.py'):
        h.update(open(os.path.join(_HERE, f), 'rb').read())
    x, e, r = cl_odds._inputs(J['cl'], J['euro'], J['cl_ratings'], year)
    cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'shootout_winner', 'competition']
    h.update(e[cols].sort_values(['date', 'home_team']).to_csv(index=False).encode())
    r = J['ratings'][J['ratings']['date'].between(x['date'].min() - pd.Timedelta(days=760), x['date'].max())]
    r = r.sort_values(['date', 'team']).copy()
    for c in ('rating', 'rating_o', 'rating_d'):
        r[c] = r[c].round(3)
    h.update(r.to_csv(index=False).encode())
    if year == J['cl_current'] and J['cl_schedule'] is not None:
        h.update(J['cl_schedule'].sort_values(['date', 'home_team']).to_csv(index=False).encode())
    return h.hexdigest()


def build_cl_cached(games, ratings, schedule, kickoffs, workers=None, log=print):
    """{season label: days} of Champions League matches, 2003-04 on."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    g = games.dropna(subset=['home_score', 'away_score']).copy()
    g['date'] = pd.to_datetime(g['date'])
    r_full = ratings[['date', 'team', 'rating', 'rank', 'rating_o', 'rating_d']].copy()
    r_full['date'] = pd.to_datetime(r_full['date'])
    r_full = r_full.drop_duplicates(['date', 'team']).sort_values('date')
    L = g[g['competition'].isin(league_odds.BIG5)]
    big5 = set(L['home_team']) | set(L['away_team'])
    country = {t: lg for lg, x in L.groupby('competition') for t in set(x['home_team']) | set(x['away_team'])}
    cl = g[g['competition'] == 'Champions League']
    euro = g[g['competition'].isin(cl_odds.EURO)]
    years = sorted({int(s[:4]) for s in cl['comp_season'].dropna().unique() if int(s[:4]) >= cl_odds.FIRST_SEASON})
    current = max(years)
    if schedule is not None and len(schedule):
        schedule = schedule.copy()
        schedule['date'] = pd.to_datetime(schedule['date'])
        schedule = schedule[schedule['comp_season'] == f"{current}-{str(current + 1)[-2:]}"]
    # The CL sim reads only rating_o / rating_d; the full table also gives
    # each club's ZIDANE rank for the matchup.
    _JOB.update(cl=cl, euro=euro, cl_ratings=r_full, ratings=r_full,
                big5=big5, country=country, cl_schedule=schedule, cl_current=current, kickoffs=kickoffs)
    out, todo, sigs = {}, [], {}
    for y in years:
        sigs[y] = _cl_fingerprint(y, _JOB)
        path = os.path.join(CACHE_DIR, f'CL_{y}.pkl')
        if os.path.exists(path):
            try:
                sig, days = pickle.load(open(path, 'rb'))
                if sig == sigs[y]:
                    out[y] = days
                    continue
            except Exception:
                pass
        todo.append(y)
    log(f'  daily matchups: {len(years) - len(todo)} CL seasons from cache, computing {len(todo)}')
    if todo:
        with _mp.get_context('fork').Pool(workers or os.cpu_count()) as pool:
            for y, days in pool.imap_unordered(_cl_one, sorted(todo, reverse=True)):
                out[y] = days
                pickle.dump((sigs[y], days), open(os.path.join(CACHE_DIR, f'CL_{y}.pkl'), 'wb'))
    return {f"{y}-{str(y + 1)[-2:]}": out[y] for y in years}


def merge_by_season(per_league):
    """{(league, season): days} -> {season: days, all leagues merged by date}."""
    merged = {}
    for (league, season), days in per_league.items():
        by = merged.setdefault(season, {})
        for d in days:
            by.setdefault(d['date'], []).extend(d['games'])
    return {s: [{'date': d, 'games': g} for d, g in sorted(by.items())] for s, by in merged.items()}
