"""Datos SINTÉTICOS solo para probar el código (no son resultados reales).

El "mercado" sintético se construye con las probabilidades verdaderas más ruido,
por lo que aquí el mercado debería ganarle o igualar al modelo. Esto valida la
tubería, NO la capacidad predictiva con datos reales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson


def _round_robin(n: int):
    teams = list(range(n))
    rounds = []
    for _ in range(n - 1):
        rounds.append([(teams[i], teams[n - 1 - i]) for i in range(n // 2)])
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds + [[(b, a) for a, b in r] for r in rounds]


def make_synthetic(n_teams=20, n_seasons=5, seed=1, market_noise=0.04, margin=0.05, start_year=2018):
    rng = np.random.default_rng(seed)
    att = rng.normal(0, 0.25, n_teams)
    dfn = rng.normal(0, 0.25, n_teams)
    gamma, mu0 = 0.25, 0.2
    names = [f"Team{i:02d}" for i in range(n_teams)]
    g = np.arange(11)
    rows = []
    for s in range(n_seasons):
        att = att + rng.normal(0, 0.05, n_teams)
        dfn = dfn + rng.normal(0, 0.05, n_teams)
        att, dfn = att - att.mean(), dfn - dfn.mean()
        season_start = pd.Timestamp(year=start_year + s, month=8, day=10)
        for r, matches in enumerate(_round_robin(n_teams)):
            for hi, ai in matches:
                lam = np.exp(mu0 + att[hi] + dfn[ai] + gamma)
                mu = np.exp(mu0 + att[ai] + dfn[hi])
                hg, ag = rng.poisson(lam), rng.poisson(mu)
                m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
                p = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
                p = p * np.exp(rng.normal(0, market_noise, 3))
                p = p / p.sum()
                o = 1.0 / (p * (1 + margin))
                rows.append(
                    (season_start + pd.Timedelta(days=7 * r + int(rng.integers(0, 3))), "SYN",
                     f"{start_year + s}", names[hi], names[ai], hg, ag, o[0], o[1], o[2], "sintético")
                )
    cols = ["date", "div", "season", "home", "away", "hg", "ag", "o_h", "o_d", "o_a", "odds_source"]
    return pd.DataFrame(rows, columns=cols).sort_values("date").reset_index(drop=True)
