"""Modelo Dixon-Coles (Poisson con corrección de marcadores bajos y peso temporal).

lambda (goles local) = exp(mu0 + ataque_local + defensa_visitante + ventaja_local)
mu     (goles visita) = exp(mu0 + ataque_visita + defensa_local)

"defensa" alta = concede más. Ataque y defensa se centran (media 0) para que el
modelo sea identificable. Un pequeño ridge (reg) estabiliza equipos con pocos partidos.
La verosimilitud y su gradiente son analíticos (ver tests/test_dixon_coles.py).

Filas de la matriz de marcadores = goles del local; columnas = goles del visitante.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

TAU_FLOOR = 1e-6


def _tau_terms(x, y, lam, mu, rho):
    """tau de Dixon-Coles y derivadas de log(tau) respecto a log(lam), log(mu) y rho."""
    tau = np.ones_like(lam)
    d_ll = np.zeros_like(lam)
    d_lm = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    ok = tau > TAU_FLOOR
    t = np.where(ok, tau, 1.0)
    d_ll[m00] = -lam[m00] * mu[m00] * rho / t[m00]
    d_lm[m00] = d_ll[m00]
    d_rho[m00] = -lam[m00] * mu[m00] / t[m00]
    d_ll[m01] = lam[m01] * rho / t[m01]
    d_rho[m01] = lam[m01] / t[m01]
    d_lm[m10] = mu[m10] * rho / t[m10]
    d_rho[m10] = mu[m10] / t[m10]
    d_rho[m11] = -1.0 / t[m11]
    for d in (d_ll, d_lm, d_rho):
        d[~ok] = 0.0
    return np.maximum(tau, TAU_FLOOR), d_ll, d_lm, d_rho


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Distribución conjunta de marcadores (filas = local, columnas = visitante)."""
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= max(1 - lam * mu * rho, TAU_FLOOR)
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0.0, None)
    return m / m.sum()


@dataclass
class DCModel:
    teams: list
    attack: np.ndarray
    defense: np.ndarray
    home_adv: float
    rho: float
    mu0: float
    xi: float
    n_train: int
    converged: bool

    def __post_init__(self):
        self.idx = {t: i for i, t in enumerate(self.teams)}
        # Equipo no visto (p. ej. ascendido): ataque débil y defensa floja del plantel
        self.att_new = float(np.quantile(self.attack, 0.25))
        self.def_new = float(np.quantile(self.defense, 0.75))

    def knows(self, team: str) -> bool:
        return team in self.idx

    def _params(self, team: str):
        i = self.idx.get(team)
        if i is None:
            return self.att_new, self.def_new
        return float(self.attack[i]), float(self.defense[i])

    def expected_goals(self, home: str, away: str):
        ah, dh = self._params(home)
        aa, da = self._params(away)
        lam = float(np.exp(self.mu0 + ah + da + self.home_adv))
        mu = float(np.exp(self.mu0 + aa + dh))
        return lam, mu

    def score_matrix(self, home: str, away: str, max_goals: int = 10) -> np.ndarray:
        lam, mu = self.expected_goals(home, away)
        return score_matrix(lam, mu, self.rho, max_goals)


def fit(matches: pd.DataFrame, ref_date, xi: float = 0.0023, reg: float = 0.05) -> DCModel:
    """Ajusta el modelo con partidos (columnas date, home, away, hg, ag).

    xi: decaimiento diario del peso (0.0023 ~ vida media de ~300 días).
    ref_date: fecha de referencia; los partidos más antiguos pesan menos.
    """
    teams = sorted(set(matches["home"]) | set(matches["away"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    h = matches["home"].map(idx).to_numpy()
    a = matches["away"].map(idx).to_numpy()
    x = matches["hg"].to_numpy(float)
    y = matches["ag"].to_numpy(float)
    age = (pd.Timestamp(ref_date) - matches["date"]).dt.days.to_numpy(float)
    w = np.exp(-xi * np.clip(age, 0, None))

    def unpack(theta):
        att = theta[:n] - theta[:n].mean()
        dfn = theta[n : 2 * n] - theta[n : 2 * n].mean()
        return att, dfn, theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]

    def objective(theta):
        att, dfn, gamma, rho, mu0 = unpack(theta)
        log_lam = mu0 + att[h] + dfn[a] + gamma
        log_mu = mu0 + att[a] + dfn[h]
        lam, mu = np.exp(log_lam), np.exp(log_mu)
        tau, d_ll, d_lm, d_rho = _tau_terms(x, y, lam, mu, rho)
        ll = w * (np.log(tau) + x * log_lam - lam + y * log_mu - mu)
        f = -ll.sum() + reg * (att @ att + dfn @ dfn)
        g_ll = w * (d_ll + x - lam)
        g_lm = w * (d_lm + y - mu)
        g_att = np.bincount(h, g_ll, n) + np.bincount(a, g_lm, n) - 2 * reg * att
        g_def = np.bincount(a, g_ll, n) + np.bincount(h, g_lm, n) - 2 * reg * dfn
        g_att = g_att - g_att.mean()  # regla de la cadena del centrado
        g_def = g_def - g_def.mean()
        grad = -np.concatenate([g_att, g_def, [g_ll.sum(), (w * d_rho).sum(), g_ll.sum() + g_lm.sum()]])
        return f, grad

    mean_goals = max(float(np.average(np.concatenate([x, y]), weights=np.concatenate([w, w]))), 0.5)
    theta0 = np.zeros(2 * n + 3)
    theta0[2 * n] = 0.25
    theta0[2 * n + 1] = -0.05
    theta0[2 * n + 2] = np.log(mean_goals)
    bounds = [(None, None)] * (2 * n + 1) + [(-0.3, 0.3), (None, None)]
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
    att, dfn, gamma, rho, mu0 = unpack(res.x)
    return DCModel(teams, att, dfn, float(gamma), float(rho), float(mu0), xi, len(matches), bool(res.success))


# ---------- mercados derivados de la matriz de marcadores ----------

def outcome_probs(m: np.ndarray) -> np.ndarray:
    """[P(local), P(empate), P(visitante)]."""
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def over_prob(m: np.ndarray, line: float) -> float:
    """P(goles totales > line), p. ej. line=2.5."""
    i, j = np.indices(m.shape)
    return float(m[(i + j) > line].sum())


def btts_prob(m: np.ndarray) -> float:
    """P(ambos equipos anotan)."""
    return float(m[1:, 1:].sum())


def top_scores(m: np.ndarray, n: int = 3):
    """Los n marcadores más probables: lista de (goles_local, goles_visita, prob)."""
    order = np.argsort(m, axis=None)[::-1][:n]
    return [(int(k // m.shape[1]), int(k % m.shape[1]), float(m.flat[k])) for k in order]


def simulate_scores(m: np.ndarray, n: int = 10000, seed: int = 0) -> np.ndarray:
    """Monte Carlo sobre la matriz: devuelve frecuencias simuladas (misma forma).
    Converge a la matriz analítica; sirve sobre todo para combinadas y escenarios."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(m.size, size=n, p=(m / m.sum()).ravel())
    return np.bincount(draws, minlength=m.size).reshape(m.shape) / n
