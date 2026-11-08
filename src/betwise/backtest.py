"""Backtest walk-forward sin fuga de información y métricas de validación.

Regla central: las predicciones de un lote con fecha de corte T usan SOLO
partidos con fecha < T. Los resultados del propio lote nunca entran al ajuste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dixon_coles import fit, outcome_probs, over_prob, btts_prob
from .odds import remove_margin

EPS = 1e-12


def walk_forward(matches, test_start, test_end=None, step_days=14, window_days=1460,
                 xi=0.0023, reg=0.05, min_train=150) -> pd.DataFrame:
    """Predice cada partido del periodo de prueba con un modelo ajustado antes del lote."""
    out = []
    test_start = pd.Timestamp(test_start)
    for div, g in matches.groupby("div"):
        g = g.sort_values("date")
        test = g[g["date"] >= test_start]
        if test_end is not None:
            test = test[test["date"] <= pd.Timestamp(test_end)]
        if test.empty:
            continue
        cutoff, last = test["date"].min().normalize(), test["date"].max()
        step = pd.Timedelta(days=step_days)
        while cutoff <= last:
            batch = test[(test["date"] >= cutoff) & (test["date"] < cutoff + step)]
            train = g[(g["date"] < cutoff) & (g["date"] >= cutoff - pd.Timedelta(days=window_days))]
            if len(batch) and len(train) >= min_train:
                model = fit(train, ref_date=cutoff, xi=xi, reg=reg)
                freq = np.array([(train.hg > train.ag).mean(), (train.hg == train.ag).mean(), (train.hg < train.ag).mean()])
                for r in batch.itertuples(index=False):
                    m = model.score_matrix(r.home, r.away)
                    p = outcome_probs(m)
                    out.append({
                        "date": r.date, "div": div, "home": r.home, "away": r.away, "hg": r.hg, "ag": r.ag,
                        "outcome": 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2),
                        "p_h": p[0], "p_d": p[1], "p_a": p[2],
                        "p_over25": over_prob(m, 2.5), "p_btts": btts_prob(m),
                        "b_h": freq[0], "b_d": freq[1], "b_a": freq[2],
                        "o_h": r.o_h, "o_d": r.o_d, "o_a": r.o_a, "odds_source": r.odds_source,
                        "cutoff": cutoff, "known_teams": model.knows(r.home) and model.knows(r.away),
                    })
            cutoff += step
    df = pd.DataFrame(out)
    if df.empty:
        return df
    odds = df[["o_h", "o_d", "o_a"]].to_numpy(float)
    has = np.isfinite(odds).all(axis=1)
    q = np.full_like(odds, np.nan)
    q[has] = remove_margin(odds[has])
    df["q_h"], df["q_d"], df["q_a"] = q[:, 0], q[:, 1], q[:, 2]
    return df.sort_values("date").reset_index(drop=True)


def _probs(df, prefix):
    return df[[f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"]].to_numpy(float)


def brier_score(P, y) -> float:
    onehot = np.eye(3)[np.asarray(y, dtype=int)]
    return float(((P - onehot) ** 2).sum(axis=1).mean())


def log_losses(P, y) -> np.ndarray:
    P = np.clip(P, EPS, 1.0)
    return -np.log(P[np.arange(len(P)), np.asarray(y, dtype=int)])


def compare_models(df: pd.DataFrame) -> dict:
    """Modelo vs mercado sin margen vs frecuencias históricas, sobre los MISMOS partidos."""
    d = df.dropna(subset=["q_h", "q_d", "q_a"])
    y = d["outcome"].to_numpy()
    res = {"n": int(len(d))}
    ll = {}
    for name, pre in (("modelo", "p"), ("mercado", "q"), ("base", "b")):
        P = _probs(d, pre)
        ll[name] = log_losses(P, y)
        res[name] = {"brier": brier_score(P, y), "log_loss": float(ll[name].mean())}
    diff = ll["mercado"] - ll["modelo"]  # >0: el modelo es mejor
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    res["modelo_vs_mercado"] = {"dif_logloss_media": float(diff.mean()), "error_estandar": float(se),
                                "z": float(diff.mean() / se) if se and se > 0 else float("nan")}
    return res


def calibration_table(df: pd.DataFrame, prefix: str = "p", bins: int = 10) -> pd.DataFrame:
    """Curva de calibración agrupada (los 3 resultados de cada partido)."""
    d = df.dropna(subset=[f"{prefix}_h"])
    P = _probs(d, prefix).ravel()
    hit = np.eye(3)[d["outcome"].to_numpy(int)].ravel()
    cut = pd.cut(P, np.linspace(0, 1, bins + 1), include_lowest=True)
    t = pd.DataFrame({"p": P, "hit": hit, "bin": cut}).groupby("bin", observed=True).agg(
        prob_media=("p", "mean"), frecuencia_obs=("hit", "mean"), n=("hit", "size")).reset_index()
    return t


def calibration_error(table: pd.DataFrame) -> float:
    """ECE: brecha media ponderada entre probabilidad predicha y frecuencia observada."""
    return float((table["n"] * (table["prob_media"] - table["frecuencia_obs"]).abs()).sum() / table["n"].sum())


def betting_sim(df: pd.DataFrame, min_ev: float = 0.0) -> dict:
    """Apuesta plana de 1 unidad al resultado 1X2 con mayor EV por partido si EV >= min_ev.

    OJO: usa las cuotas del archivo (normalmente de cierre). Sirve para ver el orden de
    magnitud; no es una simulación de apuestas reales ni debe usarse para elegir
    el umbral con los mismos datos (sobreajuste).
    """
    d = df.dropna(subset=["o_h", "o_d", "o_a"])
    P, O = _probs(d, "p"), d[["o_h", "o_d", "o_a"]].to_numpy(float)
    ev = P * O - 1.0
    pick = ev.argmax(axis=1)
    best = ev[np.arange(len(d)), pick]
    sel = best >= min_ev
    if sel.sum() == 0:
        return {"min_ev": min_ev, "n_apuestas": 0}
    y = d["outcome"].to_numpy(int)[sel]
    o = O[np.arange(len(d)), pick][sel]
    profit = np.where(pick[sel] == y, o - 1.0, -1.0)
    n = int(sel.sum())
    se = profit.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return {"min_ev": min_ev, "n_apuestas": n, "acierto": float((pick[sel] == y).mean()),
            "roi": float(profit.mean()), "ic95": (float(profit.mean() - 1.96 * se), float(profit.mean() + 1.96 * se))}


def summarize(df: pd.DataFrame, thresholds=(0.0, 0.03, 0.05, 0.10)) -> str:
    if df.empty:
        return "Sin predicciones: revisa fechas y datos."
    c = compare_models(df)
    cal = calibration_table(df)
    L = [f"Partidos evaluados con cuotas: {c['n']}", ""]
    L.append(f"{'':10s}{'Brier':>9s}{'LogLoss':>10s}")
    for k in ("modelo", "mercado", "base"):
        L.append(f"{k:10s}{c[k]['brier']:9.4f}{c[k]['log_loss']:10.4f}")
    v = c["modelo_vs_mercado"]
    L += ["", f"Log loss: mercado - modelo = {v['dif_logloss_media']:+.4f} (EE {v['error_estandar']:.4f}, z = {v['z']:+.2f})",
          "  (positivo = el modelo es mejor; |z| < 2 no es evidencia suficiente)",
          f"Error de calibración del modelo (ECE): {calibration_error(cal):.4f}", "",
          "Simulación de apuesta plana a cuotas del archivo (sensibilidad, NO validación):"]
    for t in thresholds:
        s = betting_sim(df, t)
        if s["n_apuestas"] == 0:
            L.append(f"  EV >= {t:.0%}: sin apuestas")
        else:
            L.append(f"  EV >= {t:.0%}: n={s['n_apuestas']:5d}  acierto={s['acierto']:.1%}  ROI={s['roi']:+.1%}  "
                     f"IC95%=[{s['ic95'][0]:+.1%}, {s['ic95'][1]:+.1%}]")
    return "\n".join(L)
