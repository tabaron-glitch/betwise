"""Pronóstico de un partido con cuotas que TÚ proporcionas (no se inventan cuotas).

    python scripts/predict_match.py --league E0 --home Arsenal --away Chelsea --odds 1.95 3.60 4.00
Los nombres de equipo deben coincidir con los de football-data.co.uk (p. ej. "Man City").
Mientras el modelo no pase la validación (backtest), tómalo como información, no como señal.
"""
from __future__ import annotations

import argparse

import pandas as pd

from betwise.data import LEAGUES, load_matches
from betwise.dixon_coles import btts_prob, fit, outcome_probs, over_prob, top_scores
from betwise.odds import (ev_sensitivity, expected_value, fair_odds, min_acceptable_odds,
                          overround, remove_margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True, choices=list(LEAGUES))
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--odds", nargs=3, type=float, metavar=("LOCAL", "EMPATE", "VISITA"))
    ap.add_argument("--seasons", nargs="+", default=["2324", "2425", "2526", "2627"])
    ap.add_argument("--window-days", type=int, default=1460)
    a = ap.parse_args()

    matches = load_matches([a.league], a.seasons, refresh_seasons=[a.seasons[-1]])
    today = pd.Timestamp.today().normalize()
    train = matches[(matches.date < today) & (matches.date >= today - pd.Timedelta(days=a.window_days))]
    model = fit(train, ref_date=today)
    for t in (a.home, a.away):
        if not model.knows(t):
            print(f"AVISO: '{t}' no está en los datos; se usa un equipo 'promedio-bajo'. Revisa el nombre.")
    m = model.score_matrix(a.home, a.away)
    p = outcome_probs(m)
    lam, mu = model.expected_goals(a.home, a.away)
    print(f"\n{a.home} vs {a.away}  |  goles esperados {lam:.2f} - {mu:.2f}  |  entrenado con {model.n_train} partidos")
    print(f"1X2 modelo:  {p[0]:.1%} / {p[1]:.1%} / {p[2]:.1%}   (cuotas justas {fair_odds(p).round(2)})")
    print("Más de 1.5 / 2.5 / 3.5: " + " / ".join(f"{over_prob(m, l):.1%}" for l in (1.5, 2.5, 3.5)))
    print(f"Ambos anotan: {btts_prob(m):.1%}")
    print("Marcadores más probables: " + ", ".join(f"{h}-{x} ({q:.1%})" for h, x, q in top_scores(m, 3)))
    if a.odds:
        o = pd.Series(a.odds).to_numpy()
        q = remove_margin(o)
        print(f"\nMargen de la casa: {overround(o):.1%}  |  mercado sin margen: {q[0]:.1%} / {q[1]:.1%} / {q[2]:.1%}")
        for name, pi, oi in zip(("Local", "Empate", "Visita"), p, o):
            s = ev_sensitivity(pi, oi)
            print(f"{name:7s} EV teórico {expected_value(pi, oi):+.1%} (si p real fuera ±3 pp: "
                  f"{s['ev_pesimista']:+.1%} a {s['ev_optimista']:+.1%}) | cuota mínima para EV 0%: {min_acceptable_odds(pi):.2f}")
        print("\nSin backtest validado no hay clasificación de valor (A–E): los umbrales deben calibrarse con datos.")


if __name__ == "__main__":
    main()
