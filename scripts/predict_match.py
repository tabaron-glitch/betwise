"""Pronóstico de UN partido, con cuotas que TÚ proporcionas (nunca se inventan cuotas).

    python scripts/predict_match.py --league E0 --home "Man City" --away Arsenal --odds 1.95 3.60 4.00
    python scripts/predict_match.py --league SP1 --home Betis --away "Atletico Madrid"

Fuentes legales: openfootball (CC0) + football-data.org (temporada en curso, requiere
FOOTBALLDATA_KEY en .env; sin ella usa solo openfootball y avisa que puede estar desactualizado).
Los nombres se escriben como quieras ("Man City", "atletico"); si son ambiguos te muestra las opciones.
Usa la misma configuración del modelo que el registro diario. Esto NO queda en el libro: es una
consulta. Mientras el modelo no le gane al mercado, tómalo como información, no como señal.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from betwise import daily
from betwise.dixon_coles import btts_prob, outcome_probs, over_prob, top_scores
from betwise.odds import (ev_sensitivity, expected_value, fair_odds, min_acceptable_odds,
                          overround, remove_margin)
from betwise.report import DISCLAIMER
from betwise.sources import footballdata_org as fd
from betwise.teams import UnknownTeamError, resolve_user_team

LEAGUES = {"E0": "Premier League", "SP1": "LaLiga"}


def main():
    if hasattr(sys.stdout, "reconfigure"):  # tildes bien en consolas de Windows
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Pronóstico de un partido (consulta, no se registra).")
    ap.add_argument("--league", required=True, choices=list(LEAGUES))
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--odds", nargs=3, type=float, metavar=("LOCAL", "EMPATE", "VISITA"))
    ap.add_argument("--sin-api", action="store_true", help="no consultar football-data.org")
    a = ap.parse_args()
    if a.odds and min(a.odds) <= 1.0:
        sys.exit("Las cuotas deben ser decimales mayores que 1 (ej. 1.95 3.60 4.00).")

    now = pd.Timestamp.now(tz="UTC")
    client = None
    if not a.sin_api:
        try:
            client = fd.Client(fd.load_key())
        except Exception:
            print("AVISO: sin FOOTBALLDATA_KEY; uso solo openfootball (puede faltar la última jornada).")
    results, _, current = daily.load_all_results([a.league], now, client, verbose=False)
    try:
        home = resolve_user_team(a.home, current[a.league], a.league)
        away = resolve_user_team(a.away, current[a.league], a.league)
    except UnknownTeamError as e:
        sys.exit(f"ERROR: {e}")
    if home == away:
        sys.exit("ERROR: local y visitante son el mismo equipo.")

    model, last = daily.fit_for_cutoff(results, a.league, now)
    days_old = (now.tz_localize(None).normalize() - last).days
    if days_old > 10:
        print(f"AVISO: el último resultado cargado es de hace {days_old} días ({last.date()}); "
              "el modelo no conoce las jornadas recientes.")
    for t in (home, away):
        if not model.knows(t):
            print(f"AVISO: {t} no tiene partidos en la ventana de entrenamiento (recién ascendido): "
                  "su fuerza se estima como promedio; el pronóstico es menos confiable.")
    m = model.score_matrix(home, away)
    p = outcome_probs(m)
    lam, mu = model.expected_goals(home, away)
    print(f"\n{LEAGUES[a.league]}: {home} vs {away}")
    print(f"Datos hasta {last.date()} | entrenado con {model.n_train} partidos | parámetros {daily.PARAMS}")
    print(f"Goles esperados {lam:.2f} - {mu:.2f}")
    print(f"1X2 modelo:  {p[0]:.1%} / {p[1]:.1%} / {p[2]:.1%}   (cuotas justas {np.round(fair_odds(p), 2)})")
    print("Más de 1.5 / 2.5 / 3.5: " + " / ".join(f"{over_prob(m, l):.1%}" for l in (1.5, 2.5, 3.5)))
    print(f"Ambos anotan: {btts_prob(m):.1%}")
    print("Marcadores más probables: " + ", ".join(f"{h}-{x} ({q:.1%})" for h, x, q in top_scores(m, 3)))
    if a.odds:
        o = np.array(a.odds)
        q = remove_margin(o)
        print(f"\nCuotas que diste: margen de la casa {overround(o):.1%} | "
              f"mercado sin margen {q[0]:.1%} / {q[1]:.1%} / {q[2]:.1%}")
        for name, pi, oi in zip(("Local", "Empate", "Visita"), p, o):
            s = ev_sensitivity(pi, oi)
            print(f"{name:7s} EV teórico {expected_value(pi, oi):+.1%} (si p real fuera ±3 pp: "
                  f"{s['ev_pesimista']:+.1%} a {s['ev_optimista']:+.1%}) | cuota mínima EV 0%: "
                  f"{min_acceptable_odds(pi):.2f}")
        print("\nEl backtest NO mostró ventaja sobre el mercado (z = -1,95): si el modelo y el mercado "
              "discrepan, lo más probable es que el mercado tenga razón. Sin clasificación de valor (A–E).")
    print("\n" + DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
