"""Backtest walk-forward.

Datos reales, fuente por defecto (resultados openfootball + cierre Pinnacle de OddsPapi):
    python scripts/update_odds.py --leagues E0 SP1          # primero, bajar/actualizar cuotas
    python scripts/run_backtest.py --leagues E0 SP1 --test-start 2026-01-17
Fuente antigua (football-data.co.uk; bloqueada en Colombia por Coljuegos, no usar sin concepto legal):
    python scripts/run_backtest.py --source football-data --test-start 2024-08-01
Prueba de tubería con datos sintéticos (NO son resultados reales):
    python scripts/run_backtest.py --synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

from betwise.backtest import summarize, walk_forward
from betwise.data import LEAGUES, load_matches
from betwise.sources.merge import attach_closing, load_closing
from betwise.sources.openfootball import load_results
from betwise.synthetic import make_synthetic

SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["openfootball", "football-data"], default="openfootball")
    ap.add_argument("--leagues", nargs="+", default=list(LEAGUES))
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    ap.add_argument("--test-start", default="2026-01-17")
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.synthetic:
        matches = make_synthetic()
        test_start = "2021-08-01"
        print("*** DATOS SINTÉTICOS: solo prueba de la tubería, no son resultados reales ***\n")
    elif a.source == "openfootball":
        results = load_results(a.leagues, a.seasons, refresh_seasons=[a.seasons[-1]])
        matches, rep = attach_closing(results, load_closing(a.leagues))
        print("Unión openfootball + OddsPapi:", {k: v for k, v in rep.items() if k != "ejemplos_sin_resultado"})
        if rep.get("ejemplos_sin_resultado"):
            print("  ejemplos de cierres sin resultado en openfootball:", rep["ejemplos_sin_resultado"])
        print(f"  último resultado disponible en openfootball: {results['date'].max().date()}\n")
        test_start = a.test_start
    else:
        seasons = [s for s in a.seasons if s != "2627"]
        matches = load_matches(a.leagues, seasons, refresh_seasons=[seasons[-1]])
        test_start = a.test_start

    preds = walk_forward(matches, test_start=test_start, test_end=a.test_end, step_days=a.step_days)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    preds.to_csv(out / ("predicciones_sinteticas.csv" if a.synthetic else "predicciones_backtest.csv"), index=False)
    print(summarize(preds))


if __name__ == "__main__":
    main()
