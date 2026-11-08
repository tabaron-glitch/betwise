"""Backtest walk-forward.

Datos reales (requiere internet):
    python scripts/run_backtest.py --leagues E0 SP1 --seasons 1920 2021 2122 2223 2324 2425 2526 --test-start 2024-08-01
Prueba de tubería con datos sintéticos (NO son resultados reales):
    python scripts/run_backtest.py --synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

from betwise.backtest import summarize, walk_forward
from betwise.data import LEAGUES, load_matches
from betwise.synthetic import make_synthetic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+", default=list(LEAGUES))
    ap.add_argument("--seasons", nargs="+", default=["1920", "2021", "2122", "2223", "2324", "2425", "2526"])
    ap.add_argument("--test-start", default="2024-08-01")
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.synthetic:
        matches = make_synthetic()
        test_start = "2021-08-01"
        print("*** DATOS SINTÉTICOS: solo prueba de la tubería, no son resultados reales ***\n")
    else:
        matches = load_matches(a.leagues, a.seasons, refresh_seasons=[a.seasons[-1]])
        test_start = a.test_start

    preds = walk_forward(matches, test_start=test_start, step_days=a.step_days)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    preds.to_csv(out / ("predicciones_sinteticas.csv" if a.synthetic else "predicciones_backtest.csv"), index=False)
    print(summarize(preds))


if __name__ == "__main__":
    main()
