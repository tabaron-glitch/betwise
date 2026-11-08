"""Búsqueda de parámetros según docs/protocolo_ajuste.md (validación SIN cuotas, antes de la prueba).

    python scripts/tune.py

No toca el periodo de prueba (desde 2026-01-17). Guarda results/ajuste.csv.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from betwise.backtest import log_losses, walk_forward
from betwise.sources.openfootball import load_results

VAL_START, VAL_END = "2023-08-01", "2026-01-16"
XI = [0.001, 0.0019, 0.0023, 0.003, 0.0045, 0.006]
REG = [0.05, 1, 3, 10]
WINDOW = [730, 1460]
DEFAULT = (0.0023, 0.05, 1460)
SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]


def load():
    m = load_results(["E0", "SP1"], SEASONS, verbose=False)
    m = m[m["date"] <= VAL_END].copy()  # nada del periodo de prueba
    for c in ("o_h", "o_d", "o_a"):
        m[c] = np.nan
    m["odds_source"] = ""
    return m


def run(cfg):
    xi, reg, win = cfg
    p = walk_forward(load(), test_start=VAL_START, test_end=VAL_END, xi=xi, reg=reg, window_days=win)
    p = p.sort_values(["date", "div", "home"]).reset_index(drop=True)
    ll = log_losses(p[["p_h", "p_d", "p_a"]].to_numpy(float), p["outcome"].to_numpy())
    return cfg, ll, p[["date", "div", "home", "away"]]


def main():
    grid = list(itertools.product(XI, REG, WINDOW))
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(run, grid))
    keys = results[0][2]
    for cfg, ll, k in results:  # mismos partidos en todas las combinaciones (comparación pareada)
        assert k.equals(keys), f"partidos distintos en {cfg}"
    base_ll = dict((cfg, ll) for cfg, ll, _ in results)[DEFAULT]
    rows = []
    for cfg, ll, _ in results:
        d = base_ll - ll  # >0: mejor que la configuración actual
        rows.append({"xi": cfg[0], "reg": cfg[1], "ventana": cfg[2], "n": len(ll), "log_loss": ll.mean(),
                     "mejora_vs_actual": d.mean(), "ee_mejora": d.std(ddof=1) / np.sqrt(len(d))})
    t = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    Path("results").mkdir(exist_ok=True)
    t.to_csv("results/ajuste.csv", index=False)
    print(f"Validación {VAL_START} a {VAL_END}: {t.n.iloc[0]} partidos, {len(t)} combinaciones\n")
    print(t.head(12).to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nActual:", t[(t.xi == DEFAULT[0]) & (t.reg == DEFAULT[1]) & (t.ventana == DEFAULT[2])].to_string(header=False))
    best = t.iloc[0]
    if best.mejora_vs_actual > best.ee_mejora:
        print(f"\nELEGIDA (mejora > 1 EE): xi={best.xi}, reg={best.reg}, ventana={int(best.ventana)}")
    else:
        print(f"\nSe CONSERVA la actual {DEFAULT}: la mejor mejora {best.mejora_vs_actual:.4f} < 1 EE ({best.ee_mejora:.4f})")


if __name__ == "__main__":
    main()
