"""Descarga/actualiza las cuotas de cierre de Pinnacle desde OddsPapi (requiere .env con ODDSPAPI_KEY).

    python scripts/update_odds.py --leagues E0 SP1 --start 2026-01-01

La primera vez tarda (~5,5 s por partido, pausa exigida por la API). Luego solo baja partidos nuevos.
"""
from __future__ import annotations

import argparse

import pandas as pd

from betwise.sources import oddspapi as op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+", default=list(op.TOURNAMENTS))
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=str(pd.Timestamp.now().date()))
    ap.add_argument("--cache", default="data/raw/oddspapi")
    a = ap.parse_args()
    client = op.Client(op.load_key())
    for div in a.leagues:
        df = op.update_closing(client, div, a.start, a.end, cache_dir=a.cache)
        print(f"{div}: {len(df)} partidos en la tabla; estados: {df['status'].value_counts().to_dict()}")
    print(f"Consultas facturables usadas en esta corrida: {client.billable_calls}")


if __name__ == "__main__":
    main()
