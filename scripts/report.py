"""Informe de la fase 1 desde el libro, con exportación a CSV.

    python scripts/report.py --ledger ledger/betwise.db --csv results/informe_fase1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from betwise.ledger import Ledger
from betwise.report import build_table, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="ledger/betwise.db")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    with Ledger(a.ledger) as led:
        counts = led.verify()
        df = build_table(led)
    print(f"Libro verificado (huellas OK): {counts}\n")
    print(summary(df))
    if a.csv and not df.empty:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.csv, index=False, encoding="utf-8-sig")  # utf-8-sig: Excel lee bien las tildes
        print(f"\nExportado: {a.csv}")


if __name__ == "__main__":
    main()
