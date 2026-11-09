"""Trabajo diario de la fase 1: pronosticar los próximos partidos y liquidar los jugados.

    python scripts/daily_run.py --ledger ledger/betwise.db

Requiere en .env: ODDSPAPI_KEY y FOOTBALLDATA_KEY. Consultas facturables de OddsPapi:
2 por corrida (calendario de Premier y LaLiga). El histórico de cuotas no descuenta.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from betwise import daily
from betwise.ledger import Ledger
from betwise.sources import footballdata_org as fd
from betwise.sources import oddspapi as op

DIVS = ["E0", "SP1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="ledger/betwise.db")
    ap.add_argument("--now", default=None, help="solo para pruebas: hora UTC ISO")
    a = ap.parse_args()
    now = pd.Timestamp(a.now) if a.now else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    # 1. Resultados: openfootball (histórico) + football-data.org (temporada en curso, al día)
    results, canon, _ = daily.load_all_results(DIVS, now, fd.Client(fd.load_key()))
    print(f"Resultados: hasta {results['date'].max().date()} ({len(results)} partidos)")

    # 2. Calendario, pronósticos y liquidación
    opc = op.Client(op.load_key())
    fixtures = [{**f, "div": d} for d in DIVS for f in op.upcoming_fixtures(opc, d, now)]
    version = daily.git_version()
    if version.endswith("+dirty") or version == "desconocida":
        print(f"AVISO: versión de código '{version}': guarda los cambios en git antes de registrar en serio.")
    with Ledger(a.ledger) as led:
        new = daily.predict_upcoming(led, fixtures, results, canon, now, version)
        rep = daily.settle(led, results, lambda fid: opc.get("historical-odds", billable=False, fixtureId=fid,
                                                             bookmakers="pinnacle"), now,
                           source="openfootball+football-data.org")
        counts = led.verify()
    print(f"Pronósticos nuevos: {len(new)} | liquidación: {rep}")
    print(f"Libro verificado (huellas OK): {counts}")
    print(f"Consultas facturables de OddsPapi en esta corrida: {opc.billable_calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
