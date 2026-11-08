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
from betwise.sources.openfootball import load_results, season_teams
from betwise.teams import to_canonical

DIVS = ["E0", "SP1"]


def season_codes(now: pd.Timestamp, n_back: int = 5) -> list[str]:
    y = now.year if now.month >= 7 else now.year - 1
    return [f"{(y - k) % 100:02d}{(y - k + 1) % 100:02d}" for k in range(n_back, -1, -1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="ledger/betwise.db")
    ap.add_argument("--now", default=None, help="solo para pruebas: hora UTC ISO")
    a = ap.parse_args()
    now = pd.Timestamp(a.now) if a.now else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    seasons = season_codes(now)
    current = seasons[-1]

    # 1. Resultados: openfootball (histórico) + football-data.org (temporada en curso, al día)
    of = load_results(DIVS, seasons, refresh_seasons=[current])
    canon = {d: set().union(*(season_teams(d, s) for s in seasons)) for d in DIVS}
    fdc = fd.Client(fd.load_key())
    season_start = f"20{current[:2]}-07-01"
    extra = []
    for d in DIVS:
        r = fd.finished_matches(fdc, d, season_start, str((now + pd.Timedelta(days=1)).date()))
        r["home"] = to_canonical(r["home"], "footballdata", d, canon[d])
        r["away"] = to_canonical(r["away"], "footballdata", d, canon[d])
        extra.append(r)
    results = daily.combine_results(of, pd.concat(extra, ignore_index=True))
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
