"""Fase 1: registro diario de pronósticos y liquidación en el libro inmutable.

Regla de no fuga: el pronóstico de un partido se hace ANTES de su inicio y el modelo solo
ve resultados con fecha anterior al día del corte (data_cutoff). La liquidación nunca
modifica pronósticos: agrega resultados y cuotas en sus propias tablas.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from .dixon_coles import btts_prob, fit, outcome_probs, over_prob
from .ledger import Ledger
from .sources.oddspapi import closing_prices
from .teams import to_canonical

PARAMS = {"xi": 0.0023, "reg": 0.05, "window_days": 1460}  # ver docs/protocolo_ajuste.md
HORIZON_H = 48
SETTLE_AFTER_H = 2.5


def git_version(repo: Path | None = None) -> str:
    """Commit del código que genera el pronóstico (+dirty si hay cambios sin guardar)."""
    repo = repo or Path(__file__).resolve().parents[2]
    try:
        h = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=repo, capture_output=True,
                           text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                               capture_output=True, text=True, check=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "desconocida"


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def combine_results(primary: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Une resultados (nombres ya canónicos). Si un partido está en ambos, gana `primary`."""
    cols = ["date", "div", "home", "away", "hg", "ag"]
    if extra is None or extra.empty:
        return primary[cols].copy()
    m = extra[cols].merge(primary[cols], on=["div", "home", "away"], how="left", suffixes=("", "_p"))
    dup = (m["date"] - m["date_p"]).abs() <= pd.Timedelta(days=1)
    dup_keys = set(map(tuple, m.loc[dup, ["div", "home", "away", "date"]].to_numpy()))
    keep = [tuple(r) not in dup_keys for r in extra[["div", "home", "away", "date"]].to_numpy()]
    out = pd.concat([primary[cols], extra.loc[keep, cols]], ignore_index=True)
    return out.sort_values(["date", "div", "home"]).reset_index(drop=True)


def predict_upcoming(ledger: Ledger, fixtures: list[dict], results: pd.DataFrame, canon: dict, now,
                     model_version: str, params: dict = PARAMS, horizon_h: float = HORIZON_H,
                     table=None) -> list[int]:
    """Registra pronósticos de los partidos que empiezan en (now, now + horizon_h].

    fixtures: partidos de OddsPapi con 'div' agregado. canon: {div: set de nombres canónicos}.
    """
    now = _utc(now)
    cutoff = now.tz_localize(None).normalize()  # solo resultados de días anteriores
    created_at = now.isoformat(timespec="seconds")
    new_ids, models = [], {}
    for f in sorted(fixtures, key=lambda f: f["startTime"]):
        ko = _utc(f["startTime"])
        if not (now < ko <= now + pd.Timedelta(hours=horizon_h)):
            continue
        div = f["div"]
        if div not in models:
            tr = results[(results["div"] == div) & (results["date"] < cutoff)
                         & (results["date"] >= cutoff - pd.Timedelta(days=params["window_days"]))]
            models[div] = (fit(tr, ref_date=cutoff, xi=params["xi"], reg=params["reg"]),
                           tr["date"].max())
        model, last_result = models[div]
        home, away = to_canonical([f["participant1Name"], f["participant2Name"]], "oddspapi", div,
                                  canon[div], table)
        m = model.score_matrix(home, away)
        p = outcome_probs(m)
        rid = ledger.append("pronosticos", dict(
            fixture_id=f["fixtureId"], div=div, kickoff=ko.isoformat(), home=home, away=away,
            model_version=model_version, params=json.dumps(params, sort_keys=True),
            data_cutoff=str(last_result.date()), n_train=int(model.n_train),
            known_teams=int(model.knows(home) and model.knows(away)),
            p_h=float(p[0]), p_d=float(p[1]), p_a=float(p[2]),
            p_over25=over_prob(m, 2.5), p_btts=btts_prob(m)), created_at=created_at)
        if rid:
            new_ids.append(rid)
    return new_ids


def settle(ledger: Ledger, results: pd.DataFrame, get_hist, now, source: str = "resultados",
           settle_after_h: float = SETTLE_AFTER_H) -> dict:
    """Agrega resultado y cuotas (apertura, al pronóstico, cierre) de partidos ya jugados.

    get_hist(fixture_id) -> respuesta de historical-odds o None si aún no hay datos.
    """
    now = _utc(now)
    preds = pd.DataFrame(ledger.rows("pronosticos"))
    rep = {"resultados_nuevos": 0, "cuotas_nuevas": 0, "sin_resultado_aun": 0, "sin_cuotas_aun": 0}
    if preds.empty:
        return rep
    have_res = {r["fixture_id"] for r in ledger.rows("resultados")}
    have_odds = {r["fixture_id"] for r in ledger.rows("cuotas") if r["tipo"] == "cierre"}
    first_pred = preds.sort_values("id").groupby("fixture_id").first()
    for fid, p in first_pred.iterrows():
        ko = _utc(p["kickoff"])
        if ko > now - pd.Timedelta(hours=settle_after_h):
            continue
        if fid not in have_res:
            day = ko.tz_localize(None).normalize()
            r = results[(results["div"] == p["div"]) & (results["home"] == p["home"])
                        & (results["away"] == p["away"]) & ((results["date"] - day).abs() <= pd.Timedelta(days=1))]
            if len(r) == 1:
                ledger.append("resultados", dict(fixture_id=fid, hg=int(r.hg.iloc[0]), ag=int(r.ag.iloc[0]),
                                                 source=source))
                rep["resultados_nuevos"] += 1
            else:
                rep["sin_resultado_aun"] += 1
        if fid not in have_odds:
            hist = get_hist(fid)
            close = closing_prices(hist, ko) if hist else None
            if not close or not close["quote_time"]:
                rep["sin_cuotas_aun"] += 1
                continue
            snaps = {"apertura": closing_prices(hist, ko, which="first"),
                     "al_pronostico": closing_prices(hist, min(_utc(p["created_at"]), ko)),
                     "cierre": close}
            for tipo, q in snaps.items():
                vals = {k: (None if pd.isna(v) else float(v)) for k, v in q.items()
                        if k not in ("quote_time", "min_before_ko")}
                ledger.append("cuotas", dict(fixture_id=fid, bookmaker="pinnacle", tipo=tipo,
                                             quote_time=q["quote_time"] or None, **vals))
            rep["cuotas_nuevas"] += 1
    return rep


def season_codes(now, n_back: int = 5) -> list[str]:
    """Códigos de temporada ('2627') desde n_back temporadas atrás hasta la actual (inicia en julio)."""
    now = pd.Timestamp(now)
    y = now.year if now.month >= 7 else now.year - 1
    return [f"{(y - k) % 100:02d}{(y - k + 1) % 100:02d}" for k in range(n_back, -1, -1)]


def load_all_results(divs, now, fd_client=None, verbose: bool = True):
    """Resultados para entrenar: openfootball (histórico, CC0) + football-data.org (temporada en curso).

    Devuelve (results, canon, current_teams): canon = {div: nombres canónicos de las temporadas
    cargadas}; current_teams = {div: equipos de la temporada actual}. Sin fd_client, solo openfootball.
    """
    from .sources import footballdata_org as fd
    from .sources.openfootball import load_results, season_teams

    seasons = season_codes(now)
    current = seasons[-1]
    of = load_results(divs, seasons, refresh_seasons=[current], verbose=verbose)
    canon = {d: set().union(*(season_teams(d, s) for s in seasons)) for d in divs}
    current_teams = {d: season_teams(d, current) for d in divs}
    extra = []
    if fd_client is not None:
        start = f"20{current[:2]}-07-01"
        end = str((_utc(now) + pd.Timedelta(days=1)).date())
        for d in divs:
            r = fd.finished_matches(fd_client, d, start, end)
            r["home"] = to_canonical(r["home"], "footballdata", d, canon[d])
            r["away"] = to_canonical(r["away"], "footballdata", d, canon[d])
            extra.append(r)
    results = combine_results(of, pd.concat(extra, ignore_index=True) if extra else None)
    return results, canon, current_teams


def fit_for_cutoff(results: pd.DataFrame, div: str, cutoff, params: dict = PARAMS):
    """Ajusta el modelo de una liga con resultados de fecha < cutoff (día), dentro de la ventana.

    Es la misma regla que usa el registro diario: nada del día del pronóstico ni posterior."""
    cutoff = pd.Timestamp(cutoff).normalize()
    cutoff = cutoff.tz_localize(None) if cutoff.tzinfo is not None else cutoff
    tr = results[(results["div"] == div) & (results["date"] < cutoff)
                 & (results["date"] >= cutoff - pd.Timedelta(days=params["window_days"]))]
    if tr.empty:
        raise ValueError(f"No hay resultados de {div} antes de {cutoff.date()}")
    return fit(tr, ref_date=cutoff, xi=params["xi"], reg=params["reg"]), tr["date"].max()
