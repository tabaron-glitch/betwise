"""Une resultados de openfootball con cuotas de cierre de OddsPapi.

Emparejamiento: misma liga, mismos equipos (nombre canónico) y fecha a ±1 día
(OddsPapi da hora UTC; openfootball, fecha local). Las cuotas NO entran al modelo:
solo se usan para compararlo con el mercado.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..teams import to_canonical

SOURCE_LABEL = "Pinnacle cierre (OddsPapi)"
MAX_MARGIN = 0.10
STALE_MIN = 360  # última cotización con más de 6 h antes del inicio: se marca


def attach_closing(results: pd.DataFrame, closing: pd.DataFrame, table=None) -> tuple[pd.DataFrame, dict]:
    """Agrega o_h/o_d/o_a (y totales/ambos anotan) a `results`. Devuelve (df, informe)."""
    res = results.copy().reset_index(drop=True)
    res["_rid"] = np.arange(len(res))
    for c in ("o_h", "o_d", "o_a", "o_over25", "o_under25", "o_btts_y", "o_btts_n", "min_before_ko"):
        res[c] = np.nan
    res["odds_source"] = ""
    res["odds_stale"] = False
    report = {"cierres_totales": int(len(closing))}
    ok = closing[closing["status"] == "ok"].copy()
    report["cierres_ok"] = int(len(ok))
    if ok.empty:
        return res.drop(columns="_rid"), report

    parts = []
    for div, g in ok.groupby("div"):
        r = res[res["div"] == div]
        canon = set(r["home"]) | set(r["away"])
        g = g.copy()
        g["home"] = to_canonical(g["home_src"], "oddspapi", div, canon, table)
        g["away"] = to_canonical(g["away_src"], "oddspapi", div, canon, table)
        parts.append(g)
    ok = pd.concat(parts, ignore_index=True)
    ok["date_utc"] = pd.to_datetime(ok["start_time"], utc=True).dt.tz_localize(None).dt.normalize()

    m = ok.merge(res[["_rid", "div", "home", "away", "date"]], on=["div", "home", "away"], how="left")
    m = m[(m["date"] - m["date_utc"]).abs() <= pd.Timedelta(days=1)]
    if m["fixtureId"].duplicated().any() or m["_rid"].duplicated().any():
        raise ValueError("Emparejamiento ambiguo entre OddsPapi y openfootball")
    unmatched = ok[~ok["fixtureId"].isin(m["fixtureId"])]
    report["emparejados"] = int(len(m))
    report["cierres_sin_resultado"] = int(len(unmatched))
    report["ejemplos_sin_resultado"] = unmatched[["div", "start_time", "home", "away"]].head(5).to_dict("records")

    O = m[["o_h", "o_d", "o_a"]].to_numpy(float)
    margin = (1.0 / O).sum(axis=1) - 1.0
    bad = ~((margin >= 0) & (margin <= MAX_MARGIN))
    report["margen_fuera_de_rango"] = int(bad.sum())
    m = m[~bad]

    cols = ["o_h", "o_d", "o_a", "o_over25", "o_under25", "o_btts_y", "o_btts_n", "min_before_ko"]
    res = res.set_index("_rid")
    res.loc[m["_rid"].to_numpy(), cols] = m[cols].to_numpy(float)
    res.loc[m["_rid"].to_numpy(), "odds_source"] = SOURCE_LABEL
    res["odds_stale"] = res["min_before_ko"] > STALE_MIN
    report["con_cuota"] = int((res["odds_source"] == SOURCE_LABEL).sum())
    report["cotizacion_vieja_mas_6h"] = int(res["odds_stale"].sum())
    return res.reset_index(drop=True), report


def load_closing(divs, cache_dir="data/raw/oddspapi") -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype={"fixtureId": str}) for d in divs
              if (p := Path(cache_dir) / f"closing_{d}.csv").exists()]
    if not frames:
        raise FileNotFoundError(f"No hay cuotas en {cache_dir}: corre primero scripts/update_odds.py")
    return pd.concat(frames, ignore_index=True)
