"""Lógica del panel visual (sin Streamlit, para poder probarla).

El panel solo LEE: el libro se copia a un archivo temporal antes de abrirlo, así el
repositorio de datos nunca se modifica desde aquí.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import daily
from .dixon_coles import btts_prob, outcome_probs, over_prob, top_scores
from .ledger import Ledger
from .report import build_table
from .odds import expected_value, fair_odds, overround, remove_margin

OUTCOMES = ("Local", "Empate", "Visita")


def read_ledger(path) -> tuple[dict, dict, pd.DataFrame]:
    """Devuelve ({tabla: DataFrame}, conteos verificados, tabla del informe). LedgerError si la cadena está rota."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el libro en {path}")
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "libro.db"
        shutil.copy2(path, copy)
        with Ledger(copy) as led:
            counts = led.verify()
            tables = {t: pd.DataFrame(led.rows(t)) for t in ("pronosticos", "cuotas", "resultados")}
            settled = build_table(led)
    return tables, counts, settled


def ledger_view(tables: dict) -> pd.DataFrame:
    """Una fila por pronóstico: probabilidades del modelo, resultado si ya hay, y mercado al cierre."""
    p = tables.get("pronosticos", pd.DataFrame())
    if p.empty:
        return pd.DataFrame()
    df = p[["fixture_id", "div", "kickoff", "home", "away", "created_at", "data_cutoff",
            "p_h", "p_d", "p_a", "p_over25", "p_btts", "model_version"]].copy()
    r = tables.get("resultados", pd.DataFrame())
    if not r.empty:
        last = r.sort_values("id").groupby("fixture_id").last()[["hg", "ag"]]
        df = df.join(last, on="fixture_id")
    else:
        df["hg"] = df["ag"] = np.nan
    q = tables.get("cuotas", pd.DataFrame())
    df[["m_h", "m_d", "m_a"]] = np.nan
    if not q.empty:
        c = q[q["tipo"] == "cierre"].set_index("fixture_id")[["o_h", "o_d", "o_a"]]
        for fid, row in c.iterrows():
            o = row.to_numpy(float)
            if np.isfinite(o).all():
                df.loc[df.fixture_id == fid, ["m_h", "m_d", "m_a"]] = remove_margin(o)
    played = df["hg"].notna()
    df["estado"] = np.where(played, "Jugado", "Pendiente")
    df["marcador"] = np.where(played, df["hg"].fillna(0).astype(int).astype(str) + "-"
                              + df["ag"].fillna(0).astype(int).astype(str), "")
    res = np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))
    P = df[["p_h", "p_d", "p_a"]].to_numpy(float)
    df["prob_modelo_resultado"] = np.where(played, P[np.arange(len(df)), res], np.nan)
    return df.sort_values("kickoff", ascending=False).reset_index(drop=True)


def forecast(results: pd.DataFrame, div: str, home: str, away: str, now, odds=None) -> dict:
    """Pronóstico de consulta (no se registra), con la misma regla sin fuga del registro diario."""
    if home == away:
        raise ValueError("Local y visitante son el mismo equipo")
    model, last = daily.fit_for_cutoff(results, div, now)
    m = model.score_matrix(home, away)
    p = outcome_probs(m)
    lam, mu = model.expected_goals(home, away)
    out = {
        "home": home, "away": away, "last_result": last, "n_train": int(model.n_train),
        "unknown": [t for t in (home, away) if not model.knows(t)],
        "xg": (lam, mu), "p": p, "fair": fair_odds(p),
        "over": {l: over_prob(m, l) for l in (1.5, 2.5, 3.5)}, "btts": btts_prob(m),
        "top": top_scores(m, 5), "matrix": m[:6, :6],
    }
    if odds is not None:
        o = np.asarray(odds, float)
        if (o <= 1).any():
            raise ValueError("Las cuotas deben ser decimales mayores que 1")
        out.update(odds=o, margin=overround(o), market=remove_margin(o),
                   ev=np.array([expected_value(pi, oi) for pi, oi in zip(p, o)]))
    return out
