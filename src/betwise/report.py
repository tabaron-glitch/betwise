"""Informe de la fase 1 a partir del libro: modelo vs mercado en partidos que nadie había visto.

Referencias del mercado (Pinnacle, sin margen):
- al_pronostico: la cuota vigente cuando se registró el pronóstico (comparación justa).
- cierre: la última antes del inicio (referencia exigente: incorpora alineaciones, etc.).
Movimiento hacia el modelo: si el mercado se mueve, entre el pronóstico y el cierre, en la
dirección en que el modelo discrepaba. Positivo y sostenido sería una señal; con pocos
partidos no prueba nada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import brier_score, calibration_error, calibration_table, log_losses
from .ledger import Ledger
from .odds import remove_margin

DISCLAIMER = ("Aviso: estimaciones estadísticas, no garantizan ganancias. Juega con responsabilidad; "
              "si no hay evidencia de ventaja, la decisión es NO APOSTAR.")


def build_table(ledger: Ledger) -> pd.DataFrame:
    """Una fila por pronóstico liquidado (resultado y cuotas ya registrados)."""
    p = pd.DataFrame(ledger.rows("pronosticos"))
    r = pd.DataFrame(ledger.rows("resultados"))
    q = pd.DataFrame(ledger.rows("cuotas"))
    if p.empty or r.empty or q.empty:
        return pd.DataFrame()
    r = r.sort_values("id").groupby("fixture_id").last()[["hg", "ag"]]  # la corrección más reciente vale
    df = p.join(r, on="fixture_id", how="inner")
    for tipo, pre in (("al_pronostico", "qp"), ("cierre", "qc")):
        t = q[q["tipo"] == tipo].set_index("fixture_id")[["o_h", "o_d", "o_a"]]
        O = t.to_numpy(float)
        ok = np.isfinite(O).all(axis=1)
        Q = np.full_like(O, np.nan)
        Q[ok] = remove_margin(O[ok])
        qdf = pd.DataFrame(Q, index=t.index, columns=[f"{pre}_h", f"{pre}_d", f"{pre}_a"])
        df = df.join(qdf, on="fixture_id")
    df["outcome"] = np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))
    return df.sort_values("kickoff").reset_index(drop=True)


def _cmp(df, pre):
    d = df.dropna(subset=[f"{pre}_h", f"{pre}_d", f"{pre}_a"])
    if len(d) < 2:
        return None
    y = d["outcome"].to_numpy()
    P = d[["p_h", "p_d", "p_a"]].to_numpy(float)
    Q = d[[f"{pre}_h", f"{pre}_d", f"{pre}_a"]].to_numpy(float)
    diff = log_losses(Q, y) - log_losses(P, y)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return {"n": len(d), "ll_modelo": log_losses(P, y).mean(), "ll_mercado": log_losses(Q, y).mean(),
            "brier_modelo": brier_score(P, y), "brier_mercado": brier_score(Q, y),
            "dif": diff.mean(), "z": diff.mean() / se if se > 0 else float("nan")}


def summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "Aún no hay pronósticos liquidados en el libro.\n" + DISCLAIMER
    L = [f"Pronósticos liquidados: {len(df)} (desde {df.kickoff.min()[:10]} hasta {df.kickoff.max()[:10]})",
         f"Versiones del modelo: {sorted(df.model_version.unique())}", ""]
    for pre, name in (("qp", "mercado al pronóstico"), ("qc", "mercado al cierre")):
        c = _cmp(df, pre)
        if c is None:
            L.append(f"{name}: sin datos suficientes")
            continue
        L.append(f"vs {name} (n={c['n']}): log loss modelo {c['ll_modelo']:.4f} | mercado {c['ll_mercado']:.4f} | "
                 f"dif {c['dif']:+.4f}, z = {c['z']:+.2f}")
        L.append(f"   Brier modelo {c['brier_modelo']:.4f} | mercado {c['brier_mercado']:.4f}")
    L.append("   (dif positiva = modelo mejor; |z| < 2 no es evidencia)")
    d = df.dropna(subset=["qp_h", "qc_h"])
    if len(d):
        P = d[["p_h", "p_d", "p_a"]].to_numpy(float)
        QP = d[["qp_h", "qp_d", "qp_a"]].to_numpy(float)
        QC = d[["qc_h", "qc_d", "qc_a"]].to_numpy(float)
        move = (np.sign(P - QP) * (QC - QP)).sum(axis=1) / 2
        L.append(f"Movimiento del mercado hacia el modelo: {move.mean():+.4f} por partido "
                 f"(EE {move.std(ddof=1) / np.sqrt(len(move)) if len(move) > 1 else float('nan'):.4f})")
    if len(df) >= 30:
        L.append(f"Calibración del modelo (ECE): {calibration_error(calibration_table(df, 'p')):.4f}")
    L += ["", "Recordatorio: detectar una ventaja pequeña requiere miles de partidos; una racha corta no "
              "demuestra nada.", DISCLAIMER]
    return "\n".join(L)
