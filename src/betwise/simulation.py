"""Modo SIMULACIÓN: apuestas con dinero FICTICIO para practicar. No es una recomendación.

Reglas fijas (declaradas antes de ver resultados; no se ajustan con los datos de la simulación):
- Solo mercado 1X2 y solo con cuotas que escribe la persona.
- Probabilidad pesimista = probabilidad del modelo − 3 puntos.
- Se propone la opción con mayor EV pesimista, solo si es > 0. Si ninguna, no hay apuesta simulada.
- Monto = 1/4 de Kelly con la probabilidad pesimista, tope 2 % de la banca ficticia.
- Límites: 5 % de la banca en apuestas por día (hora Colombia), 10 % en 7 días;
  pausa si la banca cae 15 % o más desde su máximo. Sin martingala: perder no sube el monto.
- Un partido solo se puede simular antes de jugarse, y una sola vez.
- El cuaderno es de solo agregar (triggers bloquean UPDATE y DELETE), como el libro.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .odds import expected_value, fractional_kelly

OUTCOMES = ("Local", "Empate", "Visita")
PESSIMISM = 0.03
KELLY_FRACTION = 0.25
MAX_STAKE = 0.02
DAILY_LIMIT = 0.05
WEEKLY_LIMIT = 0.10
DRAWDOWN_PAUSE = 0.15
COL = timezone(timedelta(hours=-5))  # hora Colombia (sin horario de verano)

SCHEMA = """
CREATE TABLE IF NOT EXISTS config (clave TEXT PRIMARY KEY, valor TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS apuestas (
    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, div TEXT NOT NULL, home TEXT NOT NULL,
    away TEXT NOT NULL, fecha_partido TEXT NOT NULL, opcion INTEGER NOT NULL, cuota REAL NOT NULL,
    p_modelo REAL NOT NULL, p_pesimista REAL NOT NULL, ev_pesimista REAL NOT NULL,
    monto REAL NOT NULL, banca_antes REAL NOT NULL,
    UNIQUE (div, home, away, fecha_partido));
CREATE TABLE IF NOT EXISTS liquidaciones (
    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, apuesta_id INTEGER NOT NULL UNIQUE,
    hg INTEGER NOT NULL, ag INTEGER NOT NULL, ganada INTEGER NOT NULL, resultado REAL NOT NULL);
"""
TRIGGERS = [f"CREATE TRIGGER IF NOT EXISTS no_{op.lower()}_{t} BEFORE {op} ON {t} "
            f"BEGIN SELECT RAISE(ABORT, 'cuaderno de simulación inmutable'); END"
            for t in ("config", "apuestas", "liquidaciones") for op in ("UPDATE", "DELETE")]


class SimulationError(ValueError):
    pass


def propose(p, odds) -> dict | None:
    """Opción simulada según las reglas fijas, o None si ninguna tiene EV pesimista > 0."""
    p = np.asarray(p, float)
    o = np.asarray(odds, float)
    pp = np.clip(p - PESSIMISM, 1e-9, 1)
    ev = np.array([expected_value(a, b) for a, b in zip(pp, o)])
    i = int(np.argmax(ev))
    if ev[i] <= 0:
        return None
    frac = fractional_kelly(pp[i], o[i], KELLY_FRACTION, MAX_STAKE)
    if frac <= 0:
        return None
    return {"opcion": i, "nombre": OUTCOMES[i], "cuota": float(o[i]), "p_modelo": float(p[i]),
            "p_pesimista": float(pp[i]), "ev_pesimista": float(ev[i]), "fraccion": frac}


class Notebook:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)
        for t in TRIGGERS:
            self.con.execute(t)
        self.con.commit()

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ---------- configuración
    def initial_bankroll(self) -> float | None:
        r = self.con.execute("SELECT valor FROM config WHERE clave='banca_inicial'").fetchone()
        return float(r[0]) if r else None

    def start(self, bankroll: float):
        if self.initial_bankroll() is not None:
            raise SimulationError("La banca inicial ya está fijada y no se cambia")
        if bankroll <= 0:
            raise SimulationError("La banca inicial debe ser positiva")
        self.con.execute("INSERT INTO config VALUES ('banca_inicial', ?)", (str(float(bankroll)),))
        self.con.commit()

    # ---------- consultas
    def bets(self) -> pd.DataFrame:
        b = pd.read_sql("SELECT * FROM apuestas ORDER BY id", self.con)
        s = pd.read_sql("SELECT apuesta_id, hg, ag, ganada, resultado FROM liquidaciones", self.con)
        return b.merge(s, left_on="id", right_on="apuesta_id", how="left").drop(columns="apuesta_id")

    def state(self, now=None) -> dict:
        now = _now(now)
        ini = self.initial_bankroll()
        if ini is None:
            return {"iniciada": False}
        b = self.bets()
        settled = b[b["resultado"].notna()]
        curve = ini + settled.sort_values("id")["resultado"].cumsum()
        bank = float(curve.iloc[-1]) if len(curve) else ini
        peak = float(max(ini, curve.max())) if len(curve) else ini
        created = pd.to_datetime(b["created_at"], utc=True) if len(b) else pd.Series(dtype="datetime64[ns, UTC]")
        today = now.astimezone(COL).date()
        day = float(b.loc[created.dt.tz_convert(COL).dt.date == today, "monto"].sum()) if len(b) else 0.0
        week = float(b.loc[created > pd.Timestamp(now) - pd.Timedelta(days=7), "monto"].sum()) if len(b) else 0.0
        staked = float(settled["monto"].sum())
        return {"iniciada": True, "banca_inicial": ini, "banca": bank, "maximo": peak,
                "caida": 1 - bank / peak, "pausada": bank <= peak * (1 - DRAWDOWN_PAUSE),
                "hoy": day, "semana": week, "n": len(b), "pendientes": int(b["resultado"].isna().sum()),
                "ganadas": int(settled["ganada"].sum()), "liquidadas": len(settled),
                "resultado": bank - ini, "roi": (bank - ini) / staked if staked else float("nan")}

    # ---------- registrar y liquidar
    def register(self, div, home, away, match_date, proposal: dict, results: pd.DataFrame, now=None) -> dict:
        now = _now(now)
        st = self.state(now)
        if not st["iniciada"]:
            raise SimulationError("Primero fija la banca inicial ficticia")
        if st["pausada"]:
            raise SimulationError(f"Simulación en pausa: la banca cayó {st['caida']:.0%} desde su máximo "
                                  f"(límite {DRAWDOWN_PAUSE:.0%})")
        md = pd.Timestamp(match_date).normalize()
        if md.date() < now.astimezone(COL).date():
            raise SimulationError("Ese partido ya pasó: solo se simulan partidos por jugar")
        if not _find(results, div, home, away, md).empty:
            raise SimulationError("Ese partido ya tiene resultado: no se puede simular")
        amount = round(proposal["fraccion"] * st["banca"], 2)
        if st["hoy"] + amount > DAILY_LIMIT * st["banca"] + 1e-9:
            raise SimulationError(f"Superaría el límite diario ({DAILY_LIMIT:.0%} de la banca)")
        if st["semana"] + amount > WEEKLY_LIMIT * st["banca"] + 1e-9:
            raise SimulationError(f"Superaría el límite semanal ({WEEKLY_LIMIT:.0%} de la banca)")
        try:
            self.con.execute(
                "INSERT INTO apuestas (created_at, div, home, away, fecha_partido, opcion, cuota, p_modelo, "
                "p_pesimista, ev_pesimista, monto, banca_antes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (now.isoformat(timespec="seconds"), div, home, away, str(md.date()), proposal["opcion"],
                 proposal["cuota"], proposal["p_modelo"], proposal["p_pesimista"], proposal["ev_pesimista"],
                 amount, st["banca"]))
        except sqlite3.IntegrityError:
            raise SimulationError("Ese partido ya está en el cuaderno de simulación") from None
        self.con.commit()
        return {"monto": amount, **proposal}

    def settle(self, results: pd.DataFrame, now=None) -> int:
        now = _now(now)
        b = self.bets()
        n = 0
        for _, r in b[b["resultado"].isna()].iterrows():
            m = _find(results, r["div"], r["home"], r["away"], pd.Timestamp(r["fecha_partido"]))
            if len(m) != 1:
                continue
            hg, ag = int(m["hg"].iloc[0]), int(m["ag"].iloc[0])
            won = int(r["opcion"]) == (0 if hg > ag else 1 if hg == ag else 2)
            pnl = r["monto"] * (r["cuota"] - 1) if won else -r["monto"]
            self.con.execute("INSERT INTO liquidaciones (created_at, apuesta_id, hg, ag, ganada, resultado) "
                             "VALUES (?,?,?,?,?,?)", (now.isoformat(timespec="seconds"), int(r["id"]),
                                                      hg, ag, int(won), round(float(pnl), 2)))
            n += 1
        self.con.commit()
        return n


def _now(now=None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    ts = pd.Timestamp(now)
    return (ts.tz_localize("UTC") if ts.tzinfo is None else ts).to_pydatetime()


def _find(results, div, home, away, day) -> pd.DataFrame:
    dates = pd.to_datetime(results["date"])
    return results[(results["div"] == div) & (results["home"] == home) & (results["away"] == away)
                   & ((dates - day).abs() <= pd.Timedelta(days=1))]
