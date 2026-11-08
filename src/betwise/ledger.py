"""Libro (ledger) inmutable de pronósticos, cuotas y resultados en SQLite.

Reglas:
- Solo inserciones: triggers rechazan UPDATE y DELETE.
- Cada fila guarda la huella (SHA-256) de su contenido más la huella de la fila anterior
  de la misma tabla: alterar o borrar una fila rompe la cadena y `verify()` lo detecta.
- Una corrección es una fila nueva con `corrige_id` apuntando a la original (como un ajuste
  contable); nunca se reescribe el pasado.

Los triggers protegen contra errores; contra manipulación deliberada protege la huella
encadenada (y el historial de git del repositorio de datos).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64

TABLES = {
    "pronosticos": [
        ("fixture_id", "TEXT NOT NULL"), ("div", "TEXT NOT NULL"), ("kickoff", "TEXT NOT NULL"),
        ("home", "TEXT NOT NULL"), ("away", "TEXT NOT NULL"),
        ("model_version", "TEXT NOT NULL"), ("params", "TEXT NOT NULL"),
        ("data_cutoff", "TEXT NOT NULL"), ("n_train", "INTEGER NOT NULL"), ("known_teams", "INTEGER NOT NULL"),
        ("p_h", "REAL NOT NULL"), ("p_d", "REAL NOT NULL"), ("p_a", "REAL NOT NULL"),
        ("p_over25", "REAL NOT NULL"), ("p_btts", "REAL NOT NULL"),
    ],
    "cuotas": [
        ("fixture_id", "TEXT NOT NULL"), ("bookmaker", "TEXT NOT NULL"),
        ("tipo", "TEXT NOT NULL CHECK (tipo IN ('apertura','al_pronostico','cierre'))"),
        ("quote_time", "TEXT"),
        ("o_h", "REAL"), ("o_d", "REAL"), ("o_a", "REAL"), ("o_over25", "REAL"), ("o_under25", "REAL"),
        ("o_btts_y", "REAL"), ("o_btts_n", "REAL"),
    ],
    "resultados": [
        ("fixture_id", "TEXT NOT NULL"), ("hg", "INTEGER NOT NULL"), ("ag", "INTEGER NOT NULL"),
        ("source", "TEXT NOT NULL"), ("corrige_id", "INTEGER"),
    ],
}
# Unicidad: no se registra dos veces lo mismo (las correcciones de resultados van aparte)
UNIQUE = {
    "pronosticos": "CREATE UNIQUE INDEX IF NOT EXISTS u_pron ON pronosticos(fixture_id, model_version)",
    "cuotas": "CREATE UNIQUE INDEX IF NOT EXISTS u_cuo ON cuotas(fixture_id, bookmaker, tipo)",
    "resultados": "CREATE UNIQUE INDEX IF NOT EXISTS u_res ON resultados(fixture_id) WHERE corrige_id IS NULL",
}


class LedgerError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(prev: str, table: str, created_at: str, values: dict) -> str:
    body = json.dumps({"t": table, "created_at": created_at, **values}, sort_keys=True, ensure_ascii=False,
                      default=str, allow_nan=True)
    return hashlib.sha256((prev + body).encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self._init()

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _init(self):
        c = self.con
        for t, cols in TABLES.items():
            cols_sql = ", ".join(f"{n} {d}" for n, d in cols)
            c.execute(f"CREATE TABLE IF NOT EXISTS {t} (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                      f"created_at TEXT NOT NULL, {cols_sql}, prev_hash TEXT NOT NULL, row_hash TEXT NOT NULL)")
            c.execute(UNIQUE[t])
            for op in ("UPDATE", "DELETE"):
                c.execute(f"CREATE TRIGGER IF NOT EXISTS no_{op.lower()}_{t} BEFORE {op} ON {t} "
                          f"BEGIN SELECT RAISE(ABORT, 'ledger inmutable: {op} prohibido en {t}'); END")
        c.commit()

    def _last_hash(self, table: str) -> str:
        r = self.con.execute(f"SELECT row_hash FROM {table} ORDER BY id DESC LIMIT 1").fetchone()
        return r[0] if r else GENESIS

    def append(self, table: str, values: dict, created_at: str | None = None) -> int | None:
        """Inserta una fila. Devuelve su id, o None si ya existía (misma clave única)."""
        names = [n for n, _ in TABLES[table]]
        extra = set(values) - set(names)
        if extra:
            raise LedgerError(f"Columnas desconocidas para {table}: {sorted(extra)}")
        vals = {n: values.get(n) for n in names}
        created_at = created_at or _now()
        prev = self._last_hash(table)
        h = _hash(prev, table, created_at, vals)
        cols = ["created_at", *names, "prev_hash", "row_hash"]
        try:
            cur = self.con.execute(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [created_at, *vals.values(), prev, h])
        except sqlite3.IntegrityError as e:
            if "UNIQUE" in str(e):
                return None
            raise
        self.con.commit()
        return cur.lastrowid

    def rows(self, table: str, where: str = "", params=()) -> list[dict]:
        q = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "") + " ORDER BY id"
        return [dict(r) for r in self.con.execute(q, params).fetchall()]

    def verify(self) -> dict:
        """Recalcula la cadena de huellas de cada tabla. Lanza LedgerError si algo no cuadra."""
        names = {t: [n for n, _ in cols] for t, cols in TABLES.items()}
        counts = {}
        for t in TABLES:
            prev = GENESIS
            rows = self.rows(t)
            for r in rows:
                if r["prev_hash"] != prev:
                    raise LedgerError(f"{t}: cadena rota en id={r['id']} (falta o se movió una fila)")
                h = _hash(prev, t, r["created_at"], {n: r[n] for n in names[t]})
                if h != r["row_hash"]:
                    raise LedgerError(f"{t}: la fila id={r['id']} fue alterada")
                prev = h
            counts[t] = len(rows)
        return counts
