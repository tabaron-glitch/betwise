"""Resultados de la temporada en curso vía football-data.org (API v4, plan gratis).

Se usa porque openfootball se actualiza con semanas de retraso. Datos según la
documentación oficial (8-nov-2026): cabecera X-Auth-Token, 10 consultas/minuto en el plan
gratis, marcadores con retraso (no dicen cuánto). Premier = PL, LaLiga = PD.
Clave en .env: FOOTBALLDATA_KEY=...  (nunca se imprime). Datos en data/raw/ (no se publican).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.football-data.org/v4/"
COMPETITIONS = {"E0": "PL", "SP1": "PD"}
COOLDOWN_S = 6.5  # 10 consultas/minuto


class FootballDataError(RuntimeError):
    pass


def load_key(env_path: str | Path = ".env") -> str:
    key = os.environ.get("FOOTBALLDATA_KEY", "").strip()
    p = Path(env_path)
    if not key and p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("FOOTBALLDATA_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key or key == "PEGA_AQUI_TU_CLAVE":
        raise FootballDataError("No hay clave de football-data.org: escribe FOOTBALLDATA_KEY=... en .env")
    return key


class Client:
    def __init__(self, key: str, cooldown: float = COOLDOWN_S, opener=None):
        self._key = key
        self.cooldown = cooldown
        self._open = opener or urllib.request.urlopen
        self._last = 0.0

    def __repr__(self):
        return "Client(key=***)"

    def get(self, path: str, **params):
        wait = self.cooldown - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
        req = urllib.request.Request(url, headers={"X-Auth-Token": self._key, "User-Agent": "betwise-research"})
        try:
            with self._open(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise FootballDataError(f"HTTP {e.code} en {url}") from None
        except urllib.error.URLError as e:
            raise FootballDataError(f"Sin conexión con football-data.org ({url}): {e.reason}") from None
        finally:
            self._last = time.monotonic()


def parse_matches(data: dict, div: str) -> pd.DataFrame:
    """Partidos FINISHED con marcador final al esquema común (fecha = día UTC)."""
    rows = []
    for m in data.get("matches", []):
        ft = (m.get("score") or {}).get("fullTime") or {}
        if m.get("status") != "FINISHED" or ft.get("home") is None or ft.get("away") is None:
            continue
        ts = pd.Timestamp(m["utcDate"])
        rows.append({"date": ts.tz_convert("UTC").tz_localize(None).normalize(), "div": div,
                     "home": m["homeTeam"]["name"].strip(), "away": m["awayTeam"]["name"].strip(),
                     "hg": int(ft["home"]), "ag": int(ft["away"]), "utc_kickoff": ts.isoformat()})
    df = pd.DataFrame(rows, columns=["date", "div", "home", "away", "hg", "ag", "utc_kickoff"])
    return df.astype({"hg": int, "ag": int})


def finished_matches(client: Client, div: str, date_from: str, date_to: str,
                     cache_dir="data/raw/footballdata") -> pd.DataFrame:
    """Consulta (y guarda copia de) los partidos terminados entre dos fechas."""
    data = client.get(f"competitions/{COMPETITIONS[div]}/matches", status="FINISHED",
                      dateFrom=date_from, dateTo=date_to)
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{div}_{date_from}_{date_to}.json").write_text(json.dumps(data), encoding="utf-8")
    return parse_matches(data, div)
