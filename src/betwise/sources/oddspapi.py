"""Cuotas históricas de Pinnacle vía OddsPapi (https://oddspapi.io).

- La clave se lee de .env (ODDSPAPI_KEY) o de la variable de entorno; nunca se imprime.
- /v4/fixtures descuenta de la cuota mensual (250 en el plan gratis): se guarda en caché por mes.
- /v4/historical-odds no descuenta según la documentación; hay pausa obligatoria de 5 s.
- Los términos prohíben redistribuir los datos: todo se guarda en data/raw/ (ignorado por git).

Cuota de cierre = última cotización ACTIVA con createdAt < hora programada de inicio (startTime).
Las cotizaciones en vivo (después del inicio) y las suspendidas se descartan.
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

API = "https://api.oddspapi.io/v4/"
USER_AGENT = "betwise-research/0.1"
COOLDOWN_S = 5.5
TOURNAMENTS = {"E0": 17, "SP1": 8}

# marketId -> {outcomeId: columna}; confirmado con GET /v4/markets (sportId=10) el 8-nov-2026
MARKETS = {
    "101": {"101": "o_h", "102": "o_d", "103": "o_a"},          # Full Time Result (1X2)
    "1010": {"1010": "o_over25", "1011": "o_under25"},        # Over/Under 2.5
    "104": {"104": "o_btts_y", "105": "o_btts_n"},            # Ambos anotan
}
PRICE_COLS = [c for m in MARKETS.values() for c in m.values()]


class OddsPapiError(RuntimeError):
    pass


def load_key(env_path: str | Path = ".env") -> str:
    key = os.environ.get("ODDSPAPI_KEY", "").strip()
    p = Path(env_path)
    if not key and p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ODDSPAPI_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key or key == "PEGA_AQUI_TU_CLAVE":
        raise OddsPapiError("No hay clave de OddsPapi: escribe ODDSPAPI_KEY=... en el archivo .env")
    return key


class Client:
    def __init__(self, key: str, cooldown: float = COOLDOWN_S, opener=None):
        self._key = key
        self.cooldown = cooldown
        self._open = opener or urllib.request.urlopen
        self._last = 0.0
        self.billable_calls = 0

    def __repr__(self):  # nunca mostrar la clave
        return "Client(key=***)"

    def get(self, path: str, billable: bool = True, **params):
        """GET a la API. Devuelve el JSON o None si la API responde 404."""
        wait = self.cooldown - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        url = API + path + "?" + urllib.parse.urlencode({**params, "apiKey": self._key})
        safe = API + path + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with self._open(req, timeout=60) as resp:
                body = resp.read()
            if billable:
                self.billable_calls += 1
            return json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise OddsPapiError(f"HTTP {e.code} en {safe}") from None
        except urllib.error.URLError as e:
            raise OddsPapiError(f"Sin conexión con OddsPapi ({safe}): {e.reason}") from None
        finally:
            self._last = time.monotonic()


def closing_prices(hist: dict, start_time, bookmaker: str = "pinnacle", which: str = "last") -> dict:
    """Cuotas a partir de la respuesta de historical-odds, solo con cotizaciones ACTIVAS
    anteriores a `start_time`. which="last": la última (cierre, o la vigente a una hora dada);
    which="first": la primera publicada (apertura)."""
    first = which == "first"
    start = pd.Timestamp(start_time)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    out = {c: float("nan") for c in PRICE_COLS}
    last_ts = {}
    markets = ((hist or {}).get("bookmakers", {}).get(bookmaker) or {}).get("markets", {})
    for mid, outcomes in MARKETS.items():
        m = markets.get(mid) or {}
        for oid, col in outcomes.items():
            rows = ((m.get("outcomes", {}).get(oid) or {}).get("players") or {}).get("0") or []
            best = None
            for r in rows:
                if not r.get("active") or r.get("price") is None:
                    continue
                ts = pd.Timestamp(r["createdAt"])
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                if ts < start and (best is None or (ts < best[0] if first else ts > best[0])):
                    best = (ts, float(r["price"]))
            if best and best[1] > 1.0:
                out[col] = best[1]
                last_ts[col] = best[0]
    one_x_two = [last_ts[c] for c in ("o_h", "o_d", "o_a") if c in last_ts]
    ref = (max(one_x_two) if first else min(one_x_two)) if len(one_x_two) == 3 else None
    out["quote_time"] = ref.isoformat() if ref is not None else ""
    out["min_before_ko"] = (start - ref).total_seconds() / 60 if ref is not None else float("nan")
    return out


def _months(start: pd.Timestamp, end: pd.Timestamp):
    m = pd.Timestamp(year=start.year, month=start.month, day=1)
    while m <= end:
        nxt = m + pd.offsets.MonthBegin(1)
        yield m, nxt - pd.Timedelta(seconds=1)
        m = nxt


def fixtures(client: Client, div: str, start: str, end: str, cache_dir: Path) -> list:
    """Partidos terminados (statusId=2) por mes, con caché. El mes en curso se vuelve a pedir."""
    cache_dir = Path(cache_dir) / "fixtures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    out = []
    for a, b in _months(pd.Timestamp(start), min(pd.Timestamp(end), now)):
        path = cache_dir / f"{div}_{a:%Y-%m}.json"
        closed_month = b < now - pd.Timedelta(days=3)
        if path.exists() and closed_month:
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = client.get("fixtures", tournamentId=TOURNAMENTS[div], statusId=2,
                              **{"from": f"{a:%Y-%m-%dT%H:%M:%SZ}", "to": f"{b:%Y-%m-%dT%H:%M:%SZ}"}) or []
            path.write_text(json.dumps(data), encoding="utf-8")
        out.extend(data)
    return out


def upcoming_fixtures(client: Client, div: str, now=None, days: int = 7) -> list:
    """Partidos por jugar (statusId=0) de los próximos `days` días. 1 consulta facturable."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    end = now + pd.Timedelta(days=days)
    return client.get("fixtures", tournamentId=TOURNAMENTS[div], statusId=0,
                      **{"from": f"{now:%Y-%m-%dT%H:%M:%SZ}", "to": f"{end:%Y-%m-%dT%H:%M:%SZ}"}) or []


def update_closing(client: Client, div: str, start: str, end: str, cache_dir="data/raw/oddspapi",
                   verbose: bool = True) -> pd.DataFrame:
    """Actualiza la tabla de cierres de una liga. Solo consulta partidos que aún no están en ella."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"closing_{div}.csv"
    have = pd.read_csv(path, dtype={"fixtureId": str}) if path.exists() else pd.DataFrame(columns=["fixtureId"])
    done = set(have["fixtureId"])
    todo = [f for f in fixtures(client, div, start, end, cache_dir) if f["fixtureId"] not in done]
    rows = []
    for i, f in enumerate(todo, 1):
        hist = client.get("historical-odds", billable=False, fixtureId=f["fixtureId"], bookmakers="pinnacle")
        row = {"fixtureId": f["fixtureId"], "div": div, "start_time": f["startTime"],
               "home_src": f.get("participant1Name", ""), "away_src": f.get("participant2Name", "")}
        if hist is None:
            row.update({c: float("nan") for c in PRICE_COLS}, quote_time="", min_before_ko=float("nan"),
                       status="sin_datos")
        else:
            cp = closing_prices(hist, f["startTime"])
            row.update(cp, status="ok" if cp["quote_time"] else "incompleto")
        rows.append(row)
        if verbose and (i % 25 == 0 or i == len(todo)):
            print(f"OddsPapi {div}: {i}/{len(todo)} partidos consultados")
        if i % 25 == 0:  # guardar avance por si se corta la conexión
            pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True) if rows else have
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out
