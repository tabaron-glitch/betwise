"""Descarga y estandarización de CSV de football-data.co.uk.

- Uso personal/investigación. El sitio reclama el copyright de los archivos:
  por eso data/raw está en .gitignore y NO se redistribuyen en el repositorio.
- Las columnas de cuotas varían según liga y temporada; se elige, fila por fila,
  el primer conjunto completo en este orden de preferencia.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
USER_AGENT = "betwise-research/0.1"

LEAGUES = {"E0": "Premier League", "SP1": "LaLiga"}

# (etiqueta, columnas H/D/A) en orden de preferencia: cierre sharp primero
ODDS_SETS = [
    ("Pinnacle cierre", ("PSCH", "PSCD", "PSCA")),
    ("Promedio cierre", ("AvgCH", "AvgCD", "AvgCA")),
    ("Bet365 cierre", ("B365CH", "B365CD", "B365CA")),
    ("Promedio", ("AvgH", "AvgD", "AvgA")),
    ("Bet365", ("B365H", "B365D", "B365A")),
]


def download_csv(div: str, season: str, cache_dir: Path, refresh: bool = False, timeout: int = 30) -> Path:
    """Descarga (con caché) el CSV de una liga/temporada. season: '2425' = 2024/25."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{div}_{season}.csv"
    if path.exists() and not refresh:
        return path
    req = urllib.request.Request(BASE_URL.format(season=season, div=div), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        path.write_bytes(resp.read())
    time.sleep(1.0)  # cortesía con el servidor
    return path


def standardize(raw: pd.DataFrame, div: str, season: str) -> pd.DataFrame:
    """Convierte un CSV crudo al esquema común del proyecto."""
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce"),
            "div": div,
            "season": season,
            "home": raw["HomeTeam"].astype(str).str.strip(),
            "away": raw["AwayTeam"].astype(str).str.strip(),
            "hg": raw["FTHG"].astype(int),
            "ag": raw["FTAG"].astype(int),
        }
    )
    o = np.full((len(raw), 3), np.nan)
    source = np.array([""] * len(raw), dtype=object)
    for label, cols in ODDS_SETS:
        if not all(c in raw.columns for c in cols):
            continue
        vals = raw[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        ok = np.isfinite(vals).all(axis=1) & (vals > 1.0).all(axis=1) & np.isnan(o[:, 0])
        o[ok] = vals[ok]
        source[ok] = label
    out["o_h"], out["o_d"], out["o_a"] = o[:, 0], o[:, 1], o[:, 2]
    out["odds_source"] = source
    return out.dropna(subset=["date"])


def load_matches(divs, seasons, cache_dir="data/raw", refresh_seasons=()) -> pd.DataFrame:
    """Carga y concatena ligas/temporadas. `refresh_seasons`: temporadas a re-descargar
    (la temporada en curso cambia cada semana)."""
    frames = []
    for div in divs:
        for season in seasons:
            path = download_csv(div, season, Path(cache_dir), refresh=season in refresh_seasons)
            raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
            frames.append(standardize(raw, div, season))
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
