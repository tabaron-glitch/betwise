"""Resultados de openfootball/football.json (dominio público, CC0 1.0).

https://github.com/openfootball/football.json — sin cuotas: solo sirve para entrenar el modelo.
Formatos de marcador vistos en los archivos:
  - {"ft": [1, 0], "ht": [0, 0]}  (lo habitual)
  - [0, 0]                        (solo resultado final, aparece desde 2025/26)
  - sin "score"                   (partido aún no jugado o dato faltante: se descarta)
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"
USER_AGENT = "betwise-research/0.1"

# código interno de liga -> archivo de openfootball
LEAGUE_CODES = {"E0": "en.1", "SP1": "es.1"}


def season_folder(season: str) -> str:
    """'2425' -> '2024-25' (mismo código de temporada que usa el resto del proyecto)."""
    return f"20{season[:2]}-{season[2:]}"


def download_json(div: str, season: str, cache_dir: Path, refresh: bool = False, timeout: int = 30) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{div}_{season}.json"
    if path.exists() and not refresh:
        return path
    url = BASE_URL.format(season=season_folder(season), code=LEAGUE_CODES[div])
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        path.write_bytes(resp.read())
    time.sleep(0.5)
    return path


def _final_score(score):
    """Devuelve (goles local, goles visitante) o None si no hay resultado final."""
    if isinstance(score, dict):
        score = score.get("ft")
    if isinstance(score, (list, tuple)) and len(score) == 2 and all(isinstance(g, int) for g in score):
        return int(score[0]), int(score[1])
    return None


def fix_season_year(date: pd.Timestamp, season: str) -> pd.Timestamp:
    """Corrige un año mal escrito: una fecha anterior al 1 de agosto del año de inicio de la
    temporada se pasa al año siguiente (caso real: Premier 2019/20, partidos de julio de 2020
    publicados como julio de 2019). Fuera de ese caso la fecha no se toca."""
    y0 = 2000 + int(season[:2])
    if date < pd.Timestamp(y0, 8, 1):
        fixed = date + pd.DateOffset(years=1)
        if fixed <= pd.Timestamp(y0 + 1, 8, 31):
            return fixed
    return date


def parse(data: dict, div: str, season: str) -> tuple[pd.DataFrame, int]:
    """Convierte el JSON al esquema común. Devuelve (partidos jugados, n descartados sin marcador).
    El atributo df.attrs['fechas_corregidas'] cuenta las fechas con año corregido."""
    rows, skipped, fixed = [], 0, 0
    for m in data.get("matches", []):
        sc = _final_score(m.get("score"))
        if sc is None:
            skipped += 1
            continue
        raw_date = pd.Timestamp(m["date"])
        date = fix_season_year(raw_date, season)
        fixed += date != raw_date
        rows.append({
            "date": date,
            "div": div,
            "season": season,
            "home": str(m["team1"]).strip(),
            "away": str(m["team2"]).strip(),
            "hg": sc[0],
            "ag": sc[1],
            "time_local": m.get("time", ""),
        })
    cols = ["date", "div", "season", "home", "away", "hg", "ag", "time_local"]
    df = pd.DataFrame(rows, columns=cols)
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df.attrs["fechas_corregidas"] = int(fixed)
    return df, skipped


def load_results(divs, seasons, cache_dir="data/raw/openfootball", refresh_seasons=(), verbose=True) -> pd.DataFrame:
    """Carga resultados jugados de varias ligas/temporadas, ordenados por fecha."""
    frames = []
    for div in divs:
        for season in seasons:
            path = download_json(div, season, Path(cache_dir), refresh=season in refresh_seasons)
            df, skipped = parse(json.loads(path.read_text(encoding="utf-8")), div, season)
            if verbose and skipped:
                print(f"openfootball {div} {season}: {len(df)} jugados, {skipped} sin marcador (descartados)")
            if verbose and df.attrs.get("fechas_corregidas"):
                print(f"openfootball {div} {season}: {df.attrs['fechas_corregidas']} fechas con año corregido")
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    dup = out.duplicated(["div", "date", "home", "away"])
    if dup.any():
        raise ValueError(f"openfootball: {int(dup.sum())} partidos duplicados")
    return out.sort_values(["date", "div", "home"]).reset_index(drop=True)
