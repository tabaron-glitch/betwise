"""Equivalencias de nombres de equipos entre fuentes.

El nombre canónico es el de openfootball (fuente de los resultados con que se entrena el modelo).
La tabla data/team_names.csv solo lista los nombres que difieren; un nombre que no está en la
tabla se acepta tal cual SOLO si ya es canónico. Si no, se lanza error: nunca se adivina.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_TABLE = Path(__file__).resolve().parents[2] / "data" / "team_names.csv"


class UnknownTeamError(ValueError):
    pass


def load_table(path=DEFAULT_TABLE) -> pd.DataFrame:
    t = pd.read_csv(path, encoding="utf-8")
    dup = t.duplicated(["source", "div", "source_name"])
    if dup.any():
        raise ValueError(f"Nombres repetidos en {path}: {t[dup].to_dict('records')}")
    return t


def to_canonical(names, source: str, div: str, canonical: set, table: pd.DataFrame | None = None) -> list:
    """Traduce nombres de `source` al nombre canónico. Error si alguno no se puede resolver."""
    table = load_table() if table is None else table
    t = table[(table["source"] == source) & (table["div"] == div)]
    mapping = dict(zip(t["source_name"], t["canonical"]))
    bad_targets = sorted(set(mapping.values()) - set(canonical))
    out, unknown = [], set()
    for n in names:
        c = mapping.get(n, n)
        if c not in canonical:
            unknown.add(n)
        out.append(c)
    if unknown:
        raise UnknownTeamError(
            f"Equipos de {source}/{div} sin equivalencia: {sorted(unknown)}. "
            f"Agrégalos a data/team_names.csv"
            + (f" (ojo, destinos que no existen en openfootball: {bad_targets})" if bad_targets else "")
        )
    return out
