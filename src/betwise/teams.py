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


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join("".join(c if c.isalnum() else " " for c in s.lower()).split())


def resolve_user_team(query: str, canonical: set, div: str | None = None,
                      table: pd.DataFrame | None = None) -> str:
    """Nombre escrito por una persona -> nombre canónico, sin adivinar.

    1) coincidencia exacta (sin mayúsculas ni tildes) con un canónico o con un alias de la tabla;
    2) si no, cada palabra escrita debe ser el comienzo de una palabra del nombre
       ("Man City" -> "Manchester City FC"). Debe quedar UN solo candidato; si hay
       varios o ninguno, error con las opciones.
    """
    q = _norm(query)
    if not q:
        raise UnknownTeamError("Nombre de equipo vacío")
    canonical = set(canonical)
    aliases = {}
    if table is not None or DEFAULT_TABLE.exists():
        t = load_table() if table is None else table
        if div is not None:
            t = t[t["div"] == div]
        for a, c in zip(t["source_name"], t["canonical"]):
            if c in canonical:
                aliases.setdefault(_norm(a), set()).add(c)
    exact = {c for c in canonical if _norm(c) == q} | aliases.get(q, set())
    if len(exact) == 1:
        return exact.pop()
    toks = q.split()

    def ok(name):
        words = _norm(name).split()
        return all(any(w.startswith(tk) for w in words) for tk in toks)

    cands = {c for c in canonical if ok(c)} | {c for a, cs in aliases.items() if ok(a) for c in cs}
    if len(cands) == 1:
        return cands.pop()
    if cands:
        raise UnknownTeamError(f"'{query}' es ambiguo; escribe uno de: {sorted(cands)}")
    raise UnknownTeamError(f"No encuentro '{query}'. Equipos disponibles: {sorted(canonical)}")
