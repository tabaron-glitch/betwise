import pandas as pd
import pytest

from betwise.teams import UnknownTeamError, load_table, to_canonical

CANON = {"Real Betis Balompié", "Sevilla FC", "Club Atlético de Madrid"}
TABLE = pd.DataFrame([
    {"source": "oddspapi", "div": "SP1", "source_name": "Real Betis Seville", "canonical": "Real Betis Balompié"},
    {"source": "oddspapi", "div": "SP1", "source_name": "Atletico Madrid", "canonical": "Club Atlético de Madrid"},
])


def test_traduce_y_acepta_nombres_ya_canonicos():
    out = to_canonical(["Real Betis Seville", "Sevilla FC", "Atletico Madrid"], "oddspapi", "SP1", CANON, TABLE)
    assert out == ["Real Betis Balompié", "Sevilla FC", "Club Atlético de Madrid"]


def test_nombre_desconocido_lanza_error_y_lo_nombra():
    with pytest.raises(UnknownTeamError, match="Real Madrid"):
        to_canonical(["Real Madrid"], "oddspapi", "SP1", CANON, TABLE)


def test_la_tabla_es_por_liga():
    with pytest.raises(UnknownTeamError):
        to_canonical(["Real Betis Seville"], "oddspapi", "E0", CANON, TABLE)


def test_tabla_del_proyecto_es_valida_y_utf8():
    t = load_table()
    assert set(t.columns) == {"source", "div", "source_name", "canonical"}
    assert "Real Betis Balompié" in set(t.canonical)  # tildes leídas bien
