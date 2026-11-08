import numpy as np
import pandas as pd
import pytest

from betwise.sources.merge import SOURCE_LABEL, attach_closing

TABLE = pd.DataFrame([{"source": "oddspapi", "div": "SP1", "source_name": "Real Betis Seville",
                       "canonical": "Real Betis Balompié"}])

RESULTS = pd.DataFrame({
    "date": pd.to_datetime(["2026-03-01", "2026-03-01", "2026-03-08", "2025-10-01"]),
    "div": "SP1", "season": "2526",
    "home": ["Real Betis Balompié", "Getafe CF", "Sevilla FC", "Real Betis Balompié"],
    "away": ["Sevilla FC", "Girona FC", "Real Betis Balompié", "Sevilla FC"],
    "hg": [1, 0, 2, 3], "ag": [1, 0, 0, 0],
})


def closing(**over):
    base = {"fixtureId": ["a", "b", "c"], "div": "SP1",
            "start_time": ["2026-02-28T23:30:00.000Z", "2026-03-08T20:00:00.000Z", "2026-03-20T20:00:00.000Z"],
            "home_src": ["Real Betis Seville", "Sevilla FC", "Getafe CF"],
            "away_src": ["Sevilla FC", "Real Betis Seville", "Girona FC"],
            "o_h": [2.1, 1.9, 2.0], "o_d": [3.4, 3.5, 3.2], "o_a": [3.6, 4.2, 4.0],
            "o_over25": np.nan, "o_under25": np.nan, "o_btts_y": np.nan, "o_btts_n": np.nan,
            "min_before_ko": [5.0, 400.0, 3.0], "status": "ok"}
    base.update(over)
    return pd.DataFrame(base)


def test_empareja_con_tolerancia_de_un_dia_y_reporta_sobrantes():
    df, rep = attach_closing(RESULTS, closing(), TABLE)
    assert rep["emparejados"] == 2 and rep["cierres_sin_resultado"] == 1  # "c": fecha no coincide
    got = df[df.odds_source == SOURCE_LABEL]
    assert list(got.o_h) == [2.1, 1.9]  # el del 28-feb UTC casó con el del 1-mar local
    assert df.loc[3, "odds_source"] == ""  # el Betis-Sevilla de octubre no recibe la cuota de marzo
    assert rep["cotizacion_vieja_mas_6h"] == 1


def test_descarta_margen_imposible_y_estado_no_ok():
    c = closing(o_h=[1.2, 1.9, 2.0], o_d=[1.2, 3.5, 3.2], o_a=[1.2, 4.2, 4.0], status=["ok", "ok", "sin_datos"])
    df, rep = attach_closing(RESULTS, c, TABLE)
    assert rep["margen_fuera_de_rango"] == 1 and rep["con_cuota"] == 1


def test_nombre_sin_equivalencia_detiene_todo():
    with pytest.raises(ValueError, match="Atletico"):
        attach_closing(RESULTS, closing(home_src=["Atletico Madrid", "Sevilla FC", "Getafe CF"]), TABLE)
