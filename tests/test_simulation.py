import sqlite3

import pandas as pd
import pytest

from betwise.simulation import Notebook, SimulationError, propose

NOW = pd.Timestamp("2026-11-20T15:00:00Z")
EMPTY = pd.DataFrame(columns=["date", "div", "home", "away", "hg", "ag"])


def res(home, away, day, hg, ag):
    return pd.DataFrame([{"date": pd.Timestamp(day), "div": "E0", "home": home, "away": away, "hg": hg, "ag": ag}])


@pytest.fixture
def nb(tmp_path):
    with Notebook(tmp_path / "s.db") as n:
        n.start(1_000_000)
        yield n


def test_propone_solo_con_ev_pesimista_positivo():
    assert propose([0.50, 0.25, 0.25], [1.80, 3.60, 4.00]) is None          # sin valor
    p = propose([0.50, 0.25, 0.25], [2.30, 3.60, 4.00])                       # 0.47*2.30-1 = +8.1 %
    assert p["nombre"] == "Local" and p["ev_pesimista"] == pytest.approx(0.081)
    assert 0 < p["fraccion"] <= 0.02


def test_registra_liquida_y_actualiza_banca(nb):
    p = propose([0.50, 0.25, 0.25], [2.30, 3.60, 4.00])
    r = nb.register("E0", "A", "B", "2026-11-21", p, EMPTY, now=NOW)
    assert r["monto"] == pytest.approx(round(p["fraccion"] * 1_000_000, 2))
    assert nb.settle(EMPTY, now=NOW) == 0
    assert nb.settle(res("A", "B", "2026-11-21", 2, 1), now=NOW) == 1
    st = nb.state(NOW)
    assert st["banca"] == pytest.approx(1_000_000 + r["monto"] * 1.30)
    assert st["ganadas"] == 1 and st["pendientes"] == 0


def test_no_simula_pasados_ni_jugados_ni_repetidos(nb):
    p = propose([0.50, 0.25, 0.25], [2.30, 3.60, 4.00])
    with pytest.raises(SimulationError, match="ya pasó"):
        nb.register("E0", "A", "B", "2026-11-19", p, EMPTY, now=NOW)
    with pytest.raises(SimulationError, match="resultado"):
        nb.register("E0", "A", "B", "2026-11-20", p, res("A", "B", "2026-11-20", 1, 0), now=NOW)
    nb.register("E0", "A", "B", "2026-11-22", p, EMPTY, now=NOW)
    with pytest.raises(SimulationError, match="ya está"):
        nb.register("E0", "A", "B", "2026-11-22", p, EMPTY, now=NOW)


def test_limite_diario(nb):
    p = propose([0.50, 0.25, 0.25], [2.30, 3.60, 4.00])
    p = {**p, "fraccion": 0.02}
    for i in range(2):
        nb.register("E0", f"A{i}", "B", "2026-11-22", p, EMPTY, now=NOW)
    with pytest.raises(SimulationError, match="diario"):  # 2 % + 2 % + 2 % > 5 %
        nb.register("E0", "A9", "B", "2026-11-22", p, EMPTY, now=NOW)


def test_pausa_por_caida_y_sin_martingala(nb):
    p = {**propose([0.50, 0.25, 0.25], [2.30, 3.60, 4.00]), "fraccion": 0.02}
    day = pd.Timestamp("2026-11-20T15:00:00Z")
    for i in range(10):  # pierde 2 % de la banca cada vez, en días distintos para no tocar límites
        now = day + pd.Timedelta(days=8 * i)
        d = (now + pd.Timedelta(days=1)).date()
        before = nb.state(now)["banca"]
        try:
            r = nb.register("E0", f"A{i}", "B", str(d), p, EMPTY, now=now)
        except SimulationError as e:
            assert "pausa" in str(e)
            break
        assert r["monto"] == pytest.approx(0.02 * before, rel=1e-6)  # el monto baja con la banca
        nb.settle(res(f"A{i}", "B", str(d), 0, 1), now=now + pd.Timedelta(days=2))
    assert nb.state(now)["pausada"]


def test_cuaderno_inmutable_y_banca_fija(nb):
    p = propose([0.50, 0.25, 0.25], [2.30, 3.60, 4.00])
    nb.register("E0", "A", "B", "2026-11-21", p, EMPTY, now=NOW)
    with pytest.raises(sqlite3.DatabaseError, match="inmutable"):
        nb.con.execute("UPDATE apuestas SET monto = 1")
    with pytest.raises(sqlite3.DatabaseError, match="inmutable"):
        nb.con.execute("DELETE FROM apuestas")
    with pytest.raises(SimulationError):
        nb.start(5)
