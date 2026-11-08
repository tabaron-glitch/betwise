import sqlite3

import pytest

from betwise.ledger import Ledger, LedgerError

PRON = dict(fixture_id="f1", div="E0", kickoff="2026-11-10T15:00:00+00:00", home="Arsenal FC",
            away="Chelsea FC", model_version="abc1234", params='{"xi": 0.0023}',
            data_cutoff="2026-11-09", n_train=1500, known_teams=1,
            p_h=0.5, p_d=0.25, p_a=0.25, p_over25=0.55, p_btts=0.5)


@pytest.fixture
def led(tmp_path):
    with Ledger(tmp_path / "l.db") as l:
        yield l


def test_inserta_y_verifica(led):
    assert led.append("pronosticos", PRON) == 1
    led.append("resultados", dict(fixture_id="f1", hg=2, ag=1, source="prueba"))
    assert led.verify() == {"pronosticos": 1, "cuotas": 0, "resultados": 1}


def test_no_duplica_misma_clave(led):
    assert led.append("pronosticos", PRON) == 1
    assert led.append("pronosticos", {**PRON, "p_h": 0.9}) is None  # el pasado no se reescribe
    assert led.rows("pronosticos")[0]["p_h"] == 0.5


@pytest.mark.parametrize("sql", ["UPDATE pronosticos SET p_h = 0.9", "DELETE FROM pronosticos"])
def test_update_y_delete_prohibidos(led, sql):
    led.append("pronosticos", PRON)
    with pytest.raises(sqlite3.IntegrityError, match="inmutable"):
        led.con.execute(sql)


def test_detecta_alteracion_aunque_quiten_los_triggers(led):
    led.append("pronosticos", PRON)
    led.append("pronosticos", {**PRON, "fixture_id": "f2"})
    led.con.execute("DROP TRIGGER no_update_pronosticos")
    led.con.execute("UPDATE pronosticos SET p_h = 0.9 WHERE id = 1")
    with pytest.raises(LedgerError, match="alterada"):
        led.verify()


def test_detecta_fila_borrada(led):
    for f in ("f1", "f2", "f3"):
        led.append("pronosticos", {**PRON, "fixture_id": f})
    led.con.execute("DROP TRIGGER no_delete_pronosticos")
    led.con.execute("DELETE FROM pronosticos WHERE id = 2")
    with pytest.raises(LedgerError, match="cadena rota"):
        led.verify()


def test_correccion_de_resultado_es_fila_nueva(led):
    i = led.append("resultados", dict(fixture_id="f1", hg=2, ag=1, source="prueba"))
    assert led.append("resultados", dict(fixture_id="f1", hg=2, ag=2, source="prueba")) is None
    j = led.append("resultados", dict(fixture_id="f1", hg=2, ag=2, source="prueba", corrige_id=i))
    assert j and [r["ag"] for r in led.rows("resultados")] == [1, 2]
    led.verify()


def test_rechaza_columnas_desconocidas_y_tipo_invalido(led):
    with pytest.raises(LedgerError):
        led.append("pronosticos", {**PRON, "inventada": 1})
    with pytest.raises(sqlite3.IntegrityError):
        led.append("cuotas", dict(fixture_id="f1", bookmaker="pinnacle", tipo="otra"))
