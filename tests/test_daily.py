import pandas as pd
import pytest

from betwise import daily
from betwise.ledger import Ledger
from betwise.synthetic import make_synthetic

NOW = pd.Timestamp("2022-03-01T06:00:00Z")
TABLE = pd.DataFrame(columns=["source", "div", "source_name", "canonical"])


@pytest.fixture(scope="module")
def results():
    r = make_synthetic(n_seasons=4, start_year=2018)
    return r.assign(div="E0")[["date", "div", "home", "away", "hg", "ag"]]


@pytest.fixture
def led(tmp_path):
    with Ledger(tmp_path / "l.db") as l:
        yield l


def fx(fid, start, home="Team01", away="Team02"):
    return {"fixtureId": fid, "div": "E0", "startTime": start, "participant1Name": home, "participant2Name": away}


def canon(results):
    return {"E0": set(results.home) | set(results.away)}


def test_solo_pronostica_partidos_futuros_dentro_del_horizonte(led, results):
    fixtures = [fx("pasado", "2022-03-01T05:00:00Z"), fx("ok", "2022-03-02T15:00:00Z"),
                fx("lejos", "2022-03-05T15:00:00Z")]
    ids = daily.predict_upcoming(led, fixtures, results, canon(results), NOW, "v1", table=TABLE)
    rows = led.rows("pronosticos")
    assert len(ids) == 1 and rows[0]["fixture_id"] == "ok"
    assert rows[0]["created_at"] < rows[0]["kickoff"]
    assert abs(rows[0]["p_h"] + rows[0]["p_d"] + rows[0]["p_a"] - 1) < 1e-9


def test_sin_fuga_resultados_del_dia_o_futuros_no_cambian_el_pronostico(tmp_path, results):
    past = results[results.date < "2022-03-01"]
    future = pd.DataFrame([{"date": pd.Timestamp("2022-03-01"), "div": "E0", "home": "Team01", "away": "Team03",
                            "hg": 9, "ag": 0}, {"date": pd.Timestamp("2022-03-20"), "div": "E0",
                            "home": "Team02", "away": "Team01", "hg": 9, "ag": 0}])
    out = []
    for i, data in enumerate((past, pd.concat([past, future]))):
        with Ledger(tmp_path / f"{i}.db") as l:
            daily.predict_upcoming(l, [fx("ok", "2022-03-02T15:00:00Z")], data, canon(results), NOW, "v1",
                                   table=TABLE)
            out.append(l.rows("pronosticos")[0])
    for k in ("p_h", "p_d", "p_a", "p_over25", "n_train", "data_cutoff"):
        assert out[0][k] == out[1][k]
    assert out[0]["data_cutoff"] < "2022-03-01"


def test_no_repite_pronostico_de_la_misma_version(led, results):
    f = [fx("ok", "2022-03-02T15:00:00Z")]
    assert daily.predict_upcoming(led, f, results, canon(results), NOW, "v1", table=TABLE)
    assert daily.predict_upcoming(led, f, results, canon(results), NOW + pd.Timedelta(hours=1), "v1",
                                  table=TABLE) == []
    assert daily.predict_upcoming(led, f, results, canon(results), NOW, "v2", table=TABLE)


def hist(rows_by_outcome):
    return {"bookmakers": {"pinnacle": {"markets": {"101": {"outcomes": {
        o: {"players": {"0": [{"createdAt": t, "price": p, "active": True} for t, p in rs]}}
        for o, rs in rows_by_outcome.items()}}}}}}


def test_liquidacion_agrega_resultado_y_tres_cuotas_sin_tocar_pronosticos(led, results):
    daily.predict_upcoming(led, [fx("ok", "2022-03-02T15:00:00Z")], results, canon(results), NOW, "v1",
                           table=TABLE)
    before = led.rows("pronosticos")
    played = pd.DataFrame([{"date": pd.Timestamp("2022-03-02"), "div": "E0", "home": "Team01",
                            "away": "Team02", "hg": 2, "ag": 1}])
    H = hist({"101": [("2022-02-25T10:00:00Z", 2.0), ("2022-03-01T12:00:00Z", 1.9), ("2022-03-02T16:00:00Z", 9.0)],
              "102": [("2022-02-25T10:00:00Z", 3.5)], "103": [("2022-02-25T10:00:00Z", 4.0)]})
    # antes de que termine: no liquida
    assert daily.settle(led, played, lambda f: H, "2022-03-02T16:00:00Z")["resultados_nuevos"] == 0
    rep = daily.settle(led, played, lambda f: H, "2022-03-03T06:00:00Z", source="prueba")
    assert rep["resultados_nuevos"] == 1 and rep["cuotas_nuevas"] == 1
    q = {r["tipo"]: r for r in led.rows("cuotas")}
    assert q["apertura"]["o_h"] == 2.0 and q["al_pronostico"]["o_h"] == 2.0 and q["cierre"]["o_h"] == 1.9
    assert led.rows("resultados")[0]["hg"] == 2
    assert led.rows("pronosticos") == before
    assert daily.settle(led, played, lambda f: H, "2022-03-04T06:00:00Z")["cuotas_nuevas"] == 0
    led.verify()


def test_liquidacion_espera_si_aun_no_hay_datos(led, results):
    daily.predict_upcoming(led, [fx("ok", "2022-03-02T15:00:00Z")], results, canon(results), NOW, "v1",
                           table=TABLE)
    rep = daily.settle(led, results.iloc[:0], lambda f: None, "2022-03-03T06:00:00Z")
    assert rep == {"resultados_nuevos": 0, "cuotas_nuevas": 0, "sin_resultado_aun": 1, "sin_cuotas_aun": 1}


def test_combine_results_prefiere_la_fuente_principal():
    a = pd.DataFrame([{"date": pd.Timestamp("2026-09-20"), "div": "E0", "home": "X", "away": "Y", "hg": 1, "ag": 0}])
    b = pd.DataFrame([{"date": pd.Timestamp("2026-09-21"), "div": "E0", "home": "X", "away": "Y", "hg": 5, "ag": 5},
                      {"date": pd.Timestamp("2026-10-01"), "div": "E0", "home": "Y", "away": "X", "hg": 2, "ag": 2}])
    c = daily.combine_results(a, b)
    assert list(c.hg) == [1, 2]


def test_season_codes():
    assert daily.season_codes(pd.Timestamp("2026-11-08"), 2) == ["2425", "2526", "2627"]
    assert daily.season_codes(pd.Timestamp("2027-03-01"), 1) == ["2526", "2627"]


def test_fit_for_cutoff_sin_fuga_y_igual_al_registro_diario(tmp_path, results):
    future = pd.DataFrame([{"date": pd.Timestamp("2022-03-01"), "div": "E0", "home": "Team01", "away": "Team03",
                            "hg": 9, "ag": 0}])
    m1, last1 = daily.fit_for_cutoff(results[results.date < "2022-03-01"], "E0", NOW)
    m2, last2 = daily.fit_for_cutoff(pd.concat([results[results.date < "2022-03-01"], future]), "E0", NOW)
    assert last1 == last2 < pd.Timestamp("2022-03-01")
    assert (m1.score_matrix("Team01", "Team03") == m2.score_matrix("Team01", "Team03")).all()
    with Ledger(tmp_path / "x.db") as l:
        daily.predict_upcoming(l, [fx("ok", "2022-03-02T15:00:00Z")], results, canon(results), NOW, "v1",
                               table=TABLE)
        r = l.rows("pronosticos")[0]
    from betwise.dixon_coles import outcome_probs
    p = outcome_probs(m1.score_matrix("Team01", "Team02"))
    assert abs(p[0] - r["p_h"]) < 1e-12 and abs(p[2] - r["p_a"]) < 1e-12
