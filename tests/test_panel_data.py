import numpy as np
import pandas as pd
import pytest

from betwise import daily
from betwise.ledger import Ledger
from betwise.panel_data import forecast, ledger_view, read_ledger
from betwise.synthetic import make_synthetic

NOW = pd.Timestamp("2022-03-01T06:00:00Z")


@pytest.fixture(scope="module")
def results():
    return make_synthetic(n_seasons=4, start_year=2018).assign(div="E0")[["date", "div", "home", "away", "hg", "ag"]]


def _pred(fid, ko, ph=0.5, pd_=0.3, pa=0.2):
    return dict(fixture_id=fid, div="E0", kickoff=ko, home="A", away="B", model_version="v1", params="{}",
                data_cutoff="2022-02-28", n_train=100, known_teams=1, p_h=ph, p_d=pd_, p_a=pa,
                p_over25=0.5, p_btts=0.5)


def test_libro_vacio(tmp_path):
    with Ledger(tmp_path / "l.db"):
        pass
    tables, counts, settled = read_ledger(tmp_path / "l.db")
    assert counts == {"pronosticos": 0, "cuotas": 0, "resultados": 0}
    assert ledger_view(tables).empty and settled.empty


def test_vista_del_libro_y_no_modifica_el_archivo(tmp_path):
    path = tmp_path / "l.db"
    with Ledger(path) as led:
        led.append("pronosticos", _pred("f1", "2022-03-02T15:00:00+00:00"), created_at="2022-03-01T06:00:00+00:00")
        led.append("pronosticos", _pred("f2", "2022-03-03T15:00:00+00:00"), created_at="2022-03-01T06:00:00+00:00")
        led.append("resultados", dict(fixture_id="f1", hg=2, ag=2, source="t"))
        led.append("cuotas", dict(fixture_id="f1", bookmaker="pinnacle", tipo="cierre", quote_time="x",
                                  o_h=2.0, o_d=3.5, o_a=4.0))
    before = path.read_bytes()
    tables, counts, settled = read_ledger(path)
    assert path.read_bytes() == before
    v = ledger_view(tables).set_index("fixture_id")
    assert v.loc["f1", "estado"] == "Jugado" and v.loc["f1", "marcador"] == "2-2"
    assert v.loc["f1", "prob_modelo_resultado"] == pytest.approx(0.3)
    assert v.loc["f1", ["m_h", "m_d", "m_a"]].sum() == pytest.approx(1)
    assert v.loc["f2", "estado"] == "Pendiente" and np.isnan(v.loc["f2", "m_h"])
    assert len(settled) == 1


def test_forecast_igual_al_registro_y_valida_cuotas(results):
    f = forecast(results, "E0", "Team01", "Team02", NOW, odds=[2.0, 3.5, 4.0])
    model, _ = daily.fit_for_cutoff(results, "E0", NOW)
    from betwise.dixon_coles import outcome_probs
    assert np.allclose(f["p"], outcome_probs(model.score_matrix("Team01", "Team02")))
    assert f["market"].sum() == pytest.approx(1) and len(f["ev"]) == 3
    with pytest.raises(ValueError):
        forecast(results, "E0", "Team01", "Team02", NOW, odds=[1.0, 3.5, 4.0])
    with pytest.raises(ValueError):
        forecast(results, "E0", "Team01", "Team01", NOW)
