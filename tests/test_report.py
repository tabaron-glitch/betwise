import pandas as pd
import pytest

from betwise.ledger import Ledger
from betwise.report import build_table, summary


@pytest.fixture
def led(tmp_path):
    with Ledger(tmp_path / "l.db") as l:
        base = dict(div="E0", home="A", away="B", model_version="v1", params="{}", data_cutoff="2026-11-01",
                    n_train=100, known_teams=1, p_over25=0.5, p_btts=0.5)
        games = [("f1", 0.6, 0.2, 0.2, 2, 0, (1.8, 3.6, 4.5), (1.7, 3.8, 4.8)),
                 ("f2", 0.3, 0.3, 0.4, 0, 0, (2.5, 3.2, 2.9), (2.6, 3.1, 2.8)),
                 ("f3", 0.5, 0.3, 0.2, 1, 3, (2.0, 3.4, 3.8), (2.1, 3.4, 3.6))]
        for fid, ph, pd_, pa, hg, ag, op, oc in games:
            l.append("pronosticos", dict(base, fixture_id=fid, kickoff=f"2026-11-0{fid[1]}T15:00:00+00:00",
                                         p_h=ph, p_d=pd_, p_a=pa))
            l.append("resultados", dict(fixture_id=fid, hg=hg, ag=ag, source="prueba"))
            for tipo, o in (("al_pronostico", op), ("cierre", oc)):
                l.append("cuotas", dict(fixture_id=fid, bookmaker="pinnacle", tipo=tipo, o_h=o[0], o_d=o[1], o_a=o[2]))
        # corrección: f3 en realidad terminó 1-1
        rid = [r["id"] for r in l.rows("resultados") if r["fixture_id"] == "f3"][0]
        l.append("resultados", dict(fixture_id="f3", hg=1, ag=1, source="prueba", corrige_id=rid))
        yield l


def test_tabla_usa_la_correccion_y_quita_margen(led):
    df = build_table(led)
    assert len(df) == 3
    assert list(df.outcome) == [0, 1, 1]
    assert abs(df[["qp_h", "qp_d", "qp_a"]].sum(axis=1) - 1).max() < 1e-9


def test_resumen_incluye_ambas_referencias_y_aviso(led):
    s = summary(build_table(led))
    assert "al pronóstico" in s and "al cierre" in s and "NO APOSTAR" in s


def test_libro_vacio():
    assert "Aún no hay" in summary(pd.DataFrame())
