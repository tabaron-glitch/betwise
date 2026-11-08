import numpy as np
import pandas as pd
import pytest

from betwise.backtest import walk_forward
from betwise.dixon_coles import (btts_prob, fit, outcome_probs, over_prob, score_matrix,
                                 simulate_scores, top_scores)
from betwise.synthetic import make_synthetic


@pytest.fixture(scope="module")
def data():
    return make_synthetic(n_teams=12, n_seasons=3, seed=7)


def test_matriz_normalizada_y_mercados(data):
    m = score_matrix(1.6, 1.1, -0.08)
    assert m.sum() == pytest.approx(1.0)
    assert outcome_probs(m).sum() == pytest.approx(1.0)
    assert over_prob(m, 2.5) + (1 - over_prob(m, 2.5)) == pytest.approx(1.0)
    assert 0 < btts_prob(m) < 1
    top = top_scores(m, 3)
    assert top[0][2] >= top[1][2] >= top[2][2]
    assert outcome_probs(m)[0] > outcome_probs(m)[2]  # lam > mu


def test_monte_carlo_converge_a_la_matriz():
    m = score_matrix(1.5, 1.2, -0.05)
    f = simulate_scores(m, n=200_000, seed=1)
    assert np.abs(f - m).max() < 0.005


def test_gradiente_analitico(data):
    # Compara el gradiente analítico con diferencias finitas
    from scipy.optimize import approx_fprime
    import betwise.dixon_coles as dc

    train = data[data.date < "2020-08-01"]
    captured = {}
    orig_minimize = dc.minimize

    def spy(fun, x0, **kw):
        captured["fun"], captured["x0"] = fun, x0
        return orig_minimize(fun, x0, **kw)

    dc.minimize = spy
    try:
        fit(train, ref_date=pd.Timestamp("2020-08-01"))
    finally:
        dc.minimize = orig_minimize
    fun, x0 = captured["fun"], captured["x0"]
    rng = np.random.default_rng(0)
    x = x0 + rng.normal(0, 0.05, len(x0))
    x[-2] = -0.05  # rho dentro de límites
    g = fun(x)[1]
    g_num = approx_fprime(x, lambda t: fun(t)[0], 1e-6)
    assert np.allclose(g, g_num, rtol=1e-3, atol=1e-3)


def test_recupera_ventaja_local_y_ordena_equipos(data):
    model = fit(data, ref_date=data.date.max())
    assert model.converged
    assert 0.1 < model.home_adv < 0.4  # verdadero: 0.25
    assert abs(model.attack.mean()) < 1e-9
    # el equipo con más goles a favor debería tener ataque por encima de la mediana
    gf = pd.concat([data.groupby("home").hg.sum(), data.groupby("away").ag.sum()], axis=1).sum(axis=1)
    best = gf.idxmax()
    assert model.attack[model.idx[best]] > np.median(model.attack)


def test_equipo_desconocido_no_rompe(data):
    model = fit(data, ref_date=data.date.max())
    m = model.score_matrix("Team00", "Ascendido FC")
    assert m.sum() == pytest.approx(1.0)
    assert not model.knows("Ascendido FC")


def test_sin_fuga_de_informacion(data):
    """Cambiar resultados FUTUROS no puede alterar predicciones PASADAS."""
    start, cut = "2019-08-01", pd.Timestamp("2020-02-01")
    base = walk_forward(data, test_start=start, step_days=14, min_train=100)
    tampered = data.copy()
    fut = tampered.date >= cut
    tampered.loc[fut, ["hg", "ag"]] = [9, 0]
    alt = walk_forward(tampered, test_start=start, step_days=14, min_train=100)
    a = base[base.date < cut].reset_index(drop=True)
    b = alt[alt.date < cut].reset_index(drop=True)
    assert len(a) > 100
    cols = ["p_h", "p_d", "p_a", "p_over25"]
    assert np.allclose(a[cols].to_numpy(), b[cols].to_numpy())
