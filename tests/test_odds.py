import numpy as np
import pytest

from betwise.odds import (expected_value, fair_odds, fractional_kelly, implied_prob,
                          min_acceptable_odds, overround, remove_margin)


def test_ejemplo_del_prompt():
    # 60% de probabilidad con cuota 1.90 -> EV +14%
    assert expected_value(0.60, 1.90) == pytest.approx(0.14)


def test_implied_y_margen():
    odds = [2.0, 3.5, 4.0]
    assert implied_prob(odds).sum() - 1 == pytest.approx(overround(odds))
    q = remove_margin(odds)
    assert q.sum() == pytest.approx(1.0)
    assert overround(odds) > 0


def test_remove_margin_varias_filas():
    q = remove_margin(np.array([[2.0, 3.5, 4.0], [1.5, 4.0, 7.0]]))
    assert q.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_cuota_justa_y_minima():
    assert fair_odds(0.5) == pytest.approx(2.0)
    assert min_acceptable_odds(0.5, 0.05) == pytest.approx(2.1)
    assert expected_value(0.5, min_acceptable_odds(0.5, 0.05)) == pytest.approx(0.05)


def test_cuota_invalida():
    with pytest.raises(ValueError):
        implied_prob([1.0, 2.0])


def test_kelly_conservador():
    assert fractional_kelly(0.40, 2.0) == 0.0  # EV negativo -> no apostar
    assert fractional_kelly(0.90, 2.0) <= 0.02  # tope del 2%
