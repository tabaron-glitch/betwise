"""Aritmética de cuotas: probabilidad implícita, margen, EV, cuota justa.

Todas las cuotas son DECIMALES (europeas). Funciones puras, sin datos externos.
"""
from __future__ import annotations

import numpy as np


def implied_prob(odds):
    """P = 1 / cuota."""
    o = np.asarray(odds, dtype=float)
    if np.any(o <= 1.0):
        raise ValueError("Las cuotas decimales deben ser mayores que 1.0")
    return 1.0 / o


def overround(odds) -> np.ndarray | float:
    """Margen de la casa en un mercado completo: suma(1/cuota) - 1."""
    return implied_prob(odds).sum(axis=-1) - 1.0


def remove_margin(odds):
    """Normalización proporcional (aproximación; en mercados de 3 vías el
    margen no se reparte exactamente igual entre resultados)."""
    p = implied_prob(odds)
    return p / p.sum(axis=-1, keepdims=True)


def expected_value(p, odds):
    """EV por unidad apostada = p * cuota - 1. Solo para mercados sin
    devolución ni liquidación parcial."""
    return np.asarray(p, dtype=float) * np.asarray(odds, dtype=float) - 1.0


def fair_odds(p):
    """Cuota justa estimada = 1 / p."""
    return 1.0 / np.asarray(p, dtype=float)


def min_acceptable_odds(p, min_edge: float = 0.0):
    """Cuota mínima para que EV >= min_edge: (1 + min_edge) / p."""
    return (1.0 + min_edge) / np.asarray(p, dtype=float)


def ev_sensitivity(p: float, odds: float, delta: float = 0.03) -> dict:
    """EV si la probabilidad real fuera `delta` menor o mayor que la estimada."""
    return {
        "ev_pesimista": expected_value(max(p - delta, 1e-9), odds),
        "ev_base": expected_value(p, odds),
        "ev_optimista": expected_value(min(p + delta, 1.0), odds),
    }


def fractional_kelly(p: float, odds: float, fraction: float = 0.25, cap: float = 0.02) -> float:
    """Kelly fraccionado con tope (por defecto 1/4 de Kelly y máximo 2% del capital).

    Solo tiene sentido si `p` está validada; sin validación, no usar.
    """
    if odds <= 1.0:
        return 0.0
    full = (p * odds - 1.0) / (odds - 1.0)
    return float(min(max(full, 0.0) * fraction, cap))
