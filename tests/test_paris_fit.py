"""Paris fitting: exact recovery on synthetic power-law data, tail trimming."""

import numpy as np
import pytest

from afml.models.paris_fit import fit_paris_curve

C_TRUE, M_TRUE = 1e-9, 3.2


def test_exact_power_law_recovered():
    dk = np.geomspace(3, 30, 20)
    dadn = C_TRUE * dk**M_TRUE
    fit = fit_paris_curve(dk, dadn)
    assert fit["m"] == pytest.approx(M_TRUE, abs=1e-9)
    assert fit["C"] == pytest.approx(C_TRUE, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["n_points_used"] == 20


def test_threshold_tail_is_trimmed():
    dk = np.geomspace(2, 30, 25)
    dadn = C_TRUE * dk**M_TRUE
    # bend the low-ΔK tail down like near-threshold behavior
    dadn[:5] *= np.array([0.01, 0.05, 0.2, 0.5, 0.8])
    fit = fit_paris_curve(dk, dadn)
    assert fit["n_points_used"] < 25
    assert fit["m"] == pytest.approx(M_TRUE, rel=0.05)


def test_too_few_points_returns_none():
    dk = np.array([3.0, 4.0, 5.0])
    assert fit_paris_curve(dk, C_TRUE * dk**M_TRUE) is None


def test_noisy_curve_still_reasonable():
    rng = np.random.default_rng(7)
    dk = np.geomspace(3, 30, 30)
    dadn = C_TRUE * dk**M_TRUE * 10 ** rng.normal(0, 0.05, 30)
    fit = fit_paris_curve(dk, dadn)
    assert fit["m"] == pytest.approx(M_TRUE, rel=0.15)
    assert fit["r2"] > 0.9
