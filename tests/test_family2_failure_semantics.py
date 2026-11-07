"""Tests for Family 2 failure semantics matching native Cantera C++ behavior."""

import pytest
import numpy as np
import cantera_python as ct
from cantera_python.constants import CanteraError
from cantera_python.rates import (
    ArrheniusRate, FalloffRate, LindemannRate, TroeRate, SriRate, TsangRate,
    PlogRate, ChebyshevRate, BlowersMaselRate, LinearBurkeRate
)


def test_invalid_temperatures_and_pressures():
    gas = ct.Solution("sri-falloff.yaml")
    # Temperature <= 0
    with pytest.raises(CanteraError):
        gas.TP = -100.0, ct.OneAtm

    with pytest.raises(CanteraError):
        gas.TP = 0.0, ct.OneAtm

    # Non-finite temperature
    with pytest.raises(CanteraError):
        gas.TP = float("nan"), ct.OneAtm

    # Pressure <= 0
    with pytest.raises(CanteraError):
        gas.TP = 300.0, -1000.0

    with pytest.raises(CanteraError):
        gas.TP = 300.0, 0.0

    with pytest.raises(CanteraError):
        gas.TP = 300.0, float("nan")


def test_invalid_reaction_index():
    gas = ct.Solution("sri-falloff.yaml")
    with pytest.raises(IndexError):
        gas.reaction(-1)

    with pytest.raises(IndexError):
        gas.reaction(100)


def test_invalid_troe_parameters():
    k_low = ArrheniusRate(1e10, 0.0, 1e4)
    k_high = ArrheniusRate(1e12, 0.0, 2e4)

    # Empty coeffs
    with pytest.raises((ValueError, CanteraError)):
        TroeRate(k_low, k_high, ())

    # Too few coeffs (< 3)
    with pytest.raises((ValueError, CanteraError)):
        TroeRate(k_low, k_high, (0.5, 1000.0))

    # Too many coeffs (> 4)
    with pytest.raises((ValueError, CanteraError)):
        TroeRate(k_low, k_high, (0.5, 1000.0, 2000.0, 3000.0, 4000.0))


def test_invalid_sri_parameters():
    k_low = ArrheniusRate(1e10, 0.0, 1e4)
    k_high = ArrheniusRate(1e12, 0.0, 2e4)

    # Too few coeffs (< 3)
    with pytest.raises((ValueError, CanteraError)):
        SriRate(k_low, k_high, (1.0, 100.0))

    # Negative c in SRI
    with pytest.raises(CanteraError):
        SriRate(k_low, k_high, (1.0, 100.0, -50.0))


def test_invalid_chebyshev_parameters():
    # Empty coefficient matrix
    with pytest.raises((ValueError, CanteraError)):
        ChebyshevRate(300.0, 2000.0, 1000.0, 1e6, [])

    # Tmin >= Tmax
    with pytest.raises((ValueError, CanteraError)):
        ChebyshevRate(2000.0, 300.0, 1000.0, 1e6, [[1.0, 2.0], [3.0, 4.0]])

    # Pmin >= Pmax
    with pytest.raises((ValueError, CanteraError)):
        ChebyshevRate(300.0, 2000.0, 1e6, 1000.0, [[1.0, 2.0], [3.0, 4.0]])


def test_invalid_plog_parameters():
    # Negative pressure in Plog
    with pytest.raises((ValueError, CanteraError)):
        PlogRate([(-1000.0, ArrheniusRate(1e10, 0.0, 1e4))])
