"""Independent scientific sanity checks for Family 2 kinetics models."""

import pytest
import numpy as np
import cantera_python as ct


def test_mass_conservation_across_mechanisms():
    mechs = [
        ("sri-falloff.yaml", None, "H:0.1, R1A:0.2, R1B:0.2, P1:0.1, R2:0.1, P2A:0.1, P2B:0.1, R3:0.1"),
        ("tsang-falloff.yaml", None, "H:0.05, NO:0.1, OH:0.1, CN:0.05, HCN:0.1, CO2:0.1, H2O:0.1, N2:0.3, N2O:0.05, HONO:0.05"),
        ("chemically-activated-reaction.yaml", None, "ch3:0.2, oh:0.2, ch2o:0.1, h2:0.1, n2:0.4"),
        ("pdep-test.yaml", None, "H:0.1, R1A:0.1, R1B:0.1, P1:0.1, R2:0.1, P2A:0.1, P2B:0.1, R3:0.1, P3A:0.1, P3B:0.1"),
        ("linearBurke-test.yaml", None, "H:0.1, OH:0.1, H2O:0.3, O2:0.2, HO2:0.1, H2O2:0.2"),
        ("blowers-masel.yaml", "gas", "O:0.05, H2:0.4, O2:0.1, OH:0.05, H2O:0.1, CH4:0.05, AR:0.1, N2:0.15"),
        ("gri30.yaml", None, "CH4:0.095, O2:0.19, N2:0.715"),
    ]

    for mech, phase, comp in mechs:
        gas = ct.Solution(mech, phase) if phase else ct.Solution(mech)
        gas.TPX = 1500.0, ct.OneAtm, comp

        wdot = np.array(gas.net_production_rates)
        mw = np.array(gas.molecular_weights)

        # Net mass production rate: sum(wdot_k * MW_k) must equal 0 (mass conservation)
        mass_rate = np.sum(wdot * mw)
        total_scale = np.sum(np.abs(wdot) * mw) + 1e-30
        assert abs(mass_rate) / total_scale < 1e-12, f"Mass conservation breached in {mech}: {mass_rate} / {total_scale}"


def test_falloff_asymptotes():
    gas = ct.Solution("sri-falloff.yaml")

    # High pressure asymptote: rate approaches k_high
    gas.TP = 1000.0, 1e9 # 10,000 atm
    k_high_p = gas.forward_rate_constants[0]

    gas.TP = 1000.0, 1e10 # 100,000 atm
    k_very_high_p = gas.forward_rate_constants[0]

    # As P increases, rate approaches an asymptotic plateau
    rel_diff = abs(k_very_high_p - k_high_p) / k_high_p
    assert rel_diff < 0.01, f"High pressure limit not approached: {rel_diff}"


def test_cache_invalidation_temperature_and_pressure():
    gas = ct.Solution("pdep-test.yaml")
    gas.TP = 1000.0, ct.OneAtm
    k1 = np.array(gas.forward_rate_constants)

    # Change temperature: rate must change
    gas.TP = 1200.0, ct.OneAtm
    k2 = np.array(gas.forward_rate_constants)
    assert not np.allclose(k1, k2), "Rates failed to update on temperature change"

    # Change pressure: pressure-dependent rate must change
    gas.TP = 1200.0, 10.0 * ct.OneAtm
    k3 = np.array(gas.forward_rate_constants)
    assert not np.allclose(k2, k3), "Rates failed to update on pressure change"
