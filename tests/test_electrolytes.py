"""Unit tests for electrolyte thermodynamics (Debye-Huckel and HMW/Pitzer)."""

import pytest
import math
import numpy as np

from cantera_python.electrolytes import WaterProps, DebyeHuckel, HMWSoln


def test_water_props():
    T = 298.15
    rho = WaterProps.density(T)
    assert 995.0 < rho < 1000.0  # ~997.0 kg/m^3

    eps = WaterProps.dielectric_constant(T)
    assert 77.0 < eps < 80.0     # ~78.4

    A_phi = WaterProps.debye_huckel_A_phi(T)
    assert 0.38 < A_phi < 0.40   # ~0.3915 (kg/mol)^0.5

    B = WaterProps.debye_huckel_B(T)
    assert 3.0 < B < 3.5         # ~3.28


def test_debye_huckel_dilute_limit():
    dh = DebyeHuckel(species_names=["Na+", "Cl-"], charges=[1, -1])
    dh.molalities = [0.001, 0.001]
    assert math.isclose(dh.ionic_strength, 0.001, rel_tol=1e-6)

    # Activity coefficient in dilute solution should be ~0.96
    gammas = dh.activity_coefficients(form="limiting")
    assert 0.95 < gammas[0] < 0.98
    assert math.isclose(gammas[0], gammas[1], rel_tol=1e-10)

    # Osmotic coefficient should be close to 1
    phi = dh.osmotic_coefficient()
    assert 0.98 < phi < 1.0

    # Water activity should be close to 1
    aw = dh.water_activity()
    assert 0.999 < aw <= 1.0


def test_debye_huckel_b_dot():
    dh = DebyeHuckel(species_names=["Na+", "Cl-"], charges=[1, -1], b_dot=0.041)
    dh.molalities = [0.1, 0.1]
    gammas = dh.activity_coefficients(form="b_dot")
    # At 0.1m, experimental gamma_NaCl is ~0.778
    assert 0.70 < gammas[0] < 0.85


def test_hmw_pitzer_nacl():
    hmw = HMWSoln(cations=["Na+"], anions=["Cl-"])
    # Test at 1.0 mol/kg NaCl
    hmw.set_molalities({"Na+": 1.0, "Cl-": 1.0})
    assert math.isclose(hmw.ionic_strength, 1.0, rel_tol=1e-6)

    # Pitzer experimental osmotic coefficient for 1.0 m NaCl is ~0.936
    phi = hmw.osmotic_coefficient()
    assert abs(phi - 0.936) < 0.02

    # Mean activity coefficient for 1.0 m NaCl is ~0.657
    gamma_pm = hmw.mean_activity_coefficient("Na+", "Cl-")
    assert abs(gamma_pm - 0.657) < 0.03

    # Water activity should be ~0.967
    aw = hmw.water_activity()
    assert abs(aw - 0.967) < 0.01


def test_hmw_multicomponent_ionic_strength():
    hmw = HMWSoln(cations=["Na+", "Ca2+"], anions=["Cl-"])
    hmw.set_charge("Ca2+", 2)
    # 0.5m NaCl + 0.2m CaCl2: I = 0.5*(0.5*1 + 0.2*4 + (0.5 + 0.4)*1) = 0.5*(0.5 + 0.8 + 0.9) = 1.1
    hmw.set_molalities({"Na+": 0.5, "Ca2+": 0.2, "Cl-": 0.9})
    assert math.isclose(hmw.ionic_strength, 1.1, rel_tol=1e-6)
