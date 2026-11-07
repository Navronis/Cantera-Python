"""Tests for TPX pure substance thermodynamic models."""
import math
import pytest
import numpy as np

from cantera_python.tpx import (
    Water, CarbonDioxide, Nitrogen, Oxygen, Hydrogen, Methane, Heptane, HFC134a,
    PropertyPair, TPXError
)

def test_water_basic_properties():
    w = Water()
    assert w.MolWt() == 18.016
    assert abs(w.Tcrit() - 647.286) < 1e-3
    assert abs(w.Pcrit() - 22.089e6) < 1e2
    assert w.name() == "water"
    assert w.formula() == "H2O"

def test_water_saturation_pressure():
    w = Water()
    w.set_T(373.15) # 100 C
    psat = w.Ps()
    # At 100 C, saturation pressure should be close to 1 atm (101325 Pa)
    assert abs(psat - 101325.0) / 101325.0 < 0.05

def test_water_liquid_density():
    w = Water()
    w.set_T(300.0)
    w.Set(PropertyPair.TP, 300.0, 101325.0)
    rho = w.Rho
    # Saturated/subcooled liquid water density at 300K is ~996.5 kg/m3
    assert abs(rho - 996.5) / 996.5 < 0.01

def test_water_two_phase_quality():
    w = Water()
    w.Set(PropertyPair.TX, 373.15, 0.5)
    assert w.TwoPhase() == 1
    assert abs(w.x() - 0.5) < 1e-4

def test_carbon_dioxide_properties():
    co2 = CarbonDioxide()
    assert co2.MolWt() == 44.01
    assert abs(co2.Tcrit() - 304.21) < 1e-2  # pinned CarbonDioxide.cpp
    co2.Set(PropertyPair.TP, 300.0, 1e6)
    assert co2.P() > 0.0

def test_nitrogen_and_oxygen():
    n2 = Nitrogen()
    assert n2.MolWt() == 28.01348  # pinned Nitrogen.cpp
    n2.Set(PropertyPair.TP, 300.0, 101325.0)
    assert abs(n2.Rho - 101325.0 * 28.01348 / (8314.462618 * 300.0)) / n2.Rho < 0.05

    o2 = Oxygen()
    assert o2.MolWt() == 31.9994  # pinned Oxygen.cpp
    o2.Set(PropertyPair.TP, 300.0, 101325.0)
    assert abs(o2.Rho - 101325.0 * 31.9994 / (8314.462618 * 300.0)) / o2.Rho < 0.05

def test_methane_and_hydrogen():
    ch4 = Methane()
    assert ch4.MolWt() == 16.04996  # pinned Methane.cpp
    h2 = Hydrogen()
    assert h2.MolWt() == 2.0159  # pinned Hydrogen.cpp

def test_hfc134a_refrigerant():
    r134a = HFC134a()
    assert r134a.MolWt() == 102.032  # pinned HFC134a.cpp
    r134a.set_T(273.15)
    psat = r134a.Ps()
    assert psat > 1e5

def test_new_substance_factory():
    from cantera_python.tpx import newSubstance
    w = newSubstance("water")
    assert w.name() == "water"
    co2 = newSubstance("carbon-dioxide")
    assert co2.name() == "carbon-dioxide"
    ch4 = newSubstance("methane")
    assert ch4.name() == "methane"

def test_substance_derivatives():
    w = Water()
    w.Set(PropertyPair.TP, 350.0, 2e5)
    beta = w.thermalExpansionCoeff()
    kappa = w.isothermalCompressibility()
    assert np.isfinite(beta)
    assert np.isfinite(kappa)
    assert kappa >= 0.0
