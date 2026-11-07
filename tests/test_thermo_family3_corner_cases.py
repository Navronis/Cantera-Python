"""Unit tests for Thermodynamics Family 3: ThermoPhase Base and Inversion Corner Cases."""
import pytest
import numpy as np
from cantera_python.solution import Solution

def test_enforced_temperature_limits_hp_upper():
    sol = Solution("h2o2.yaml")
    thermo = sol.thermo
    if hasattr(thermo, "setTemperatureLimitsEnforced"):
        thermo.setTemperatureLimitsEnforced(True)

    T_target = thermo.maxTemp() - 0.02
    thermo.setState_TPX(T_target, 101325.0, "O2:1")
    h = thermo.enthalpy_mass

    thermo.setState_TP(thermo.minTemp() + 1.23, 101325.0)
    thermo.setState_HP(h, 101325.0)

    assert abs(thermo.temperature - T_target) < 1e-4

def test_enforced_temperature_limits_hp_lower():
    sol = Solution("h2o2.yaml")
    thermo = sol.thermo
    if hasattr(thermo, "setTemperatureLimitsEnforced"):
        thermo.setTemperatureLimitsEnforced(True)

    T_target = thermo.minTemp() + 0.01
    thermo.setState_TPX(T_target, 101325.0, "O:1")
    h = thermo.enthalpy_mass

    thermo.setState_TP(thermo.maxTemp() - 1.24, 101325.0)
    thermo.setState_HP(h, 101325.0)

    assert abs(thermo.temperature - T_target) < 1e-4

def test_enforced_temperature_limits_sp_upper():
    sol = Solution("h2o2.yaml")
    thermo = sol.thermo
    if hasattr(thermo, "setTemperatureLimitsEnforced"):
        thermo.setTemperatureLimitsEnforced(True)

    T_target = thermo.maxTemp() - 0.01
    thermo.setState_TPX(T_target, 101325.0, "O2:1")
    s = thermo.entropy_mass

    thermo.setState_TP(thermo.minTemp() + 1.23, 101325.0)
    thermo.setState_SP(s, 101325.0)

    assert abs(thermo.temperature - T_target) < 1e-4

def test_enforced_temperature_limits_sp_lower():
    sol = Solution("h2o2.yaml")
    thermo = sol.thermo
    if hasattr(thermo, "setTemperatureLimitsEnforced"):
        thermo.setTemperatureLimitsEnforced(True)

    T_target = thermo.minTemp() + 0.01
    thermo.setState_TPX(T_target, 101325.0, "O2:1")
    s = thermo.entropy_mass

    thermo.setState_TP(thermo.maxTemp() - 1.24, 101325.0)
    thermo.setState_SP(s, 101325.0)

    assert abs(thermo.temperature - T_target) < 1e-4
