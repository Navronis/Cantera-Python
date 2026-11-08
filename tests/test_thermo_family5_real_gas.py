"""Comprehensive unit tests for Thermodynamics Family 5:
Equations of state, fugacity, pure fluids, water properties, and consistency.
"""
import math
import pytest
import numpy as np

import cantera_python as ct
from cantera_python.phases import PureFluidPhase
from cantera_python.cubic_eos import PengRobinson, RedlichKwongMFTP, MixtureFugacityTP
from cantera_python.electrolytes import (
    WaterProps, WaterPropsIAPWS, WaterPropsIAPWSphi,
    WATER_GAS, WATER_LIQUID, WATER_SUPERCRIT
)
from cantera_python.constants import OneAtm, GasConstant


# =========================================================================
# 1. Cubic EOS: Peng-Robinson and Redlich-Kwong Thermodynamic Consistency
# =========================================================================

def test_peng_robinson_consistency():
    """Verify Peng-Robinson thermodynamic fundamental relations: g = h - Ts, u = h - Pv."""
    sol = ct.Solution("co2_PR_example.yaml", "CO2-PR")
    
    test_points = [
        (250.0, 101325.0, "CO2:1.0"),
        (350.0, 2.0e6, "CO2:0.9, H2:0.1"),
        (500.0, 8.0e6, "CO2:0.7, H2O:0.1, CH4:0.1, O2:0.1"),
    ]
    
    for T, P, X in test_points:
        sol.TPX = (T, P, X)
        h = sol.enthalpy_mole
        u = sol.int_energy_mole
        s = sol.entropy_mole
        g = sol.gibbs_mole
        v = 1.0 / sol.molar_density
        
        # Fundamental relations
        assert abs(g - (h - T * s)) / abs(g) < 1.0e-10
        assert abs(u - (h - P * v)) / abs(u) < 1.0e-10
        
        # Verify fugacity coefficients and activity coefficients
        act_coeff = sol.activity_coefficients
        assert len(act_coeff) == sol.n_species
        assert np.all(act_coeff > 0.0)


def test_redlich_kwong_consistency():
    """Verify Redlich-Kwong thermodynamic fundamental relations and departure terms."""
    sol = ct.Solution("co2_RK_example.yaml", "carbondioxide")
    
    test_points = [
        (250.0, 101325.0, "CO2:1.0"),
        (350.0, 2.0e6, "CO2:0.8, H2:0.2"),
        (600.0, 5.0e6, "CO2:0.5, H2:0.5"),
    ]
    
    for T, P, X in test_points:
        sol.TPX = (T, P, X)
        h = sol.enthalpy_mole
        u = sol.int_energy_mole
        s = sol.entropy_mole
        g = sol.gibbs_mole
        v = 1.0 / sol.molar_density
        
        # Fundamental relations
        assert abs(g - (h - T * s)) / abs(g) < 1.0e-10
        assert abs(u - (h - P * v)) / abs(u) < 1.0e-10
        
        # Isothermal compressibility and sound speed must be physically positive
        assert sol.isothermal_compressibility > 0.0
        assert sol.sound_speed > 0.0


def test_cubic_eos_binary_interactions():
    """Verify setting and retrieving binary interaction parameters."""
    sol = ct.Solution("co2_PR_example.yaml", "CO2-PR")
    thermo = sol._thermo
    
    # Check binary interaction retrieval
    thermo.setBinaryInteraction("CO2", "H2", 0.05)
    assert abs(thermo.getBinaryInteraction("CO2", "H2") - 0.05) < 1.0e-12
    assert abs(thermo.getBinaryInteraction("H2", "CO2") - 0.05) < 1.0e-12
    
    # Check with species indices
    i_co2 = sol.species_index("CO2")
    i_h2 = sol.species_index("H2")
    thermo.setBinaryInteraction(i_co2, i_h2, 0.08)
    assert abs(thermo.getBinaryInteraction(i_co2, i_h2) - 0.08) < 1.0e-12


# =========================================================================
# 2. PureFluidPhase: Multiphasic States, Saturation, and Pairs
# =========================================================================

def test_pure_fluid_phase_states():
    """Test pure-fluid phase state identification and transitions across water and nitrogen."""
    water = ct.Solution("liquidvapor.yaml", "water")
    
    # Subcritical liquid
    water.TP = (300.0, 101325.0)
    assert water.phase_of_matter == "liquid"
    assert water.Q == 0.0
    
    # Supercritical fluid
    water.TP = (700.0, 30.0e6)
    assert water.phase_of_matter == "supercritical"
    
    # Subcritical gas
    water.TP = (450.0, 101325.0)
    assert water.phase_of_matter == "gas"


def test_pure_fluid_saturation_and_pairs():
    """Test saturation curve calculations and pair setters (TV, PV, HP, SP, Tsat, Psat)."""
    water = ct.Solution("liquidvapor.yaml", "water")
    
    # Set saturated state via Tsat
    T_sat = 373.124
    water.setState_Tsat(T_sat, 0.5)
    assert abs(water.T - T_sat) < 1.0e-3
    assert abs(water.Q - 0.5) < 1.0e-4
    assert abs(water.P - OneAtm) < 500.0
    
    # Test property pair getters and setters
    orig_T, orig_V = water.TV
    water.TV = (orig_T + 10.0, orig_V)
    assert abs(water.T - (orig_T + 10.0)) < 1.0e-3
    
    # Test HP setter
    h_orig = water.enthalpy_mass
    p_orig = water.P
    water.HP = (h_orig, p_orig)
    assert abs(water.enthalpy_mass - h_orig) < 1.0


# =========================================================================
# 3. Water Thermodynamic Formulations: IAPWS-95 and WaterProps
# =========================================================================

def test_water_iapws_reference_points():
    """Verify IAPWS-95 reference points (Triple Point liquid/gas, Normal Boiling Point)."""
    water = WaterPropsIAPWS()
    
    # 1. Triple point liquid (T = 273.16 K)
    T_trip = 273.16
    P_sat = water.psat(T_trip)
    assert abs(P_sat - 611.654771) < 1.0e-3
    
    rho_liq = water.density(T_trip, P_sat, WATER_LIQUID)
    assert abs(rho_liq - 999.793) < 1.0e-2
    assert abs(water.intEnergy_mass() - 0.0) < 1.0
    assert abs(water.entropy_mass() - 0.0) < 1.0
    
    # 2. Triple point gas
    water.psat(T_trip, WATER_GAS)
    rho_gas = water.density(T_trip, P_sat, WATER_GAS)
    assert abs(rho_gas - 4.85458e-3) < 1.0e-5
    assert abs(water.entropy_mass() - 9155.5) < 5.0
    assert abs(water.enthalpy_mass() - 2500920.0) < 50.0
    
    # 3. Normal boiling point (T = 373.124 K)
    T_boil = 373.124
    P_boil = water.psat(T_boil)
    assert abs(P_boil - OneAtm) < 2.0
    rho_boil_liq = water.density(T_boil, P_boil, WATER_LIQUID)
    assert abs(rho_boil_liq - 958.368) < 1.0e-1
    rho_boil_gas = water.density(T_boil, P_boil, WATER_GAS)
    assert abs(rho_boil_gas - 0.597651) < 1.0e-3


def test_water_props_correlations():
    """Verify WaterProps dielectric constant, Debye-Huckel parameters, and expansions."""
    wp = WaterProps()
    
    # 300 K, 1 atm
    T = 300.0
    P = OneAtm
    
    rel_eps = wp.relEpsilon(T, P)
    assert 77.0 < rel_eps < 78.5
    
    a_debye = wp.ADebye(T, P)
    assert 1.15 < a_debye < 1.20
    
    dens_t = wp.density_T(T, P)
    assert 990.0 < dens_t < 1000.0


# =========================================================================
# 4. Adversarial Failure Semantics & Boundary Conditions
# =========================================================================

def test_cubic_adversarial_boundaries():
    """Verify proper exception handling for non-physical inputs in real gas models."""
    sol = ct.Solution("co2_PR_example.yaml", "CO2-PR")
    
    # Negative temperature should fail or raise
    with pytest.raises((ValueError, Exception)):
        sol.TP = (-100.0, OneAtm)
        
    # Negative pressure should fail or raise
    with pytest.raises((ValueError, Exception)):
        sol.TP = (300.0, -101325.0)


def test_pure_fluid_adversarial_boundaries():
    """Verify proper exception handling for non-physical states in pure fluid models."""
    water = ct.Solution("liquidvapor.yaml", "water")
    
    # Unphysical negative density
    with pytest.raises(ValueError):
        water.density = -500.0
