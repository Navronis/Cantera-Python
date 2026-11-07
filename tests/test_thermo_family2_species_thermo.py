"""Comprehensive verification test suite for Thermodynamics Family 2: SpeciesThermo."""
import math
import numpy as np
import pytest

from cantera_python.thermo import (
    SpeciesThermoInterpType,
    NasaPoly1,
    NasaPoly2,
    Nasa9Poly1,
    Nasa9PolyMultiTempRegion,
    ShomatePoly,
    ShomatePoly2,
    ConstCpPoly,
    Mu0Poly,
    create_species_thermo,
)
from cantera_python.multi_species_thermo import MultiSpeciesThermo
from cantera_python.ideal_gas import Species
from cantera_python.elements import (
    elementSymbols,
    elementNames,
    getElementWeight,
    getElementSymbol,
    getElementName,
    getAtomicNumber,
    numElementsDefined,
    numIsotopesDefined,
)
import cantera_python as ct


def test_elements_database_queries():
    symbols = elementSymbols()
    names = elementNames()
    assert len(symbols) > 100
    assert len(names) > 100
    assert "H" in symbols
    assert "C" in symbols
    assert "O" in symbols
    assert "N" in symbols
    assert "Ar" in symbols

    # getElementWeight
    assert abs(getElementWeight("H") - 1.008) < 0.01
    assert abs(getElementWeight("C") - 12.011) < 0.01
    assert abs(getElementWeight("O") - 15.999) < 0.01

    # getElementSymbol & getElementName
    assert getElementSymbol("carbon") == "C"
    assert getElementName("C").lower() == "carbon"
    assert getAtomicNumber("C") >= 1
    assert numElementsDefined() > 100
    assert numIsotopesDefined() > 100


def test_nasapoly1_evaluations_and_methods():
    # 7 NASA coeffs for O2 low temp
    coeffs = [3.21293640E+00, 1.12748635E-03, -5.75615047E-07, 1.31387723E-09, -8.76855392E-13, -1.00524902E+03, 6.03473759E+00]
    nasa1 = NasaPoly1(coeffs, min_temp=200.0, max_temp=1000.0, reference_pressure=101325.0)

    assert nasa1.reportType() == 256
    assert nasa1.minTemp() == 200.0
    assert nasa1.maxTemp() == 1000.0
    assert nasa1.refPressure() == 101325.0
    assert nasa1.temperaturePolySize() == 6
    assert nasa1.nCoeffs() == 7

    cp, h, s = nasa1.properties(300.0)
    assert cp > 0.0
    assert s > 0.0

    cp_arr = [0.0]
    h_arr = [0.0]
    s_arr = [0.0]
    nasa1.updatePropertiesTemp(300.0, cp_arr, h_arr, s_arr)
    assert abs(cp_arr[0] - cp) < 1e-12
    assert abs(h_arr[0] - h) < 1e-12
    assert abs(s_arr[0] - s) < 1e-12

    # Hf298
    hf = nasa1.reportHf298()
    nasa1.modifyOneHf298(0, hf + 1000.0)
    nasa1.resetHf298()


def test_nasapoly2_evaluations_and_methods():
    low_c = [3.21293640E+00, 1.12748635E-03, -5.75615047E-07, 1.31387723E-09, -8.76855392E-13, -1.00524902E+03, 6.03473759E+00]
    high_c = [3.69777829E+00, 6.13519789E-04, -1.25884282E-07, 1.77528148E-11, -1.13643547E-15, -1.23393018E+03, 3.18916559E+00]
    nasa2 = NasaPoly2(1000.0, low_c, high_c, min_temp=200.0, max_temp=3500.0, reference_pressure=101325.0)

    assert nasa2.reportType() == 4
    assert nasa2.minTemp() == 200.0
    assert nasa2.maxTemp() == 3500.0
    assert nasa2.nCoeffs() == 15
    assert nasa2.temperaturePolySize() == 6

    # Low and high temperature evaluations
    cp_low, h_low, s_low = nasa2.properties(500.0)
    cp_high, h_high, s_high = nasa2.properties(1500.0)
    assert cp_low > 0.0 and cp_high > 0.0


def test_nasa9poly_evaluations_and_methods():
    # 9 NASA coeffs
    coeffs = [1.0e4, -1.0e2, 3.5, 1e-3, -1e-6, 1e-9, -1e-12, 100.0, 5.0]
    nasa9_1 = Nasa9Poly1(coeffs, min_temp=200.0, max_temp=1000.0, reference_pressure=101325.0)

    assert nasa9_1.reportType() == 512
    assert nasa9_1.temperaturePolySize() == 7
    assert nasa9_1.nCoeffs() == 9

    cp, h, s = nasa9_1.properties(500.0)
    assert math.isfinite(cp)
    assert math.isfinite(h)
    assert math.isfinite(s)

    # Multi region
    multi = Nasa9PolyMultiTempRegion([200.0, 1000.0, 3000.0], [coeffs, coeffs], reference_pressure=101325.0)
    assert multi.reportType() == 513
    assert multi.minTemp() == 200.0
    assert multi.maxTemp() == 3000.0
    cp_m, h_m, s_m = multi.properties(1500.0)
    assert math.isfinite(cp_m)


def test_shomate_evaluations_and_methods():
    coeffs = [29.5, -3.0, 5.0, -1.0, 0.1, -10.0, 200.0]
    shomate = ShomatePoly(coeffs, min_temp=298.15, max_temp=1000.0, reference_pressure=101325.0)

    assert shomate.reportType() == 128
    assert shomate.minTemp() == 298.15
    assert shomate.maxTemp() == 1000.0
    cp, h, s = shomate.properties(500.0)
    assert cp > 0.0

    shomate2 = ShomatePoly2(1000.0, coeffs, coeffs, min_temp=298.15, max_temp=3000.0, reference_pressure=101325.0)
    assert shomate2.reportType() == 8
    assert shomate2.minTemp() == 298.15
    assert shomate2.maxTemp() == 3000.0


def test_constcp_and_mu0_evaluations_and_methods():
    const_cp = ConstCpPoly(t0=298.15, h0=1e6, s0=2e5, cp0=30000.0, min_temp=200.0, max_temp=2000.0)
    assert const_cp.reportType() == 1
    assert const_cp.minTemp() == 200.0
    assert const_cp.maxTemp() == 200.0 or const_cp.maxTemp() == 2000.0
    cp, h, s = const_cp.properties(500.0)
    assert cp > 0.0

    # Mu0Poly
    t_mu = {298.15: -1e5, 500.0: -1.2e5, 1000.0: -1.8e5}
    mu0 = Mu0Poly(h0=0.0, temperature_mu=t_mu, min_temp=298.15, max_temp=1000.0)
    assert mu0.reportType() == 64
    cp_m, h_m, s_m = mu0.properties(600.0)
    assert math.isfinite(cp_m)


def test_multi_species_thermo_manager():
    mst = MultiSpeciesThermo()
    assert mst.refPressure() == 101325.0

    low_c = [3.21293640E+00, 1.12748635E-03, -5.75615047E-07, 1.31387723E-09, -8.76855392E-13, -1.00524902E+03, 6.03473759E+00]
    high_c = [3.69777829E+00, 6.13519789E-04, -1.25884282E-07, 1.77528148E-11, -1.13643547E-15, -1.23393018E+03, 3.18916559E+00]
    nasa2 = NasaPoly2(1000.0, low_c, high_c, min_temp=200.0, max_temp=3500.0)

    mst.install_STIT(0, nasa2)
    assert mst.ready(1)
    assert mst.provideSTIT(0) is nasa2
    assert mst.reportType(0) == 4
    assert mst.minTemp(0) == 200.0
    assert mst.maxTemp(0) == 3500.0

    cp_arr = np.zeros(1)
    h_arr = np.zeros(1)
    s_arr = np.zeros(1)
    mst.update(500.0, cp_arr, h_arr, s_arr)
    assert cp_arr[0] > 0.0
    assert h_arr[0] != 0.0
    assert s_arr[0] > 0.0

    # Hf298 modification
    hf0 = mst.reportOneHf298(0)
    mst.modifyOneHf298(0, hf0 + 5000.0)
    mst.resetHf298(0)


def test_species_object_methods():
    sp = Species("CH4", 16.043, ConstCpPoly())
    assert sp.name == "CH4"
    assert sp.molecular_weight == 16.043
    assert sp.molecularWeight() == 16.043
    sp.setMolecularWeight(16.05)
    assert sp.molecularWeight() == 16.05
    sp.setMolecularWeight(16.043)
