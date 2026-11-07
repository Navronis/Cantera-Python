"""Scientific and physical invariance checks for remaining Transport models:
- UnityLewisTransport
- WaterTransport
- IonGasTransport
"""
import pytest
import numpy as np

import cantera_python as ct
from cantera_python.transport import (
    UnityLewisTransport, IonGasTransport, WaterTransport
)

def test_unity_lewis_scientific_invariants():
    gas = ct.Solution("gri30.yaml", "gri30")
    gas.TPX = 1200.0, 101325.0, "CH4:0.095, O2:0.19, N2:0.715"
    gas.transport_model = "unity-Lewis-number"
    tran = gas.transport

    # 1. D_km = lambda / (rho * cp)
    lam = float(tran.thermal_conductivity)
    rho = float(gas.density)
    cp = float(gas.cp_mass)
    expected_D = lam / (rho * cp)

    d_mix = np.zeros(gas.n_species)
    tran.getMixDiffCoeffs(d_mix)
    for k in range(gas.n_species):
        assert pytest.approx(d_mix[k], rel=1e-12) == expected_D

    # 2. Thermal diffusion coefficients are strictly zero
    dt = np.zeros(gas.n_species)
    tran.getThermalDiffCoeffs(dt)
    assert np.all(dt == 0.0)

def test_water_transport_scientific_invariants():
    class StateMock:
        def __init__(self, T, rho, kappa=0.0, alpha=0.0):
            self.temperature = T
            self.density = rho
            self._isothermal_compressibility = kappa
            self._thermal_expansion_coeff = alpha

    # 1. Liquid water at 298.15 K, 1 atm
    m_liq = StateMock(298.15, 997.0476, 4.5246e-10, 0.000257)
    wt_liq = WaterTransport(m_liq)
    mu_liq = float(wt_liq.viscosity)
    lam_liq = float(wt_liq.thermalConductivity())
    assert 0.00085 < mu_liq < 0.00095  # ~ 0.89 mPa*s
    assert 0.55 < lam_liq < 0.65       # ~ 0.60 W/(m*K)
    assert wt_liq.bulkViscosity() == 0.0

    # 2. Near-critical enhancement
    m_crit = StateMock(647.1, 287.854, 2.6487e-05, 6.7397)
    wt_crit = WaterTransport(m_crit)
    lam_crit = float(wt_crit.thermalConductivity())
    # Critical anomaly causes lambda to be enhanced significantly (> 1.0 W/(m*K))
    assert lam_crit > 1.0

def test_ion_gas_scientific_invariants():
    sol = ct.Solution("ch4_ion.yaml", "gas")
    sol.TPX = 1500.0, 101325.0, "CO:0.05, CO2:0.05, H2O:0.15, N2:0.74, E:0.005, H3O+:0.005"
    sol.transport_model = "ionized-gas"
    tran = sol.transport

    # 1. Electron mobility dominates heavy ion mobility by orders of magnitude
    mobi = np.zeros(sol.n_species)
    tran.getMobilities(mobi)
    k_e = sol.species_index("E")
    k_h3o = sol.species_index("H3O+")
    assert mobi[k_e] == 0.4
    assert 1e-5 < mobi[k_h3o] < 1e-2
    assert mobi[k_e] / mobi[k_h3o] > 100.0

    # 2. Electrical conductivity is strictly positive when charged species are present
    sigma = float(tran.electricalConductivity())
    assert sigma > 0.0

    # 3. All mixture diffusion coefficients are strictly positive
    d_mix = np.zeros(sol.n_species)
    tran.getMixDiffCoeffs(d_mix)
    assert np.all(d_mix > 0.0)
