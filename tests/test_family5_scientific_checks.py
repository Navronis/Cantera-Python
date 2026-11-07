"""Scientific invariant tests for Kinetics Family 5 (Plasma, Vibrational, MultiRate).

Verifies fundamental physical and mathematical invariants:
1. Two-temperature plasma thermal equilibrium limit:
   When Te == T, TwoTempPlasmaRate reduces identically to a generalized Arrhenius
   form with total temperature exponent (b + b_gas) and energy Ea.
2. Constant vibrational relaxation rate temperature invariance:
   ConstantVibrationalRelaxationRate(A) == A for all physical temperatures.
3. Landau-Teller vibrational relaxation scaling:
   VibrationalRelaxationRate with C13 < 0 exhibits monotonic increase with T.
4. Electron collision plasma rate integral properties:
   - Strictly linear with respect to EEDF amplitude: k(levels, c * f) == c * k(levels, f)
   - Strictly linear with respect to cross-section amplitude
   - Non-negative for physical non-negative inputs
5. CustomFunc1Rate delegation fidelity:
   Arbitrary analytical functions are evaluated with floating-point precision.
6. Kinetics temperature derivative mass conservation:
   sum_k(W_k * d_wdot_k / dT) == 0 (total mass rate derivative vanishes).
"""

import math
import numpy as np
import pytest
import cantera_python as ct
from cantera_python.rates import (
    TwoTempPlasmaRate, TwoTempPlasmaData,
    ElectronCollisionPlasmaRate, ElectronCollisionPlasmaData,
    ConstantVibrationalRelaxationRate, StarikovskiyVibrationalRelaxationRate,
    CastelaVibrationalRelaxationRate, MultiStateResolvedVibrationalRelaxationRate,
    CustomFunc1Rate, GAS_CONSTANT
)


def test_two_temp_plasma_thermal_equilibrium_limit():
    """When Te == T, TwoTempPlasmaRate matches generalized Arrhenius."""
    A = 2.5e12
    b = 0.5
    bg = 0.2
    Ea = 5.0e7  # J/kmol
    EE = 3.0e7
    Tinv = 5000.0

    r2t = TwoTempPlasmaRate(A=A, b=b, Ea=Ea, EE=EE, bg=bg, Tinv=Tinv)

    for T in [300.0, 500.0, 1000.0, 2000.0, 3500.0]:
        val_plasma = r2t(T, Te=T)
        # Analytical expression at Te == T:
        # EE term vanishes since (Te - T) == 0
        expected = A * math.exp(
            (bg + b) * math.log(T)
            - (Ea / GAS_CONSTANT) / T
            - T / Tinv
        )
        assert abs(val_plasma - expected) / expected < 1e-14


def test_constant_vibrational_relaxation_temperature_invariance():
    """ConstantVibrationalRelaxationRate is invariant to temperature."""
    A_val = 4.82e7
    rate = ConstantVibrationalRelaxationRate(A=A_val)
    for T in [200.0, 300.0, 800.0, 1500.0, 3000.0, 5000.0]:
        assert rate(T) == A_val


def test_landau_teller_scaling_monotonicity():
    """Landau-Teller vibrational rate with negative C13 increases monotonically with T."""
    # Typical Landau-Teller: C13 < 0 implies exp(C13 * T^(-1/3)) increases as T increases
    rate = StarikovskiyVibrationalRelaxationRate(A=1.0e8, b=0.0, C0=0.0, C13=-10.0)
    temps = np.linspace(300.0, 3000.0, 20)
    vals = [rate(float(T)) for T in temps]
    for i in range(len(vals) - 1):
        assert vals[i + 1] > vals[i], f"Landau-Teller rate failed monotonicity between {temps[i]} and {temps[i+1]}"


def test_electron_collision_integral_linearity_and_positivity():
    """Electron collision rates are strictly linear and positive-definite."""
    energy_grid = [0.0, 1.0, 2.0, 5.0, 10.0]
    cross_sec = [0.0, 1.0e-20, 2.5e-20, 4.0e-20, 5.0e-20]
    rate = ElectronCollisionPlasmaRate(energy_grid, cross_sec)

    eedf = [1.5, 1.2, 0.8, 0.2, 0.01]

    base_k = rate(energy_grid, eedf)
    assert base_k > 0.0, "Rate must be strictly positive for positive inputs"

    # Linearity w.r.t. EEDF scaling
    scaled_eedf = [3.0 * x for x in eedf]
    scaled_k = rate(energy_grid, scaled_eedf)
    assert abs(scaled_k - 3.0 * base_k) / (3.0 * base_k) < 1e-14

    # Linearity w.r.t. cross-section scaling
    scaled_cs = [2.0 * x for x in cross_sec]
    rate_scaled_cs = ElectronCollisionPlasmaRate(energy_grid, scaled_cs)
    cs_scaled_k = rate_scaled_cs(energy_grid, eedf)
    assert abs(cs_scaled_k - 2.0 * base_k) / (2.0 * base_k) < 1e-14


def test_custom_func1_exact_delegation():
    """CustomFunc1Rate exactly evaluates arbitrary user-supplied functions."""
    cfr = CustomFunc1Rate()
    cfr.set_rate_function(lambda T: 1.5e5 * (T / 300.0) ** 1.75 + math.cos(T))
    for T in [300.0, 750.0, 1200.0, 2500.0]:
        expected = 1.5e5 * (T / 300.0) ** 1.75 + math.cos(T)
        assert abs(cfr(T) - expected) < 1e-12


def test_kinetics_temperature_derivative_mass_conservation():
    """Sum of molar mass * d(wdot)/dT must equal 0 due to mass conservation."""
    sol = ct.Solution("h2o2.yaml")
    sol.TPX = 1400.0, 303975.0, "H2:0.4, O2:0.2, AR:0.4"
    kin = sol.kinetics
    d_wdot = kin.getNetProductionRates_ddT()
    mw = np.asarray(sol.thermo.molecular_weights)  # kg/kmol

    # Total mass production rate derivative
    total_mass_rate_ddT = float(np.sum(mw * d_wdot))
    # Relative to maximum component magnitude
    max_component = float(np.max(np.abs(mw * d_wdot)))
    assert abs(total_mass_rate_ddT) < 1e-8 * max_component or abs(total_mass_rate_ddT) < 1e-10, (
        f"Mass conservation violated for d(wdot)/dT: net = {total_mass_rate_ddT}"
    )
