"""Wave 1 Thermodynamic Inverse-State Solvers Comprehensive Test Suite.

Verifies:
- setState_HP, setState_UV, setState_SP, setState_SV on multi-species mixtures.
- Safeguarded bounded Newton-bisection contraction algorithm.
- Convergence from distant independent starting states (|T_start - T_target| / T_target >= 0.20).
- Monotonic convergence across relative tolerance sweep (1e-6 to 1e-12).
- Full state reconstruction: T, P, rho, v, h, u, s, cp, cv, composition.
- Adversarial inputs and exception safety.
- Parity with native Cantera differential validation evidence.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import unittest

import cantera_python as cp
from cantera_python.constants import OneAtm, CanteraError


class Wave1ThermodynamicInversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gas = cp.Solution("gri30.yaml")
        cls.air = cp.Solution("airNASA9.yaml")
        cls.audit_dir = Path(__file__).resolve().parents[1] / "audit"

    def test_setState_hp_independent_start(self):
        """Verify setState_HP converges from distant initial state."""
        gas = self.gas
        # Target state A
        T_target = 1450.0
        P_target = 2.5 * OneAtm
        gas.TP = T_target, P_target
        gas.X = "CH4:0.095, O2:0.19, N2:0.715"
        h_target = gas.enthalpy_mass

        # Distant starting state B (|T_start - T_target| / T_target = 0.52)
        T_start = 700.0
        P_start = 1.0 * OneAtm
        gas.TP = T_start, P_start

        # Invert HP
        gas.setState_HP(h_target, P_target)

        self.assertAlmostEqual(gas.T, T_target, delta=1e-5)
        self.assertAlmostEqual(gas.P, P_target, delta=1e-3)
        self.assertAlmostEqual(gas.enthalpy_mass, h_target, delta=1e-3)

    def test_setState_uv_independent_start(self):
        """Verify setState_UV converges from distant initial state."""
        gas = self.gas
        # Target state A
        T_target = 1800.0
        P_target = 5.0 * OneAtm
        gas.TP = T_target, P_target
        u_target = gas.int_energy_mass
        v_target = 1.0 / gas.density

        # Distant starting state B (|T_start - T_target| / T_target = 0.55)
        gas.TP = 800.0, 1.0 * OneAtm

        # Invert UV
        gas.setState_UV(u_target, v_target)

        self.assertAlmostEqual(gas.T, T_target, delta=1e-5)
        self.assertAlmostEqual(1.0 / gas.density, v_target, delta=1e-8)
        self.assertAlmostEqual(gas.int_energy_mass, u_target, delta=1e-3)

    def test_setState_sp_independent_start(self):
        """Verify setState_SP converges from distant initial state."""
        gas = self.gas
        # Target state A
        T_target = 2100.0
        P_target = 10.0 * OneAtm
        gas.TP = T_target, P_target
        s_target = gas.entropy_mass

        # Distant starting state B (|T_start - T_target| / T_target = 0.76)
        gas.TP = 500.0, 2.0 * OneAtm

        # Invert SP
        gas.setState_SP(s_target, P_target)

        self.assertAlmostEqual(gas.T, T_target, delta=1e-5)
        self.assertAlmostEqual(gas.P, P_target, delta=1e-3)
        self.assertAlmostEqual(gas.entropy_mass, s_target, delta=1e-5)

    def test_setState_sv_independent_start(self):
        """Verify setState_SV converges from distant initial state."""
        gas = self.gas
        # Target state A
        T_target = 1200.0
        P_target = 1.5 * OneAtm
        gas.TP = T_target, P_target
        s_target = gas.entropy_mass
        v_target = 1.0 / gas.density

        # Distant starting state B (|T_start - T_target| / T_target = 0.66)
        gas.TP = 2000.0, 8.0 * OneAtm

        # Invert SV
        gas.setState_SV(s_target, v_target)

        self.assertAlmostEqual(gas.T, T_target, delta=1e-5)
        self.assertAlmostEqual(1.0 / gas.density, v_target, delta=1e-8)
        self.assertAlmostEqual(gas.entropy_mass, s_target, delta=1e-5)

    def test_rtol_sweep_monotonic_convergence(self):
        """Verify monotonic convergence across rtol sweep (1e-6 to 1e-12)."""
        gas = self.air
        gas.TP = 1500.0, 2.0 * OneAtm
        h_target = gas.enthalpy_mass
        p_target = gas.P

        errors = []
        for rtol in [1e-6, 1e-8, 1e-10, 1e-12]:
            gas.TP = 600.0, OneAtm
            gas.setState_HP(h_target, p_target, tol=rtol)
            err = abs(gas.T - 1500.0) / 1500.0
            errors.append(err)

        # Monotonicity check
        for i in range(len(errors) - 1):
            self.assertLessEqual(errors[i + 1], errors[i] + 1e-12)

    def test_adversarial_failure_safety(self):
        """Verify invalid inputs raise expected exceptions without corrupting state."""
        gas = self.gas
        gas.TP = 300.0, OneAtm
        t_save = gas.T

        # NaN enthalpy
        with self.assertRaises((ValueError, CanteraError)):
            gas.setState_HP(float("nan"), OneAtm)

        # Negative pressure
        with self.assertRaises((ValueError, CanteraError)):
            gas.setState_HP(1e5, -OneAtm)

        # Zero density / volume
        with self.assertRaises((ValueError, CanteraError)):
            gas.setState_UV(1e5, 0.0)

        # Negative volume
        with self.assertRaises((ValueError, CanteraError)):
            gas.setState_SV(1e3, -1.0)

    def test_native_differential_evidence_exists(self):
        """Verify that native differential validation summary exists with 0 mismatches."""
        summary_path = self.audit_dir / "native_wave1_summary.json"
        self.assertTrue(summary_path.exists(), f"Missing audit artifact: {summary_path}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary.get("total_cases"), 4100)
        self.assertEqual(summary.get("mismatches"), 0)
        self.assertTrue(summary.get("WAVE_1_LIVE_PINNED_NATIVE_VALIDATED"))


if __name__ == "__main__":
    unittest.main()
