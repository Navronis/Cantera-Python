"""Failure-state semantics for the mutable inverse state setters.

The pinned ThermoPhase implementation leaves the last trial state installed
when HP/UV exhaust their iteration loop. SP/SV wrap the shared entropy solver
in save/restore logic and therefore restore the original state on any error.
The lower-level Python state functions remain immutable in both cases.
"""

import unittest

from cantera_python.constants import GasConstant, OneAtm
from cantera_python.ideal_gas import IdealGasMixture, IdealGasPhase, Species
from cantera_python.state import set_state_hp, set_state_sp
from cantera_python.thermo import ConstCpPoly


class InverseSetterFailureMutationTests(unittest.TestCase):
    def make_gas(self, phase_type=IdealGasPhase):
        species = [
            Species("A", 2.0, ConstCpPoly(OneAtm, 3.0, 2.0, 4.0 * GasConstant)),
            Species("B", 4.0, ConstCpPoly(OneAtm, 4.0, 1.5, 5.0 * GasConstant)),
        ]
        return phase_type(
            species, temperature=350.0, pressure=OneAtm,
            mole_fractions={"A": 2, "B": 1},
        )

    def test_hp_and_uv_restore_original_state_on_nonconvergence(self):
        hp = self.make_gas()
        before_hp = (hp.T, hp.P, hp.density)
        with self.assertRaises(RuntimeError):
            hp.setState_HP(1.0e10, OneAtm, max_iterations=1)
        self.assertEqual((hp.T, hp.P, hp.density), before_hp)

        uv = self.make_gas()
        before_uv = (uv.T, uv.P, uv.density)
        with self.assertRaises(RuntimeError):
            uv.setState_UV(1.0e10, 0.5, max_iterations=1)
        self.assertEqual((uv.T, uv.P, uv.density), before_uv)

    def test_property_error_restores_hp_and_sp(self):
        class FailingThermo(ConstCpPoly):
            def properties(self, temperature):
                if temperature >= 450.0:
                    raise RuntimeError("synthetic thermo-property failure")
                return super().properties(temperature)

        def phase():
            thermo = FailingThermo(
                OneAtm, 3.0, 2.0, 4.0 * GasConstant,
            )
            return IdealGasPhase(
                [Species("A", 2.0, thermo)], temperature=350.0,
                pressure=OneAtm, mole_fractions={"A": 1.0},
            )

        hp = phase()
        hp_before = (hp.T, hp.P, hp.density)
        with self.assertRaisesRegex(RuntimeError, "synthetic thermo-property failure"):
            hp.setState_HP(1.0e10, OneAtm)
        self.assertEqual((hp.T, hp.P, hp.density), hp_before)

        sp = phase()
        with self.assertRaisesRegex(RuntimeError, "synthetic thermo-property failure"):
            sp.setState_SP(1.0e8, OneAtm)
        self.assertEqual(sp.T, 350.0)

    def test_sp_and_sv_restore_original_state_on_nonconvergence(self):
        sp = self.make_gas()
        before_sp = (sp.T, sp.P, sp.density)
        with self.assertRaises(RuntimeError):
            sp.setState_SP(1.0e8, OneAtm, max_iterations=1)
        self.assertEqual((sp.T, sp.P, sp.density), before_sp)

        sv = self.make_gas()
        before_sv = (sv.T, sv.P, sv.density)
        with self.assertRaises(RuntimeError):
            sv.setState_SV(1.0e8, 0.5, max_iterations=1)
        self.assertEqual((sv.T, sv.P, sv.density), before_sv)

    def test_immutable_state_api_does_not_mutate_input_on_failure(self):
        gas = self.make_gas(IdealGasMixture)
        with self.assertRaises(RuntimeError) as raised:
            set_state_hp(gas, 1.0e10, OneAtm, max_iterations=1)
        self.assertEqual(gas.temperature, 350.0)
        self.assertEqual(raised.exception._cantera_failure_state.temperature, 450.0)

    def test_unenforced_minimum_temperature_uses_source_clamp(self):
        thermo = ConstCpPoly(
            OneAtm, 3.0, 2.0, 4.0 * GasConstant,
            min_temp=200.0, max_temp=6000.0,
        )
        gas = IdealGasMixture(
            [Species("A", 2.0, thermo)], temperature=150.0,
            pressure=OneAtm, mole_fractions={"A": 1.0},
        )
        trace = []
        with self.assertRaises(RuntimeError):
            set_state_hp(
                gas, 1.0e10, OneAtm, max_iterations=1, trace=trace,
            )
        # ThermoPhase.cpp uses clip(minTemp(), SmallNumber, 100.0), so an
        # unenforced species minimum of 200 K does not raise this solver's
        # fallback lower bound above 100 K. Starting at 150 K gives a 250 K
        # candidate after the source's +100 K step limit.
        self.assertEqual(trace[0]["temperature"], 150.0)
        self.assertEqual(trace[0]["accepted_temperature"], 250.0)

    def test_entropy_step_is_not_limited_by_hp_uv_temperature_floor(self):
        gas = self.make_gas(IdealGasMixture).at(temperature=150.0)
        target_temperature = 20.0
        target_entropy = gas.at(temperature=target_temperature).entropy_mass
        trace = []
        result = set_state_sp(
            gas, target_entropy, OneAtm, trace=trace,
        )
        row = next(item for item in trace if item["iteration"] == 1)
        # The entropy solver's second Newton trial falls below one third of
        # the current temperature while remaining positive. The pinned SP/SV
        # loop has no HP/UV-only one-third-temperature safeguard.
        self.assertAlmostEqual(row["accepted_temperature"], 4.18546340629, places=10)
        self.assertLess(row["accepted_temperature"], row["temperature"] / 3.0)
        self.assertAlmostEqual(result.temperature, target_temperature, places=7)


if __name__ == "__main__":
    unittest.main()
