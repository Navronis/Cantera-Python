"""Regression tests for the shared ThermoPhase inverse solver on cubic EOS."""

import unittest

from cantera_python.constants import CanteraError, GasConstant, OneAtm
from cantera_python.cubic_eos import PengRobinson
from cantera_python.ideal_gas import Species
from cantera_python.solution import Solution
from cantera_python.thermo import ConstCpPoly


class CubicInverseSourceTests(unittest.TestCase):
    def make_phase(self):
        species = Species(
            "CO2", 44.0095,
            ConstCpPoly(OneAtm, 3.0e8, 2.1e5, 3.8e4 * GasConstant,
                        min_temp=200.0, max_temp=6000.0),
        )
        phase = PengRobinson(
            [species], temperature=350.0, pressure=OneAtm,
            mole_fractions={"CO2": 1.0},
        )
        tc, pc, acentric = 304.1282, 7.3773e6, 0.22394
        phase.setSpeciesCoeffs(
            "CO2",
            phase.omega_a * (GasConstant * tc) ** 2 / pc,
            phase.omega_b * GasConstant * tc / pc,
            acentric,
        )
        phase.setState_TP(350.0, OneAtm)
        return phase

    def test_hp_and_uv_recover_source_targets(self):
        hp = self.make_phase()
        hp.setState_TP(525.0, 2.0e6)
        h_target = float(hp.enthalpy_mass)
        hp.setState_TP(350.0, OneAtm)
        hp.setState_HP(h_target, 2.0e6)
        self.assertAlmostEqual(float(hp.temperature), 525.0, places=3)

        uv = self.make_phase()
        rho = 180.0
        uv.setState_TD(525.0, rho)
        u_target = float(uv.int_energy_mass)
        volume = 1.0 / rho
        uv.setState_TD(350.0, rho)
        uv.setState_UV(u_target, volume)
        self.assertAlmostEqual(float(uv.temperature), 525.0, places=3)
        self.assertAlmostEqual(float(uv.density), rho, places=10)

    def test_sp_and_sv_recover_targets_and_restore_after_errors(self):
        sp = self.make_phase()
        sp.setState_TP(525.0, 2.0e6)
        s_target = float(sp.entropy_mass)
        sp.setState_TP(350.0, OneAtm)
        sp.setState_SP(s_target, 2.0e6)
        self.assertAlmostEqual(float(sp.temperature), 525.0, places=7)

        sv = self.make_phase()
        rho = 180.0
        sv.setState_TD(525.0, rho)
        s_target = float(sv.entropy_mass)
        volume = 1.0 / rho
        sv.setState_TD(350.0, rho)
        sv.setState_SV(s_target, volume)
        self.assertAlmostEqual(float(sv.temperature), 525.0, places=7)

        for setter, args in ((sp.setState_SP, (1.0e8, OneAtm)),
                             (sv.setState_SV, (1.0e8, volume))):
            before = (float(sp.temperature), float(sp.density), float(sp.pressure)) if setter == sp.setState_SP else (
                float(sv.temperature), float(sv.density), float(sv.pressure))
            with self.assertRaises(CanteraError):
                setter(*args, max_iterations=1)
            after = (float(sp.temperature), float(sp.density), float(sp.pressure)) if setter == sp.setState_SP else (
                float(sv.temperature), float(sv.density), float(sv.pressure))
            self.assertEqual(after, before)

    def test_hp_uv_keep_failed_trial_and_solution_forwards_solver_options(self):
        hp = self.make_phase()
        with self.assertRaises(CanteraError):
            hp.setState_HP(1.0e12, OneAtm, max_iterations=1)
        self.assertAlmostEqual(float(hp.temperature), 450.0)

        uv = self.make_phase()
        volume = 1.0 / float(uv.density)
        with self.assertRaises(CanteraError):
            uv.setState_UV(1.0e12, volume, max_iterations=1)
        self.assertAlmostEqual(float(uv.temperature), 450.0)

        phase = self.make_phase()
        solution = Solution(thermo=phase)
        phase.setState_TP(500.0, 2.0e6)
        h_target = float(phase.enthalpy_mass)
        phase.setState_TP(350.0, OneAtm)
        solution.setState_HP(h_target, 2.0e6, rtol=1.0e-9, max_iterations=500)
        self.assertAlmostEqual(float(solution.T), 500.0, places=4)


if __name__ == "__main__":
    unittest.main()
