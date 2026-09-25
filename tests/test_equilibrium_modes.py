"""Verification tests for all equilibrium solver modes (TP, HP, SP, UV, SV)."""
import math
import unittest
from pathlib import Path
import numpy as np

import cantera_python as ct

from cantera_python import DATA_DIR; GRI30 = DATA_DIR / 'gri30.yaml'


class EquilibriumModesTests(unittest.TestCase):
    def setUp(self):
        self.gas = ct.Solution(GRI30)
        self.feed_h2 = 'H2:2, O2:1, N2:3.76'
        self.feed_ch4 = 'CH4:1, O2:2, N2:7.52'

    def _element_totals(self, gas):
        mechanism = gas._mechanism
        composition = {s.name: s.composition for s in mechanism.species}
        return {
            element: sum(x * composition[name].get(element, 0.0)
                         for name, x in zip(gas.species_names, gas.X))
            for element in {e for values in composition.values() for e in values}
        }

    def _check_element_conservation(self, gas, before):
        after = self._element_totals(gas)
        before_sum = sum(before.values())
        after_sum = sum(after.values())
        max_err = max(abs(after[e] / after_sum - before[e] / before_sum) for e in before)
        self.assertLess(max_err, 1e-9, msg=f"Element conservation failed: err={max_err}")

    def test_tp_mode(self):
        self.gas.TPX = 2200.0, ct.OneAtm, self.feed_h2
        el_before = self._element_totals(self.gas)
        t_before = self.gas.T
        p_before = self.gas.P
        self.gas.equilibrate('TP')
        self.assertAlmostEqual(self.gas.T, t_before, places=5)
        self.assertAlmostEqual(self.gas.P, p_before, places=5)
        self._check_element_conservation(self.gas, el_before)
        self.assertGreater(self.gas.X[self.gas.species_index('H2O')], 0.2)

    def test_hp_mode(self):
        self.gas.TPX = 300.0, ct.OneAtm, self.feed_h2
        el_before = self._element_totals(self.gas)
        h_before = self.gas.enthalpy_mass
        p_before = self.gas.P
        self.gas.equilibrate('HP')
        self.assertAlmostEqual(self.gas.P, p_before, places=4)
        self.assertAlmostEqual(self.gas.enthalpy_mass, h_before, delta=max(1.0, abs(h_before) * 1e-6))
        self._check_element_conservation(self.gas, el_before)
        self.assertAlmostEqual(self.gas.T, 2380.8, delta=2.0)

    def test_uv_mode(self):
        self.gas.TPX = 300.0, ct.OneAtm, self.feed_h2
        el_before = self._element_totals(self.gas)
        u_before = self.gas.int_energy_mass
        rho_before = self.gas.density
        self.gas.equilibrate('UV')
        self.assertAlmostEqual(self.gas.density, rho_before, delta=rho_before * 1e-6)
        self.assertAlmostEqual(self.gas.int_energy_mass, u_before, delta=max(1.0, abs(u_before) * 1e-6))
        self._check_element_conservation(self.gas, el_before)
        self.assertGreater(self.gas.T, 2500.0)

    def test_sp_mode(self):
        self.gas.TPX = 1500.0, 5.0 * ct.OneAtm, self.feed_ch4
        el_before = self._element_totals(self.gas)
        s_before = self.gas.entropy_mass
        p_before = self.gas.P
        self.gas.equilibrate('SP')
        self.assertAlmostEqual(self.gas.P, p_before, places=4)
        self.assertAlmostEqual(self.gas.entropy_mass, s_before, delta=abs(s_before) * 1e-6)
        self._check_element_conservation(self.gas, el_before)

    def test_sv_mode(self):
        self.gas.TPX = 1500.0, 5.0 * ct.OneAtm, self.feed_ch4
        el_before = self._element_totals(self.gas)
        s_before = self.gas.entropy_mass
        rho_before = self.gas.density
        self.gas.equilibrate('SV')
        self.assertAlmostEqual(self.gas.density, rho_before, delta=rho_before * 1e-6)
        self.assertAlmostEqual(self.gas.entropy_mass, s_before, delta=abs(s_before) * 1e-6)
        self._check_element_conservation(self.gas, el_before)


if __name__ == '__main__':
    unittest.main()
