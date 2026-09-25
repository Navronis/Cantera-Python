import unittest
from pathlib import Path

import numpy as np

import cantera_python as ct


from cantera_python import DATA_DIR; GRI30 = DATA_DIR / 'gri30.yaml'


class IdealGasEquilibriumSolverTests(unittest.TestCase):
    def setUp(self):
        self.gas = ct.Solution(GRI30)
        self.feed = 'H2:2, O2:1, N2:3.76'

    def _element_totals(self):
        mechanism = self.gas._mechanism
        composition = {species.name: species.composition
                       for species in mechanism.species}
        return {
            element: sum(x * composition[name].get(element, 0.0)
                         for name, x in zip(self.gas.species_names, self.gas.X))
            for element in {e for values in composition.values() for e in values}
        }

    def test_tp_equilibrium_matches_released_reference_and_conserves_elements(self):
        self.gas.TPX = 2500.0, ct.OneAtm, self.feed
        elements_before = self._element_totals()
        self.gas.equilibrate('TP')
        reference = {
            'H2': 0.022313894256282045, 'O2': 0.006895228635628145,
            'H2O': 0.3115945582315578, 'OH': 0.010858235601810663,
            'H': 0.0037589847342708355, 'O': 0.00120233012636467,
            'N2': 0.6396063593168504, 'NO': 0.003766955170977291,
        }
        for name, expected in reference.items():
            self.assertAlmostEqual(self.gas.X[self.gas.species_index(name)], expected,
                                   delta=max(1e-10, expected * 2e-6))
        elements_after = self._element_totals()
        before_sum = sum(elements_before.values())
        after_sum = sum(elements_after.values())
        self.assertLess(max(abs(elements_after[e] / after_sum
                                - elements_before[e] / before_sum)
                            for e in elements_before), 1e-10)

    def test_hp_equilibrium_recovers_adiabatic_temperature_and_enthalpy(self):
        self.gas.TPX = 300.0, ct.OneAtm, self.feed
        target_enthalpy = self.gas.enthalpy_mass
        self.gas.equilibrate('HP')
        self.assertAlmostEqual(self.gas.T, 2380.806278078447, delta=1.0)
        self.assertAlmostEqual(self.gas.enthalpy_mass, target_enthalpy,
                               delta=max(1e-5, abs(target_enthalpy) * 1e-8))

    def test_hp_equilibrium_at_thirty_atmospheres_matches_reference(self):
        self.gas.TPX = 300.0, 30.0 * ct.OneAtm, self.feed
        self.gas.equilibrate('HP')
        self.assertAlmostEqual(self.gas.T, 2460.770255633361, delta=1.0)
        reference = {
            'H2': 0.00700768035494948, 'O2': 0.0018519414454392115,
            'H2O': 0.33693553119757197, 'OH': 0.0030665821868392806,
            'H': 0.00032278334215344554, 'O': 9.347678214976257e-05,
            'N2': 0.6488845890951358, 'NO': 0.0018333880204016283,
        }
        for name, expected in reference.items():
            self.assertAlmostEqual(self.gas.X[self.gas.species_index(name)], expected,
                                   delta=max(1e-10, expected * 2e-6))

    def test_invalid_mode_and_iteration_failure_are_explicit(self):
        self.gas.TPX = 300.0, ct.OneAtm, self.feed
        with self.assertRaisesRegex(ValueError, 'TP, HP, SP, UV, or SV'):
            self.gas.equilibrate('XYZ')
        with self.assertRaisesRegex(RuntimeError, 'converge'):
            self.gas.equilibrate('TP', max_steps=0)


if __name__ == '__main__':
    unittest.main()
