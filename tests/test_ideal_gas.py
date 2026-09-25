import math
import unittest

from cantera_python.ideal_gas import IdealGasMixture, Species
from cantera_python.thermo import ConstCpPoly
from cantera_python.constants import GasConstant, OneAtm


class IdealGasTests(unittest.TestCase):
    def setUp(self):
        # cp/R, h/RT and s/R at the reference state; values are deliberately
        # different so mixture weighting and pressure/composition terms show.
        thermo_a = ConstCpPoly(101325.0, 3.5, 1.2, 2.0)
        thermo_b = ConstCpPoly(101325.0, 4.0, 2.0, 1.5)
        self.gas = IdealGasMixture([
            Species('A', 2.0, thermo_a), Species('B', 4.0, thermo_b)
        ], temperature=600.0, pressure=2 * OneAtm, mole_fractions={'A': 1, 'B': 3})

    def test_state_normalization_and_density(self):
        self.assertEqual(self.gas.X, (0.25, 0.75))
        self.assertAlmostEqual(self.gas.mean_molecular_weight, 3.5)
        self.assertAlmostEqual(self.gas.molar_density, 2 * OneAtm / (GasConstant * 600))
        self.assertAlmostEqual(self.gas.density, self.gas.molar_density * 3.5)
        self.assertAlmostEqual(sum(self.gas.Y), 1.0)

    def test_composition_string(self):
        gas = self.gas.at(mole_fractions='A: 1, B: 3')
        self.assertEqual(gas.X, (0.25, 0.75))

    def test_thermodynamic_identities(self):
        self.assertAlmostEqual(self.gas.cv_mole, self.gas.cp_mole - GasConstant)
        self.assertAlmostEqual(self.gas.int_energy_mole,
                               self.gas.enthalpy_mole - GasConstant * 600)
        self.assertAlmostEqual(self.gas.gibbs_mole,
                               self.gas.enthalpy_mole - 600 * self.gas.entropy_mole)
        self.assertAlmostEqual(self.gas.cp_mass,
                               self.gas.cp_mole / self.gas.mean_molecular_weight)

    def test_partial_properties_and_state_copy(self):
        self.assertAlmostEqual(sum(x * h for x, h in zip(self.gas.X, self.gas.partial_molar_enthalpies)),
                               self.gas.enthalpy_mole)
        self.assertEqual(self.gas.at().X, self.gas.X)
        changed = self.gas.at(temperature=900, mole_fractions=[1, 1])
        self.assertEqual(changed.X, (0.5, 0.5))
        self.assertEqual(changed.temperature, 900)

    def test_additive_species_enthalpy_offset_preserves_cp_and_entropy(self):
        delta_h = 2.5e6
        shifted = self.gas.with_enthalpy_offset(1, delta_h)
        self.assertEqual(shifted.cp_mole, self.gas.cp_mole)
        self.assertEqual(shifted.entropy_mole, self.gas.entropy_mole)
        self.assertAlmostEqual(
            shifted.partial_molar_enthalpies[1]
            - self.gas.partial_molar_enthalpies[1], delta_h)
        self.assertAlmostEqual(
            shifted.enthalpy_mole - self.gas.enthalpy_mole,
            self.gas.X[1] * delta_h)
        self.assertAlmostEqual(
            shifted.standard_chemical_potentials[1]
            - self.gas.standard_chemical_potentials[1], delta_h)
        # Thermodynamic perturbations survive ordinary state updates.
        moved = shifted.at(temperature=900.0)
        self.assertAlmostEqual(
            moved.partial_molar_enthalpies[1]
            - self.gas.at(temperature=900.0).partial_molar_enthalpies[1],
            delta_h)

    def test_domain_validation(self):
        with self.assertRaises(ValueError):
            self.gas.at(pressure=0)
        with self.assertRaises(ValueError):
            self.gas.at(mole_fractions={'missing': 1})
        with self.assertRaises(ValueError):
            IdealGasMixture(self.gas.species, mole_fractions=[0, 0])


if __name__ == '__main__':
    unittest.main()
