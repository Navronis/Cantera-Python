import unittest

from cantera_python.constants import GasConstant, OneAtm
from cantera_python.ideal_gas import Species
from cantera_python.stoich import StoichiometricSubstance
from cantera_python.thermo import ConstCpPoly


class StoichiometricSubstanceTests(unittest.TestCase):
    def setUp(self):
        self.species = Species('A(s)', 10.0, ConstCpPoly(OneAtm, 3.0, 2.0, 1.5))
        self.phase = StoichiometricSubstance(
            self.species, density=2.0, temperature=600.0, pressure=4.0 * OneAtm)

    def test_constant_volume_pressure_setter(self):
        shifted = self.phase.with_pressure(7.0 * OneAtm)
        self.assertEqual(shifted.density, self.phase.density)
        self.assertEqual(shifted.molar_volume, self.phase.molar_volume)
        self.assertAlmostEqual(shifted.enthalpy_mole - self.phase.enthalpy_mole,
                               3.0 * OneAtm * self.phase.molar_volume)
        self.assertAlmostEqual(shifted.gibbs_mole - self.phase.gibbs_mole,
                               3.0 * OneAtm * self.phase.molar_volume)
        self.assertAlmostEqual(shifted.int_energy_mole, self.phase.int_energy_mole)

    def test_single_species_properties_and_activities(self):
        self.assertEqual(self.phase.activities, (1.0,))
        self.assertEqual(self.phase.activity_concentrations, (1.0,))
        self.assertEqual(self.phase.standard_concentration, 1.0)
        self.assertEqual(self.phase.log_standard_concentration, 0.0)
        self.assertEqual(self.phase.isothermal_compressibility, 0.0)
        self.assertEqual(self.phase.thermal_expansion_coeff, 0.0)
        self.assertEqual(self.phase.internal_pressure, 0.0)
        self.assertAlmostEqual(self.phase.cv_mole, self.phase.cp_mole)
        self.assertAlmostEqual(self.phase.gibbs_mole,
                               self.phase.enthalpy_mole - 600.0 * self.phase.entropy_mole)
        self.assertEqual(self.phase.chemical_potentials,
                         self.phase.standard_chemical_potentials)

    def test_reference_internal_energy_equation(self):
        _, h_rt, _ = self.species.thermo.properties(600.0)
        expected = GasConstant * 600.0 * h_rt - OneAtm * self.phase.molar_volume
        self.assertAlmostEqual(self.phase.int_energy_mole, expected)
        self.assertEqual(self.phase.partial_molar_int_energies,
                         (self.phase.int_energy_mole,))

    def test_input_validation_and_volume_input(self):
        volume_form = StoichiometricSubstance(self.species, molar_volume=5.0)
        self.assertEqual(volume_form.density, 2.0)
        with self.assertRaises(ValueError):
            StoichiometricSubstance(self.species, density=2.0, molar_volume=5.0)
        with self.assertRaises(ValueError):
            StoichiometricSubstance(self.species, density=0.0)
        with self.assertRaises(ValueError):
            self.phase.with_pressure(0.0)


if __name__ == '__main__':
    unittest.main()
