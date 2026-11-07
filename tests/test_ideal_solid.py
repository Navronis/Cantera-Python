import math
import unittest

from cantera_python.constants import GasConstant, OneAtm
from cantera_python.ideal_gas import Species
from cantera_python.ideal_solid import IdealSolidSolution
from cantera_python.thermo import ConstCpPoly


class IdealSolidSolutionTests(unittest.TestCase):
    def setUp(self):
        self.species = (
            Species('A', 10.0, ConstCpPoly(OneAtm, 3.0, 2.0, 1.0)),
            Species('B', 20.0, ConstCpPoly(OneAtm, 4.0, 1.5, 2.0)),
        )
        self.phase = IdealSolidSolution(
            self.species, (2.0, 5.0), temperature=600.0,
            pressure=3.0 * OneAtm, mole_fractions={'A': 1, 'B': 3},
            standard_concentration='species-molar-volume')

    def test_incompressible_density_and_concentrations(self):
        self.assertEqual(self.phase.X, (0.25, 0.75))
        self.assertAlmostEqual(self.phase.mean_molecular_weight, 17.5)
        self.assertAlmostEqual(self.phase.molar_volume, 4.25)
        self.assertAlmostEqual(self.phase.density, 17.5 / 4.25)
        self.assertEqual(self.phase.isothermal_compressibility, 0.0)
        self.assertEqual(self.phase.thermal_expansion_coeff, 0.0)
        self.assertEqual(self.phase.activity_concentrations, (0.125, 0.15))
        self.assertEqual(self.phase.standard_concentrations, (0.5, 0.2))

    def test_pressure_and_ideal_mixing_equations(self):
        ref = self.phase.at(pressure=OneAtm)
        delta_p = self.phase.pressure - OneAtm
        self.assertAlmostEqual(self.phase.enthalpy_mole - ref.enthalpy_mole,
                               delta_p * self.phase.molar_volume)
        self.assertAlmostEqual(self.phase.gibbs_mole - ref.gibbs_mole,
                               delta_p * self.phase.molar_volume)
        self.assertAlmostEqual(self.phase.int_energy_mole, ref.int_energy_mole)
        expected_s = GasConstant * (
            sum(x * p.thermo.properties(600.0)[2] for x, p in zip(self.phase.X, self.species))
            - sum(x * math.log(x) for x in self.phase.X))
        self.assertAlmostEqual(self.phase.entropy_mole, expected_s)
        self.assertAlmostEqual(self.phase.gibbs_mole,
                               self.phase.enthalpy_mole - 600.0 * self.phase.entropy_mole)

    def test_partial_properties_and_standard_concentration_modes(self):
        h = self.phase.partial_molar_enthalpies
        self.assertAlmostEqual(sum(x * value for x, value in zip(self.phase.X, h)),
                               self.phase.enthalpy_mole)
        for mu, mu0, x in zip(self.phase.chemical_potentials,
                              self.phase.standard_chemical_potentials, self.phase.X):
            self.assertAlmostEqual(mu, mu0 + GasConstant * 600.0 * math.log(x))
        solvent = self.phase.at(standard_concentration='solvent_volume')
        self.assertEqual(solvent.activity_concentrations, (0.05, 0.15))
        self.assertEqual(solvent.standard_concentrations, (0.2, 0.2))
        self.assertEqual(self.phase.at().X, self.phase.X)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            IdealSolidSolution(self.species, (2.0,), mole_fractions=[1, 1])
        with self.assertRaises(ValueError):
            IdealSolidSolution(self.species, (2.0, 0.0), mole_fractions=[1, 1])
        with self.assertRaises(ValueError):
            self.phase.at(mole_fractions={'missing': 1})
        with self.assertRaises(ValueError):
            self.phase.at(standard_concentration='invalid')


if __name__ == '__main__':
    unittest.main()
