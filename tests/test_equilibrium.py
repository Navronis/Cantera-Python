import math
import unittest

from cantera_python.constants import GasConstant, OneAtm
from cantera_python.equilibrium import IdealGasEquilibrium, ReactionStoichiometry
from cantera_python.ideal_gas import Species
from cantera_python.thermo import ConstCpPoly


class IdealGasEquilibriumTests(unittest.TestCase):
    def setUp(self):
        self.species = (
            Species('A', 10.0, ConstCpPoly(OneAtm, 3.0, 2.0, 1.0)),
            Species('B', 20.0, ConstCpPoly(OneAtm, 4.0, 1.5, 2.0)),
            Species('C', 30.0, ConstCpPoly(OneAtm, 5.0, 1.0, 3.0)),
        )
        self.eq = IdealGasEquilibrium(self.species, temperature=500.0)

    def test_standard_state_changes_and_kp_identity(self):
        reaction = ReactionStoichiometry({'A': 1.0, 'B': 0.5}, {'C': 1.5})
        h = self.eq.delta_standard_enthalpy(reaction)
        s = self.eq.delta_standard_entropy(reaction)
        g = self.eq.delta_standard_gibbs(reaction)
        self.assertAlmostEqual(g, h - 500.0 * s)
        self.assertAlmostEqual(self.eq.equilibrium_constant_pressure(reaction),
                               math.exp(-g / (GasConstant * 500.0)))

    def test_source_concentration_equilibrium_equation_with_fractional_delta_n(self):
        # Matches BulkKinetics.cpp: exp(-delta_g0/RT + delta_n*log(Pref/RT)).
        reaction = ReactionStoichiometry({'A': 1.0}, {'B': 1.4, 'C': 0.2})
        expected = (self.eq.equilibrium_constant_pressure(reaction)
                    * (OneAtm / (GasConstant * 500.0)) ** 0.6)
        self.assertAlmostEqual(reaction.delta_n, 0.6)
        self.assertAlmostEqual(self.eq.equilibrium_constant_concentration(reaction), expected)
        self.assertAlmostEqual(self.eq.log_equilibrium_constant_concentration(reaction),
                               math.log(expected))

    def test_zero_delta_n_and_temperature_state_copy(self):
        reaction = ReactionStoichiometry({'A': 1}, {'B': 1})
        self.assertEqual(reaction.delta_n, 0.0)
        self.assertAlmostEqual(self.eq.equilibrium_constant_concentration(reaction),
                               self.eq.equilibrium_constant_pressure(reaction))
        changed = self.eq.at(temperature=800.0)
        self.assertEqual(changed.temperature, 800.0)
        self.assertNotEqual(changed.delta_standard_gibbs(reaction),
                            self.eq.delta_standard_gibbs(reaction))

    def test_validation(self):
        with self.assertRaises(ValueError):
            ReactionStoichiometry({}, {'A': 1})
        with self.assertRaises(ValueError):
            ReactionStoichiometry({'A': 0}, {'B': 1})
        with self.assertRaises(ValueError):
            self.eq.equilibrium_constant_pressure(ReactionStoichiometry({'A': 1}, {'missing': 1}))
        with self.assertRaises(ValueError):
            IdealGasEquilibrium(self.species, temperature=0)


if __name__ == '__main__':
    unittest.main()
