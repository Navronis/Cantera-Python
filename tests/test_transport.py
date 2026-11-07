from types import SimpleNamespace
import unittest

from cantera_python.solution import Solution
from cantera_python.transport import GasTransportData, MixTransport


class ThermoStub:
    def __init__(self, molecular_weights, mole_fractions):
        self.temperature = 800.0
        self.pressure = 202650.0
        # GasTransport builds its fitted collision polynomials across the
        # thermodynamic phase's declared temperature interval.
        self.min_temp = 200.0
        self.max_temp = 3500.0
        self.species = tuple(
            SimpleNamespace(name=f"S{k}", molecular_weight=mw)
            for k, mw in enumerate(molecular_weights))
        self.X = tuple(mole_fractions)
        self.molecular_weights = tuple(molecular_weights)

    @property
    def n_species(self):
        return len(self.species)

    @property
    def species_names(self):
        return tuple(species.name for species in self.species)

    def species_name(self, index):
        return self.species[index].name

    @property
    def mean_molecular_weight(self):
        return sum(x * mw for x, mw in zip(self.X, self.molecular_weights))


class MixtureDiffusionTests(unittest.TestCase):
    def setUp(self):
        thermo = ThermoStub((2.016, 16.0, 31.998, 39.948),
                            (0.15, 0.2, 0.25, 0.4))
        data = {
            "S0": GasTransportData(diameter=2.92, well_depth=38.0),
            "S1": GasTransportData(diameter=2.75, well_depth=80.0),
            "S2": GasTransportData(diameter=3.46, well_depth=107.4),
            "S3": GasTransportData(diameter=3.33, well_depth=136.5),
        }
        self.gas = Solution(thermo=thermo, transport=MixTransport(thermo, data))

    def test_distinct_mixture_diffusion_definitions_match_pinned_equations(self):
        x = self.gas.X
        mw = self.gas.molecular_weights
        mean_mw = self.gas.mean_molecular_weight
        binary = self.gas.transport.binary_diffusion_coefficients()

        expected = []
        expected_mole = []
        expected_mass = []
        for k in range(self.gas.n_species):
            inverse = sum(x[j] / binary[j][k]
                          for j in range(self.gas.n_species) if j != k)
            weighted = sum(x[j] * mw[j] / binary[j][k]
                           for j in range(self.gas.n_species) if j != k)
            expected.append(
                (mean_mw - x[k] * mw[k]) / (mean_mw * inverse))
            expected_mole.append((1.0 - x[k]) / inverse)
            expected_mass.append(1.0 / (
                inverse + x[k] * weighted / (mean_mw - x[k] * mw[k])))

        for actual, reference in zip(self.gas.mix_diff_coeffs, expected):
            self.assertAlmostEqual(actual, reference, places=15)
        for actual, reference in zip(self.gas.mix_diff_coeffs_mole, expected_mole):
            self.assertAlmostEqual(actual, reference, places=15)
        for actual, reference in zip(self.gas.mix_diff_coeffs_mass, expected_mass):
            self.assertAlmostEqual(actual, reference, places=15)

        self.assertNotEqual(self.gas.mix_diff_coeffs, self.gas.mix_diff_coeffs_mole)
        self.assertNotEqual(self.gas.mix_diff_coeffs, self.gas.mix_diff_coeffs_mass)

    def test_single_species_uses_binary_self_diffusion(self):
        thermo = ThermoStub((39.948,), (1.0,))
        transport = MixTransport(
            thermo, {"S0": GasTransportData(diameter=3.33, well_depth=136.5)})
        gas = Solution(thermo=thermo, transport=transport)
        expected = gas.transport.binary_diffusion_coefficients()[0][0]
        self.assertEqual(gas.mix_diff_coeffs, (expected,))
        self.assertEqual(gas.mix_diff_coeffs_mole, (expected,))
        self.assertEqual(gas.mix_diff_coeffs_mass, (expected,))


if __name__ == "__main__":
    unittest.main()
