import unittest

from cantera_python.constants import GasConstant, OneAtm
from cantera_python.ideal_gas import IdealGasMixture, Species
from cantera_python.state import set_state_td, set_state_hp, set_state_sp, set_state_uv
from cantera_python.thermo import ConstCpPoly


class StateInversionTests(unittest.TestCase):
    def setUp(self):
        self.gas = IdealGasMixture([
            # Cv must be positive for the UV path; use cp values above R.
            Species('A', 2.0, ConstCpPoly(OneAtm, 3.0, 2.0, 4.0 * GasConstant)),
            Species('B', 4.0, ConstCpPoly(OneAtm, 4.0, 1.5, 5.0 * GasConstant)),
        ], temperature=350.0, pressure=OneAtm, mole_fractions={'A': 2, 'B': 1})

    def test_hp_recovers_temperature_pressure_and_composition(self):
        expected = self.gas.at(temperature=1250.0, pressure=3.0 * OneAtm)
        recovered = set_state_hp(self.gas, expected.enthalpy_mass, expected.pressure)
        self.assertAlmostEqual(recovered.temperature, expected.temperature, places=8)
        self.assertEqual(recovered.pressure, expected.pressure)
        self.assertEqual(recovered.X, self.gas.X)
        self.assertAlmostEqual(recovered.enthalpy_mass, expected.enthalpy_mass, places=6)

    def test_source_step_cap_handles_large_temperature_change(self):
        expected = self.gas.at(temperature=2400.0, pressure=2.0 * OneAtm)
        recovered = set_state_hp(self.gas, expected.enthalpy_mass, expected.pressure)
        self.assertAlmostEqual(recovered.temperature, 2400.0, places=8)

    def test_sp_recovers_temperature_pressure_and_composition(self):
        expected = self.gas.at(temperature=1400.0, pressure=0.5 * OneAtm)
        recovered = set_state_sp(self.gas, expected.entropy_mass, expected.pressure)
        self.assertAlmostEqual(recovered.temperature, expected.temperature, places=8)
        self.assertEqual(recovered.pressure, expected.pressure)
        self.assertEqual(recovered.X, self.gas.X)
        self.assertAlmostEqual(recovered.entropy_mass, expected.entropy_mass, places=8)

    def test_sp_step_cap_and_validation(self):
        expected = self.gas.at(temperature=2400.0, pressure=2.0 * OneAtm)
        self.assertAlmostEqual(
            set_state_sp(self.gas, expected.entropy_mass, expected.pressure).temperature,
            2400.0, places=8)
        with self.assertRaises(ValueError):
            set_state_sp(self.gas, float('nan'), OneAtm)
        with self.assertRaises(RuntimeError):
            set_state_sp(self.gas, 1e8, OneAtm, max_iterations=1)

    def test_uv_recovers_temperature_and_density_coupled_pressure(self):
        volume = 0.5
        target_temperature = 1500.0
        target_pressure = (GasConstant * target_temperature
                           / (self.gas.mean_molecular_weight * volume))
        expected = self.gas.at(temperature=target_temperature, pressure=target_pressure)
        recovered = set_state_uv(self.gas, expected.int_energy_mass, volume)
        self.assertAlmostEqual(recovered.temperature, target_temperature, places=8)
        self.assertAlmostEqual(recovered.pressure, target_pressure, places=6)
        self.assertAlmostEqual(1.0 / recovered.density, volume, places=12)
        self.assertEqual(recovered.X, self.gas.X)
        self.assertAlmostEqual(recovered.int_energy_mass, expected.int_energy_mass, places=6)

    def test_uv_step_cap_and_validation(self):
        volume = 0.5
        temperature = 2500.0
        expected = self.gas.at(
            temperature=temperature,
            pressure=GasConstant * temperature / (self.gas.mean_molecular_weight * volume))
        self.assertAlmostEqual(
            set_state_uv(self.gas, expected.int_energy_mass, volume).temperature,
            temperature, places=8)
        with self.assertRaises(ValueError):
            set_state_uv(self.gas, 1.0, 0.0)
        with self.assertRaises(RuntimeError):
            set_state_uv(self.gas, 1e10, volume, max_iterations=1)

    def test_td_sets_ideal_gas_pressure_from_density(self):
        state = set_state_td(self.gas, 900.0, 0.75)
        expected_pressure = 0.75 * GasConstant * 900.0 / self.gas.mean_molecular_weight
        self.assertAlmostEqual(state.pressure, expected_pressure)
        self.assertAlmostEqual(state.density, 0.75)
        self.assertAlmostEqual(state.temperature, 900.0)
        self.assertEqual(state.X, self.gas.X)

    def test_td_validation(self):
        with self.assertRaises(ValueError):
            set_state_td(self.gas, 0.0, 1.0)
        with self.assertRaises(ValueError):
            set_state_td(self.gas, 300.0, float('nan'))

    def test_validation_and_iteration_failure(self):
        with self.assertRaises(ValueError):
            set_state_hp(self.gas, 1.0, 0.0)
        with self.assertRaises(ValueError):
            set_state_hp(self.gas, float('nan'), OneAtm)
        with self.assertRaises(ValueError):
            set_state_hp(self.gas, 1.0, OneAtm, max_iterations=0)
        with self.assertRaises(RuntimeError):
            set_state_hp(self.gas, 1e10, OneAtm, max_iterations=1)


if __name__ == '__main__':
    unittest.main()
