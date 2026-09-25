"""Source-linked regression tests for IdealGasPhase::setState_DP.

Pinned sources:
* include/cantera/thermo/IdealGasPhase.h:369-376
* src/thermo/Phase.cpp:607-616
* include/cantera/thermo/Phase.h:647-655
"""
import math
from pathlib import Path
import unittest

import cantera_python as ct
from cantera_python.constants import GasConstant, OneAtm
from cantera_python.ideal_gas import IdealGasPhase, IdealGasMixture, Species
from cantera_python.thermo import ConstCpPoly


from cantera_python import DATA_DIR
GRI30_PATH = DATA_DIR / 'gri30.yaml'


class SetStateDPSemanticsTests(unittest.TestCase):
    def _phase_factories(self):
        return (
            ('IdealGasPhase', lambda: IdealGasPhase.from_mechanism(GRI30_PATH)),
            ('Solution', lambda: ct.Solution(GRI30_PATH)),
        )

    @staticmethod
    def _initialized(factory):
        gas = factory()
        gas.TPX = 700.0, 2.0 * OneAtm, 'H2:1, O2:1'
        return gas

    def test_pressure_is_validated_before_density_and_state_is_unchanged(self):
        # IdealGasPhase.h checks p <= 0 before Phase::setDensity(rho).
        for name, factory in self._phase_factories():
            with self.subTest(api=name):
                gas = self._initialized(factory)
                before = gas.T, gas.P, gas.density
                with self.assertRaisesRegex(ValueError, 'pressure must be positive'):
                    gas.DP = -1.0, -1.0
                self.assertEqual((gas.T, gas.P, gas.density), before)

    def test_density_failure_occurs_after_positive_pressure_check(self):
        # Phase.cpp accepts density only when density > 0; NaN therefore fails.
        for name, factory in self._phase_factories():
            with self.subTest(api=name):
                gas = self._initialized(factory)
                before = gas.T, gas.P, gas.density
                with self.assertRaisesRegex(ValueError, 'density must be positive'):
                    gas.setState_DP(float('nan'), OneAtm)
                self.assertEqual((gas.T, gas.P, gas.density), before)

    def test_nan_pressure_leaves_new_density_and_old_temperature(self):
        # NaN bypasses `p <= 0`; density is installed, then derived NaN T is
        # rejected by Phase::setTemperature. The C++ object is not rolled back.
        for name, factory in self._phase_factories():
            with self.subTest(api=name):
                gas = self._initialized(factory)
                old_temperature = gas.T
                rho = 2.0
                mean_mw = gas.mean_molecular_weight
                with self.assertRaisesRegex(ValueError, 'temperature must be positive'):
                    gas.DP = rho, float('nan')
                self.assertEqual(gas.T, old_temperature)
                self.assertEqual(gas.density, rho)
                self.assertEqual(gas.P, GasConstant * (rho / mean_mw) * old_temperature)

    def test_positive_infinite_density_mutates_before_temperature_failure(self):
        # Phase::setDensity tests only > 0 and therefore stores +inf. The
        # finite-pressure derived temperature is zero and is then rejected.
        for name, factory in self._phase_factories():
            with self.subTest(api=name):
                gas = self._initialized(factory)
                old_temperature = gas.T
                with self.assertRaisesRegex(ValueError, 'temperature must be positive'):
                    gas.setState_DP(float('inf'), OneAtm)
                self.assertEqual(gas.T, old_temperature)
                self.assertTrue(math.isinf(gas.density))
                self.assertTrue(math.isinf(gas.P))

    def test_positive_infinite_pressure_is_accepted(self):
        # Both source guards are strict `> 0`/`<= 0` comparisons, so +inf P
        # produces and installs +inf T at the requested finite density.
        for name, factory in self._phase_factories():
            with self.subTest(api=name):
                gas = self._initialized(factory)
                gas.DP = 2.0, float('inf')
                self.assertTrue(math.isinf(gas.T))
                self.assertEqual(gas.density, 2.0)
                self.assertTrue(math.isinf(gas.P))

    def test_both_infinite_inputs_leave_density_mutated(self):
        # inf / inf derives NaN temperature; setTemperature rejects it after
        # the independent density has already changed.
        for name, factory in self._phase_factories():
            with self.subTest(api=name):
                gas = self._initialized(factory)
                old_temperature = gas.T
                with self.assertRaisesRegex(ValueError, 'temperature must be positive'):
                    gas.DP = float('inf'), float('inf')
                self.assertEqual(gas.T, old_temperature)
                self.assertTrue(math.isinf(gas.density))

    def test_immutable_state_uses_same_validation_order_and_infinity_domain(self):
        species = Species(
            'A', 2.0, ConstCpPoly(OneAtm, 298.15, 0.0, 4.0 * GasConstant))
        gas = IdealGasMixture([species], temperature=700.0, pressure=OneAtm)
        with self.assertRaisesRegex(ValueError, 'pressure must be positive'):
            gas.setState_DP(-1.0, -1.0)
        result = gas.setState_DP(2.0, float('inf'))
        self.assertTrue(math.isinf(result.temperature))
        self.assertEqual(result.density, 2.0)


if __name__ == '__main__':
    unittest.main()
