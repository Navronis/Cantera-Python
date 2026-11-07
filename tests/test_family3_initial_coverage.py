"""Initial focused tests for Family 3: Heterogeneous and Surface Kinetics."""
import unittest
import numpy as np

from cantera_python.kinetics import InterfaceKinetics, EdgeKinetics
from cantera_python.surface import IdealSurfacePhase, SurfaceSpecies
from cantera_python.ideal_gas import IdealGasMixture, Species
from cantera_python.thermo import ConstCpPoly
from cantera_python.rates import (
    InterfaceRate, InterfaceData, StickingRate, StickingCoverage,
)
from cantera_python.constants import GasConstant, OneAtm


class Family3InitialCoverageTests(unittest.TestCase):
    def test_interface_kinetics_type_and_basic_properties(self):
        gas_sp = [Species("H2", 2.016, ConstCpPoly()), Species("O2", 32.0, ConstCpPoly())]
        gas = IdealGasMixture(gas_sp, temperature=1000.0, pressure=OneAtm)

        surf_sp = [SurfaceSpecies("Pt(b)", 195.0, ConstCpPoly(), size=1.0),
                   SurfaceSpecies("H(s)", 1.008, ConstCpPoly(), size=1.0)]
        surf = IdealSurfacePhase(surf_sp, site_density=2.7e-9, temperature=1000.0)

        kin = InterfaceKinetics(phases=[surf, gas], reactions=[])
        self.assertEqual(kin.kineticsType(), "InterfaceKinetics")
        self.assertEqual(kin.nPhases(), 2)
        self.assertEqual(kin.nTotalSpecies(), 4)
        self.assertEqual(kin.nReactions(), 0)

    def test_edge_kinetics_type(self):
        surf_sp = [SurfaceSpecies("Pt(b)", 195.0, ConstCpPoly(), size=1.0)]
        surf = IdealSurfacePhase(surf_sp, site_density=2.7e-9, temperature=1000.0)
        edge_kin = EdgeKinetics(phases=[surf], reactions=[])
        self.assertEqual(edge_kin.kineticsType(), "Edge")

    def test_sticking_coverage_dependence(self):
        cov = StickingCoverage()
        self.assertFalse(cov.motzWiseCorrection())
        cov.setMotzWiseCorrection(True)
        self.assertTrue(cov.motzWiseCorrection())
        cov.setStickingParameters({"motz-wise": True, "species": "H2"})
        params = cov.getStickingParameters()
        self.assertEqual(params.get("species"), "H2")

    def test_sticking_rate_creation_and_evaluation(self):
        rate = StickingRate(A=0.1, b=0.0, Ea=0.0)
        self.assertEqual(rate.preExponentialFactor(), 0.1)
        self.assertEqual(rate.activationEnergy(), 0.0)
        self.assertFalse(rate.motzWiseCorrection())
        rate.setMotzWiseCorrection(True)
        self.assertTrue(rate.motzWiseCorrection())

        class MockData:
            temperature = 1000.0

        val = rate.evalFromStruct(MockData())
        self.assertEqual(val, 0.1)

    def test_interface_rate_creation(self):
        rate = InterfaceRate(A=1.0e13, b=0.5, Ea=50000.0)
        self.assertEqual(rate.preExponentialFactor(), 1.0e13)
        self.assertEqual(rate.beta(), 0.5)
        self.assertEqual(rate.activationEnergy(), 50000.0)

        class MockData:
            temperature = 1000.0

        val = rate.evalFromStruct(MockData())
        expected = 1.0e13 * (1000.0 ** 0.5) * np.exp(-50000.0 / (GasConstant * 1000.0))
        self.assertAlmostEqual(val, expected, places=5)


if __name__ == "__main__":
    unittest.main()
