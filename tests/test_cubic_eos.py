"""Unit tests for Cubic Equations of State (Peng-Robinson and Redlich-Kwong).

Ported directly from Cantera native C++ test suites:
- test/thermo/PengRobinson_Test.cpp
- test/thermo/RedlichKwongMFTP_Test.cpp
"""
from __future__ import annotations

import math
from pathlib import Path
import unittest
import numpy as np

import cantera_python as ct
from cantera_python.constants import GasConstant, OneAtm, CanteraError
from cantera_python.cubic_eos import (
    PengRobinson, RedlichKwongMFTP, MixtureFugacityTP,
    FLUID_UNDEFINED, FLUID_GAS, FLUID_LIQUID_0
)


TEST_DATA_DIR = Path(__file__).resolve().parents[2] / 'cantera-original' / 'test' / 'data'
DATA_DIR = Path(__file__).resolve().parents[2] / 'cantera-original' / 'data'
THERMO_MODELS_YAML = TEST_DATA_DIR / 'thermo-models.yaml'
CO2_RK_EXAMPLE_YAML = TEST_DATA_DIR / 'co2_RK_example.yaml'
CO2_RK_LOOKUP_YAML = TEST_DATA_DIR / 'co2_RK_lookup.yaml'


class TestPengRobinson(unittest.TestCase):
    """Peng-Robinson EOS test suite matching native PengRobinson_Test.cpp."""

    def test_chem_potentials(self):
        """Verify chemical potentials across composition range against native C++ test vectors."""
        pr = PengRobinson.from_mechanism(THERMO_MODELS_YAML, 'CO2-PR')
        pr.setState_TP(298.15, 101325.0)

        expected_result = [
            -4.5733812970445037e+008,
            -4.5732707887912911e+008,
            -4.5731721431077951e+008,
            -4.5730835465227401e+008,
            -4.5730035374028891e+008,
            -4.5729309245485628e+008,
            -4.5728647273969948e+008,
            -4.5728041314238912e+008,
            -4.5727484544186872e+008,
        ]

        mu = np.zeros(pr.n_species)
        for i in range(9):
            r = 0.6 + i * 0.0375
            mf = [0.0] * pr.n_species
            mf[0] = r
            mf[2] = 1.0 - r
            pr.setMoleFractions(mf)
            pr.getChemPotentials(mu)
            self.assertAlmostEqual(mu[0], expected_result[i], delta=1e-6)

    def test_standard_concentrations(self):
        """Verify standard concentration identity C^0 = P / (R * T)."""
        pr = PengRobinson.from_mechanism(THERMO_MODELS_YAML, 'CO2-PR')
        pr.setState_TP(298.15, 101325.0)

        expected_c0 = pr.pressure() / (pr.temperature() * GasConstant)
        self.assertAlmostEqual(pr.standardConcentration(0), expected_c0, places=10)
        self.assertAlmostEqual(pr.standardConcentration(1), expected_c0, places=10)

    def test_set_tp_isobars(self):
        """Verify phase diagram isobars including subcooled liquid, phase change, and supercritical."""
        rho1 = [
            6.6603507723749249e+002,
            1.6824762614489907e+002,
            1.6248354709581241e+002,
            1.5746729362032696e+002,
            1.5302217175386241e+002,
            1.4902908974486667e+002
        ]
        rho2 = [
            7.5732259810273172e+002,
            7.2766981078381912e+002,
            6.935475475396446e+002,
            6.5227027102964917e+002,
            5.9657442842753153e+002,
            3.9966973266966875e+002
        ]
        rho3 = [
            8.0601205067780199e+002,
            7.8427655940884574e+002,
            7.6105347579146576e+002,
            7.3605202492828505e+002,
            7.0887891410210011e+002,
            6.7898591969734434e+002
        ]

        pr = PengRobinson.from_mechanism(THERMO_MODELS_YAML, 'CO2-PR')
        for i in range(6):
            temp = 294.0 + i * 2.0
            mf = [0.0] * pr.n_species
            mf[0] = 0.999
            mf[2] = 0.001
            pr.setMoleFractions(mf)

            pr.setState_TP(temp, 5542027.5)
            self.assertAlmostEqual(pr.density(), rho1[i], delta=1e-6)

            pr.setState_TP(temp, 7388370.0)
            self.assertAlmostEqual(pr.density(), rho2[i], delta=1e-6)

            pr.setState_TP(temp, 9236712.5)
            self.assertAlmostEqual(pr.density(), rho3[i], delta=1e-6)

    def test_get_pressure_theoretical(self):
        """Verify pressure calculation against analytical Peng-Robinson EOS formula across temperatures."""
        pr = PengRobinson.from_mechanism(THERMO_MODELS_YAML, 'CO2-PR')

        a_coeff = 3.958095109e+5
        b_coeff = 26.62616317 / 1000.0
        acc_factor = 0.228
        rho = 1.0737
        kappa = 0.37464 + 1.54226 * acc_factor - 0.26992 * (acc_factor ** 2)

        for i in range(15):
            temp = 296.0 + i * 50.0
            mf = [0.0] * pr.n_species
            mf[0] = 1.0
            pr.setMoleFractions(mf)
            pr.setState_TD(temp, rho)

            tcrit = pr.critTemperature()
            mv = (1.0 / rho) * pr.mean_molecular_weight
            alpha = (1.0 + kappa * (1.0 - math.sqrt(temp / tcrit))) ** 2
            pres_theoretical = ((GasConstant * temp) / (mv - b_coeff)
                                - (a_coeff * alpha) / (mv * mv + 2.0 * b_coeff * mv - b_coeff * b_coeff))
            self.assertAlmostEqual(pr.pressure(), pres_theoretical, delta=3.0)

    def test_coolprop_validate(self):
        """Validate Peng-Robinson density against tabulated values."""
        rhoCoolProp = [
            9.067928191884574,
            8.318322900591179,
            7.6883521740498155,
            7.150504298001246,
            6.685330199667018,
            6.278630757480957,
            5.919763108091383,
            5.600572727499541,
            5.314694056926007,
            5.057077678380463
        ]
        p = 5.0e5
        pr = PengRobinson.from_mechanism(THERMO_MODELS_YAML, 'CO2-PR')
        for i in range(10):
            temp = 300.0 + i * 25.0
            mf = [0.0] * pr.n_species
            mf[0] = 1.0
            pr.setMoleFractions(mf)
            pr.setState_TP(temp, p)
            self.assertAlmostEqual(pr.density(), rhoCoolProp[i], delta=1e-5)

    def test_forced_gas_branch_prevents_crossing(self):
        """Verify forced branch selection rejects liquid state crossing."""
        pr = PengRobinson.from_mechanism(THERMO_MODELS_YAML, 'CO2-PR')
        mf = [0.0] * pr.n_species
        mf[0] = 1.0
        pr.setMoleFractions(mf)

        pr.setForcedSolutionBranch(FLUID_UNDEFINED)
        pr.setState_TP(400.0, 1.0e6)
        self.assertLess(pr.reportSolnBranchActual(), FLUID_LIQUID_0)

        pr.setForcedSolutionBranch(FLUID_GAS)
        with self.assertRaises(CanteraError) as cm:
            pr.setState_TP(240.0, 4.0e7)
        self.assertIn('wrong state', str(cm.exception))

        pr.setForcedSolutionBranch(FLUID_UNDEFINED)
        pr.setState_TP(240.0, 4.0e7)
        self.assertGreaterEqual(pr.reportSolnBranchActual(), FLUID_LIQUID_0)

    def test_density_calc_filters_near_ideal_roots(self):
        """Verify density calculation filters near-ideal roots properly."""
        test = ct.Solution({'phases': [{'name': 'test', 'species': [{'gri30.yaml/species': ['CO2']}],
                                       'thermo': 'Peng-Robinson'}]})
        test.setSpeciesCoeffs('CO2', 1.0e-4, 1.0e-11, 0.0)
        test.setState_TPX(300.0, 1.0e5, 'CO2:1.0')

        rhoGas = test.densityCalc(300.0, 1.0e5, FLUID_GAS, -1.0)
        rhoIdeal = test.mean_molecular_weight * 1.0e5 / (GasConstant * 300.0)
        self.assertTrue(math.isfinite(rhoGas))
        self.assertGreater(rhoGas, 0.0)
        self.assertAlmostEqual(rhoGas, rhoIdeal, delta=1e-8 * rhoIdeal)
        self.assertLess(test.densityCalc(300.0, 1.0e5, FLUID_LIQUID_0, -1.0), 0.0)

    def test_lookup_species_properties(self):
        """Verify correspondence to critical properties database for GRI30 species."""
        test = ct.Solution({'phases': [{'name': 'test', 'species': [{'gri30.yaml/species': ['CO2', 'CH4', 'N2']}],
                                       'thermo': 'Peng-Robinson'}]})

        test.setState_TPX(330.0, 100.0 * OneAtm, 'CH4: 1.0')
        self.assertAlmostEqual(test.critPressure(), 4.63e+06, delta=1e-2)
        self.assertAlmostEqual(test.critTemperature(), 190.7, delta=1e-3)

        test.setState_TPX(330.0, 180.0 * OneAtm, 'N2: 1.0')
        self.assertAlmostEqual(test.critPressure(), 3.39e+06, delta=1e-2)
        self.assertAlmostEqual(test.critTemperature(), 126.2, delta=1e-3)

    def test_lookup_species_properties_missing(self):
        """Verify that species missing from critical-properties database raises an error."""
        with self.assertRaises(ValueError):
            ct.Solution({'phases': [{'name': 'test', 'species': [{'gri30.yaml/species': ['CO2', 'CH3', 'N2']}],
                                    'thermo': 'Peng-Robinson'}]})

    def test_local_crit_properties(self):
        """Verify calculation based on local critical-parameters in species definition."""
        test = PengRobinson.from_mechanism(THERMO_MODELS_YAML, 'CO2-PR-params')
        test.setState_TPX(400.0, 1.2e6, 'CO2: 1.0')
        self.assertAlmostEqual(test.critTemperature(), 304.128, delta=1e-5)
        self.assertAlmostEqual(test.critPressure(), 7.3773e6, delta=1e-4)

        test.setState_TPX(400.0, 1.2e6, 'H2O: 1.0')
        self.assertAlmostEqual(test.critTemperature(), 647.096, delta=1e-5)
        self.assertAlmostEqual(test.critPressure(), 22.064e6, delta=1e-4)


class TestRedlichKwongMFTP(unittest.TestCase):
    """Redlich-Kwong EOS test suite matching native RedlichKwongMFTP_Test.cpp."""

    def test_chem_potentials(self):
        """Verify chemical potentials across composition range against native C++ test vectors."""
        rk = RedlichKwongMFTP.from_mechanism(CO2_RK_EXAMPLE_YAML)
        rk.setState_TP(298.15, 101325.0)

        expected_result = [
            -4.5735784132470691e+008,
            -4.5734715010829216e+008,
            -4.5733760789206791e+008,
            -4.5732903883366525e+008,
            -4.5732130124096912e+008,
            -4.5731427966336435e+008,
            -4.5730787908411121e+008,
            -4.5730202059007066e+008,
            -4.5729663809807611e+008,
        ]

        mu = np.zeros(rk.n_species)
        for i in range(9):
            r = 0.6 + i * 0.0375
            mf = [0.0] * rk.n_species
            mf[0] = r
            mf[2] = 1.0 - r
            rk.setMoleFractions(mf)
            rk.getChemPotentials(mu)
            self.assertAlmostEqual(mu[0], expected_result[i], delta=1e-6)

    def test_standard_concentrations(self):
        """Verify standard concentration identity C^0 = P / (R * T)."""
        rk = RedlichKwongMFTP.from_mechanism(CO2_RK_EXAMPLE_YAML)
        rk.setState_TP(298.15, 101325.0)

        expected_c0 = rk.pressure() / (rk.temperature() * GasConstant)
        self.assertAlmostEqual(rk.standardConcentration(0), expected_c0, places=10)
        self.assertAlmostEqual(rk.standardConcentration(1), expected_c0, places=10)

    def test_internal_pressure(self):
        """Verify internal pressure thermodynamic identity pi_T = T (dp/dT)_v - P."""
        rk = RedlichKwongMFTP.from_mechanism(CO2_RK_EXAMPLE_YAML)
        mf = [0.0] * rk.n_species
        mf[0] = 0.73
        mf[2] = 0.27
        rk.setMoleFractions(mf)
        rk.setState_TP(450.0, 9.0e6)

        alpha_p = rk.thermalExpansionCoeff()
        beta_T = rk.isothermalCompressibility()
        pi_T_ref = rk.temperature() * alpha_p / beta_T - rk.pressure()

        self.assertAlmostEqual(rk.internalPressure(), pi_T_ref, delta=1e-7 * abs(pi_T_ref))

    def test_partial_molar_int_energies_tv_identity(self):
        """Verify thermodynamic identity utilde_k = ubar_k - pi_T * vbar_k."""
        rk = RedlichKwongMFTP.from_mechanism(CO2_RK_EXAMPLE_YAML)
        mf = [0.0] * rk.n_species
        mf[0] = 0.73
        mf[2] = 0.27
        rk.setMoleFractions(mf)
        rk.setState_TP(450.0, 9.0e6)

        kk = rk.n_species
        ubar = np.zeros(kk)
        utilde = np.zeros(kk)
        vbar = np.zeros(kk)
        rk.getPartialMolarIntEnergies(ubar)
        rk.getPartialMolarIntEnergies_TV(utilde)
        rk.getPartialMolarVolumes(vbar)
        pi_t = rk.internalPressure()

        for k in range(kk):
            expected = ubar[k] - pi_t * vbar[k]
            atol = 1e-12 * max(1.0, abs(expected))
            self.assertAlmostEqual(utilde[k], expected, delta=atol)

    def test_partial_molar_cv_tv_finite_difference(self):
        """Verify partial molar Cv_TV against central difference of internal energy."""
        rk = RedlichKwongMFTP.from_mechanism(CO2_RK_EXAMPLE_YAML)
        mf = [0.0] * rk.n_species
        mf[0] = 0.67
        mf[2] = 0.33
        rk.setMoleFractions(mf)
        rk.setState_TP(400.0, 7.0e6)

        kk = rk.n_species
        T0 = float(rk.temperature())
        rho0 = float(rk.density())
        dT = 1e-4 * T0

        cv_tilde = np.zeros(kk)
        uplus = np.zeros(kk)
        uminus = np.zeros(kk)

        rk.getPartialMolarCv_TV(cv_tilde)

        rk.setState_TD(T0 + dT, rho0)
        rk.getPartialMolarIntEnergies_TV(uplus)

        rk.setState_TD(T0 - dT, rho0)
        rk.getPartialMolarIntEnergies_TV(uminus)

        for k in range(kk):
            cv_fd = (uplus[k] - uminus[k]) / (2.0 * dT)
            atol = 1e-4 * max(1.0, abs(cv_tilde[k]))
            self.assertAlmostEqual(cv_tilde[k], cv_fd, delta=atol)

    def test_set_tp_isobars(self):
        """Verify phase diagram isobars across temperatures and pressures against native test vectors."""
        rho1 = [
            1.5870830380619182e+002,
            1.5419384162620102e+002,
            1.5016078232989273e+002,
            1.4651351852180966e+002,
            1.4318315080653846e+002,
            1.4011821957432278e+002
        ]
        rho2 = [
            6.2669819090204760e+002,
            5.9931065632330956e+002,
            5.6593959797702098e+002,
            5.1995461110601525e+002,
            3.3929302641053914e+002,
            2.7562068824891088e+002
        ]
        rho3 = [
            6.8411632182418634e+002,
            6.6686672949843251e+002,
            6.4850120074098390e+002,
            6.2879881554424378e+002,
            6.0746376039603331e+002,
            5.8409057903881308e+002
        ]

        rk = RedlichKwongMFTP.from_mechanism(CO2_RK_EXAMPLE_YAML)
        for i in range(6):
            temp = 294.0 + i * 2.0
            mf = [0.0] * rk.n_species
            mf[0] = 0.99
            mf[2] = 0.01
            rk.setMoleFractions(mf)

            rk.setState_TP(temp, 5542027.5)
            self.assertAlmostEqual(rk.density(), rho1[i], delta=1e-6)

            rk.setState_TP(temp, 7389370.0)
            self.assertAlmostEqual(rk.density(), rho2[i], delta=1e-6)

            rk.setState_TP(temp, 9236712.5)
            self.assertAlmostEqual(rk.density(), rho3[i], delta=1e-6)

    def test_crit_prop_lookup(self):
        """Verify tabulated critical property calculation for pure fluids in RedlichKwongMFTP."""
        rk = RedlichKwongMFTP.from_mechanism(CO2_RK_LOOKUP_YAML)

        # CO2 - matches tabulated values in critical-properties.yaml
        rk.X = 'CO2: 1.0'
        self.assertAlmostEqual(rk.critTemperature(), 304.2, places=1)
        self.assertAlmostEqual(rk.critPressure(), 7390000.0, places=0)

        # H2 - calculated from pureFluidParameters
        rk.X = 'H2: 1.0'
        self.assertAlmostEqual(rk.critTemperature(), 33.001, delta=1e-3)
        self.assertAlmostEqual(rk.critPressure(), 1347700.0, delta=100.0)

    def test_local_crit_properties(self):
        """Verify critical properties from local critical-parameters in YAML species definition."""
        rk = RedlichKwongMFTP.from_mechanism(THERMO_MODELS_YAML, 'CO2-RK-params')
        rk.setState_TPX(400.0, 1.2e6, 'CO2: 1.0')
        self.assertAlmostEqual(rk.critTemperature(), 304.128, delta=1e-5)
        self.assertAlmostEqual(rk.critPressure(), 7.3773e6, delta=1e-4)

        rk.setState_TPX(400.0, 1.2e6, 'H2O: 1.0')
        self.assertAlmostEqual(rk.critTemperature(), 647.096, delta=1e-5)
        self.assertAlmostEqual(rk.critPressure(), 22.064e6, delta=1e-4)

    def test_forced_gas_branch_prevents_crossing(self):
        """Verify forced branch selection rejects liquid state crossing in Redlich-Kwong."""
        rk = RedlichKwongMFTP.from_mechanism(CO2_RK_EXAMPLE_YAML)
        mf = [0.0] * rk.n_species
        mf[0] = 1.0
        rk.setMoleFractions(mf)

        rk.setForcedSolutionBranch(FLUID_UNDEFINED)
        rk.setState_TP(400.0, 1.0e6)
        self.assertLess(rk.reportSolnBranchActual(), FLUID_LIQUID_0)

        rk.setForcedSolutionBranch(FLUID_GAS)
        with self.assertRaises(CanteraError) as cm:
            rk.setState_TP(240.0, 4.0e7)
        self.assertIn('wrong state', str(cm.exception))

        rk.setForcedSolutionBranch(FLUID_UNDEFINED)
        rk.setState_TP(240.0, 4.0e7)
        self.assertGreaterEqual(rk.reportSolnBranchActual(), FLUID_LIQUID_0)

    def test_density_calc_filters_near_ideal_roots(self):
        """Verify near-ideal roots filtering in RedlichKwongMFTP."""
        test = RedlichKwongMFTP.from_mechanism(CO2_RK_EXAMPLE_YAML)
        test.setSpeciesCoeffs('CO2', 1.0e-4, 0.0, 1.0e-12)
        test.setState_TPX(300.0, 1.0e5, 'CO2:1.0')

        rhoGas = test.densityCalc(300.0, 1.0e5, FLUID_GAS, -1.0)
        rhoIdeal = test.mean_molecular_weight * 1.0e5 / (GasConstant * 300.0)
        self.assertTrue(math.isfinite(rhoGas))
        self.assertGreater(rhoGas, 0.0)
        self.assertAlmostEqual(rhoGas, rhoIdeal, delta=1e-8 * rhoIdeal)
        self.assertLess(test.densityCalc(300.0, 1.0e5, FLUID_LIQUID_0, -1.0), 0.0)

    def test_solution_integration(self):
        """Verify Redlich-Kwong loads seamlessly through ct.Solution interface."""
        h2o2 = DATA_DIR / 'h2o2.yaml'
        gas = ct.Solution(h2o2, phase_name='ohmech-RK')
        self.assertEqual(gas.name, 'ohmech-RK')
        self.assertEqual(gas.thermo_model, 'Redlich-Kwong')
        self.assertAlmostEqual(gas.T, 300.0)
        self.assertAlmostEqual(gas.P, OneAtm)
        self.assertEqual(gas.n_species, 10)
        self.assertTrue(math.isfinite(gas.density))
        self.assertTrue(math.isfinite(gas.cp_mole))


if __name__ == '__main__':
    unittest.main()
