import math
import unittest
from cantera_python.kinetics import (ElementaryReaction, FalloffReaction, GasKinetics,
                                     ThirdBody, create_reaction)
from cantera_python.rates import (ArrheniusRate, BlowersMaselRate, ChebyshevRate, LindemannRate,
                                  PlogRate, SriRate, TroeRate, TsangRate)
from cantera_python.mechanism import load_mechanism
from cantera_python.ideal_gas import IdealGasMixture, Species
from cantera_python.thermo import ConstCpPoly
from cantera_python.constants import GasConstant, OneAtm


class KineticsTests(unittest.TestCase):
    def test_blowers_masel_reaction_uses_live_thermo_enthalpy(self):
        class Thermo:
            species_names = ('A', 'B')
            partial_molar_enthalpies = (0.0, -50.0 * GasConstant)

        thermo = Thermo()
        rate = BlowersMaselRate(2.0, 0.0, 10.0 * GasConstant,
                                100.0 * GasConstant)
        reaction = ElementaryReaction(
            rate, {'A': 1.0}, {'B': 1.0}, rate_thermo=thermo)
        self.assertEqual(reaction.forward_rate(1000.0, {'A': 1.0}), 2.0)
        self.assertEqual(rate.delta_enthalpy, -50.0 * GasConstant)

        thermo.partial_molar_enthalpies = (0.0, 50.0 * GasConstant)
        expected = 2.0 * math.exp(-50.0 / 1000.0)
        self.assertAlmostEqual(
            reaction.forward_rate(1000.0, {'A': 1.0}), expected)
        self.assertEqual(rate.delta_enthalpy, 50.0 * GasConstant)

    def _third_body_duplicate_mechanism(self, policy='warn', skip=False,
                                        undeclared_efficiency=False):
        phase = {
            'name': 'gas', 'thermo': 'ideal-gas', 'kinetics': 'bulk',
            'transport': 'none', 'species': ['A', 'B', 'D'],
            'reactions': 'all',
            'explicit-third-body-duplicates': policy,
        }
        if skip:
            phase['skip-undeclared-third-bodies'] = True
        efficiencies = {'D': 1.0}
        if undeclared_efficiency:
            efficiencies['X'] = 2.0
        return load_mechanism({
            'phases': [phase],
            'species': [
                {'name': 'A', 'composition': {'H': 1},
                 'thermo': {'model': 'constant-cp'}},
                {'name': 'B', 'composition': {'H': 2},
                 'thermo': {'model': 'constant-cp'}},
                {'name': 'D', 'composition': {'Ar': 1},
                 'thermo': {'model': 'constant-cp'}},
            ],
            'reactions': [
                {'equation': '2 A + D => B + D',
                 'rate-constant': {'A': 1.0}},
                {'equation': '2 A + M => B + M', 'type': 'three-body',
                 'default-efficiency': 0.0, 'efficiencies': efficiencies,
                 'rate-constant': {'A': 1.0}},
            ],
        }).for_phase('gas')

    def test_explicit_third_body_duplicate_policies(self):
        with self.assertWarnsRegex(UserWarning, 'Undeclared duplicate third body'):
            warned = GasKinetics.from_mechanism(
                self._third_body_duplicate_mechanism())
        self.assertFalse(any(reaction.duplicate for reaction in warned.reactions))

        with self.assertRaisesRegex(ValueError, 'Undeclared duplicate reactions'):
            GasKinetics.from_mechanism(
                self._third_body_duplicate_mechanism('error'))

        marked = GasKinetics.from_mechanism(
            self._third_body_duplicate_mechanism('mark-duplicate'))
        self.assertTrue(all(reaction.duplicate for reaction in marked.reactions))

        modified = GasKinetics.from_mechanism(
            self._third_body_duplicate_mechanism('modify-efficiency'))
        general = next(reaction for reaction in modified.reactions
                       if reaction.third_body.name == 'M')
        self.assertEqual(general.third_body.efficiencies['D'], 0.0)

    def test_skip_undeclared_third_body_efficiencies(self):
        with self.assertRaisesRegex(ValueError, 'undeclared species'):
            GasKinetics.from_mechanism(self._third_body_duplicate_mechanism(
                'modify-efficiency', undeclared_efficiency=True))
        kinetics = GasKinetics.from_mechanism(self._third_body_duplicate_mechanism(
            'modify-efficiency', skip=True, undeclared_efficiency=True))
        general = next(reaction for reaction in kinetics.reactions
                       if reaction.third_body.name == 'M')
        self.assertNotIn('X', general.third_body.efficiencies)

    def test_pressure_dependent_factories_use_live_phase_pressure(self):
        thermo = IdealGasMixture([
            Species('A', 1.0, ConstCpPoly()),
            Species('B', 1.0, ConstCpPoly()),
        ], temperature=800.0, pressure=2.0 * OneAtm, mole_fractions={'A': 1.0})

        cheb_mech = load_mechanism({
            'species': [
                {'name': 'A', 'composition': {'H': 1}},
                {'name': 'B', 'composition': {'H': 1}},
            ],
            'reactions': [{'equation': 'A => B', 'type': 'Chebyshev',
                           'temperature-range': ['300 K', '2000 K'],
                           'pressure-range': ['0.1 atm', '10 atm'],
                           'data': [[2.0, 1.0]]}],
        })
        cheb = create_reaction(cheb_mech.reactions[0], thermo=thermo)
        self.assertIsInstance(cheb.rate, ChebyshevRate)
        self.assertAlmostEqual(
            cheb.forward_rate(800.0, {'A': 1.0}),
            cheb.rate(800.0, 2.0 * OneAtm))
        kinetics = GasKinetics(thermo, [cheb])
        self.assertAlmostEqual(kinetics.forward_rate_constants()[0],
                               cheb.rate(800.0, 2.0 * OneAtm))

        plog_mech = load_mechanism({
            'species': [
                {'name': 'A', 'composition': {'H': 1}},
                {'name': 'B', 'composition': {'H': 1}},
            ],
            'reactions': [{'equation': 'A => B', 'type': 'pressure-dependent-Arrhenius',
                           'rate-constants': [
                               {'P': '1 atm', 'A': 1.0, 'b': 0.0, 'Ea': 0.0},
                               {'P': '10 atm', 'A': 10.0, 'b': 0.0, 'Ea': 0.0},
                           ]}],
        })
        plog = create_reaction(plog_mech.reactions[0], thermo=thermo)
        self.assertIsInstance(plog.rate, PlogRate)
        self.assertAlmostEqual(
            plog.forward_rate(800.0, {'A': 1.0}),
            plog.rate(800.0, 2.0 * OneAtm))

    def test_factory_negative_a_opt_in_and_unknown_type(self):
        base = {
            'species': [
                {'name': 'A', 'composition': {'H': 1}},
                {'name': 'B', 'composition': {'H': 1}},
            ],
        }
        allowed = load_mechanism({**base, 'reactions': [
            {'equation': 'A => B', 'negative-A': True,
             'rate-constant': {'A': -2.0, 'b': 0.0, 'Ea': 0.0}},
        ]})
        self.assertEqual(create_reaction(allowed.reactions[0]).rate(500.0), -2.0)

        unsupported = load_mechanism({**base, 'reactions': [
            {'equation': 'A => B', 'type': 'custom-rate'},
        ]})
        with self.assertRaisesRegex(NotImplementedError, 'custom-rate'):
            create_reaction(unsupported.reactions[0])

    def test_reaction_order_opt_ins_and_duplicate_declarations(self):
        species_records = [
            {'name': 'A', 'composition': {'H': 1}},
            {'name': 'B', 'composition': {'H': 1}},
            {'name': 'C', 'composition': {'H': 1}},
        ]
        thermo = IdealGasMixture([
            Species('A', 1.0, ConstCpPoly()),
            Species('B', 1.0, ConstCpPoly()),
            Species('C', 1.0, ConstCpPoly()),
        ])
        ordered = load_mechanism({
            'species': species_records,
            'reactions': [{'equation': 'A => B', 'orders': {'C': -1},
                           'nonreactant-orders': True, 'negative-orders': True,
                           'rate-constant': {'A': 1.0}}],
        })
        reaction = create_reaction(ordered.reactions[0], thermo)
        self.assertEqual(reaction.orders, {'C': -1})
        self.assertEqual(reaction.forward_orders, {'A': 1.0, 'C': -1})
        self.assertEqual(reaction.rate.order, 0.0)
        self.assertEqual(reaction.forward_rate(500.0, {'A': 1, 'C': 2}), 0.5)
        self.assertEqual(reaction.forward_rate(500.0, {'A': 4, 'C': 2}), 2.0)

        dimensioned = load_mechanism({
            'units': {'length': 'cm', 'quantity': 'mol'},
            'species': species_records,
            'reactions': [{'equation': 'A => B', 'orders': {'A': 2},
                           'rate-constant': {'A': 1.0}}],
        })
        dimensioned_rate = create_reaction(
            dimensioned.reactions[0], thermo,
            default_units=dimensioned.units).rate
        self.assertEqual(dimensioned_rate.order, 2.0)
        self.assertAlmostEqual(dimensioned_rate.A, 1.0e-3)

        invalid_order = load_mechanism({
            'species': species_records,
            'reactions': [{'equation': 'A => B', 'orders': {'C': -1},
                           'rate-constant': {'A': 1.0}}],
        })
        with self.assertRaisesRegex(ValueError, 'non-reactant'):
            create_reaction(invalid_order.reactions[0], thermo)

        undeclared = load_mechanism({
            'species': species_records,
            'reactions': [
                {'equation': 'A => B', 'rate-constant': {'A': 1.0}},
                {'equation': '2 A => 2 B', 'rate-constant': {'A': 2.0}},
            ],
        })
        with self.assertRaisesRegex(ValueError, 'Undeclared duplicate'):
            GasKinetics.from_mechanism(undeclared, thermo)

        declared = load_mechanism({
            'species': species_records,
            'reactions': [
                {'equation': 'A => B', 'duplicate': True,
                 'rate-constant': {'A': 1.0}},
                {'equation': '2 A => 2 B', 'duplicate': True,
                 'rate-constant': {'A': 2.0}},
            ],
        })
        self.assertEqual(GasKinetics.from_mechanism(declared, thermo).n_reactions, 2)

    def test_rate_factory_uses_general_unit_system_conversion(self):
        mechanism = load_mechanism({
            'units': {'length': 'mm', 'quantity': 'mol', 'time': 'min'},
            'species': [
                {'name': 'A', 'composition': {'H': 1}},
                {'name': 'B', 'composition': {'H': 2}},
            ],
            'reactions': [{'equation': '2 A => B',
                           'rate-constant': {'A': 60.0, 'b': 0.0, 'Ea': 0.0}}],
        })
        reaction = create_reaction(mechanism.reactions[0], default_units=mechanism.units)
        # 60 mm^3/mol/min = 1e-6 m^3/kmol/s
        self.assertAlmostEqual(reaction.rate.A, 1.0e-6)

    def test_mass_action_and_species_source_terms(self):
        rxn = ElementaryReaction(ArrheniusRate(2.0, 0.0, 0.0), {'A': 1, 'B': 2}, {'C': 1})
        rates = rxn.species_rates(500.0, {'A': 3.0, 'B': 2.0})
        self.assertAlmostEqual(rxn.forward_rate(500.0, {'A': 3, 'B': 2}), 24.0)
        self.assertEqual(rates, {'A': -24.0, 'B': -48.0, 'C': 24.0})

    def test_reversible_progress(self):
        rxn = ElementaryReaction(ArrheniusRate(2, 0, 0), {'A': 1}, {'B': 1}, True, ArrheniusRate(1, 0, 0))
        self.assertAlmostEqual(rxn.net_rate_of_progress(300, {'A': 3, 'B': 1}), 5.0)

    def test_third_body_default_compensated_efficiencies(self):
        collider = ThirdBody({'A': 2.0, 'B': 0.5}, default_efficiency=1.0)
        concentrations = {'A': 3.0, 'B': 4.0, 'C': 5.0}
        self.assertAlmostEqual(collider.effective_concentration(concentrations), 13.0)
        self.assertEqual(collider.efficiency('C'), 1.0)

    def test_mass_action_third_body_scales_both_progress_rates(self):
        rxn = ElementaryReaction(
            ArrheniusRate(2, 0, 0), {'A': 1}, {'B': 1}, True,
            ArrheniusRate(1, 0, 0), third_body=ThirdBody({'A': 2}, 1.0))
        self.assertAlmostEqual(rxn.net_rate_of_progress(300, {'A': 3, 'B': 1}), 35.0)

    def test_falloff_style_collider_is_reported_but_not_mass_action_scaled(self):
        rxn = ElementaryReaction(ArrheniusRate(2, 0, 0), {'A': 1}, {'B': 1},
                                 third_body=ThirdBody({'A': 2}, 1.0, mass_action=False))
        self.assertEqual(rxn.effective_third_body_concentration({'A': 3, 'B': 1}), 7.0)
        self.assertEqual(rxn.forward_rate(300, {'A': 3, 'B': 1}), 6.0)

    def test_irreversible_empirical_orders(self):
        rxn = ElementaryReaction(ArrheniusRate(3, 0, 0), {'A': 1, 'B': 1}, {'C': 1},
                                 orders={'A': 0.5, 'B': 2.0})
        self.assertAlmostEqual(rxn.forward_rate(300, {'A': 4, 'B': 2}), 24.0)

    def test_reaction_order_validation_matches_reaction_check(self):
        with self.assertRaisesRegex(ValueError, 'irreversible'):
            ElementaryReaction(ArrheniusRate(1, 0, 0), {'A': 1}, {'B': 1}, True,
                               ArrheniusRate(1, 0, 0), orders={'A': 2})
        with self.assertRaisesRegex(ValueError, 'non-reactant'):
            ElementaryReaction(ArrheniusRate(1, 0, 0), {'A': 1}, {'B': 1}, orders={'B': 2})
        with self.assertRaisesRegex(ValueError, 'Negative'):
            ElementaryReaction(ArrheniusRate(1, 0, 0), {'A': 1}, {'B': 1}, orders={'A': -1})
        rxn = ElementaryReaction(ArrheniusRate(1, 0, 0), {'A': 1}, {'B': 1},
                                 orders={'C': -1}, allow_nonreactant_orders=True,
                                 allow_negative_orders=True)
        self.assertAlmostEqual(rxn.forward_rate(300, {'A': 1, 'C': 2}), 0.5)
        with self.assertRaisesRegex(ValueError, 'zero concentration'):
            rxn.forward_rate(300, {'A': 1, 'C': 0})

    def test_lindemann_reaction_consumes_effective_collider_once(self):
        rate = LindemannRate(ArrheniusRate(10, 0, 0), ArrheniusRate(2, 0, 0))
        collider = ThirdBody({'A': 2.0}, default_efficiency=1.0, mass_action=False)
        rxn = FalloffReaction(rate, {'A': 1}, {'B': 1}, third_body=collider)
        concentrations = {'A': 3.0, 'B': 1.0}  # M_eff = 7
        self.assertAlmostEqual(rxn.forward_rate(500, concentrations), rate(500, 7.0) * 3.0)

    def test_troe_and_sri_reactions_use_effective_collider(self):
        collider = ThirdBody({'AR': 0.7}, default_efficiency=1.0, mass_action=False)
        concentrations = {'A': 2.0, 'AR': 3.0, 'N2': 4.0}
        M = 2.0 + 0.7 * 3.0 + 4.0
        low, high = ArrheniusRate(1.2e5, 0.1, 300.0), ArrheniusRate(2.0e3, 0.2, 100.0)
        for rate in (TroeRate(low, high, (0.5, 1000.0, 2000.0)),
                     SriRate(low, high, (0.8, 300.0, 1000.0))):
            rxn = FalloffReaction(rate, {'A': 1}, {'B': 1}, third_body=collider)
            self.assertAlmostEqual(rxn.forward_rate(800, concentrations), rate(800, M) * 2.0)

    def test_tsang_reaction_uses_collider_once_at_dilute_and_dense_limits(self):
        rate = TsangRate(ArrheniusRate(1.0e5, 0.1, 300.0),
                         ArrheniusRate(2.0e3, 0.2, 100.0), (0.4, 2.0e-4))
        # A zero default isolates the named collider and permits direct limit tests.
        collider = ThirdBody({'M': 1.0}, default_efficiency=0.0, mass_action=False)
        rxn = FalloffReaction(rate, {'A': 1}, {'B': 1}, third_body=collider)
        for concentration in (1.0e-12, 1.0e8):
            with self.subTest(collider_concentration=concentration):
                state = {'A': 1.0, 'M': concentration}
                self.assertAlmostEqual(rxn.effective_third_body_concentration(state), concentration)
                self.assertAlmostEqual(rxn.forward_rate(900.0, state), rate(900.0, concentration))

    def test_falloff_reaction_requires_falloff_collider_notation(self):
        rate = LindemannRate(ArrheniusRate(1, 0, 0), ArrheniusRate(1, 0, 0))
        with self.assertRaisesRegex(ValueError, 'requires a ThirdBody'):
            FalloffReaction(rate, {'A': 1}, {'B': 1})
        with self.assertRaisesRegex(ValueError, 'non-mass-action'):
            FalloffReaction(rate, {'A': 1}, {'B': 1}, third_body=ThirdBody())

    def test_equilibrium_constant_and_detailed_balance_for_ideal_gas(self):
        temperature = 700.0
        gas = IdealGasMixture([
            Species('A', 2.0, ConstCpPoly(298.15, 0.0, 0.0, 3.0 * GasConstant)),
            Species('B', 4.0, ConstCpPoly(298.15, 1.5e6, 100.0, 3.5 * GasConstant)),
        ], temperature=temperature, pressure=2.0 * OneAtm, mole_fractions={'A': 1, 'B': 1})
        rxn = ElementaryReaction(ArrheniusRate(4.0, 0.0, 0.0), {'A': 1}, {'B': 2},
                                 reversible=True, equilibrium_thermo=gas)
        delta_g0 = 2.0 * gas.standard_chemical_potentials[1] - gas.standard_chemical_potentials[0]
        expected_kc = math.exp(-delta_g0 / (GasConstant * temperature)
                               + math.log(gas.standard_concentration))
        self.assertAlmostEqual(rxn.delta_standard_gibbs(), delta_g0)
        self.assertAlmostEqual(rxn.equilibrium_constant(), expected_kc)
        # For A => 2 B, C_B**2 / C_A == Kc is the concentration-unit equilibrium state.
        concentrations = {'A': 1.0, 'B': expected_kc ** 0.5}
        self.assertAlmostEqual(rxn.net_rate_of_progress(temperature, concentrations), 0.0,
                               places=12)
        self.assertAlmostEqual(rxn.equilibrium_reverse_rate_coefficient(temperature),
                               4.0 / expected_kc)

    def test_equilibrium_reverse_rate_requires_matching_state_temperature(self):
        gas = IdealGasMixture([Species('A', 2.0, ConstCpPoly()), Species('B', 3.0, ConstCpPoly())],
                              temperature=500.0, mole_fractions={'A': 1, 'B': 1})
        rxn = ElementaryReaction(ArrheniusRate(1.0, 0.0, 0.0), {'A': 1}, {'B': 1},
                                 reversible=True, equilibrium_thermo=gas)
        with self.assertRaisesRegex(ValueError, 'temperature'):
            rxn.equilibrium_reverse_rate_coefficient(600.0)


if __name__ == '__main__':
    unittest.main()


class PinnedRegressionTests(unittest.TestCase):
    """Regressions for pinned-native differential findings (MTR-005, MTR-006)."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        cls.gri30 = Path(__file__).parents[2] / 'cantera-original' / 'data' / 'gri30.yaml'

    def test_mtr005_mechanism_units_converted_to_si(self):
        """GRI-30 declares cm/mol/cal units; kf must land in SI (m3/kmol, J/kmol).

        MTR-005: Solution-path construction dropped the units header, leaving
        kf up to 1e6 too large. Oracle kf[0] at 1200 K is exactly 1e8.
        """
        from cantera_python.kinetics import GasKinetics
        from cantera_python.ideal_gas import IdealGasPhase
        thermo = IdealGasPhase.from_mechanism(self.gri30)
        thermo.TPX = 1200.0, 101325.0, 'CH4:1, O2:2, N2:7.52'
        k = GasKinetics.from_mechanism(self.gri30, thermo=thermo)
        kf = k.forward_rate_constants()
        self.assertAlmostEqual(kf[0], 1.0e8, delta=1.0e8 * 1e-9)

    def test_mtr006_kc_computed_for_irreversible_reactions(self):
        """Pinned BulkKinetics::getEquilibriumConstants computes Kc for every
        reaction; only m_rkcn (reverse rate constant) is zeroed for
        irreversible ones (BulkKinetics.cpp:163-175, 482)."""
        from cantera_python.kinetics import GasKinetics
        from cantera_python.ideal_gas import IdealGasPhase
        thermo = IdealGasPhase.from_mechanism(self.gri30)
        thermo.TPX = 1200.0, 101325.0, 'CH4:1, O2:2, N2:7.52'
        k = GasKinetics.from_mechanism(self.gri30, thermo=thermo)
        kc = k.equilibrium_constants()
        self.assertEqual(len(kc), k.n_reactions)
        irrev = [i for i, r in enumerate(k.reactions) if not r.reversible]
        self.assertTrue(irrev, 'GRI-30 must contain irreversible reactions')
        for i in irrev[:5]:
            self.assertGreater(kc[i], 0.0)
            self.assertTrue(math.isfinite(kc[i]))

    def test_mechanism_carries_units_header(self):
        from cantera_python.mechanism import load_mechanism
        m = load_mechanism(self.gri30)
        self.assertIsNotNone(m.units)
        self.assertEqual(m.units.get('length'), 'cm')
        self.assertEqual(m.units.get('quantity'), 'mol')


class ChaperonRegressionTests(unittest.TestCase):
    """MTR-007: chaperon colliders must be extracted from stoichiometry."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        cls.gri30 = Path(__file__).parents[2] / 'cantera-original' / 'data' / 'gri30.yaml'

    def _kinetics(self):
        from cantera_python.ideal_gas import IdealGasPhase
        from cantera_python.kinetics import GasKinetics
        gas = IdealGasPhase.from_mechanism(self.gri30)
        k = GasKinetics.from_mechanism(self.gri30, thermo=gas)
        return gas, k

    def test_chaperon_removed_from_stoichiometry(self):
        gas, k = self._kinetics()
        r = k.reactions[35]  # H + O2 + N2 <=> HO2 + N2
        self.assertNotIn('N2', r.reactants)
        self.assertNotIn('N2', r.products)
        self.assertEqual(r.reactants, {'H': 1.0, 'O2': 1.0})
        self.assertEqual(r.products, {'HO2': 1.0})
        self.assertIsNotNone(r.third_body)
        self.assertEqual(r.third_body.efficiencies, {'N2': 1.0})
        self.assertEqual(r.third_body.default_efficiency, 0.0)

    def test_chaperon_creation_rates_conserve_elements(self):
        gas, k = self._kinetics()
        gas.TPX = 1500.0, 101325.0, 'H2:2, O2:1, N2:4, OH:1e-6, H:1e-8'
        creation = k.creation_rates()
        destruction = k.destruction_rates()
        net = k.net_production_rates()
        i_n2 = gas.species_index('N2')
        # Pinned oracle gives wdot[N2] = -4.972e-9 at this state (chaperon
        # reactions 33/35/37 create/destroy N2 asymmetrically); match it.
        self.assertAlmostEqual(net[i_n2], -4.972046563571641e-09, places=18)
        # creation - destruction must equal net for every species
        for c, d, n in zip(creation, destruction, net):
            self.assertAlmostEqual(c - d, n, delta=abs(n) * 1e-9 + 1e-15)

    def test_explicit_elementary_type_not_extracted(self):
        """GRI-30 rxn 40 (2 H + H2 <=> 2 H2) omits 'type:', so the pinned
        oracle treats it as an explicit-chaperon reaction and so must the
        port (verified: oracle extracts H2 as third body)."""
        gas, k = self._kinetics()
        # YAML comments number reactions from 1, while the Python sequence is
        # zero-based: comment "Reaction 40" is reactions[39].
        r = k.reactions[39]
        self.assertNotIn('H2', r.reactants)
        self.assertEqual(r.reactants, {'H': 2.0})
        # product side had H2:2; pinned decrements coefficients >1 instead of
        # erasing them (Reaction.cpp:543-546), leaving H2:1
        self.assertEqual(r.products, {'H2': 1.0})
        self.assertIsNotNone(r.third_body)
        self.assertEqual(r.third_body.efficiencies, {'H2': 1.0})
        # Guard the indexing distinction: reactions[40] is YAML Reaction 41
        # and its explicit chaperon is H2O.
        self.assertEqual(k.reactions[40].third_body.efficiencies, {'H2O': 1.0})
