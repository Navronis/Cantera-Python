"""Cache-semantics regression tests: stale data must not survive state changes.

Covers the StateRevisions/RevisionCache contract in
cantera_python/state_revision.py plus live-binding behavior of the phase and
kinetics layers (MTR-003 live thermo binding).
"""
import unittest

from cantera_python.state_revision import (
    ASPECTS, RevisionCache, StateRevisions, RevisionedState,
)


class RevisionTests(unittest.TestCase):
    def test_revisions_independent_counters(self):
        r = StateRevisions()
        base = r.snapshot
        self.assertEqual(base, tuple([0] * len(ASPECTS)))
        t0 = r.bump_temperature()
        p0 = r.bump_pressure()
        c0 = r.bump_composition()
        self.assertEqual((t0, p0, c0), (1, 1, 1))
        t1 = r.bump_temperature()
        self.assertEqual(t1, 2)
        self.assertEqual(r['pressure'], 1)

    def test_unknown_aspect_rejected(self):
        r = StateRevisions()
        with self.assertRaises(KeyError):
            r.bump('bogus')

    def test_snapshot_changes_on_any_mutation(self):
        r = StateRevisions()
        s0 = r.snapshot
        for aspect in ASPECTS:
            r.bump(aspect)
            self.assertNotEqual(r.snapshot, s0)
            s0 = r.snapshot

    def test_cache_returns_stale_free_values(self):
        r = StateRevisions()
        c = RevisionCache(r)
        values = iter([10.0, 20.0])
        compute = lambda: next(values)
        self.assertEqual(c.get('cp', compute), 10.0)
        # no mutation: deterministic repeated read, no recompute
        self.assertEqual(c.get('cp', compute), 10.0)
        # mutation: must recompute, never return the stale 10.0
        r.bump_temperature()
        self.assertEqual(c.get('cp', compute), 20.0)

    def test_cache_keys_are_independent(self):
        r = StateRevisions()
        c = RevisionCache(r)
        self.assertEqual(c.get('a', lambda: 1.0), 1.0)
        self.assertEqual(c.get('b', lambda: 2.0), 2.0)
        r.bump_pressure()
        self.assertEqual(c.get('a', lambda: 3.0), 3.0)
        self.assertEqual(c.get('b', lambda: 4.0), 4.0)

    def test_cache_clear_drops_all(self):
        r = StateRevisions()
        c = RevisionCache(r)
        c.get('x', lambda: 1.0)
        c.clear()
        calls = []
        c.get('x', lambda: calls.append(1) or 2.0)
        self.assertEqual(calls, [1])

    def test_revisioned_state_mixin(self):
        class Phase(RevisionedState):
            def __init__(self):
                self._init_revisions()

        p = Phase()
        self.assertEqual(p.state_revisions.snapshot, tuple([0] * len(ASPECTS)))
        p._note_temperature_change()
        p._note_composition_change()
        self.assertEqual(p.state_revisions['temperature'], 1)
        self.assertEqual(p.state_revisions['composition'], 1)
        self.assertEqual(p.state_revisions['pressure'], 0)


class LiveBindingTests(unittest.TestCase):
    """Phase/kinetics must see current state; no stale results (MTR-003)."""

    @classmethod
    def setUpClass(cls):
        try:
            from cantera_python.kinetics import GasKinetics
        except ImportError:
            raise unittest.SkipTest("Kinetics is not part of Wave 1 release")
        from pathlib import Path
        from cantera_python import DATA_DIR; cls.gri30 = DATA_DIR / 'gri30.yaml'

    def test_temperature_change_updates_thermo_and_kinetics(self):
        from cantera_python.ideal_gas import IdealGasPhase
        from cantera_python.kinetics import GasKinetics
        gas = IdealGasPhase.from_mechanism(self.gri30)
        k = GasKinetics.from_mechanism(self.gri30, thermo=gas)
        gas.TPX = 300.0, 101325.0, 'H2:2, O2:1, N2:4'
        cp_low = gas.cp_mole
        kf_low = k.forward_rate_constants()[0]
        gas.TPX = 2000.0, 101325.0, 'H2:2, O2:1, N2:4'
        cp_high = gas.cp_mole
        kf_high = k.forward_rate_constants()[0]
        self.assertNotAlmostEqual(cp_low, cp_high)
        # reaction 0 (2 O+M <=> O2+M) has a negative temperature exponent,
        # so its kf decreases with T; assert the live value changed at all
        self.assertNotAlmostEqual(kf_low, kf_high)
        # a positive-activation-energy reaction must increase with T
        kf2_low = k.forward_rate_constants()[2]
        kf2_high = k.forward_rate_constants()[2]
        gas.TP = 300.0, 101325.0
        kf2_low = k.forward_rate_constants()[2]
        gas.TP = 2000.0, 101325.0
        kf2_high = k.forward_rate_constants()[2]
        self.assertGreater(kf2_high, kf2_low)

    def test_composition_change_updates_mw_concentrations_kinetics(self):
        from cantera_python.ideal_gas import IdealGasPhase
        from cantera_python.kinetics import GasKinetics
        gas = IdealGasPhase.from_mechanism(self.gri30)
        k = GasKinetics.from_mechanism(self.gri30, thermo=gas)
        gas.TPX = 1200.0, 101325.0, 'CH4:1, O2:1'
        mw1 = gas.mean_molecular_weight
        i_ch4 = gas.species_index('CH4')
        c1 = gas.concentrations[i_ch4]
        w1 = k.net_production_rates()
        gas.TPX = 1200.0, 101325.0, 'N2:1'
        mw2 = gas.mean_molecular_weight
        c2 = gas.concentrations[i_ch4]
        w2 = k.net_production_rates()
        self.assertNotAlmostEqual(mw1, mw2)
        self.assertNotAlmostEqual(c1, c2)
        self.assertNotEqual(tuple(w1), tuple(w2))

    def test_pressure_change_updates_density_and_standard_conc(self):
        from cantera_python.ideal_gas import IdealGasPhase
        gas = IdealGasPhase.from_mechanism(self.gri30)
        gas.TPX = 1200.0, 101325.0, 'N2:1'
        d1, sc1 = gas.density, gas.standard_concentration
        gas.TPX = 1200.0, 202650.0, 'N2:1'
        d2, sc2 = gas.density, gas.standard_concentration
        self.assertAlmostEqual(d2 / d1, 2.0, places=10)
        self.assertAlmostEqual(sc2 / sc1, 2.0, places=10)

    def test_state_roundtrip_reproduces_values(self):
        from cantera_python.ideal_gas import IdealGasPhase
        gas = IdealGasPhase.from_mechanism(self.gri30)
        gas.TPX = 1500.0, 101325.0, 'CO:1, O2:1'
        h1, s1, cp1 = gas.enthalpy_mass, gas.entropy_mass, gas.cp_mass
        gas.TPX = 500.0, 202650.0, 'N2:1'
        gas.TPX = 1500.0, 101325.0, 'CO:1, O2:1'
        h2, s2, cp2 = gas.enthalpy_mass, gas.entropy_mass, gas.cp_mass
        self.assertAlmostEqual(h1, h2, places=8)
        self.assertAlmostEqual(s1, s2, places=8)
        self.assertAlmostEqual(cp1, cp2, places=8)

    def test_independent_phases_do_not_corrupt(self):
        from cantera_python.ideal_gas import IdealGasPhase
        a = IdealGasPhase.from_mechanism(self.gri30)
        b = IdealGasPhase.from_mechanism(self.gri30)
        a.TPX = 1000.0, 101325.0, 'H2:1'
        b.TPX = 300.0, 101325.0, 'N2:1'
        # mutate a; b must be untouched
        a.TPX = 2500.0, 5066250.0, 'O2:1'
        self.assertAlmostEqual(b.temperature, 300.0, places=8)
        self.assertAlmostEqual(b.pressure, 101325.0, places=6)
        self.assertAlmostEqual(b.X[b.species_index('N2')], 1.0, places=12)

    def test_reaction_kc_uses_live_state_not_construction_state(self):
        from cantera_python.ideal_gas import IdealGasPhase
        from cantera_python.kinetics import GasKinetics
        gas = IdealGasPhase.from_mechanism(self.gri30)
        k = GasKinetics.from_mechanism(self.gri30, thermo=gas)
        gas.TPX = 300.0, 101325.0, 'H2:1, O2:1, H2O:1'
        kc_300 = k.equilibrium_constants()[0]
        gas.TPX = 2000.0, 101325.0, 'H2:1, O2:1, H2O:1'
        kc_2000 = k.equilibrium_constants()[0]
        self.assertNotAlmostEqual(kc_300, kc_2000)


if __name__ == '__main__':
    unittest.main()
