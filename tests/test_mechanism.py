import unittest

from cantera_python.mechanism import MechanismError, load_mechanism


class MechanismLoaderTests(unittest.TestCase):
    def test_basic_species_reaction_and_balance(self):
        mechanism = load_mechanism({
            'species': [
                {'name': 'H2', 'composition': {'H': 2}},
                {'name': 'O2', 'composition': {'O': 2}},
                {'name': 'H2O', 'composition': {'H': 2, 'O': 1}, 'thermo': {'model': 'NASA7'}},
            ],
            'reactions': [{'equation': '2 H2 + O2 => 2 H2O', 'rate-constant': {'A': 1}}],
        })
        self.assertEqual(mechanism.species_names, ('H2', 'O2', 'H2O'))
        self.assertAlmostEqual(mechanism.species[2].molecular_weight, 18.015)
        self.assertFalse(mechanism.reactions[0].equation.reversible)
        self.assertEqual(mechanism.reactions[0].rate_data['rate-constant']['A'], 1)

    def test_json_and_validation_errors(self):
        mechanism = load_mechanism('{"species": [{"name": "H", "composition": {"H": 1}}], "reactions": []}')
        self.assertEqual(mechanism.species_names, ('H',))
        with self.assertRaisesRegex(MechanismError, 'Unbalanced'):
            load_mechanism({'species': [{'name': 'H2', 'composition': {'H': 2}}, {'name': 'H', 'composition': {'H': 1}}],
                            'reactions': [{'equation': 'H2 => H'}]})
        with self.assertRaisesRegex(MechanismError, 'Unknown reaction species'):
            load_mechanism({'species': [{'name': 'H', 'composition': {'H': 1}}],
                            'reactions': [{'equation': 'H => X'}]})

    def test_load_yaml_string_and_files(self):
        from pathlib import Path
        yaml_text = """
species:
- name: H2
  composition: {H: 2}
- name: O2
  composition: {O: 2}
- name: H2O
  composition: {H: 2, O: 1}
reactions:
- equation: 2 H2 + O2 <=> 2 H2O
  rate-constant: {A: 1.0e8, b: 0.0, Ea: 0.0}
"""
        mech = load_mechanism(yaml_text)
        self.assertEqual(len(mech.species), 3)
        self.assertEqual(len(mech.reactions), 1)

        # Test loading actual GRI-30 YAML from cantera-original
        from cantera_python import DATA_DIR; gri30_path = DATA_DIR / 'gri30.yaml'
        if gri30_path.is_file():
            gri = load_mechanism(gri30_path)
            self.assertEqual(len(gri.species), 53)
            self.assertEqual(len(gri.reactions), 325)
            self.assertIn('CH4', gri.species_names)
            self.assertIn('NO', gri.species_names)

    def test_load_multiphase_cross_file_imports(self):
        from pathlib import Path
        from cantera_python import DATA_DIR; data_dir = DATA_DIR
        gri30_ion = data_dir / 'gri30_ion.yaml'
        if gri30_ion.is_file():
            mech = load_mechanism(gri30_ion, search_paths=[data_dir])
            self.assertEqual(len(mech.species), 56)
            self.assertEqual(len(mech.reactions), 6)
            self.assertIn('HCO+', mech.species_names)
            self.assertIn('CH', mech.species_names)

    def test_phase_selection_preserves_scope_and_transport(self):
        document = {
            'phases': [
                {'name': 'gas-a', 'thermo': 'ideal-gas', 'kinetics': 'gas',
                 'transport': 'mixture-averaged', 'species': ['A', 'B'],
                 'reactions': ['gas-a-reactions']},
                {'name': 'gas-b', 'thermo': 'ideal-gas', 'kinetics': 'none',
                 'species': ['B'], 'reactions': 'none'},
            ],
            'species': [
                {'name': 'A', 'composition': {'H': 1},
                 'transport': {'geometry': 'atom', 'diameter': 2.5, 'well-depth': 80}},
                {'name': 'B', 'composition': {'H': 1}},
                {'name': 'C', 'composition': {'O': 1}},
            ],
            'gas-a-reactions': [{'equation': 'A => B', 'rate-constant': {'A': 1}}],
        }
        mechanism = load_mechanism(document)
        self.assertEqual([p.name for p in mechanism.phases], ['gas-a', 'gas-b'])
        with self.assertRaisesRegex(MechanismError, 'phase_name is required'):
            mechanism.for_phase()
        selected = mechanism.for_phase('gas-a')
        self.assertEqual(selected.species_names, ('A', 'B'))
        self.assertEqual(len(selected.reactions), 1)
        self.assertEqual(selected.species[0].transport['diameter'], 2.5)
        inert = mechanism.for_phase('gas-b')
        self.assertEqual(inert.species_names, ('B',))
        self.assertEqual(inert.reactions, ())
        with self.assertRaisesRegex(MechanismError, 'Unknown phase'):
            mechanism.for_phase('missing')

    def test_phase_state_is_preserved(self):
        document = {
            'phases': [{'name': 'gas', 'thermo': 'ideal-gas', 'kinetics': 'none',
                        'transport': 'none', 'species': ['H2'],
                        'state': {'T': '450 K', 'P': '2 atm', 'X': {'H2': 1}}}],
            'species': [{'name': 'H2', 'composition': {'H': 2}}],
        }
        phase = load_mechanism(document).phase('gas')
        self.assertEqual(phase.state['T'], '450 K')
        self.assertEqual(phase.state['X'], {'H2': 1})

    def test_phase_third_body_policies_are_validated_and_serialized(self):
        document = {
            'phases': [{'name': 'gas', 'thermo': 'ideal-gas', 'kinetics': 'bulk',
                        'species': ['H2'], 'reactions': 'none',
                        'skip-undeclared-third-bodies': True,
                        'explicit-third-body-duplicates': 'error'}],
            'species': [{'name': 'H2', 'composition': {'H': 2}}],
        }
        phase = load_mechanism(document).phase('gas')
        self.assertTrue(phase.skip_undeclared_third_bodies)
        self.assertEqual(phase.explicit_third_body_duplicates, 'error')
        restored = load_mechanism(load_mechanism(document).to_yaml()).phase('gas')
        self.assertEqual(restored, phase)
        document['phases'][0]['explicit-third-body-duplicates'] = 'eggs'
        with self.assertRaisesRegex(MechanismError, 'Invalid explicit-third-body'):
            load_mechanism(document)

    def test_electron_element_infers_species_charge(self):
        mechanism = load_mechanism({'species': [
            {'name': 'ion+', 'composition': {'H': 1, 'E': -1}},
            {'name': 'E', 'composition': {'E': 1}},
        ]})
        self.assertEqual([species.charge for species in mechanism.species], [1.0, -1.0])

    def test_reaction_section_rules_match_declared_species_semantics(self):
        document = {
            'phases': [
                {'name': 'filtered', 'thermo': 'ideal-gas', 'kinetics': 'gas',
                 'species': ['A', 'B'],
                 'reactions': [{'reactions': 'declared-species'}]},
                {'name': 'strict', 'thermo': 'ideal-gas', 'kinetics': 'gas',
                 'species': ['A', 'B'], 'reactions': 'all'},
            ],
            'species': [
                {'name': 'A', 'composition': {'H': 1}},
                {'name': 'B', 'composition': {'H': 1}},
                {'name': 'C', 'composition': {'H': 1}},
            ],
            'reactions': [
                {'equation': 'A => B'},
                {'equation': 'A => C'},
            ],
        }
        mechanism = load_mechanism(document)
        self.assertEqual(len(mechanism.for_phase('filtered').reactions), 1)
        with self.assertRaisesRegex(MechanismError, 'undeclared species'):
            mechanism.for_phase('strict')

        imported_subset = load_mechanism({
            'phases': [{'name': 'subset', 'thermo': 'ideal-gas', 'kinetics': 'gas',
                        'species': ['A', 'B'],
                        'reactions': [{'reactions': 'declared-species'}]}],
            'species': [
                {'name': 'A', 'composition': {'H': 1}},
                {'name': 'B', 'composition': {'H': 1}},
            ],
            'reactions': [
                {'equation': 'A => B'},
                {'equation': 'A => C'},
            ],
        })
        self.assertEqual(len(imported_subset.for_phase('subset').reactions), 1)

    def test_duplicate_and_schema_errors_are_rejected(self):
        with self.assertRaisesRegex(MechanismError, 'Duplicate species'):
            load_mechanism({'species': [
                {'name': 'H', 'composition': {'H': 1}},
                {'name': 'H', 'composition': {'H': 1}},
            ]})
        with self.assertRaisesRegex(MechanismError, 'state must be a mapping'):
            load_mechanism({'phases': [
                {'name': 'gas', 'thermo': 'ideal-gas', 'state': 300}
            ]})

    def test_yaml_serialization_round_trip(self):
        document = {
            'units': {'length': 'cm'},
            'phases': [{'name': 'gas', 'thermo': 'ideal-gas', 'kinetics': 'gas',
                        'transport': 'none', 'species': ['H2', 'H'],
                        'reactions': 'all', 'state': {'T': 500.0, 'P': '1 atm'}}],
            'species': [
                {'name': 'H2', 'composition': {'H': 2}},
                {'name': 'H', 'composition': {'H': 1}},
            ],
            'reactions': [{'equation': 'H2 => 2 H', 'duplicate': True,
                           'rate-constant': {'A': 1.0}}],
        }
        original = load_mechanism(document)
        restored = load_mechanism(original.to_yaml())
        self.assertEqual(restored.species_names, original.species_names)
        self.assertEqual(restored.phase('gas'), original.phase('gas'))
        self.assertEqual(restored.reactions[0].equation,
                         original.reactions[0].equation)
        self.assertTrue(restored.reactions[0].duplicate)

    def test_yaml_serialization_preserves_nested_user_metadata(self):
        document = {
            'description': 'round-trip metadata',
            'custom-header': {'owner': 'lab', 'nested': {'flags': [1, 2, 3]}},
            'phases': [{
                'name': 'gas', 'thermo': 'ideal-gas', 'kinetics': 'gas',
                'transport': 'none', 'species': ['H2', 'H'],
                'reactions': 'all',
                'custom-phase-data': {'solver': {'rtol': 1e-9}},
            }],
            'species': [
                {
                    'name': 'H2', 'composition': {'H': 2},
                    'note': 'retain this note',
                    'equation-of-state': {
                        'model': 'constant-volume', 'density': '1 kg/m^3'},
                },
                {'name': 'H', 'composition': {'H': 1}},
            ],
            'reactions': [{
                'equation': 'H2 => 2 H',
                'rate-constant': {'A': 1.0},
                'custom-reaction-data': {'source': ['paper', 7]},
            }],
        }
        serialized = load_mechanism(document).to_yaml()
        restored = load_mechanism(serialized)
        emitted = restored.to_dict()

        self.assertEqual(emitted['custom-header'], document['custom-header'])
        self.assertEqual(emitted['phases'][0]['custom-phase-data'],
                         document['phases'][0]['custom-phase-data'])
        self.assertEqual(emitted['species'][0]['note'], 'retain this note')
        self.assertEqual(emitted['species'][0]['equation-of-state'],
                         document['species'][0]['equation-of-state'])
        self.assertEqual(emitted['reactions'][0]['custom-reaction-data'],
                         document['reactions'][0]['custom-reaction-data'])


if __name__ == '__main__':
    unittest.main()
