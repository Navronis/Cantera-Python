"""Tests for VCS Multiphase Chemical Equilibrium solver."""
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

import cantera_python as ct
from cantera_python.vcs import (MultiPhase, VCSError, vcs_basopt_components,
                                VCS_SOLVE, vcs_VolPhase, vcs_tmoles,
                                vcs_elab, vcs_elabcheck, vcs_elem_rearrange,
                                vcs_formation_stoichiometry, vcs_deltag,
                                vcs_phase_reaction_changes,
                                vcs_reaction_step_sizes,
                                vcs_hessian_act_coeff_diag,
                                vcs_hessian_diag_adj,
                                vcs_dfe_species_free_energies,
                                vcs_delta_species_step,
                                VCS_SPECIES_ZEROEDMS,
                                VCS_SPECIES_DELETED,
                                VCS_SPECIES_INTERFACIALVOLTAGE,
                                VCS_SPECIES_MINOR,
                                VCS_SPECIES_MAJOR, VCS_SPECIES_COMPONENT,
                                VCS_SPECIES_ZEROEDSS,
                                VCS_SPECIES_ACTIVEBUTZERO,
                                VCS_SPECIES_STOICHZERO,
                                VCS_SPECIES_ZEROEDPHASE,
                                vcs_equilibrate)

def test_multiphase_setup_and_properties():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1500.0, ct.OneAtm, "CH4:1.0, O2:2.0"

    mp = MultiPhase()
    mp.add_phase(gas, 1.0)

    assert mp.n_phases == 1
    assert mp.T == 1500.0
    assert mp.P == ct.OneAtm
    assert mp.total_enthalpy() != 0.0

def test_multiphase_tp_equilibrium():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1200.0, ct.OneAtm, "CH4:1.0, O2:2.0, N2:7.52"

    mp = MultiPhase()
    mp.add_phase(gas, 1.0)

    vcs_equilibrate(mp, "TP")
    # At equilibrium, combustion products should dominate (CO2, H2O, N2)
    assert gas.X[gas.species_index("CO2")] > 0.05
    assert gas.X[gas.species_index("H2O")] > 0.10
    assert gas.X[gas.species_index("CH4")] < 1e-4

def test_multiphase_hp_equilibrium():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 300.0, ct.OneAtm, "CH4:1.0, O2:2.0, N2:7.52"

    mp = MultiPhase()
    mp.add_phase(gas, 1.0)

    h_init = mp.total_enthalpy()
    vcs_equilibrate(mp, "HP")
    # Adiabatic flame temperature for stoichiometric methane/air should be ~2200K
    assert 2000.0 < mp.T < 2400.0
    assert abs(mp.total_enthalpy() - h_init) / abs(h_init) < 1e-3

def test_multiphase_vcs_equilibrate_multiphase_gas_graphite():
    """VCS multiphase equilibrium accurately oxidizes graphite into the gas phase."""
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1200.0, ct.OneAtm, "CH4:1.0, O2:2.0, N2:7.52"
    graphite = ct.Solution("graphite.yaml")
    graphite.TP = gas.TP
    assert graphite.element_names == ["C"]
    assert graphite.n_atoms("C(gr)", "C") == 1.0

    mp = MultiPhase()
    mp.add_phase(gas, 1.0)
    mp.add_phase(graphite, 0.2)

    vcs_equilibrate(mp, "TP")
    # 0.2 mol of solid graphite is fully oxidized into the gas phase, conserving all carbon
    assert np.isclose(mp.phase_moles[1], 0.0, atol=1e-10)
    assert np.isclose(mp.phase_moles[0], 1.2, rtol=1e-4)
    assert gas.X[gas.species_index("CO2")] > 0.03
    assert gas.X[gas.species_index("CO")] > 0.15

def test_vcs_solve_interface_does_not_claim_a_fake_iteration_count():
    mp = MultiPhase()
    mp.add_phase(ct.Solution("gri30.yaml"), 1.0)
    solver = ct.VCS_SOLVE()
    ret = solver.vcs_solve_TP(mp)
    assert ret == 0
    assert solver._iterations > 0
    assert solver._prepared_problem["component_basis"]["num_components"] > 0


def test_vcs_problem_preparation_builds_element_conserving_component_basis():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1200.0, ct.OneAtm, "CH4:1.0, O2:2.0, N2:7.52"
    graphite = ct.Solution("graphite.yaml")
    graphite.TP = gas.TP
    mixture = MultiPhase()
    mixture.add_phase(gas, 1.0)
    mixture.add_phase(graphite, 0.2)
    gas_x = np.asarray(gas.X).copy()
    graphite_x = np.asarray(graphite.X).copy()

    prepared = ct.VCS_SOLVE().prepare_problem(mixture)
    matrix = prepared["formula_matrix"]
    amounts = prepared["mole_numbers"]
    unknown = prepared["species_unknown_types"]
    np.testing.assert_allclose(
        prepared["element_abundance_goals"],
        vcs_elab(matrix, amounts, unknown), rtol=0.0, atol=0.0)
    basis = prepared["component_basis"]
    assert len(basis["component_species"]) == basis["num_components"]
    assert len(set(basis["component_species"])) == basis["num_components"]
    assert "C(gr)" in {
        prepared["species_names"][i] for i in basis["component_species"]
    }
    assert np.linalg.matrix_rank(
        matrix[list(basis["component_species"]), :]) == basis["num_components"]
    nc = basis["num_components"]
    formation = prepared["formation_reaction_stoichiometry"]
    assert formation.shape == (nc, len(gas.species_names) + 1 - nc)
    np.testing.assert_allclose(
        prepared["basis_ordered_formula_matrix"][:nc, :].T @ formation
        + prepared["basis_ordered_formula_matrix"][nc:, :].T,
        0.0, rtol=0.0, atol=2e-12)
    np.testing.assert_array_equal(np.asarray(gas.X), gas_x)
    np.testing.assert_array_equal(np.asarray(graphite.X), graphite_x)


def test_vcs_component_calculation_reoptimizes_basis_and_rebuilds_reactions():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1200.0, ct.OneAtm, "CH4:1.0, O2:2.0, N2:7.52"
    graphite = ct.Solution("graphite.yaml")
    graphite.TP = gas.TP
    mixture = MultiPhase()
    mixture.add_phase(gas, 1.0)
    mixture.add_phase(graphite, 0.2)

    solver = VCS_SOLVE()
    prepared = solver.prepare_problem(mixture)
    solver.initialize_thermodynamic_state()
    all_minor_zeroed = solver.solve_tp_component_calc()

    basis = prepared["component_basis"]
    ncomp = basis["num_components"]
    nsp = len(prepared["species_names"])
    formula = prepared["basis_ordered_formula_matrix"]
    stoichiometry = prepared["formation_reaction_stoichiometry"]
    assert 0 <= ncomp <= nsp
    assert len(basis["species_permutation"]) == nsp
    assert sorted(basis["species_permutation"]) == list(range(nsp))
    assert stoichiometry.shape == (ncomp, nsp - ncomp)
    np.testing.assert_allclose(
        formula[:ncomp].T @ stoichiometry + formula[ncomp:].T,
        0.0, rtol=0.0, atol=2e-12)

    state = solver._old_thermodynamic_state
    np.testing.assert_array_equal(state["species_status"], solver._species_status)
    abundances = vcs_elab(
        formula, state["mole_numbers_old"],
        prepared["basis_ordered_species_unknown_types"])
    np.testing.assert_allclose(
        abundances, prepared["reordered_element_abundance_goals"],
        rtol=0.0, atol=2e-12)
    assert isinstance(all_minor_zeroed, bool)
    solver.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_OLD)
    assert all(not phase._moles_current for phase in prepared["phase_objects"])
    solver.vcs_updateVP(vcs_VolPhase._STATE_OLD)
    assert all(phase._moles_current for phase in prepared["phase_objects"])


def test_vcs_basopt_reselects_basis_from_current_species_amounts():
    class PhaseMap:
        def __init__(self, slot):
            self.slot = slot

        def nSpecies(self):
            return 1

        def spGlobalIndexVCS(self, local):
            return self.slot

        def setSpGlobalIndexVCS(self, local, slot):
            self.slot = slot

    formula = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    phases = tuple(PhaseMap(k) for k in range(3))
    basis = {
        "num_components": 2, "species_permutation": (0, 1, 2),
        "component_species": (0, 1), "remaining_species": (2,),
        "working_mole_numbers": np.array([-1.0, 1.0, 1.0]),
        "orthogonal_basis_squared_norms": np.ones(2),
    }
    prepared = {
        "species_names": ("A", "B", "AB"),
        "element_names": ("A", "B"),
        "formula_matrix": formula.copy(),
        "element_abundance_goals": np.array([1.0, 2.0]),
        "element_active": np.ones(2, dtype=bool),
        "element_types": np.zeros(2, dtype=int),
        "basis_ordered_formula_matrix": formula.copy(),
        "basis_ordered_mole_numbers": np.array([0.0, 1.0, 1.0]),
        "basis_ordered_species_phase_ids": np.array([0, 1, 2]),
        "basis_ordered_species_unknown_types": np.zeros(3, dtype=int),
        "basis_ordered_species_single_phase": np.ones(3, dtype=bool),
        "basis_ordered_species_charges": np.zeros(3),
        "basis_ordered_standard_gibbs_rt": np.zeros(3),
        "basis_ordered_species_names": ("A", "B", "AB"),
        "reordered_element_abundance_goals": np.array([1.0, 2.0]),
        "reordered_element_active": np.ones(2, dtype=bool),
        "reordered_element_types": np.zeros(2, dtype=int),
        "formation_reaction_species": (2,),
        "formation_reaction_stoichiometry": np.array([[-1.0], [-1.0]]),
        "formation_reaction_size": np.array([2.0]),
        "delta_mol_num_phase": np.zeros((3, 1)),
        "phase_participation": np.zeros((3, 1), dtype=int),
        "num_reactions_reduced": 1,
        "phase_objects": phases,
        "component_basis": basis,
    }
    state = {
        "mole_numbers_old": np.array([0.0, 1.0, 1.0]),
        "mole_numbers_new": np.array([0.0, 1.0, 1.0]),
        "standard_gibbs_rt": np.zeros(3),
        "ln_mnaught_species": np.zeros(3),
        "activity_coefficients_old": np.ones(3),
        "activity_coefficients_new": np.ones(3),
        "species_status": np.ones(3, dtype=int),
        "species_free_energies_old": np.array([0.0, 0.0, 1.0]),
        "species_free_energies_new": np.array([0.0, 0.0, 1.0]),
        "delta_g_rxn_old": np.array([1.0]),
        "delta_g_rxn_new": np.array([1.0]),
    }
    solver = VCS_SOLVE()
    solver._prepared_problem = prepared
    solver._old_thermodynamic_state = state
    solver._species_status = state["species_status"].copy()
    solver._num_species_reduced = 3
    solver._num_reactions_reduced = 1
    solver._delta_mol_num_species = np.zeros(3)
    solver._delta_phase_moles = np.zeros(3)

    solver.vcs_basopt()

    assert basis["species_permutation"] == (2, 1, 0)
    assert prepared["basis_ordered_species_names"][0] == "AB"
    ncomp = basis["num_components"]
    ordered_formula = prepared["basis_ordered_formula_matrix"]
    stoichiometry = prepared["formation_reaction_stoichiometry"]
    np.testing.assert_allclose(
        ordered_formula[:ncomp].T @ stoichiometry
        + ordered_formula[ncomp:].T,
        0.0, rtol=0.0, atol=2e-12)
    stoichiometry_before = stoichiometry.copy()
    assert solver.vcs_basopt(do_just_components=True) == ncomp
    np.testing.assert_array_equal(
        prepared["formation_reaction_stoichiometry"], stoichiometry_before)

def test_chemequil_solver():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1400.0, ct.OneAtm, "CH4:1.0, O2:2.0"
    solver = ct.ChemEquil(gas)
    ret = solver.equilibrate(XY="TP")
    assert ret == 0
    assert solver.nIterations() > 0
    assert gas.X[gas.species_index("CO2")] > 0.05

def test_vcs_volphase_and_multiphase_equil():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1300.0, ct.OneAtm, "CH4:1.0, O2:2.0"
    vol_phase = ct.vcs_VolPhase(gas, phase_num=0)
    assert vol_phase.phaseNum() == 0
    # The pinned vcs_VolPhase constructor starts at zero moles for ordinary
    # phases; VCS_SOLVE later transfers the MultiPhase amount.
    assert vol_phase.exists() == 0
    assert vol_phase.nSpecies() == gas.n_species
    assert len(vol_phase.activityCoefficients()) == gas.n_species
    np.testing.assert_allclose(vol_phase.moleFractions(), gas.X, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        vol_phase._formula_matrix[:, :len(gas.element_names)],
        [[gas.n_atoms(name, element) for element in gas.element_names]
         for name in gas.species_names], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        [vol_phase.GStar_calc_one(k) for k in range(gas.n_species)],
        gas.standard_chemical_potentials, rtol=0.0, atol=0.0)

    mp = MultiPhase()
    mp.add_phase(gas, 1.0)
    mpe = ct.MultiPhaseEquil(mp)
    with pytest.raises(NotImplementedError, match="MultiPhaseEquil is not implemented"):
        mpe.equilibrate("TP")


def test_vcs_volphase_charge_and_electron_balance_rows_match_pinned_source():
    class ChargedPhase:
        name = "ions"
        species_names = ("A+", "A-")
        element_names = ("A", "E")
        species = (
            type("Species", (), {"charge": 1.0, "composition": {"A": 1.0}})(),
            type("Species", (), {"charge": -1.0, "composition": {"A": 1.0}})(),
        )
        X = np.array([0.75, 0.25])
        T = 300.0
        P = ct.OneAtm
        electric_potential = 0.0

        def __init__(self, require_neutrality):
            self.require_neutrality = require_neutrality

        def n_atoms(self, species, element):
            if isinstance(species, str):
                species = self.species_names.index(species)
            return self.species[species].composition.get(element, 0.0)

        def chargeNeutralityNecessary(self):
            return self.require_neutrality

        @property
        def TPX(self):
            return self.T, self.P, self.X

        @TPX.setter
        def TPX(self, state):
            self.T, self.P, composition = state
            self.X = np.asarray(composition, dtype=float)
            self.X = self.X / self.X.sum()

    neutral = ct.vcs_VolPhase(ChargedPhase(True))
    assert neutral._element_names == ("A", "E", "cn_ions")
    assert neutral.elementType(1) == 1  # VCS_ELEM_TYPE_ELECTRONCHARGE
    assert neutral.elementActive(1) is False
    assert neutral.elementType(2) == 2  # VCS_ELEM_TYPE_CHARGENEUTRALITY
    np.testing.assert_array_equal(
        neutral._formula_matrix,
        [[1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])

    unconstrained = ct.vcs_VolPhase(ChargedPhase(False))
    assert unconstrained._element_names == ("A", "E")
    assert unconstrained.elementType(1) == 1
    # The pinned source only disables the explicit electron balance when a
    # separate charge-neutrality constraint is active.
    assert unconstrained.elementActive(1) is True
    np.testing.assert_array_equal(
        unconstrained._formula_matrix, [[1.0, -1.0], [1.0, 1.0]])


def test_vcs_elab_excludes_voltage_unknowns():
    formula = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    amounts = np.array([2.0, 1.0, 100.0])
    unknown_types = np.array([0, 0, -5])
    np.testing.assert_array_equal(
        vcs_elab(formula, amounts, unknown_types), [5.0, 8.0])


def test_vcs_elabcheck_uses_component_bounds_and_zero_element_tolerances():
    formula = np.array([[1.0, 1.0], [0.0, -1.0]])
    amounts = np.array([1.0, 1.0])
    actual = vcs_elab(formula, amounts)
    goals = np.array([1.0, 0.0])
    active = np.array([True, True])
    kinds = np.array([0, 0])
    assert vcs_elabcheck(actual, goals, 1, active, kinds, formula, amounts)
    assert not vcs_elabcheck(
        [1.0, 3.0e-11], goals, 2, active, kinds, formula, amounts)
    assert vcs_elabcheck(
        [1.0, 1.0e-11], goals, 2, active, kinds, formula, amounts)
    assert not vcs_elabcheck(
        [1.0, 3.0e-11], goals, 2, active, kinds, formula, amounts)
    active[1] = False
    assert vcs_elabcheck(
        [1.0, 3.0e-11], goals, 2, active, kinds, formula, amounts)


def test_vcs_elabcheck_handles_electron_and_charge_neutrality_constraints():
    electron_formula = np.array([[-1.0], [1.0]])
    electron_moles = np.array([2.0, 2.0])
    electron_type = np.array([1])
    assert vcs_elabcheck(
        [2.0e-11], [0.0], 1, [True], electron_type,
        electron_formula, electron_moles)
    assert not vcs_elabcheck(
        [4.0e-11], [0.0], 1, [True], electron_type,
        electron_formula, electron_moles)
    with pytest.raises(VCSError, match="Problem with charge neutrality condition"):
        vcs_elabcheck([0.9], [1.0], 1, [True], [2], [[1.0]], [0.9])


def test_vcs_basopt_selects_independent_species_by_abundance_and_gram_schmidt():
    basis = vcs_basopt_components(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [1.0, 1.0, 0.0], [1.0, 1.0], [True, True], [0, 0],
        species_sizes=[1.0, 1.0, 2.0])
    assert basis["num_components"] == 2
    assert basis["component_species"] == (0, 1)
    assert basis["remaining_species"] == (2,)
    assert np.all(basis["orthogonal_basis_squared_norms"] > 1.0e-6)


def test_vcs_basopt_rejects_dependent_rows_and_uses_minor_growth_candidate():
    dependent = vcs_basopt_components(
        [[1.0, 0.0], [2.0, 0.0]], [1.0, 0.0], [1.0, 0.0],
        [True, True], [0, 0])
    assert dependent["num_components"] == 1
    assert dependent["component_species"] == (0,)

    growth = vcs_basopt_components(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [1.0, 0.0, 0.0], [1.0, 0.0], [True, True], [0, 0],
        species_sizes=[1.0, 1.0, 2.0])
    assert growth["component_species"] == (0, 2)
    assert growth["remaining_species"] == (1,)


def test_vcs_element_rearrangement_and_formation_stoichiometry_balance():
    # The first element is inactive and absent from the component, matching
    # the pinned source's documented rank-deficient leading-row case.
    formula = np.array([[0.0, 1.0], [0.0, 2.0]])
    reordered, order, goals, active, _ = vcs_elem_rearrange(
        formula, [0.0, 2.0], [False, True], [0, 0], 1)
    assert order == (1, 0)
    assert active.tolist() == [True, False]
    np.testing.assert_array_equal(goals, [2.0, 0.0])
    stoich = vcs_formation_stoichiometry(reordered, 1)
    np.testing.assert_allclose(stoich, [[-2.0]], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        reordered[0, :] * stoich[0, 0] + reordered[1, :],
        [0.0, 0.0], rtol=0.0, atol=0.0)


def test_vcs_deltag_major_minor_all_and_zero_component_rules():
    free = np.array([5.0, 2.0, -10.0, -1.0])
    stoich = np.array([[-1.0, 0.0], [0.0, -1.0]])
    status = np.array([1, 1, 1, 0])
    moles = np.array([0.0, 1.0, 0.0, 0.0])
    initial = np.array([99.0, 99.0])
    args = (initial, free, stoich, [2, 3], status, 2, moles)
    np.testing.assert_array_equal(vcs_deltag(*args, L=-1), [0.0, 99.0])
    np.testing.assert_array_equal(vcs_deltag(*args, L=0), [0.0, -3.0])
    np.testing.assert_array_equal(vcs_deltag(*args, L=1), [99.0, -3.0])

    # Minor filtering uses old component moles even when the selected state is
    # new, as the pinned source does in the L > 0 branch.
    new_moles = np.array([1.0, 1.0, 0.0, 0.0])
    old_moles = np.array([0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        vcs_deltag(initial, free, stoich, [2, 3], status, 2, new_moles,
                   L=1, old_species_moles=old_moles), [99.0, -3.0])


def test_vcs_dfe_species_free_energies_source_branches_and_filters():
    from cantera_python.vcs import VCS_DELETE_MINORSPECIES_CUTOFF

    standard = np.arange(1.0, 8.0)
    ln_mnaught = np.array([0.0, 0.25, 0.5, 0.0, 1.0, 0.0, 0.1])
    moles = np.array([0.75, 0.0, 0.0, 0.4, 0.0, 0.2, 0.0])
    phase_moles = np.array([0.75, 0.4, 0.0, 0.0, 0.0])
    phase_ids = np.array([0, 0, 0, 1, 2, 3, 4])
    activity = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 7.0])
    charges = np.array([0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0])
    potentials = np.array([0.0, 0.1, 0.0, 0.2, 0.0])
    status = np.array([2, VCS_SPECIES_MINOR, VCS_SPECIES_ZEROEDMS,
                       2, 2, 1, VCS_SPECIES_MINOR])
    single = np.array([False, False, False, True, False, False, False])
    unknown = np.array([0, 0, 0, 0, 0,
                        VCS_SPECIES_INTERFACIALVOLTAGE, 0])
    rxn_species = np.arange(1, 7)
    args = (standard, ln_mnaught, moles, phase_moles, phase_ids,
            activity, charges, potentials, status, single, unknown,
            rxn_species, 1)

    all_species = vcs_dfe_species_free_energies(
        *args, faraday_dim=2.0, L=0)
    expected = standard.copy()
    expected[1] = 2.0 + np.log(2.0 * VCS_DELETE_MINORSPECIES_CUTOFF / 0.75) - 0.25
    expected[2] = 3.0 - 0.5
    expected[3] = 4.0 + 1.0 * 2.0 * 0.1
    expected[4] = 5.0 - 1.0
    expected[5] = 6.0 - 1.0 * 2.0 * 0.2
    expected[6] = 7.0 - 0.1
    np.testing.assert_allclose(all_species, expected, rtol=0.0, atol=1e-14)

    major = vcs_dfe_species_free_energies(
        *args, faraday_dim=2.0, L=-1)
    assert major[0] == all_species[0]
    assert major[1] == standard[1]  # minor is untouched
    assert major[2] == all_species[2]
    assert major[3] == all_species[3]
    assert major[4] == all_species[4]
    assert major[5] == all_species[5]
    assert major[6] == standard[6]  # minor is untouched

    minor = vcs_dfe_species_free_energies(
        *args, faraday_dim=2.0, L=1)
    assert minor[1] == all_species[1]
    assert minor[6] == all_species[6]
    assert minor[2] == standard[2]

    inconsistent_totals = phase_moles.copy()
    inconsistent_totals[0] += 1.0
    with pytest.raises(VCSError, match="phase mole totals"):
        vcs_dfe_species_free_energies(
            standard, ln_mnaught, moles, inconsistent_totals, phase_ids,
            activity, charges, potentials, status, single, unknown,
            rxn_species, 1, faraday_dim=2.0)


def test_vcs_tmoles_excludes_voltage_unknown_and_preserves_native_sum_order():
    moles = np.array([1.0, -0.2, 0.75, 0.25, 123.0])
    phases = np.array([0, 0, 1, 1, 1])
    unknown = np.array([0, 0, 0, 0, -5])
    phase_totals, total = vcs_tmoles(moles, phases, unknown, 2)
    np.testing.assert_array_equal(phase_totals, [0.8, 1.0])
    assert total == 1.8


def test_vcs_phase_state_upload_tracks_basis_map_and_old_new_cache():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1200.0, ct.OneAtm, "CH4:1, O2:2, N2:7.52"
    graphite = ct.Solution("graphite.yaml")
    graphite.TP = gas.TP
    mixture = MultiPhase()
    mixture.add_phase(gas, 1.0)
    mixture.add_phase(graphite, 0.2)
    solver = VCS_SOLVE()
    prepared = solver.prepare_problem(mixture)
    names = prepared["basis_ordered_species_names"]
    nsp = len(names)

    # Native vcs_basopt permutes global species indices; phase-local mappings
    # must point back into the basis-ordered solver state.
    for phase in prepared["phase_objects"]:
        for local in range(phase.nSpecies()):
            global_index = phase.spGlobalIndexVCS(local)
            assert names[global_index] == phase.speciesName(local)

    moles = np.zeros(nsp)
    moles[names.index("CH4")] = 2.0
    moles[names.index("O2")] = 1.0
    moles[names.index("C(gr)")] = 3.0
    result = solver.update_phase_state(moles, state_calc=0)
    np.testing.assert_array_equal(result["phase_moles"], [3.0, 3.0])
    assert result["total_moles"] == 6.0
    gas_vp, graphite_vp = prepared["phase_objects"]
    assert gas_vp.totalMoles() == 3.0
    assert graphite_vp.totalMoles() == 3.0
    assert gas_vp.creationMoleNumbers(gas.species_index("CH4")) == pytest.approx(2.0 / 3.0)

    activity = np.full(nsp, np.nan)
    gas_vp.updateActCoeffCache(0, moles, activity)
    for local in range(gas_vp.nSpecies()):
        assert activity[gas_vp.spGlobalIndexVCS(local)] == pytest.approx(1.0)

    old_x = gas_vp.moleFractions()
    changed = moles.copy()
    changed[names.index("CH4")] = 0.0
    changed[names.index("O2")] = 3.0
    # Same-state current caches are intentionally not refreshed.
    gas_vp.updateFromVCS_MoleNumbers(changed, 0)
    np.testing.assert_array_equal(gas_vp.moleFractions(), old_x)
    gas_vp.setMolesOutOfDate()
    gas_vp.updateFromVCS_MoleNumbers(changed, 0)
    assert gas_vp.moleFraction(gas.species_index("CH4")) == 0.0
    assert gas_vp.moleFraction(gas.species_index("O2")) == 1.0
    old_creation = gas_vp._creation_mole_numbers.copy()

    new_state = changed.copy()
    gas_vp.setMolesOutOfDate(1)
    gas_vp.updateFromVCS_MoleNumbers(new_state, 1)
    np.testing.assert_array_equal(gas_vp._creation_mole_numbers, old_creation)
    assert gas_vp._state_status == 1

    empty = np.zeros(nsp)
    x_before_empty = gas_vp.moleFractions()
    gas_vp.setMolesOutOfDate(0)
    gas_vp.updateFromVCS_MoleNumbers(empty, 0)
    assert gas_vp.totalMoles() == 0.0
    assert gas_vp.exists() == vcs_VolPhase._EXIST_NO
    np.testing.assert_array_equal(gas_vp.moleFractions(), x_before_empty)


def test_vcs_phase_upload_handles_interfacial_voltage_unknown():
    gas = ct.Solution("gri30.yaml")
    phase = vcs_VolPhase(gas)
    voltage_local = gas.species_index("H+") if "H+" in gas.species_names else 0
    phase.setPhiVarIndex(voltage_local)
    nsp = gas.n_species
    for local in range(nsp):
        phase.setSpGlobalIndexVCS(local, local + 1)
    state = np.zeros(nsp + 1)
    state[voltage_local + 1] = 0.375
    state[2] = 2.0
    phase.setMolesFromVCS(0, state)
    assert phase.electricPotential() == pytest.approx(0.375)
    assert phase.moleFraction(voltage_local) == 0.0
    assert phase.totalMoles() == pytest.approx(2.0)


def test_vcs_phase_stability_successive_substitution_matches_pinned_reference():
    root = Path(__file__).resolve().parents[1]
    reference = root / "tests" / "fixtures" / "vcs_phase_stability_gri_pinned.txt"
    comparator = root / "tools" / "compare_vcs_phase_stability_probe.py"
    result = subprocess.run(
        [sys.executable, str(comparator), "--native", str(reference)],
        cwd=root, capture_output=True, text=True, check=True)
    assert "Pinned native VCS phase-stability differential: PASS" in result.stdout
    assert "converged True iterations 21" in result.stdout


def test_vcs_owned_old_thermodynamic_state_matches_pinned_reference():
    root = Path(__file__).resolve().parents[1]
    reference = root / "tests" / "fixtures" / "vcs_initial_state_gri_pinned.txt"
    comparator = root / "tools" / "compare_vcs_initial_state_probe.py"
    result = subprocess.run(
        [sys.executable, str(comparator), "--native", str(reference)],
        cwd=root, capture_output=True, text=True, check=True)
    assert "Pinned native OLD-state construction: PASS" in result.stdout


def test_vcs_evalss_tp_matches_native_state_update_contract():
    class Phase:
        def __init__(self, indices, standard_gibbs):
            self.indices = indices
            self.standard_gibbs = np.asarray(standard_gibbs, dtype=float)
            self.state = None

        def setState_TP(self, temperature, pressure):
            self.state = (temperature, pressure)

        def sendToVCS_GStar(self, output):
            output[self.indices] = self.standard_gibbs

    from cantera_python.constants import GasConstant
    from cantera_python.vcs import VCS_SUCCESS

    temperature = 1200.0
    pressure = 2.0e5
    phases = (
        Phase([0, 2], [GasConstant * temperature, 3 * GasConstant * temperature]),
        Phase([1], [2 * GasConstant * temperature]),
    )
    solver = VCS_SOLVE()
    solver._temperature = temperature
    solver._pressure = pressure
    solver._prepared_problem = {
        "species_names": ("A", "B", "C"),
        "phase_objects": phases,
        "component_basis": {"species_permutation": np.asarray([1, 2, 0])},
    }

    # The pinned C++ body uses solver members for T/P and does not read these
    # four call arguments; the Python port preserves that source behavior.
    result = solver.vcs_evalSS_TP(9, 8, 300.0, 1.0)

    assert result == VCS_SUCCESS
    assert [phase.state for phase in phases] == [
        (temperature, pressure), (temperature, pressure)]
    np.testing.assert_allclose(
        solver._prepared_problem["basis_ordered_standard_gibbs_rt"],
        [1.0, 2.0, 3.0], rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        solver._prepared_problem["species_standard_gibbs_rt"],
        [3.0, 1.0, 2.0], rtol=0.0, atol=2e-15)


def test_vcs_species_type_translates_pinned_classification_order():
    class Phase:
        def __init__(self, existence):
            self.existence = existence

        def exists(self):
            return self.existence

    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "species_names": ("C", "A", "B"),
        "component_basis": {"num_components": 1},
        "phase_objects": (Phase(vcs_VolPhase._EXIST_YES),),
        "basis_ordered_species_phase_ids": np.array([0, 0, 0]),
        "basis_ordered_species_unknown_types": np.array([0, 0, 0]),
        "basis_ordered_species_single_phase": np.array([False, False, False]),
        "basis_ordered_formula_matrix": np.zeros((3, 1)),
        "reordered_element_types": np.array([vcs_VolPhase._ELEM_TYPE_ABSPOS]),
        "reordered_element_abundance_goals": np.array([1.0]),
        "formation_reaction_stoichiometry": np.zeros((1, 2)),
        "formation_reaction_size": np.ones(2),
    }
    n = np.array([1.0, 1.0e-4, 1.0e-4])
    totals = np.array([1.0002])
    delta_g = np.array([1.0, 1.0])
    n_before, totals_before, dg_before = n.copy(), totals.copy(), delta_g.copy()

    # Components are major; a minor species remains minor below both the
    # phase-fraction threshold and the stoichiometric component threshold.
    assert solver.vcs_species_type(0, n, totals, delta_g) == VCS_SPECIES_MAJOR
    assert solver.vcs_species_type(2, n, totals, delta_g) == VCS_SPECIES_MINOR

    # Zero-mole species in an existing multispecies phase retain the native
    # active-zero classification when their formation reaction is uphill.
    n[1] = 0.0
    assert solver.vcs_species_type(1, n, totals, delta_g) == VCS_SPECIES_ACTIVEBUTZERO

    # A positive atom count with no available absolute-positive element is
    # eliminated by the stoichiometric-zero guard before the Gibbs test.
    p = solver._prepared_problem
    p["basis_ordered_formula_matrix"][1, 0] = 1.0
    p["reordered_element_abundance_goals"][0] = 0.0
    assert solver.vcs_species_type(1, n, totals, delta_g) == VCS_SPECIES_STOICHZERO

    # Single-species phases use their own status path; interfacial voltage
    # unknowns are classified before any mole-number logic.
    p["basis_ordered_formula_matrix"][1, 0] = 0.0
    p["reordered_element_abundance_goals"][0] = 1.0
    p["basis_ordered_species_single_phase"][1] = True
    assert solver.vcs_species_type(1, n, np.array([0.0]), delta_g) == VCS_SPECIES_ZEROEDSS
    p["basis_ordered_species_unknown_types"][2] = vcs_VolPhase._UNKNOWN_VOLTAGE
    assert solver.vcs_species_type(2, n, totals, delta_g) == VCS_SPECIES_INTERFACIALVOLTAGE

    np.testing.assert_array_equal(n[:1], n_before[:1])
    np.testing.assert_array_equal(totals, totals_before)
    np.testing.assert_array_equal(delta_g, dg_before)


def test_vcs_species_type_checks_state_vector_dimensions():
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "species_names": ("C",), "component_basis": {"num_components": 1},
        "phase_objects": (type("Phase", (), {"exists": lambda self: 2})(),),
        "basis_ordered_species_phase_ids": np.array([0]),
        "basis_ordered_species_unknown_types": np.array([0]),
        "basis_ordered_species_single_phase": np.array([False]),
        "basis_ordered_formula_matrix": np.zeros((1, 1)),
        "reordered_element_types": np.array([0]),
        "reordered_element_abundance_goals": np.array([1.0]),
        "formation_reaction_stoichiometry": np.zeros((1, 0)),
        "formation_reaction_size": np.zeros(0),
    }
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        solver.vcs_species_type(0, [], [1.0], [])


def test_vcs_evaluate_species_type_builds_status_vector_and_zeroed_count():
    class Phase:
        def exists(self):
            return vcs_VolPhase._EXIST_YES

    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "species_names": ("C", "A", "B"),
        "component_basis": {"num_components": 1},
        "phase_objects": (Phase(),),
        "basis_ordered_species_phase_ids": np.array([0, 0, 0]),
        "basis_ordered_species_unknown_types": np.array([0, 0, 0]),
        "basis_ordered_species_single_phase": np.array([False, False, False]),
        "basis_ordered_formula_matrix": np.zeros((3, 1)),
        "reordered_element_types": np.array([vcs_VolPhase._ELEM_TYPE_ABSPOS]),
        "reordered_element_abundance_goals": np.array([1.0]),
        "formation_reaction_stoichiometry": np.zeros((1, 2)),
        "formation_reaction_size": np.ones(2),
    }
    moles = np.array([1.0, 1e-4, 1e-4])
    phase_moles = np.array([1.0002])
    delta_g = np.array([1.0, 1.0])

    assert solver.vcs_evaluate_speciesType(moles, phase_moles, delta_g) is True
    np.testing.assert_array_equal(
        solver._species_status,
        [VCS_SPECIES_MAJOR, VCS_SPECIES_MINOR, VCS_SPECIES_MINOR])
    assert solver._num_rxn_minor_zeroed == 2

    # Promoting one reaction species to major reduces the non-major reaction
    # count below the total reduced reaction count, so the native boolean is false.
    moles[1] = 0.01
    assert solver.vcs_evaluate_speciesType(moles, phase_moles, delta_g) is False
    assert solver._num_rxn_minor_zeroed == 1
    solver._prepared_problem["num_reactions_reduced"] = 1
    assert solver.vcs_evaluate_speciesType(moles, phase_moles, delta_g) is True

    with pytest.raises(ValueError, match="all three state vectors"):
        solver.vcs_evaluate_speciesType(moles, phase_moles, None)


def test_vcs_switch_pos_swaps_species_and_reaction_state_consistently():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1200.0, ct.OneAtm, "CH4:1.0, O2:2.0, N2:7.52"
    graphite = ct.Solution("graphite.yaml")
    graphite.TP = gas.TP
    mixture = MultiPhase()
    mixture.add_phase(gas, 1.0)
    mixture.add_phase(graphite, 0.2)
    solver = VCS_SOLVE()
    prepared = solver.prepare_problem(mixture)
    solver.initialize_thermodynamic_state()
    ncomp = prepared["component_basis"]["num_components"]
    nsp = len(prepared["basis_ordered_species_names"])
    assert nsp - ncomp >= 2

    # Native ifunc=false performs only the species-indexed swaps.
    k1, k2 = 0, ncomp
    species_keys = (
        "basis_ordered_species_phase_ids",
        "basis_ordered_species_unknown_types",
        "basis_ordered_species_single_phase",
        "basis_ordered_species_charges",
        "basis_ordered_standard_gibbs_rt",
        "basis_ordered_formula_matrix",
        "basis_ordered_mole_numbers",
        "basis_ordered_species_names",
    )
    before = {key: np.asarray(prepared[key]).copy() for key in species_keys}
    state_keys = (
        "mole_numbers_old", "mole_numbers_new", "standard_gibbs_rt",
        "ln_mnaught_species", "activity_coefficients_old",
        "activity_coefficients_new", "species_status",
        "species_free_energies_old", "species_free_energies_new",
    )
    before_state = {key: solver._old_thermodynamic_state[key].copy()
                    for key in state_keys}
    old_permutation = tuple(prepared["component_basis"]["species_permutation"])
    old_phase_maps = [tuple(phase.spGlobalIndexVCS(k)
                            for k in range(phase.nSpecies()))
                      for phase in prepared["phase_objects"]]
    solver._species_status = np.arange(nsp, dtype=int)
    before_status = solver._species_status.copy()

    solver.vcs_switch_pos(False, k1, k2)

    for key in species_keys:
        expected = before[key].copy()
        expected[[k1, k2]] = expected[[k2, k1]]
        np.testing.assert_array_equal(prepared[key], expected)
    for key in state_keys:
        expected = before_state[key].copy()
        expected[[k1, k2]] = expected[[k2, k1]]
        np.testing.assert_array_equal(solver._old_thermodynamic_state[key], expected)
    expected_permutation = list(old_permutation)
    expected_permutation[k1], expected_permutation[k2] = (
        expected_permutation[k2], expected_permutation[k1])
    basis = prepared["component_basis"]
    assert basis["species_permutation"] == tuple(expected_permutation)
    assert basis["component_species"] == tuple(expected_permutation[:ncomp])
    assert basis["remaining_species"] == tuple(expected_permutation[ncomp:])
    expected_status = before_status.copy()
    expected_status[k1], expected_status[k2] = expected_status[k2], expected_status[k1]
    np.testing.assert_array_equal(solver._species_status, expected_status)
    for iph, phase in enumerate(prepared["phase_objects"]):
        expected_map = list(old_phase_maps[iph])
        for local, global_index in enumerate(expected_map):
            if global_index == k1:
                expected_map[local] = k2
            elif global_index == k2:
                expected_map[local] = k1
        assert tuple(phase.spGlobalIndexVCS(k) for k in range(phase.nSpecies())) == tuple(expected_map)

    # Native ifunc=true also swaps formation-reaction columns and Gibbs data.
    r1, r2 = ncomp, ncomp + 1
    before_stoich = prepared["formation_reaction_stoichiometry"].copy()
    before_phase_delta = prepared["delta_mol_num_phase"].copy()
    before_phase_participation = prepared["phase_participation"].copy()
    before_size = prepared["formation_reaction_size"].copy()
    before_reaction_species = tuple(prepared["formation_reaction_species"])
    before_delta_g = solver._old_thermodynamic_state["delta_g_rxn_old"].copy()
    before_delta_g_new = solver._old_thermodynamic_state["delta_g_rxn_new"].copy()
    solver._delta_mol_num_species = np.arange(nsp, dtype=float) + 10.0
    before_delta_moles = solver._delta_mol_num_species.copy()
    solver.vcs_switch_pos(True, r1, r2)
    expected_stoich = before_stoich.copy()
    expected_stoich[:, [0, 1]] = expected_stoich[:, [1, 0]]
    np.testing.assert_array_equal(prepared["formation_reaction_stoichiometry"], expected_stoich)
    expected_phase_delta = before_phase_delta.copy()
    expected_phase_delta[:, [0, 1]] = expected_phase_delta[:, [1, 0]]
    np.testing.assert_array_equal(prepared["delta_mol_num_phase"], expected_phase_delta)
    expected_participation = before_phase_participation.copy()
    expected_participation[:, [0, 1]] = expected_participation[:, [1, 0]]
    np.testing.assert_array_equal(prepared["phase_participation"], expected_participation)
    expected_size = before_size.copy()
    expected_size[[0, 1]] = expected_size[[1, 0]]
    np.testing.assert_array_equal(prepared["formation_reaction_size"], expected_size)
    expected_species = list(before_reaction_species)
    expected_species[0], expected_species[1] = expected_species[1], expected_species[0]
    assert prepared["formation_reaction_species"] == tuple(expected_species)
    expected_delta_g = before_delta_g.copy()
    expected_delta_g[[0, 1]] = expected_delta_g[[1, 0]]
    np.testing.assert_array_equal(solver._old_thermodynamic_state["delta_g_rxn_old"], expected_delta_g)
    expected_delta_g_new = before_delta_g_new.copy()
    expected_delta_g_new[[0, 1]] = expected_delta_g_new[[1, 0]]
    np.testing.assert_array_equal(
        solver._old_thermodynamic_state["delta_g_rxn_new"], expected_delta_g_new)
    expected_delta_moles = before_delta_moles.copy()
    expected_delta_moles[[r1, r2]] = expected_delta_moles[[r2, r1]]
    np.testing.assert_array_equal(solver._delta_mol_num_species,
                                  expected_delta_moles)


def test_vcs_phase_reaction_changes_matches_native_phase_accumulation_and_cutoff():
    stoich = np.array([[-1.0, 1.0e-7], [0.5, -0.5]])
    delta_phase, participation = vcs_phase_reaction_changes(
        stoich, np.array([0, 1, 1, 1]), num_components=2, num_phases=2)
    np.testing.assert_array_equal(delta_phase, [[-1.0, 0.0], [1.5, 0.5]])
    np.testing.assert_array_equal(participation, [[1, 0], [2, 2]])
    assert stoich[0, 1] == 0.0


def test_vcs_reaction_step_regular_newton_step_and_component_damping():
    common = dict(
        stoichiometric_coefficients=np.array([[-1.0]]),
        mole_numbers_old=np.array([10.0, 1.0]),
        species_status=np.array([VCS_SPECIES_COMPONENT, VCS_SPECIES_MAJOR]),
        species_unknown_types=np.zeros(2, dtype=int),
        species_single_phase=np.zeros(2, dtype=bool),
        species_phase_ids=np.array([0, 0]),
        phase_moles_old=np.array([11.0]),
        phase_single_species=np.array([False]),
        delta_mol_num_phase=np.array([[0.0]]),
        total_moles=11.0,
        phase_exists=np.array([vcs_VolPhase._EXIST_YES]),
    )
    result = vcs_reaction_step_sizes(**common, delta_g_rxn_new=np.array([0.55]))
    assert result["delta_mol_num_species"][1] == pytest.approx(-0.5)
    assert result["deleted_phase"] is None
    assert result["force_component_calc"] == 0

    damped = vcs_reaction_step_sizes(**common, delta_g_rxn_new=np.array([-100.0]))
    # Native component-depletion damping caps consumption at its available
    # component amount for this unit stoichiometric coefficient.
    assert damped["delta_mol_num_species"][1] == pytest.approx(10.0)

    jacobian = np.array([[0.0, 0.0], [0.0, 11.0]])
    adjusted = vcs_reaction_step_sizes(
        **common, delta_g_rxn_new=np.array([0.55]),
        use_activity_jacobian=True, ln_act_coeff_jacobian=jacobian)
    assert adjusted["delta_mol_num_species"][1] == pytest.approx(-0.55 / 2.1)


def test_vcs_activity_hessian_diagonal_and_native_stabilization_branches():
    stoich = np.array([[-1.0]])
    reaction_species = np.array([1])
    phase_ids = np.array([0, 0])
    phase_moles = np.array([10.0])
    single = np.array([False, False])
    jacobian = np.array([[2.0, 3.0], [5.0, 7.0]])
    # Native formula: 7/10 + 2/10 - (3+5)/10 = 0.1.
    assert vcs_hessian_act_coeff_diag(
        0, stoich, reaction_species, phase_ids, phase_moles,
        single, jacobian) == pytest.approx(0.1)
    assert vcs_hessian_diag_adj(
        0, 1.0, stoich, reaction_species, phase_ids, phase_moles,
        single, jacobian) == pytest.approx(1.1)

    for value, expected in ((0.5, 1.5), (-0.5, 0.5), (-1.0, 0.3334)):
        diagonal = np.zeros((2, 2))
        diagonal[1, 1] = value * phase_moles[0]
        assert vcs_hessian_diag_adj(
            0, 1.0, stoich, reaction_species, phase_ids, phase_moles,
            single, diagonal) == pytest.approx(expected)
    with pytest.raises(VCSError, match="must be positive"):
        vcs_hessian_diag_adj(
            0, 0.0, stoich, reaction_species, phase_ids, phase_moles,
            single, np.zeros((2, 2)))


def test_vcs_calc_activity_jacobian_scatter_skips_ideal_phase():
    class Phase:
        def __init__(self, mapping, ideal, local_jacobian=None):
            self.mapping = tuple(mapping)
            self.ideal = ideal
            self.local_jacobian = local_jacobian
            self.moles_seen = None

        def nSpecies(self):
            return len(self.mapping)

        def isIdealSoln(self):
            return self.ideal

        def setMolesFromVCS(self, state, moles):
            assert state == vcs_VolPhase._STATE_OLD
            self.moles_seen = np.asarray(moles).copy()

        def sendToVCS_LnActCoeffJac(self, output):
            for i, gi in enumerate(self.mapping):
                for j, gj in enumerate(self.mapping):
                    output[gi, gj] = self.local_jacobian[i, j]

    local = np.array([[1.0, 2.0], [3.0, 4.0]])
    nonideal = Phase((0, 2), False, local)
    ideal = Phase((1, 3), True)
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "species_names": ("A", "B", "C", "D"),
        "phase_objects": (nonideal, ideal),
    }
    moles = np.array([0.1, 0.2, 0.3, 0.4])
    jacobian = solver.vcs_CalcLnActCoeffJac(moles)
    expected = np.zeros((4, 4))
    expected[np.ix_([0, 2], [0, 2])] = local
    np.testing.assert_array_equal(jacobian, expected)
    np.testing.assert_array_equal(nonideal.moles_seen, moles)


def test_vcs_reaction_step_zeroed_species_and_phase_birth_branches():
    common = dict(
        stoichiometric_coefficients=np.array([[-1.0]]),
        species_unknown_types=np.zeros(2, dtype=int),
        species_single_phase=np.zeros(2, dtype=bool),
        species_phase_ids=np.array([0, 1]),
        phase_moles_old=np.array([10.0, 1.0e-14]),
        phase_single_species=np.array([False, False]),
        delta_mol_num_phase=np.zeros((2, 1)),
        total_moles=10.0,
        phase_exists=np.array([vcs_VolPhase._EXIST_YES,
                               vcs_VolPhase._EXIST_YES]),
    )
    moles = np.array([10.0, 0.0])
    statuses = np.array([VCS_SPECIES_COMPONENT, VCS_SPECIES_MAJOR])
    birth = vcs_reaction_step_sizes(
        **common, delta_g_rxn_new=np.array([-1.0e-3]),
        mole_numbers_old=moles, species_status=statuses)
    assert birth["delta_mol_num_species"][1] == pytest.approx(
        1.0e-23, rel=1.0e-14, abs=1.0e-36)

    # Once the phase is appreciable, the stoichiometric-zero status keeps the
    # smaller of the two native phase-birth nudges.
    appreciable = dict(common)
    appreciable["phase_moles_old"] = np.array([9.0, 1.0])
    appreciable["phase_exists"] = np.array([vcs_VolPhase._EXIST_YES,
                                              vcs_VolPhase._EXIST_YES])
    stoich_zero = vcs_reaction_step_sizes(
        **appreciable, delta_g_rxn_new=np.array([-1.0e-3]),
        mole_numbers_old=moles,
        species_status=np.array([VCS_SPECIES_COMPONENT, VCS_SPECIES_STOICHZERO]))
    assert stoich_zero["delta_mol_num_species"][1] == pytest.approx(
        1.0e-24, rel=1.0e-14, abs=1.0e-36)

    statuses[1] = VCS_SPECIES_ZEROEDPHASE
    zeroed = vcs_reaction_step_sizes(
        **common, delta_g_rxn_new=np.array([-1.0]),
        mole_numbers_old=moles, species_status=statuses,
        delta_mol_num_species=np.array([0.0, 7.0]))
    assert zeroed["delta_mol_num_species"][1] == 0.0


def test_vcs_reaction_step_single_species_phase_deletion():
    result = vcs_reaction_step_sizes(
        stoichiometric_coefficients=np.array([[-1.0]]),
        delta_g_rxn_new=np.array([0.5]),
        mole_numbers_old=np.array([1.0, 1.0]),
        species_status=np.array([VCS_SPECIES_COMPONENT, VCS_SPECIES_MAJOR]),
        species_unknown_types=np.zeros(2, dtype=int),
        species_single_phase=np.ones(2, dtype=bool),
        species_phase_ids=np.array([0, 1]),
        phase_moles_old=np.array([1.0, 1.0]),
        phase_single_species=np.ones(2, dtype=bool),
        delta_mol_num_phase=np.array([[-1.0], [1.0]]),
        total_moles=2.0,
        phase_exists=np.array([vcs_VolPhase._EXIST_YES,
                               vcs_VolPhase._EXIST_YES]),
    )
    np.testing.assert_array_equal(result["delta_mol_num_species"], [1.0, -1.0])
    assert result["deleted_phase"] == 1
    assert result["special_species"] == 1
    assert result["force_component_calc"] == 0

    reactant_limited = vcs_reaction_step_sizes(
        stoichiometric_coefficients=np.array([[-1.0]]),
        delta_g_rxn_new=np.array([-0.5]),
        mole_numbers_old=np.array([1.0, 1.0]),
        species_status=np.array([VCS_SPECIES_COMPONENT, VCS_SPECIES_MAJOR]),
        species_unknown_types=np.zeros(2, dtype=int),
        species_single_phase=np.ones(2, dtype=bool),
        species_phase_ids=np.array([0, 1]),
        phase_moles_old=np.array([1.0, 1.0]),
        phase_single_species=np.ones(2, dtype=bool),
        delta_mol_num_phase=np.array([[-1.0], [1.0]]),
        total_moles=2.0,
        phase_exists=np.array([vcs_VolPhase._EXIST_YES,
                               vcs_VolPhase._EXIST_YES]),
    )
    np.testing.assert_array_equal(reactant_limited["delta_mol_num_species"], [-1.0, 1.0])
    assert reactant_limited["deleted_phase"] == 0
    assert reactant_limited["special_species"] == 0
    assert reactant_limited["force_component_calc"] == 1


def test_vcs_switch_pos_rejects_out_of_range_without_mutating_state():
    solver = VCS_SOLVE()
    with pytest.raises(VCSError, match="must be prepared"):
        solver.vcs_switch_pos(False, 0, 1)
    gas = ct.Solution("gri30.yaml")
    mixture = MultiPhase()
    mixture.add_phase(gas, 1.0)
    prepared = solver.prepare_problem(mixture)
    before = np.asarray(prepared["basis_ordered_mole_numbers"]).copy()
    with pytest.raises(IndexError, match="out of range"):
        solver.vcs_switch_pos(False, 0, len(before))
    np.testing.assert_array_equal(prepared["basis_ordered_mole_numbers"], before)


def test_vcs_delta_species_applies_and_limits_formation_extent():
    stoich = np.array([[-1.0], [0.5]])
    unknown = np.zeros(3, dtype=int)
    phase_ids = np.array([0, 0, 1])
    old_moles = np.array([1.0, 0.2, 0.1])
    old_phase_moles = np.array([1.2, 0.1])

    applied = vcs_delta_species_step(
        stoich, old_moles, old_phase_moles, unknown, phase_ids, 2, 0.4)
    assert applied["success"] == 1
    assert applied["delta"] == 0.4
    np.testing.assert_allclose(applied["mole_numbers"], [0.6, 0.4, 0.5])
    np.testing.assert_allclose(applied["phase_moles"], [1.0, 0.5])
    assert applied["total_moles"] == pytest.approx(1.5)

    limited = vcs_delta_species_step(
        stoich, np.array([0.25, 0.3, 0.0]), np.array([0.55, 0.0]),
        unknown, phase_ids, 2, 0.6)
    assert limited["success"] == 0
    assert limited["delta"] == pytest.approx(0.25)
    np.testing.assert_allclose(limited["mole_numbers"], [0.0, 0.425, 0.25])
    np.testing.assert_allclose(limited["phase_moles"], [0.425, 0.25])

    blocked = vcs_delta_species_step(
        stoich, np.array([0.0, 0.3, 0.0]), np.array([0.3, 0.0]),
        unknown, phase_ids, 2, 0.6)
    assert blocked["success"] == 0
    assert blocked["delta"] == 0.0
    np.testing.assert_array_equal(blocked["mole_numbers"], [0.0, 0.3, 0.0])
    np.testing.assert_array_equal(blocked["phase_moles"], [0.3, 0.0])


def test_vcs_delta_species_voltage_unknown_does_not_change_moles():
    stoich = np.array([[-1.0], [0.5]])
    unknown = np.array([0, 0, VCS_SPECIES_INTERFACIALVOLTAGE])
    result = vcs_delta_species_step(
        stoich, np.array([1.0, 0.2, 0.0]), np.array([1.2, 0.0]),
        unknown, np.array([0, 0, 1]), 2, 0.3)
    assert result["success"] == 1
    assert result["delta"] == 0.3
    np.testing.assert_array_equal(result["mole_numbers"], [1.0, 0.2, 0.0])
    np.testing.assert_array_equal(result["phase_moles"], [1.2, 0.0])


def test_vcs_delta_species_updates_owned_old_state_and_invalidates_phases():
    class PhaseCache:
        def __init__(self):
            self.invalidated = []

        def setMolesOutOfDate(self, state):
            self.invalidated.append(state)

    phase_caches = (PhaseCache(), PhaseCache(), PhaseCache())
    solver = VCS_SOLVE()
    stoich = np.array([[-1.0], [0.5]])
    moles = np.array([1.0, 0.2, 0.1])
    phase_moles = np.array([1.2, 0.1, 0.0])
    solver._prepared_problem = {
        "formation_reaction_stoichiometry": stoich,
        "basis_ordered_species_unknown_types": np.zeros(3, dtype=int),
        "basis_ordered_species_phase_ids": np.array([0, 0, 1]),
        "basis_ordered_mole_numbers": moles.copy(),
        "phase_objects": phase_caches,
    }
    solver._old_thermodynamic_state = {
        "mole_numbers_old": moles.copy(),
        "phase_moles_old": phase_moles.copy(),
        "total_moles_old": 1.3,
    }
    result = solver.delta_species(2, 0.4)
    assert result["success"] == 1
    np.testing.assert_allclose(
        solver._old_thermodynamic_state["mole_numbers_old"], [0.6, 0.4, 0.5])
    np.testing.assert_allclose(
        solver._prepared_problem["basis_ordered_mole_numbers"], [0.6, 0.4, 0.5])
    assert result["invalidated_phase_ids"] == (1, 0, 0)
    assert [phase.invalidated for phase in phase_caches] == [
        [vcs_VolPhase._STATE_OLD, vcs_VolPhase._STATE_OLD],
        [vcs_VolPhase._STATE_OLD],
        [],
    ]


def test_vcs_delta_species_does_not_invalidate_untouched_phase():
    class PhaseCache:
        def __init__(self):
            self.invalidated = []

        def setMolesOutOfDate(self, state):
            self.invalidated.append(state)

    phase_caches = tuple(PhaseCache() for _ in range(4))
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "formation_reaction_stoichiometry": np.array([[-1.0], [0.5]]),
        "basis_ordered_species_unknown_types": np.zeros(3, dtype=int),
        "basis_ordered_species_phase_ids": np.array([0, 1, 2]),
        "basis_ordered_mole_numbers": np.array([1.0, 0.2, 0.1]),
        "phase_objects": phase_caches,
    }
    solver._old_thermodynamic_state = {
        "mole_numbers_old": np.array([1.0, 0.2, 0.1]),
        "phase_moles_old": np.array([1.0, 0.2, 0.1, 0.0]),
        "total_moles_old": 1.3,
    }

    result = solver.delta_species(2, 0.4)

    assert result["invalidated_phase_ids"] == (2, 0, 1)
    assert phase_caches[0].invalidated == [vcs_VolPhase._STATE_OLD]
    assert phase_caches[1].invalidated == [vcs_VolPhase._STATE_OLD]
    assert phase_caches[2].invalidated == [vcs_VolPhase._STATE_OLD]
    assert phase_caches[3].invalidated == []


def test_vcs_zero_species_uses_native_delta_and_noop_branches():
    class PhaseCache:
        def __init__(self):
            self.invalidated = []

        def setMolesOutOfDate(self, state):
            self.invalidated.append(state)

    def make_solver(unknown, moles):
        caches = (PhaseCache(), PhaseCache())
        solver = VCS_SOLVE()
        solver._prepared_problem = {
            "formation_reaction_stoichiometry": np.array([[-1.0]]),
            "basis_ordered_species_unknown_types": np.asarray(unknown, dtype=int),
            "basis_ordered_species_phase_ids": np.array([0, 1]),
            "basis_ordered_mole_numbers": np.asarray(moles, dtype=float).copy(),
            "phase_objects": caches,
        }
        solver._old_thermodynamic_state = {
            "mole_numbers_old": np.asarray(moles, dtype=float).copy(),
            "phase_moles_old": np.asarray(moles, dtype=float).copy(),
            "total_moles_old": float(sum(moles)),
        }
        return solver, caches

    solver, caches = make_solver([0, 0], [1.0, 0.25])
    assert solver.vcs_zero_species(1) == 1
    np.testing.assert_array_equal(
        solver._old_thermodynamic_state["mole_numbers_old"], [1.25, 0.0])
    assert caches[1].invalidated == [vcs_VolPhase._STATE_OLD]
    assert caches[0].invalidated == [vcs_VolPhase._STATE_OLD]

    already_zero, zero_caches = make_solver([0, 0], [1.0, 0.0])
    assert already_zero.vcs_zero_species(1) == 1
    np.testing.assert_array_equal(
        already_zero._old_thermodynamic_state["mole_numbers_old"], [1.0, 0.0])
    assert zero_caches[0].invalidated == []
    assert zero_caches[1].invalidated == []

    voltage, voltage_caches = make_solver(
        [0, VCS_SPECIES_INTERFACIALVOLTAGE], [1.0, 0.75])
    assert voltage.vcs_zero_species(1) == 1
    np.testing.assert_array_equal(
        voltage._old_thermodynamic_state["mole_numbers_old"], [1.0, 0.75])
    assert voltage_caches[0].invalidated == []
    assert voltage_caches[1].invalidated == []


def test_vcs_delete_species_compacts_active_species_and_reaction_state():
    class PhaseCache:
        def __init__(self, global_species, *, always=False):
            self.global_species = global_species
            self.always = always
            self.invalidated = []
            self.synced = None

        def nSpecies(self):
            return 1

        def spGlobalIndexVCS(self, local):
            assert local == 0
            return self.global_species

        def setSpGlobalIndexVCS(self, local, global_species):
            assert local == 0
            self.global_species = global_species

        def exists(self):
            return (vcs_VolPhase._EXIST_ALWAYS if self.always
                    else vcs_VolPhase._EXIST_YES)

        def setMolesOutOfDate(self, state):
            self.invalidated.append(state)

        def setMolesFromVCSCheck(self, state, moles, phase_moles):
            self.synced = (state, np.asarray(moles).copy(),
                           np.asarray(phase_moles).copy())

    phases = tuple(PhaseCache(i, always=(i == 1)) for i in range(3))
    solver = VCS_SOLVE()
    prepared = {
        "basis_ordered_species_names": ("A", "B", "C"),
        "basis_ordered_species_phase_ids": np.array([0, 1, 2]),
        "basis_ordered_species_unknown_types": np.zeros(3, dtype=int),
        "basis_ordered_species_single_phase": np.array([False, True, True]),
        "basis_ordered_species_charges": np.zeros(3),
        "basis_ordered_standard_gibbs_rt": np.array([1.0, 2.0, 3.0]),
        "basis_ordered_formula_matrix": np.ones((3, 1)),
        "basis_ordered_mole_numbers": np.array([1.0, 0.25, 0.1]),
        "formation_reaction_stoichiometry": np.array([[-1.0, -1.0]]),
        "formation_reaction_size": np.array([1.0, 1.0]),
        "formation_reaction_species": (1, 2),
        "delta_mol_num_phase": np.zeros((3, 2)),
        "phase_participation": np.zeros((3, 2), dtype=int),
        "phase_objects": phases,
        "num_reactions_reduced": 2,
        "component_basis": {
            "num_components": 1,
            "species_permutation": (0, 1, 2),
            "component_species": (0,),
            "remaining_species": (1, 2),
            "working_mole_numbers": np.array([1.0, 0.25, 0.1]),
        },
    }
    state = {
        "mole_numbers_old": np.array([1.0, 0.25, 0.1]),
        "mole_numbers_new": np.array([1.0, 0.25, 0.1]),
        "phase_moles_old": np.array([1.0, 0.25, 0.1]),
        "total_moles_old": 1.35,
        "standard_gibbs_rt": np.array([1.0, 2.0, 3.0]),
        "ln_mnaught_species": np.zeros(3),
        "activity_coefficients_old": np.ones(3),
        "activity_coefficients_new": np.ones(3),
        "phase_potentials_old": np.zeros(3),
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_MINOR, VCS_SPECIES_MAJOR]),
        "species_free_energies_old": np.array([1.0, 2.0, 3.0]),
        "species_free_energies_new": np.array([1.0, 2.0, 3.0]),
        "delta_g_rxn_old": np.array([0.1, 0.2]),
        "delta_g_rxn_new": np.array([0.3, 0.4]),
    }
    solver._prepared_problem = prepared
    solver._old_thermodynamic_state = state
    solver._species_status = state["species_status"].copy()
    solver._num_species_reduced = 3
    solver._num_reactions_reduced = 2
    solver._num_rxn_minor_zeroed = 1
    solver._delta_mol_num_species = np.zeros(3)

    assert solver.vcs_delete_species(1) == 0
    assert solver._num_species_reduced == 2
    assert solver._num_reactions_reduced == 1
    assert prepared["num_species_reduced"] == 2
    assert prepared["num_reactions_reduced"] == 1
    assert prepared["basis_ordered_species_names"] == ("A", "C", "B")
    np.testing.assert_allclose(state["mole_numbers_old"], [1.25, 0.1, 0.0])
    assert solver._species_status[2] == VCS_SPECIES_DELETED
    assert solver._num_rxn_minor_zeroed == 0
    assert phases[1].synced is not None


def test_vcs_delete_species_annihilates_phase_and_reactivates_zeroed_species():
    class PhaseCache:
        def __init__(self, global_species):
            self.global_species = global_species
            self.invalidated = []
            self.total_moles = 0.25 if global_species == 1 else 1.0

        def exists(self):
            return vcs_VolPhase._EXIST_YES

        def setMolesOutOfDate(self, state):
            self.invalidated.append(state)

        def setMolesFromVCSCheck(self, state, moles, phase_moles):
            self.synced = (state, np.asarray(moles).copy(),
                           np.asarray(phase_moles).copy())

        def setTotalMoles(self, total):
            self.total_moles = float(total)

    phases = (PhaseCache(0), PhaseCache(1))
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "basis_ordered_species_names": ("A", "B"),
        "basis_ordered_species_phase_ids": np.array([0, 1]),
        "basis_ordered_species_unknown_types": np.zeros(2, dtype=int),
        "basis_ordered_species_single_phase": np.array([False, False]),
        "phase_objects": phases,
        "formation_reaction_stoichiometry": np.array([[-1.0]]),
        "component_basis": {"num_components": 1},
        "num_reactions_reduced": 1,
    }
    solver._old_thermodynamic_state = {
        "mole_numbers_old": np.array([1.0, 0.25]),
        "mole_numbers_new": np.array([1.0, 0.25]),
        "phase_moles_old": np.array([1.0, 0.25]),
        "phase_moles_new": np.array([1.0, 0.25]),
        "total_moles_old": 1.25,
        "delta_g_rxn_old": np.array([0.2]),
        "delta_g_rxn_new": np.array([0.3]),
        "species_free_energies_old": np.array([1.0, 2.0]),
        "species_free_energies_new": np.array([1.0, 2.0]),
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_MAJOR]),
    }
    solver._species_status = solver._old_thermodynamic_state[
        "species_status"].copy()
    solver._num_species_reduced = 2
    solver._num_reactions_reduced = 1
    solver._delta_mol_num_species = np.zeros(2)
    solver._delta_phase_moles = np.zeros(2)

    assert solver.vcs_delete_species(1) == 0

    state = solver._old_thermodynamic_state
    np.testing.assert_array_equal(state["mole_numbers_old"], [1.25, 0.0])
    np.testing.assert_array_equal(state["mole_numbers_new"], [1.0, 0.0])
    np.testing.assert_array_equal(state["phase_moles_old"], [1.25, 0.0])
    np.testing.assert_array_equal(state["phase_moles_new"], [1.0, 0.0])
    assert state["species_status"][1] == VCS_SPECIES_ZEROEDMS
    assert solver._num_species_reduced == 2
    assert solver._num_reactions_reduced == 1
    assert phases[1].total_moles == 0.0


def test_vcs_delete_multiphase_rejects_nonzero_component_before_mutation():
    class Phase:
        def __init__(self):
            self.total_moles = 4.0

        def setTotalMoles(self, value):
            self.total_moles = float(value)

    phase = Phase()
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "component_basis": {"num_components": 1},
        "basis_ordered_species_names": ("A", "B"),
        "basis_ordered_species_phase_ids": np.array([0, 0]),
        "phase_objects": (phase,),
    }
    solver._old_thermodynamic_state = {
        "mole_numbers_old": np.array([1.0, 0.0]),
    }
    before = solver._old_thermodynamic_state["mole_numbers_old"].copy()

    with pytest.raises(NotImplementedError, match="permanent-reaction rebalance"):
        solver.vcs_delete_multiphase(0)

    np.testing.assert_array_equal(
        solver._old_thermodynamic_state["mole_numbers_old"], before)
    assert phase.total_moles == 4.0


def test_vcs_reinsert_deleted_adds_species_with_native_relative_seed():
    class PhaseCache:
        def __init__(self):
            self.total_moles = 1.0
            self.existence = vcs_VolPhase._EXIST_YES

        def exists(self):
            return self.existence

        def setMolesOutOfDate(self, state):
            pass

        def setMolesFromVCSCheck(self, state, moles, phase_moles):
            self.total_moles = float(phase_moles[0])

        def setExistence(self, value):
            self.existence = int(value)

    phase = PhaseCache()
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "basis_ordered_species_names": ("A", "B"),
        "basis_ordered_species_phase_ids": np.array([0, 0]),
        "basis_ordered_species_single_phase": np.array([False, False]),
        "basis_ordered_species_unknown_types": np.array([0, 0]),
        "phase_objects": (phase,),
        "formation_reaction_stoichiometry": np.array([[-1.0]]),
        "component_basis": {"num_components": 1},
        "num_species_reduced": 1,
        "num_reactions_reduced": 0,
    }
    state = {
        "mole_numbers_old": np.array([1.0, 0.0]),
        "mole_numbers_new": np.array([1.0, 0.0]),
        "phase_moles_old": np.array([1.0]),
        "phase_moles_new": np.array([1.0]),
        "total_moles_old": 1.0,
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_DELETED]),
        "activity_coefficients_old": np.ones(2),
        "activity_coefficients_new": np.ones(2),
        "phase_potentials_old": np.zeros(1),
        "species_free_energies_old": np.zeros(2),
        "species_free_energies_new": np.zeros(2),
        "delta_g_rxn_old": np.zeros(1),
        "delta_g_rxn_new": np.zeros(1),
    }
    solver._old_thermodynamic_state = state
    solver._species_status = state["species_status"].copy()
    solver._num_species_reduced = 1
    solver._num_reactions_reduced = 0
    solver._num_rxn_minor_zeroed = 0
    solver._delta_mol_num_species = np.zeros(2)

    solver.vcs_reinsert_deleted(1)

    expected_seed = 1.0e-63
    assert state["mole_numbers_old"][1] == pytest.approx(expected_seed)
    assert state["species_status"][1] == VCS_SPECIES_MINOR
    assert solver._num_species_reduced == 2
    assert solver._num_reactions_reduced == 1
    assert solver._num_rxn_minor_zeroed == 1
    assert solver._prepared_problem["num_species_reduced"] == 2
    assert solver._prepared_problem["num_reactions_reduced"] == 1


def test_vcs_recheck_deleted_reinserts_favorable_species_in_live_phase():
    class PhaseCache:
        def __init__(self):
            self.total_moles = 1.0
            self.existence = vcs_VolPhase._EXIST_YES

        def exists(self):
            return self.existence

        def setMolesOutOfDate(self, state):
            pass

        def setMolesFromVCSCheck(self, state, moles, phase_moles):
            self.total_moles = float(phase_moles[0])

        def setExistence(self, value):
            self.existence = int(value)

    phase = PhaseCache()
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "basis_ordered_species_names": ("A", "B"),
        "basis_ordered_species_phase_ids": np.array([0, 0]),
        "basis_ordered_species_single_phase": np.array([False, False]),
        "basis_ordered_species_unknown_types": np.array([0, 0]),
        "basis_ordered_species_charges": np.zeros(2),
        "phase_objects": (phase,),
        "formation_reaction_stoichiometry": np.array([[-1.0]]),
        "formation_reaction_species": (1,),
        "component_basis": {"num_components": 1},
        "temperature": 1000.0,
        "num_species_reduced": 1,
        "num_reactions_reduced": 0,
    }
    state = {
        "mole_numbers_old": np.array([1.0, 0.0]),
        "mole_numbers_new": np.array([1.0, 0.0]),
        "phase_moles_old": np.array([1.0]),
        "phase_moles_new": np.array([1.0]),
        "total_moles_old": 1.0,
        "standard_gibbs_rt": np.array([0.0, -1.0]),
        "ln_mnaught_species": np.zeros(2),
        "activity_coefficients_old": np.ones(2),
        "activity_coefficients_new": np.ones(2),
        "phase_potentials_old": np.zeros(1),
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_DELETED]),
        "species_free_energies_old": np.zeros(2),
        "species_free_energies_new": np.zeros(2),
        "delta_g_rxn_old": np.zeros(1),
        "delta_g_rxn_new": np.zeros(1),
    }
    solver._old_thermodynamic_state = state
    solver._species_status = state["species_status"].copy()
    solver._num_species_reduced = 1
    solver._num_reactions_reduced = 0
    solver._num_rxn_minor_zeroed = 0
    solver._delta_mol_num_species = np.zeros(2)

    assert solver.vcs_recheck_deleted() == 1
    assert state["delta_g_rxn_new"][0] == pytest.approx(-1.0)
    assert state["mole_numbers_old"][1] == pytest.approx(1.0e-63)
    assert solver._num_species_reduced == 2
    assert solver._num_reactions_reduced == 1


def test_vcs_add_all_deleted_estimates_and_applies_minor_species_amount():
    class PhaseCache:
        def __init__(self):
            self.total_moles = 1.0
            self.existence = vcs_VolPhase._EXIST_YES

        def nSpecies(self):
            return 2

        def setMolesOutOfDate(self, state):
            pass

        def setTotalMoles(self, total):
            self.total_moles = float(total)
            self.existence = (vcs_VolPhase._EXIST_YES if total > 0.0
                              else vcs_VolPhase._EXIST_NO)

        def setMolesOutOfDate(self, state):
            pass

        def setMolesFromVCSCheck(self, state, moles, phase_moles):
            self.total_moles = float(phase_moles[0])
            self.existence = (vcs_VolPhase._EXIST_YES if self.total_moles > 0.0
                              else vcs_VolPhase._EXIST_NO)

        def sendToVCS_ActCoeff(self, state, activity, moles=None):
            activity[:] = 1.0

        def electricPotential(self):
            return 0.0

    phase = PhaseCache()
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "basis_ordered_species_names": ("A", "B"),
        "basis_ordered_species_phase_ids": np.array([0, 0]),
        "basis_ordered_species_single_phase": np.array([False, False]),
        "basis_ordered_species_unknown_types": np.array([0, 0]),
        "basis_ordered_species_unknown_types": np.array([0, 0]),
        "basis_ordered_species_charges": np.zeros(2),
        "basis_ordered_standard_gibbs_rt": np.array([0.0, 20.0]),
        "phase_objects": (phase,),
        "formation_reaction_stoichiometry": np.array([[-1.0]]),
        "formation_reaction_species": (1,),
        "component_basis": {"num_components": 1},
        "temperature": 1000.0,
        "num_species_reduced": 1,
        "num_reactions_reduced": 0,
    }
    state = {
        "mole_numbers_old": np.array([1.0, 0.0]),
        "mole_numbers_new": np.array([1.0, 0.0]),
        "phase_moles_old": np.array([1.0]),
        "phase_moles_new": np.array([1.0]),
        "total_moles_old": 1.0,
        "standard_gibbs_rt": np.array([0.0, 20.0]),
        "ln_mnaught_species": np.zeros(2),
        "activity_coefficients_old": np.ones(2),
        "activity_coefficients_new": np.ones(2),
        "phase_potentials_old": np.zeros(1),
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_DELETED]),
        "species_free_energies_old": np.zeros(2),
        "species_free_energies_new": np.zeros(2),
        "delta_g_rxn_old": np.zeros(1),
        "delta_g_rxn_new": np.zeros(1),
    }
    solver._old_thermodynamic_state = state
    solver._species_status = state["species_status"].copy()
    solver._num_species_reduced = 1
    solver._num_reactions_reduced = 0
    solver._num_rxn_minor_zeroed = 0
    solver._delta_mol_num_species = np.zeros(2)

    assert solver.vcs_add_all_deleted() == 0
    assert state["mole_numbers_old"][1] == pytest.approx(np.exp(-20.0))
    assert state["delta_g_rxn_old"][0] == pytest.approx(0.0, abs=1e-6)


def test_vcs_global_step_damping_uses_pinned_parabolic_slope_factor():
    class PhaseCache:
        def __init__(self):
            self.total_moles = 1.0

        def nSpecies(self):
            return 2

        def setTotalMoles(self, total):
            self.total_moles = float(total)

        def setMolesOutOfDate(self, state):
            pass

        def setMolesFromVCSCheck(self, state, moles, phase_moles):
            calculated = float(np.sum(moles))
            assert calculated == pytest.approx(float(phase_moles[0]))
            self.total_moles = calculated

        def sendToVCS_ActCoeff(self, state, activity, moles=None):
            activity[:] = 1.0

        def electricPotential(self):
            return 0.0

    phase = PhaseCache()
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "basis_ordered_species_names": ("A", "B"),
        "basis_ordered_species_phase_ids": np.array([0, 0]),
        "basis_ordered_species_single_phase": np.array([False, False]),
        "basis_ordered_species_unknown_types": np.array([0, 0]),
        "basis_ordered_species_charges": np.zeros(2),
        "basis_ordered_standard_gibbs_rt": np.array([0.0, 1.0]),
        "phase_objects": (phase,),
        "formation_reaction_stoichiometry": np.array([[-1.0]]),
        "formation_reaction_species": (1,),
        "component_basis": {"num_components": 1},
        "temperature": 1000.0,
    }
    state = {
        "mole_numbers_old": np.array([0.9, 0.1]),
        "mole_numbers_new": np.array([0.9, 0.1]),
        "phase_moles_old": np.array([1.0]),
        "phase_moles_new": np.array([1.0]),
        "total_moles_old": 1.0,
        "standard_gibbs_rt": np.array([0.0, 1.0]),
        "ln_mnaught_species": np.zeros(2),
        "activity_coefficients_old": np.ones(2),
        "activity_coefficients_new": np.ones(2),
        "phase_potentials_old": np.zeros(1),
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_MAJOR]),
        "species_free_energies_old": np.zeros(2),
        "species_free_energies_new": np.zeros(2),
        "delta_g_rxn_old": np.array([-1.0]),
        "delta_g_rxn_new": np.array([0.5]),
    }
    solver._old_thermodynamic_state = state
    solver._species_status = state["species_status"].copy()
    solver._num_species_reduced = 2
    solver._num_reactions_reduced = 1
    solver._delta_mol_num_species = np.array([-0.1, 0.1])
    solver._delta_phase_moles = np.array([0.0])

    assert solver.vcs_glob_step_damp() is True
    np.testing.assert_allclose(state["mole_numbers_new"], [5.0 / 6.0, 1.0 / 6.0])
    assert state["delta_g_rxn_new"][0] == pytest.approx(
        1.0 + np.log(1.0 / 6.0) - np.log(5.0 / 6.0))

    original_trial = state["mole_numbers_new"].copy()
    state["delta_g_rxn_old"][0] = 1.0
    assert solver.vcs_glob_step_damp() is False
    np.testing.assert_array_equal(state["mole_numbers_new"], original_trial)


def test_vcs_minor_alt_calc_covers_power_law_cutoff_and_voltage_branches():
    from cantera_python.constants import Faraday, GasConstant

    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "basis_ordered_species_names": ("A", "B"),
        "basis_ordered_species_phase_ids": np.array([0, 0]),
        "basis_ordered_species_unknown_types": np.array([0, 0]),
        "component_basis": {"num_components": 1},
        "phase_objects": (object(),),
        "temperature": 1000.0,
    }
    state = {
        "mole_numbers_old": np.array([1.0, 0.1]),
        "phase_moles_old": np.array([1.0]),
        "delta_g_rxn_old": np.array([0.12]),
    }
    solver._old_thermodynamic_state = state
    jacobian = np.zeros((2, 2))
    jacobian[1, 1] = 2.0

    delta, do_delete = solver.vcs_minor_alt_calc(1, 0, jacobian)
    assert do_delete is False
    assert 0.1 + delta == pytest.approx(0.1 * np.exp(-0.1))

    state["mole_numbers_old"][1] = 1.0e-150
    state["delta_g_rxn_old"][0] = 30.0
    delta, do_delete = solver.vcs_minor_alt_calc(1, 0, jacobian)
    assert do_delete is True
    assert delta == pytest.approx(-1.0e-150)

    solver._prepared_problem["basis_ordered_species_unknown_types"][1] = (
        VCS_SPECIES_INTERFACIALVOLTAGE)
    state["delta_g_rxn_old"][0] = 0.25
    delta, do_delete = solver.vcs_minor_alt_calc(1, 0, jacobian)
    assert do_delete is False
    assert delta == pytest.approx(
        0.25 * GasConstant * 1000.0 / Faraday)


def test_vcs_apply_minor_trial_updates_components_and_skips_later_iterations():
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "basis_ordered_species_names": ("A", "B"),
        "basis_ordered_species_phase_ids": np.array([0, 0]),
        "basis_ordered_species_unknown_types": np.array([0, 0]),
        "basis_ordered_species_single_phase": np.array([False, False]),
        "basis_ordered_species_charges": np.zeros(2),
        "basis_ordered_standard_gibbs_rt": np.zeros(2),
        "phase_objects": (object(),),
        "formation_reaction_stoichiometry": np.array([[-1.0]]),
        "formation_reaction_species": (1,),
        "delta_mol_num_phase": np.zeros((1, 1)),
        "component_basis": {"num_components": 1},
        "temperature": 1000.0,
    }
    state = {
        "mole_numbers_old": np.array([0.9, 0.1]),
        "mole_numbers_new": np.array([0.9, 0.1]),
        "phase_moles_old": np.array([1.0]),
        "delta_g_rxn_old": np.array([0.1]),
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_MINOR]),
    }
    solver._old_thermodynamic_state = state
    solver._species_status = state["species_status"].copy()
    solver._num_species_reduced = 2
    solver._num_reactions_reduced = 1
    solver._delta_mol_num_species = np.zeros(2)
    solver._delta_phase_moles = np.zeros(1)

    result = solver.vcs_apply_minor_trial(1, 0, iteration=0,
                                          ln_act_coeff_jacobian=np.zeros((2, 2)))
    expected = 0.1 * (np.exp(-0.1) - 1.0)
    assert result["delta"] == pytest.approx(expected)
    assert state["mole_numbers_new"][1] == pytest.approx(0.1 + expected)
    assert solver._delta_mol_num_species[0] == pytest.approx(-expected)
    assert solver._delta_mol_num_species[1] == pytest.approx(expected)

    skipped = solver.vcs_apply_minor_trial(1, 0, iteration=1,
                                           ln_act_coeff_jacobian=np.zeros((2, 2)))
    assert skipped["skipped_minor"] is True
    assert skipped["delta"] == 0.0
    assert state["mole_numbers_new"][1] == pytest.approx(0.1)
    assert solver._delta_mol_num_species[1] == 0.0


def test_vcs_apply_major_trial_clamps_and_requests_single_phase_deletion():
    multi = VCS_SOLVE()
    multi._prepared_problem = {
        "basis_ordered_species_names": ("A", "B"),
        "basis_ordered_species_phase_ids": np.array([0, 0]),
        "basis_ordered_species_unknown_types": np.array([0, 0]),
        "basis_ordered_species_single_phase": np.array([False, False]),
        "formation_reaction_stoichiometry": np.array([[-1.0]]),
        "formation_reaction_species": (1,),
        "delta_mol_num_phase": np.zeros((1, 1)),
        "component_basis": {"num_components": 1},
        "phase_objects": (object(),),
    }
    state = {
        "mole_numbers_old": np.array([0.9, 0.1]),
        "mole_numbers_new": np.array([0.9, 0.1]),
        "phase_moles_old": np.array([1.0]),
        "total_moles_old": 1.0,
        "delta_g_rxn_new": np.array([1.0]),
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_MAJOR]),
    }
    multi._old_thermodynamic_state = state
    multi._species_status = state["species_status"].copy()
    multi._num_species_reduced = 2
    multi._num_reactions_reduced = 1
    multi._delta_mol_num_species = np.array([0.0, -0.2])
    multi._delta_phase_moles = np.zeros(1)

    state["delta_g_rxn_new"][0] = 1.0e-11
    superconverged = multi.vcs_apply_major_trial(1, 0)
    assert superconverged["superconverged"] is True
    assert state["mole_numbers_new"][1] == pytest.approx(0.1)
    assert multi._delta_mol_num_species[1] == 0.0

    state["delta_g_rxn_new"][0] = 1.0
    multi._delta_mol_num_species[1] = -0.2
    result = multi.vcs_apply_major_trial(1, 0)
    assert result["delta"] == pytest.approx(-0.09)
    assert result["phase_delete"] is None
    assert state["mole_numbers_new"][1] == pytest.approx(0.01)
    assert multi._delta_mol_num_species[0] == pytest.approx(0.09)

    single = VCS_SOLVE()
    single._prepared_problem = {
        "basis_ordered_species_names": ("A", "B"),
        "basis_ordered_species_phase_ids": np.array([0, 1]),
        "basis_ordered_species_unknown_types": np.array([0, 0]),
        "basis_ordered_species_single_phase": np.array([False, True]),
        "formation_reaction_stoichiometry": np.array([[-1.0]]),
        "formation_reaction_species": (1,),
        "delta_mol_num_phase": np.array([[-1.0], [1.0]]),
        "component_basis": {"num_components": 1},
        "phase_objects": (object(), object()),
    }
    single_state = {
        "mole_numbers_old": np.array([0.9, 0.1]),
        "mole_numbers_new": np.array([0.9, 0.1]),
        "phase_moles_old": np.array([0.9, 0.1]),
        "total_moles_old": 1.0,
        "delta_g_rxn_new": np.array([1.0]),
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_MAJOR]),
    }
    single._old_thermodynamic_state = single_state
    single._species_status = single_state["species_status"].copy()
    single._num_species_reduced = 2
    single._num_reactions_reduced = 1
    single._delta_mol_num_species = np.array([0.0, -0.2])
    single._delta_phase_moles = np.zeros(2)

    result = single.vcs_apply_major_trial(1, 0)
    assert result["delta"] == pytest.approx(-0.1)
    assert result["phase_delete"] == 1
    assert result["all_minor_zeroed"] is True
    assert single_state["mole_numbers_new"][1] == 0.0
    assert single._species_status[1] == VCS_SPECIES_ZEROEDSS
    np.testing.assert_allclose(single._delta_mol_num_species, [0.1, -0.1])
    np.testing.assert_allclose(single._delta_phase_moles, [0.1, -0.1])


def test_vcs_apply_zeroed_trial_reactivates_favorable_species_with_element_guard():
    class Phase:
        def __init__(self, existence):
            self.existence = existence

        def exists(self):
            return self.existence

    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "basis_ordered_species_names": ("A", "B"),
        "basis_ordered_species_phase_ids": np.array([0, 1]),
        "basis_ordered_species_unknown_types": np.array([0, 0]),
        "basis_ordered_formula_matrix": np.array([[1.0], [1.0]]),
        "reordered_element_types": np.array([vcs_VolPhase._ELEM_TYPE_ABSPOS]),
        "reordered_element_abundance_goals": np.array([1.0]),
        "formation_reaction_stoichiometry": np.array([[-1.0]]),
        "formation_reaction_species": (1,),
        "delta_mol_num_phase": np.array([[-1.0], [1.0]]),
        "component_basis": {"num_components": 1},
        "phase_objects": (Phase(vcs_VolPhase._EXIST_YES),
                          Phase(vcs_VolPhase._EXIST_NO)),
    }
    state = {
        "mole_numbers_old": np.array([0.5, 0.0]),
        "mole_numbers_new": np.array([0.5, 0.0]),
        "total_moles_old": 0.5,
        "delta_g_rxn_new": np.array([-1.0]),
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_ZEROEDMS]),
    }
    solver._old_thermodynamic_state = state
    solver._species_status = state["species_status"].copy()
    solver._num_species_reduced = 2
    solver._num_reactions_reduced = 1
    solver._delta_mol_num_species = np.array([0.0, 0.2])
    solver._delta_phase_moles = np.zeros(2)

    result = solver.vcs_apply_zeroed_trial(1, 0)
    assert result["resurrected"] is True
    assert result["species_status"] == VCS_SPECIES_MAJOR
    assert result["delta"] == pytest.approx(0.002)
    assert state["mole_numbers_new"][1] == pytest.approx(0.002)
    np.testing.assert_allclose(solver._delta_mol_num_species, [-0.002, 0.002])
    np.testing.assert_allclose(solver._delta_phase_moles, [-0.002, 0.002])

    state["mole_numbers_new"][1] = 0.0
    state["species_status"][1] = VCS_SPECIES_ZEROEDMS
    solver._species_status[1] = VCS_SPECIES_ZEROEDMS
    solver._delta_mol_num_species[:] = [0.0, 0.2]
    solver._delta_phase_moles[:] = 0.0
    solver._prepared_problem["reordered_element_abundance_goals"][0] = 1.0e-150
    result = solver.vcs_apply_zeroed_trial(1, 0)
    assert result["resurrected"] is False
    assert result["blocked_by_element_bound"] is True
    assert solver._delta_mol_num_species[1] == 0.0


def test_vcs_apply_zeroed_trial_matches_native_rejection_and_minor_phase_paths():
    class Phase:
        def exists(self):
            return vcs_VolPhase._EXIST_YES

    def make_solver(delta_g, proposed_delta):
        solver = VCS_SOLVE()
        solver._prepared_problem = {
            "basis_ordered_species_names": ("A", "B"),
            "basis_ordered_species_phase_ids": np.array([0, 1]),
            "basis_ordered_species_unknown_types": np.array([0, 0]),
            "basis_ordered_formula_matrix": np.array([[1.0], [1.0]]),
            "reordered_element_types": np.array(
                [vcs_VolPhase._ELEM_TYPE_ABSPOS]),
            "reordered_element_abundance_goals": np.array([1.0]),
            "formation_reaction_stoichiometry": np.array([[-1.0]]),
            "formation_reaction_species": (1,),
            "delta_mol_num_phase": np.array([[-1.0], [1.0]]),
            "component_basis": {"num_components": 1},
            "phase_objects": (Phase(), Phase()),
        }
        state = {
            "mole_numbers_old": np.array([0.5, 0.0]),
            "mole_numbers_new": np.array([0.5, 0.4]),
            "total_moles_old": 0.5,
            "delta_g_rxn_new": np.array([delta_g]),
            "species_status": np.array([VCS_SPECIES_COMPONENT,
                                         VCS_SPECIES_ZEROEDMS]),
        }
        solver._old_thermodynamic_state = state
        solver._species_status = state["species_status"].copy()
        solver._num_species_reduced = 2
        solver._num_reactions_reduced = 1
        solver._delta_mol_num_species = np.array([0.0, proposed_delta])
        solver._delta_phase_moles = np.zeros(2)
        return solver, state

    # Native solve_tp_inner rejects a zeroed species with nonnegative dG and
    # restores its NEW amount to the OLD amount while clearing its trial step.
    rejected, rejected_state = make_solver(0.0, 0.2)
    result = rejected.vcs_apply_zeroed_trial(1, 0)
    assert result["resurrected"] is False
    assert rejected_state["mole_numbers_new"][1] == 0.0
    assert rejected._delta_mol_num_species[1] == 0.0
    assert rejected._species_status[1] == VCS_SPECIES_ZEROEDMS

    # A favorable species in an already-present phase returns to MINOR, and
    # only receives the source's one-percent positive trial step.
    minor, minor_state = make_solver(-0.5, 0.2)
    result = minor.vcs_apply_zeroed_trial(1, 0)
    assert result["resurrected"] is True
    assert result["species_status"] == VCS_SPECIES_MINOR
    assert result["delta"] == pytest.approx(0.002)
    assert minor_state["mole_numbers_new"][1] == pytest.approx(0.002)
    np.testing.assert_allclose(minor._delta_mol_num_species, [-0.002, 0.002])
    np.testing.assert_allclose(minor._delta_phase_moles, [-0.002, 0.002])


def test_vcs_elcorr_restores_single_species_element_inventory():
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "basis_ordered_formula_matrix": np.array([[1.0]]),
        "basis_ordered_species_unknown_types": np.array([0]),
        "basis_ordered_species_phase_ids": np.array([0]),
        "basis_ordered_species_single_phase": np.array([True]),
        "reordered_element_abundance_goals": np.array([1.0]),
        "reordered_element_active": np.array([True]),
        "reordered_element_types": np.array(
            [vcs_VolPhase._ELEM_TYPE_ABSPOS]),
        "component_basis": {"num_components": 1},
        "phase_objects": (object(),),
    }
    state = {
        "mole_numbers_old": np.array([0.5]),
        "mole_numbers_new": np.array([0.5]),
        "phase_moles_old": np.array([0.5]),
        "phase_moles_new": np.array([0.5]),
        "total_moles_old": 0.5,
        "total_moles_new": 0.5,
        "species_status": np.array([VCS_SPECIES_COMPONENT]),
    }
    solver._old_thermodynamic_state = state
    solver._species_status = state["species_status"].copy()
    solver._num_species_reduced = 1

    result = solver.vcs_elcorr()

    assert result == 1
    assert state["mole_numbers_old"][0] == pytest.approx(1.0)
    assert state["phase_moles_old"][0] == pytest.approx(1.0)
    # vcs_elcorr updates only the OLD VCS phase totals; NEW vectors are not
    # touched by the native routine.
    assert state["phase_moles_new"][0] == pytest.approx(0.5)
    assert state["total_moles_new"] == pytest.approx(0.5)
    assert state["element_abundances_old"][0] == pytest.approx(1.0)


def test_vcs_elcorr_solves_component_basis_correction_with_python_lu():
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "basis_ordered_formula_matrix": np.array(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        "basis_ordered_species_unknown_types": np.zeros(3, dtype=int),
        "basis_ordered_species_phase_ids": np.zeros(3, dtype=int),
        "basis_ordered_species_single_phase": np.zeros(3, dtype=bool),
        "reordered_element_abundance_goals": np.array([0.5, 0.5]),
        "reordered_element_active": np.ones(2, dtype=bool),
        "reordered_element_types": np.full(
            2, vcs_VolPhase._ELEM_TYPE_ABSPOS, dtype=int),
        "component_basis": {"num_components": 2},
        "phase_objects": (object(),),
    }
    state = {
        "mole_numbers_old": np.array([0.2, 0.2, 0.1]),
        "mole_numbers_new": np.array([0.2, 0.2, 0.1]),
        "phase_moles_old": np.array([0.5]),
        "phase_moles_new": np.array([0.5]),
        "total_moles_old": 0.5,
        "total_moles_new": 0.5,
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_MINOR]),
    }
    solver._old_thermodynamic_state = state
    solver._species_status = state["species_status"].copy()
    solver._num_species_reduced = 3

    result = solver.vcs_elcorr()

    assert result == 1
    np.testing.assert_allclose(state["mole_numbers_old"], [0.1, 0.4, 0.1])
    np.testing.assert_allclose(state["element_abundances_old"], [0.5, 0.5])


def test_vcs_elcorr_enforces_absolute_positive_element_upper_bound():
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "basis_ordered_formula_matrix": np.array([[1.0], [1.0]]),
        "basis_ordered_species_unknown_types": np.zeros(2, dtype=int),
        "basis_ordered_species_phase_ids": np.array([0, 1]),
        "basis_ordered_species_single_phase": np.zeros(2, dtype=bool),
        "reordered_element_abundance_goals": np.array([0.5]),
        "reordered_element_active": np.array([True]),
        "reordered_element_types": np.array(
            [vcs_VolPhase._ELEM_TYPE_ABSPOS]),
        "component_basis": {"num_components": 1},
        "phase_objects": (object(), object()),
    }
    state = {
        "mole_numbers_old": np.array([0.2, 0.8]),
        "mole_numbers_new": np.array([0.2, 0.8]),
        "phase_moles_old": np.array([0.2, 0.8]),
        "phase_moles_new": np.array([0.2, 0.8]),
        "total_moles_old": 1.0,
        "total_moles_new": 1.0,
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_MINOR]),
    }
    solver._old_thermodynamic_state = state
    solver._species_status = state["species_status"].copy()
    solver._num_species_reduced = 2

    result = solver.vcs_elcorr()

    assert result == 1
    assert state["mole_numbers_old"][0] == pytest.approx(0.0)
    assert state["mole_numbers_old"][1] == pytest.approx(0.5)
    assert state["element_abundances_old"][0] == pytest.approx(0.5)

    # The same source bound also zeroes a species below the minor cutoff and
    # assigns ACTIVEBUTZERO for a multispecies phase.
    tiny = VCS_SOLVE()
    tiny._prepared_problem = {
        "basis_ordered_formula_matrix": np.array([[1.0], [1.0]]),
        "basis_ordered_species_unknown_types": np.zeros(2, dtype=int),
        "basis_ordered_species_phase_ids": np.array([0, 1]),
        "basis_ordered_species_single_phase": np.zeros(2, dtype=bool),
        "reordered_element_abundance_goals": np.array([1.0e-150]),
        "reordered_element_active": np.array([True]),
        "reordered_element_types": np.array(
            [vcs_VolPhase._ELEM_TYPE_ABSPOS]),
        "component_basis": {"num_components": 1},
        "phase_objects": (object(), object()),
    }
    tiny_state = {
        "mole_numbers_old": np.array([0.2, 0.8]),
        "mole_numbers_new": np.array([0.2, 0.8]),
        "phase_moles_old": np.array([0.2, 0.8]),
        "phase_moles_new": np.array([0.2, 0.8]),
        "total_moles_old": 1.0,
        "total_moles_new": 1.0,
        "species_status": np.array([VCS_SPECIES_COMPONENT,
                                     VCS_SPECIES_MINOR]),
    }
    tiny._old_thermodynamic_state = tiny_state
    tiny._species_status = tiny_state["species_status"].copy()
    tiny._num_species_reduced = 2

    assert tiny.vcs_elcorr() == 1
    assert tiny_state["mole_numbers_old"][1] == 0.0
    assert tiny._species_status[1] == VCS_SPECIES_ACTIVEBUTZERO
    assert tiny_state["species_status"][1] == VCS_SPECIES_ACTIVEBUTZERO


def test_vcs_total_gibbs_uses_active_species_and_skips_voltage_unknowns():
    solver = VCS_SOLVE()
    solver._prepared_problem = {
        "basis_ordered_species_unknown_types": np.array(
            [0, VCS_SPECIES_INTERFACIALVOLTAGE, 0, 0]),
    }
    solver._num_species_reduced = 3
    solver._old_thermodynamic_state = {
        "mole_numbers_old": np.array([2.0, 3.0, 5.0, 7.0]),
        "species_free_energies_old": np.array([10.0, 100.0, 4.0, 2.0]),
    }

    assert solver.vcs_total_gibbs() == pytest.approx(40.0)
    assert solver.vcs_total_gibbs(active_species_count=4) == pytest.approx(54.0)
