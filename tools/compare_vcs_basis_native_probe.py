"""Compare Python VCS preparation with a pinned-native internal probe."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cantera_python as ct
from cantera_python.vcs import (MultiPhase, VCS_SOLVE, vcs_deltag,
                                vcs_dfe_species_free_energies)


def _read_native(path: Path):
    elements = []
    species = []
    stoich = {}
    delta_g = {}
    dfe_inputs = {}
    dfe_phases = {}
    dfe_phase_states = {}
    dfe_phase_species = {}
    dfe_reactions = {}
    dfe_results = {}
    dfe_dimensions = None
    components = total_species = total_elements = None
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "components":
            components = int(fields[1])
        elif fields[0] == "species" and len(fields) == 2:
            total_species = int(fields[1])
        elif fields[0] == "elements":
            total_elements = int(fields[1])
        elif fields[0] == "element":
            elements.append(fields[2])
        elif fields[0] == "species":
            species.append(fields[2])
        elif fields[0] == "stoich":
            stoich[(fields[1], fields[2])] = float(fields[3])
        elif fields[0] == "deltag":
            delta_g[(int(fields[1]), int(fields[2]))] = float(fields[3])
        elif fields[0] == "dfe_input":
            dfe_inputs[int(fields[1])] = tuple(map(float, fields[2:7])) + tuple(
                map(int, fields[7:11]))
        elif fields[0] == "dfe_phase":
            dfe_phases[int(fields[1])] = (float(fields[2]), float(fields[3]))
        elif fields[0] == "dfe_phase_state":
            dfe_phase_states[int(fields[1])] = (
                float(fields[2]), float(fields[3]), int(fields[4]))
        elif fields[0] == "dfe_phase_species":
            dfe_phase_species[(int(fields[1]), int(fields[2]))] = (
                int(fields[3]), fields[4], float(fields[5]), float(fields[6]))
        elif fields[0] == "dfe_reaction":
            dfe_reactions[int(fields[1])] = int(fields[2])
        elif fields[0] == "dfe_dimensions":
            dfe_dimensions = tuple(map(int, fields[1:4]))
        elif fields[0] == "dfe_result":
            dfe_results[(int(fields[1]), int(fields[2]))] = float(fields[3])
    if (components is None or total_species != len(species)
            or total_elements != len(elements)):
        raise ValueError("native probe output is incomplete or malformed")
    if dfe_dimensions is None:
        raise ValueError("native probe output is missing vcs_dfe records")
    return (components, elements, species, stoich, delta_g, dfe_inputs,
            dfe_phases, dfe_phase_states, dfe_phase_species, dfe_reactions,
            dfe_dimensions, dfe_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native", type=Path,
        default=Path("audit/current_tree_parity/native_vcs_basis_gri_graphite.txt"))
    args = parser.parse_args()

    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1200.0, ct.OneAtm, "CH4:1, O2:2, N2:7.52"
    graphite = ct.Solution("graphite.yaml")
    graphite.TP = gas.TP
    mixture = MultiPhase()
    mixture.add_phase(gas, 1.0)
    mixture.add_phase(graphite, 0.2)
    solver = VCS_SOLVE()
    prepared = solver.prepare_problem(mixture)
    basis = prepared["component_basis"]
    ncomp = basis["num_components"]
    py_species = prepared["basis_ordered_species_names"]
    py_elements = prepared["element_order_after_basis_rearrangement"]
    py_matrix = prepared["formation_reaction_stoichiometry"]

    (native_components, native_elements, native_species, native_stoich,
     native_delta_g, dfe_inputs, dfe_phases, dfe_phase_states,
     dfe_phase_species, dfe_reactions, dfe_dimensions, dfe_results) = _read_native(args.native)
    if ncomp != native_components:
        raise AssertionError(f"component count differs: Python {ncomp}, native {native_components}")
    if tuple(py_elements) != tuple(native_elements):
        raise AssertionError(f"element ordering differs: Python {py_elements}, native {native_elements}")
    if tuple(py_species) != tuple(native_species):
        raise AssertionError("species permutation differs between Python and native")

    errors = []
    for j, comp_name in enumerate(py_species[:ncomp]):
        for i, species_name in enumerate(py_species[ncomp:]):
            expected = native_stoich[(comp_name, species_name)]
            actual = float(py_matrix[j, i])
            errors.append(abs(actual - expected))
    max_abs = max(errors, default=0.0)
    if max_abs > 2e-13:
        raise AssertionError(f"formation matrix mismatch; maximum absolute error {max_abs:.17g}")

    free = np.zeros(len(py_species), dtype=float)
    free[:ncomp] = [5.0, 2.0, *([0.0] * max(0, ncomp - 2))]
    if len(py_species) > ncomp:
        free[ncomp] = -10.0
    if len(py_species) > ncomp + 1:
        free[ncomp + 1] = -1.0
    statuses = np.ones(len(py_species), dtype=int)
    if len(py_species) > ncomp + 1:
        statuses[ncomp + 1] = 0
    moles = np.asarray(prepared["basis_ordered_mole_numbers"], dtype=float).copy()
    moles[0] = 0.0
    moles[1] = 1.0
    for filter_value in (-1, 0, 1):
        values = vcs_deltag(
            np.full(py_matrix.shape[1], 99.0), free, py_matrix,
            range(ncomp, len(py_species)), statuses, ncomp, moles,
            L=filter_value)
        for irxn in (0, 1):
            expected = native_delta_g[(filter_value, irxn)]
            if abs(float(values[irxn]) - expected) > 2e-13:
                raise AssertionError(
                    f"deltag mismatch for L={filter_value}, reaction={irxn}: "
                    f"Python {values[irxn]:.17g}, native {expected:.17g}")

    dfe_nc, dfe_nrxn, dfe_nsp = dfe_dimensions
    if dfe_nsp != len(dfe_inputs) or dfe_nrxn != len(dfe_reactions):
        raise AssertionError("native vcs_dfe dimensions do not match emitted arrays")
    input_rows = [dfe_inputs[i] for i in range(dfe_nsp)]
    phase_rows = [dfe_phases[i] for i in range(len(dfe_phases))]
    standard = np.asarray([r[0] for r in input_rows])
    ln_mnaught = np.asarray([r[1] for r in input_rows])
    moles = np.asarray([r[2] for r in input_rows])
    activity = np.asarray([r[3] for r in input_rows])
    charges = np.asarray([r[4] for r in input_rows])
    phase_ids = np.asarray([r[5] for r in input_rows], dtype=int)
    statuses = np.asarray([r[6] for r in input_rows], dtype=int)
    unknown = np.asarray([r[7] for r in input_rows], dtype=int)
    single = np.asarray([r[8] for r in input_rows], dtype=bool)
    phase_moles = np.asarray([r[0] for r in phase_rows])
    potentials = np.asarray([r[1] for r in phase_rows])
    reaction_species = np.asarray([dfe_reactions[i] for i in range(dfe_nrxn)])
    dfe_errors = []
    for filter_value in (-1, 0, 1):
        actual = vcs_dfe_species_free_energies(
            standard, ln_mnaught, moles, phase_moles,
            phase_ids, activity, charges, potentials, statuses, single,
            unknown, reaction_species, dfe_nc, faraday_dim=0.0,
            L=filter_value,
            initial_free_energies=np.full(dfe_nsp, -900.0))
        for k in range(dfe_nsp):
            expected = dfe_results[(filter_value, k)]
            error = abs(float(actual[k]) - expected)
            dfe_errors.append(error)
            if error > 2e-13:
                raise AssertionError(
                    f"vcs_dfe mismatch for L={filter_value}, species={k}: "
                    f"Python {actual[k]:.17g}, native {expected:.17g}")

    phase_sync = solver.update_phase_state(moles, state_calc=0)
    if not np.allclose(phase_sync["phase_moles"], phase_moles,
                       rtol=0.0, atol=1e-12):
        raise AssertionError("Python phase totals differ from native vcs_tmoles")
    python_activity = np.full(dfe_nsp, np.nan, dtype=float)
    phase_objects = prepared["phase_objects"]
    if len(phase_objects) != len(dfe_phase_states):
        raise AssertionError("native and Python VCS phase counts differ")
    for iph, phase in enumerate(phase_objects):
        native_total, native_phi, native_existence = dfe_phase_states[iph]
        if abs(phase.totalMoles() - native_total) > 1e-12:
            raise AssertionError(f"phase {iph} total-mole update differs")
        if abs(phase.electricPotential() - native_phi) > 2e-14:
            raise AssertionError(f"phase {iph} electric potential differs")
        if phase.exists() != native_existence:
            raise AssertionError(f"phase {iph} existence status differs")
        phase.sendToVCS_ActCoeff(0, python_activity, moles=moles)
        for local in range(phase.nSpecies()):
            (global_index, name, native_x, native_ac) = dfe_phase_species[(iph, local)]
            if phase.spGlobalIndexVCS(local) != global_index:
                raise AssertionError(f"phase {iph} species global-index mapping differs")
            if phase.speciesName(local) != name:
                raise AssertionError(f"phase {iph} species name differs")
            if abs(phase.moleFraction(local) - native_x) > 2e-14:
                raise AssertionError(f"phase {iph} species mole fraction differs")
            if abs(float(python_activity[global_index]) - native_ac) > 2e-14:
                raise AssertionError(f"phase {iph} activity coefficient differs")
    print(f"Pinned-native VCS kernel parity: PASS ({len(py_species)} species, "
          f"{ncomp} components, {len(errors)} stoichiometric coefficients, "
          "6 major/minor deltag outputs)")
    print(f"elements: {', '.join(py_elements)}")
    print(f"components: {', '.join(py_species[:ncomp])}")
    print(f"maximum absolute stoichiometric error: {max_abs:.17g}")
    print("deltag L=-1/0/1, reactions 0-1: exact match")
    print(f"vcs_dfe L=-1/0/1: PASS ({len(dfe_errors)} species outputs, "
          f"maximum absolute error {max(dfe_errors, default=0.0):.17g})")
    print(f"phase uploads: PASS ({len(dfe_phase_states)} phase states, "
          f"{len(dfe_phase_species)} species mole-fraction/activity records)")


if __name__ == "__main__":
    main()
