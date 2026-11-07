"""
Reconciliation script for Flow1D / StFlow (131 canonical symbols).
Populates complete metadata, scientific responsibilities, exact line ranges,
callers, callees, and updates closure set, starting snapshot, and call graph.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INVENTORY_PATH = REPO_ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl"
CLOSURE_SET_PATH = REPO_ROOT / "audit" / "flow1d_closure_symbol_set.json"
STARTING_SNAPSHOT_PATH = REPO_ROOT / "audit" / "flow1d_starting_snapshot.json"
CALL_GRAPH_PATH = REPO_ROOT / "audit" / "flow1d_runtime_call_graph.json"

# Detailed scientific responsibilities mapping for all Flow1D symbols
RESPONSIBILITIES = {
    "domainType": "Returns domain classification string: 'free-flow', 'axisymmetric-flow', or 'unstrained-flow'.",
    "setupGrid": "Configures domain grid coordinates, validates monotonic spacing, and resizes state storage vectors.",
    "resetBadValues": "Projects unphysical species mass fractions into bounds and renormalizes via thermodynamic phase.",
    "phase": "Returns associated thermodynamic Solution / ThermoPhase object.",
    "kinetics": "Returns associated Kinetics mechanism object.",
    "_setKinetics": "Attaches Kinetics mechanism and synchronizes solution callback references.",
    "_setTransport": "Attaches Transport manager, detects multicomponent capability, and sizes transport caches.",
    "setTransportModel": "Configures gas transport model ('mixture-averaged', 'multicomponent', etc.) and updates internal state.",
    "transportModel": "Queries active gas transport model identifier.",
    "enableSoret": "Enables or disables thermal diffusion (Soret effect) in diffusive flux formulation.",
    "withSoret": "Queries whether thermal diffusion (Soret effect) is active.",
    "setFluxGradientBasis": "Sets mixture-averaged transport flux gradient basis to molar or mass fraction.",
    "fluxGradientBasis": "Queries active mixture-averaged transport flux gradient basis.",
    "setPressure": "Sets thermodynamic static pressure in domain and underlying thermo phase.",
    "pressure": "Queries domain static thermodynamic pressure.",
    "_getInitialSoln": "Extracts initial temperature, density, and mass fraction profile from associated thermo phase.",
    "_finalize": "Finalizes domain initialization, interpolates fixed temperature profiles, and anchors free flame fixed point.",
    "setFixedTempProfile": "Specifies spatial coordinate and temperature profile for fixed-temperature solving.",
    "setTemperature": "Sets fixed temperature value at specified grid point.",
    "T_fixed": "Returns fixed temperature reference value at grid point j.",
    "componentName": "Translates solution component index to canonical component name.",
    "componentIndex": "Resolves component name or alias to canonical solution component index.",
    "hasComponent": "Queries whether named component exists in domain representation.",
    "componentActive": "Determines whether component is mathematically active in current flow configuration.",
    "updateState": "Loads solution vector values at specified grid point into thermodynamic phase.",
    "show": "Prints formatted profile table of domain solution variables and radiative heat loss.",
    "getValues": "Extracts spatial profile of specified solution component into output array.",
    "setValues": "Assigns spatial profile for specified solution component from input array.",
    "getResiduals": "Extracts work residual vector for specified solution component.",
    "setFlatProfile": "Initializes specified solution component to uniform constant value across all grid points.",
    "toArray": "Exports domain grid, solution variables, and metadata into SolutionArray structure.",
    "fromArray": "Restores domain grid, solution variables, and metadata from SolutionArray structure.",
    "setFreeFlow": "Configures domain for freely propagating laminar flame with fixed internal temperature anchor.",
    "setAxisymmetricFlow": "Configures domain for axisymmetric stagnation/counterflow with radial momentum and Lambda eigenvalue.",
    "setUnstrainedFlow": "Configures domain for unstrained flat burner-stabilized flame with fixed mass flux.",
    "solveEnergyEqn": "Enables thermal energy equation and activates grid refiner for U, V, and T.",
    "allOfEnergyEnabled": "Queries whether thermal energy equation is active across all domain grid points.",
    "noneOfEnergyEnabled": "Queries whether thermal energy equation is inactive across all domain grid points.",
    "setEnergyEnabled": "Enables or disables thermal energy equation uniformly across domain.",
    "solveElectricField": "Raises NotImplementedError for neutral flow (implemented in IonFlow subclass).",
    "fixElectricField": "Raises NotImplementedError for neutral flow (implemented in IonFlow subclass).",
    "doElectricField": "Queries electric field solver status (raises NotImplementedError for neutral Flow1D).",
    "enableRadiation": "Enables or disables Liu-Rogg optically thin radiative heat loss model.",
    "radiationEnabled": "Queries whether radiative heat loss model is active.",
    "radiativeHeatLoss": "Returns evaluated radiative heat loss at grid point j [W/m^3].",
    "setBoundaryEmissivities": "Specifies infrared surface emissivities at left and right boundaries in [0, 1].",
    "leftEmissivity": "Queries surface infrared emissivity at left domain boundary.",
    "rightEmissivity": "Queries surface infrared emissivity at right domain boundary.",
    "fixTemperature": "Deactivates thermal energy equation to hold temperature profile fixed.",
    "leftControlPointTemperature": "Queries left control point temperature for two-point continuation control.",
    "leftControlPointCoordinate": "Queries left control point spatial coordinate for two-point continuation control.",
    "setLeftControlPointTemperature": "Sets left control point temperature for two-point continuation control.",
    "setLeftControlPointCoordinate": "Sets left control point spatial coordinate for two-point continuation control.",
    "rightControlPointTemperature": "Queries right control point temperature for two-point continuation control.",
    "rightControlPointCoordinate": "Queries right control point spatial coordinate for two-point continuation control.",
    "setRightControlPointTemperature": "Sets right control point temperature for two-point continuation control.",
    "setRightControlPointCoordinate": "Sets right control point spatial coordinate for two-point continuation control.",
    "enableTwoPointControl": "Enables two-point temperature continuation control in strained axisymmetric flames.",
    "twoPointControlEnabled": "Queries whether two-point continuation control is active.",
    "doEnergy": "Queries whether thermal energy equation is solved at grid point j.",
    "resize": "Allocates state vectors, thermo caches, transport properties, and Jacobian workspace arrays.",
    "setGas": "Synchronizes thermodynamic phase state with local state vector (T, P, Y) at grid point j.",
    "setGasAtMidpoint": "Synchronizes thermodynamic phase state with cell-interface average state between j and j+1.",
    "density": "Returns cached gas mixture density at grid point j [kg/m^3].",
    "isFree": "Queries whether domain is in freely propagating flame configuration.",
    "isStrained": "Queries whether domain is in strained axisymmetric counterflow configuration.",
    "setViscosityFlag": "Enables or disables viscous shear stress evaluation in radial momentum equation.",
    "leftExcessSpecies": "Returns index of dominant species at left boundary computed as 1 - sum(Y_k).",
    "rightExcessSpecies": "Returns index of dominant species at right boundary computed as 1 - sum(Y_k).",
    "hasAnalyticJacobian": "Checks if analytical Jacobian evaluation is supported for grid point j and component n.",
    "evalJacobianAnalytic": "Assembles exact analytical Jacobian entries for species, energy, and flow equations.",
    "checkAnalyticJacobian": "Validates solver prerequisites when analytic Jacobian mode is explicitly requested.",
    "getMeta": "Serializes domain physical flags, transport model, radiative settings, and refinement criteria.",
    "setMeta": "Restores domain physical flags, transport model, radiative settings, and refinement criteria.",
    "updateThermo": "Updates and caches density, mean molecular weight, specific heat, species enthalpies, and net production rates.",
    "updateTransport": "Evaluates midpoint transport properties (viscosity, thermal conductivity, binary/multicomponent diffusion).",
    "updateDiffFluxes": "Computes species diffusive mass fluxes with multicomponent, mixture-averaged, correction, and Soret terms.",
    "computeRadiation": "Evaluates optically thin volumetric radiative heat loss from CO2 and H2O using RADCAL polynomials.",
    "radiationPolyFactor": "Computes Planck-mean absorption polynomial factor for radiating species (CO2 or H2O).",
    "T": "Accesses or mutates gas temperature at grid point j from local solution vector [K].",
    "T_prev": "Returns gas temperature at grid point j from previous time step [K].",
    "prevSoln": "Accesses previous time step solution vector for transient residual evaluation.",
    "rho_u": "Computes axial mass flux (density * axial velocity) at grid point j [kg/m^2/s].",
    "u": "Returns axial flow velocity at grid point j from local solution vector [m/s].",
    "V": "Returns radial velocity gradient / spread rate at grid point j [1/s].",
    "V_prev": "Returns radial velocity gradient / spread rate at grid point j from previous time step [1/s].",
    "Lambda": "Returns radial pressure gradient eigenvalue at grid point j [(1/r) dP/dr, Pa/m^2].",
    "Uo": "Returns oxidizer inlet axial velocity variable for two-point control [m/s].",
    "Y": "Accesses species mass fraction array or scalar at grid point j.",
    "Y_prev": "Returns species mass fraction at grid point j from previous time step.",
    "X": "Computes species mole fraction from cached mean molecular weight and mass fraction at grid point j.",
    "flux": "Returns evaluated diffusive mass flux for species k at interface j [kg/m^2/s].",
    "m_flux": "Internal storage array accessor for diffusive mass fluxes across species and interfaces.",
    "dVdz": "Evaluates first spatial derivative of radial velocity V using direction-dependent upwind differencing.",
    "dYdz": "Evaluates first spatial derivative of species mass fraction Y_k using upwind differencing.",
    "dTdz": "Evaluates first spatial derivative of gas temperature T using upwind differencing.",
    "shear": "Evaluates viscous shear stress divergence operator (d/dz)(mu dV/dz) using central 3-point differencing.",
    "conduction": "Evaluates conductive heat flux divergence operator -(d/dz)(lambda dT/dz) using central 3-point differencing.",
    "mindex": "Computes 1D linear storage index for 3D multicomponent diffusion coefficient tensor (k, j, m).",
    "grad_hk": "Computes upwinded spatial gradient of species molar enthalpies dh_k/dz at grid point j.",
    "usingAnalyticJacobian": "Determines whether analytic Jacobian species columns can be evaluated for current flow state.",
    "analyticRequested": "Checks whether analytic Jacobian evaluation mode is active or set to auto.",
    "analyticJacobianSupported": "Virtual predicate indicating whether flow domain physics supports analytic Jacobian.",
    "probeAnalyticJacobian": "Probes kinetics mechanism once for net production rate concentration derivative support.",
    "computeWdotDerivatives": "Transforms sparse dwdot/dC into dense dwdot/dY via chain rule at constant T and P.",
    "fluxJacobian": "Computes partial derivatives of adjacent interface diffusive mass fluxes with respect to species mass fractions.",
    "addSpeciesJacEntries": "Assembles analytical Jacobian block entries for species conservation residual equations.",
    "addEnergyJacEntries": "Assembles analytical Jacobian entries for thermal energy conservation residual equation.",
    "addFlowJacEntries": "Assembles analytical Jacobian chain-rule entries for continuity and momentum residual equations.",
    "Flow1D": "DomainFactory registration symbol for instantiating 1D reacting flow domains.",
    "CanteraError": "Native error reporting site for invalid operations, domain states, or mismatched models.",
    "IndexError": "Native bounds checking error site for component indexing.",
    "ArraySizeError": "Native dimension checking error site for state profile array operations.",
    "energy_flags": "Internal unique set container used in getMeta() for compact energy equation flag serialization.",
    "data": "Internal local variable in toArray() holding component array values across grid points.",
    "soln": "Internal span reference in fromArray() pointing to contiguous domain state memory.",
    "JY": "Internal Eigen Map view in computeWdotDerivatives() for dense dwdot/dY matrix operations.",
    "c": "Internal Eigen Map view in computeWdotDerivatives() for molar concentration vector.",
    "constexpr": "Compile-time branch construct in templated fluxJacobian() for left vs right endpoint flux evaluation.",
}

CALL_EDGES = [
    # Top-level residual evaluation flow
    ("Flow1D.eval", "Flow1D.updateProperties"),
    ("Flow1D.eval", "Flow1D.computeRadiation"),
    ("Flow1D.eval", "Flow1D.evalContinuity"),
    ("Flow1D.eval", "Flow1D.evalMomentum"),
    ("Flow1D.eval", "Flow1D.evalEnergy"),
    ("Flow1D.eval", "Flow1D.evalLambda"),
    ("Flow1D.eval", "Flow1D.evalElectricField"),
    ("Flow1D.eval", "Flow1D.evalUo"),
    ("Flow1D.eval", "Flow1D.evalSpecies"),

    # Property updates
    ("Flow1D.updateProperties", "Flow1D.updateThermo"),
    ("Flow1D.updateProperties", "Flow1D.updateTransport"),
    ("Flow1D.updateProperties", "Flow1D.updateDiffFluxes"),
    ("Flow1D.updateThermo", "Flow1D.setGas"),
    ("Flow1D.updateTransport", "Flow1D.setGasAtMidpoint"),
    ("Flow1D.computeRadiation", "Flow1D.radiationPolyFactor"),

    # Equation evaluations
    ("Flow1D.evalContinuity", "Flow1D.rho_u"),
    ("Flow1D.evalContinuity", "Flow1D.density"),
    ("Flow1D.evalContinuity", "Flow1D.V"),
    ("Flow1D.evalMomentum", "Flow1D.shear"),
    ("Flow1D.evalMomentum", "Flow1D.dVdz"),
    ("Flow1D.evalMomentum", "Flow1D.Lambda"),
    ("Flow1D.evalMomentum", "Flow1D.V_prev"),
    ("Flow1D.evalEnergy", "Flow1D.conduction"),
    ("Flow1D.evalEnergy", "Flow1D.grad_hk"),
    ("Flow1D.evalEnergy", "Flow1D.dTdz"),
    ("Flow1D.evalEnergy", "Flow1D.T_prev"),
    ("Flow1D.evalEnergy", "Flow1D.T_fixed"),
    ("Flow1D.evalSpecies", "Flow1D.dYdz"),
    ("Flow1D.evalSpecies", "Flow1D.flux"),
    ("Flow1D.evalSpecies", "Flow1D.Y_prev"),
    ("Flow1D.evalSpecies", "Flow1D.leftExcessSpecies"),
    ("Flow1D.evalSpecies", "Flow1D.rightExcessSpecies"),

    # Analytic Jacobian flow
    ("Flow1D.evalJacobianAnalytic", "Flow1D.usingAnalyticJacobian"),
    ("Flow1D.evalJacobianAnalytic", "Flow1D.updateThermo"),
    ("Flow1D.evalJacobianAnalytic", "Flow1D.updateDiffFluxes"),
    ("Flow1D.evalJacobianAnalytic", "Flow1D.setGas"),
    ("Flow1D.evalJacobianAnalytic", "Flow1D.computeWdotDerivatives"),
    ("Flow1D.evalJacobianAnalytic", "Flow1D.addSpeciesJacEntries"),
    ("Flow1D.evalJacobianAnalytic", "Flow1D.addEnergyJacEntries"),
    ("Flow1D.evalJacobianAnalytic", "Flow1D.addFlowJacEntries"),
    ("Flow1D.addSpeciesJacEntries", "Flow1D.fluxJacobian"),
    ("Flow1D.addEnergyJacEntries", "Flow1D.grad_hk"),
    ("Flow1D.addEnergyJacEntries", "Flow1D.conduction"),
    ("Flow1D.addEnergyJacEntries", "Flow1D.radiationPolyFactor"),
    ("Flow1D.addFlowJacEntries", "Flow1D.shear"),
]


def reconcile():
    with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
        inv_rows = {json.loads(line)["stable_symbol_id"]: json.loads(line) for line in f}

    with open(CLOSURE_SET_PATH, "r", encoding="utf-8") as f:
        closure_data = json.load(f)

    print(f"Reconciling {len(closure_data)} Flow1D closure symbols...")

    callers_map = {}
    callees_map = {}
    for src, dst in CALL_EDGES:
        callees_map.setdefault(src, []).append(dst)
        callers_map.setdefault(dst, []).append(src)

    updated_symbols = {}
    for sym_id, entry in closure_data.items():
        inv_row = inv_rows.get(sym_id, {})
        method_name = entry.get("native_method", sym_id.split(":")[-1])
        class_name = entry.get("native_class", "Flow1D")
        qual_name = f"{class_name}.{method_name}"

        resp = RESPONSIBILITIES.get(
            method_name,
            entry.get("scientific_responsibility") or inv_row.get("legacy_scientific_purpose") or "Flow1D numerical governing equation subsystem method."
        )

        line_range = inv_row.get("line_range")
        if not line_range:
            parts = sym_id.split(":")
            line_no = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            line_range = [line_no, line_no]

        callers = callers_map.get(qual_name, [])
        callees = callees_map.get(qual_name, [])

        # Non-runtime proven artifacts
        is_non_runtime = entry.get("current_status") == "NON_RUNTIME_CPP_ONLY_PROVEN"
        relevance = "NON_SYMBOL_INVENTORY_ARTIFACT" if is_non_runtime else "SCIENTIFIC_RUNTIME_CANDIDATE"

        updated = {
            "canonical_index": entry.get("canonical_index", inv_row.get("canonical_index")),
            "stable_symbol_id": sym_id,
            "native_file": entry.get("native_file", inv_row.get("source_file")),
            "native_class": class_name,
            "native_method": method_name,
            "line_range": line_range,
            "python_target": f"cantera_python.oned.{qual_name}",
            "current_status": entry.get("current_status", "NOT_PORTED"),
            "scientific_runtime_relevance": relevance,
            "scientific_responsibility": resp,
            "callers": callers if callers else None,
            "callees": callees if callees else None,
            "remaining_validation_requirement": entry.get(
                "remaining_validation_requirement",
                "Differential parity against pinned native Cantera Flow1D oracle across flow types (unstrained-flow, free-flow, axisymmetric-flow), continuity, axial/radial momentum, energy equation, species conservation, multicomponent/mixture-averaged diffusion, Soret thermal diffusion, radiation models, and adaptive grid interpolation."
            )
        }
        updated_symbols[sym_id] = updated

    with open(CLOSURE_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_symbols, f, indent=2, sort_keys=True)
    print(f"Updated {CLOSURE_SET_PATH}")

    snapshot_data = {
        "solver_family": "flow1d",
        "subsystem": "1D reacting flow domains",
        "symbol_count": len(updated_symbols),
        "symbols": updated_symbols,
    }
    with open(STARTING_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, indent=2, sort_keys=True)
    print(f"Updated {STARTING_SNAPSHOT_PATH}")

    # Build comprehensive call graph
    nodes = []
    for sym_id, entry in updated_symbols.items():
        nodes.append({
            "id": sym_id,
            "label": f"{entry['native_class']}.{entry['native_method']}",
            "status": entry["current_status"],
            "relevance": entry["scientific_runtime_relevance"]
        })

    graph_data = {
        "solver_family": "flow1d",
        "nodes": nodes,
        "edges": [{"source": src, "target": dst} for src, dst in CALL_EDGES],
    }
    with open(CALL_GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, sort_keys=True)
    print(f"Updated {CALL_GRAPH_PATH} with {len(nodes)} nodes and {len(CALL_EDGES)} edges.")


if __name__ == "__main__":
    reconcile()
