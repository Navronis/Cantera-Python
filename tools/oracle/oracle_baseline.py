"""Oracle baseline: deterministic reference values from the pinned build.

Writes JSON to stdout; the caller stores it under provenance/.
"""
import json
import sys

import cantera as ct

out = {"oracle_version": ct.__version__, "oracle_commit": ct.__git_commit__}

gas = ct.Solution('gri30.yaml')
out["gri30"] = {"n_species": gas.n_species, "n_reactions": gas.n_reactions}

# Species thermo: NASA cp/h/s over a temperature sweep for a few species
thermo_ref = {}
for name in ['H2O', 'CO2', 'OH', 'N2', 'CH4']:
    sp = gas.species(name)
    rows = []
    for T in (300.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0):
        rows.append({
            "T": T,
            "cp": sp.thermo.cp(T),
            "h": sp.thermo.h(T),
            "s": sp.thermo.s(T),
        })
    thermo_ref[name] = rows
out["species_thermo"] = thermo_ref

# Phase state + kinetics at two states
states = {}
for label, (T, P, X) in {
    "state_1200_ch4": (1200.0, 101325.0, 'CH4:1, O2:2, N2:7.52'),
    "state_2000_dilute": (2000.0, 101325.0, 'O2:0.3, N2:0.7'),
}.items():
    gas.TPX = T, P, X
    states[label] = {
        "T": gas.T, "P": gas.P, "density": gas.density,
        "mean_MW": gas.mean_molecular_weight,
        "standard_concentration": (
            gas.standard_concentration(0)
            if callable(getattr(gas, "standard_concentration", None))
            else gas.P / (ct.gas_constant * gas.T)
        ),
        "cp_mass": gas.cp_mass,
        "enthalpy_mass": gas.enthalpy_mass,
        "entropy_mass": gas.entropy_mass,
        "partial_molar_enthalpies": list(gas.partial_molar_enthalpies),
        # Pinned API exposes standard-state Gibbs nondimensionally: mu0_k/(R*T)
        "standard_gibbs_RT": list(gas.standard_gibbs_RT),
        "forward_rate_constants": list(gas.forward_rate_constants),
        "reverse_rate_constants": list(gas.reverse_rate_constants),
        "equilibrium_constants": list(gas.equilibrium_constants),
        "net_production_rates": list(gas.net_production_rates),
        "creation_rates": list(gas.creation_rates),
        "destruction_rates": list(gas.destruction_rates),
        "forward_rates_of_progress": list(gas.forward_rates_of_progress),
        "reverse_rates_of_progress": list(gas.reverse_rates_of_progress),
    }
out["states"] = states

# Falloff sample: rate constant vs pressure for a Troe reaction in GRI-30
gas.TPX = 1500.0, 101325.0, 'CH4:0.1, O2:0.1, N2:0.8'
falloff = {}
for i, rxn in enumerate(gas.reactions()):
    if rxn.rate.__class__.__name__ in ('TroeRate', 'LindemannRate', 'SriRate'):
        falloff[str(i)] = {
            "type": rxn.rate.__class__.__name__,
            "kf": gas.forward_rate_constants[i],
        }
        if len(falloff) >= 3:
            break
out["falloff_sample"] = falloff

json.dump(out, sys.stdout, indent=1)
