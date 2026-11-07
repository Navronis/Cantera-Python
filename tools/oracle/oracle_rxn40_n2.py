import cantera as ct
import json, sys

gas = ct.Solution('gri30.yaml')
gas.TPX = 1500.0, 101325.0, 'H2:2, O2:1, N2:4, OH:1e-6, H:1e-8'
i = gas.species_index('N2')
out = {
    "wdot_N2": gas.net_production_rates[i],
    "rxn40": {
        "equation": gas.reaction(40).equation,
        "reactants": dict(gas.reaction(40).reactants),
        "products": dict(gas.reaction(40).products),
        "third_body": gas.reaction(40).third_body.name if gas.reaction(40).third_body else None,
        "third_body_type": type(gas.reaction(40).third_body).__name__ if gas.reaction(40).third_body else None,
    },
    "wdot_all_max_abs": max(abs(w) for w in gas.net_production_rates),
}
json.dump(out, sys.stdout)
