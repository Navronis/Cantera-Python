import cantera as ct
import json, sys

gas = ct.Solution('gri30.yaml')
for i in (33, 35):
    r = gas.reaction(i)
    out = {
        "index": i,
        "equation": r.equation,
        "reactants": dict(r.reactants),
        "products": dict(r.products),
        "reversible": r.reversible,
        "rate_type": type(r.rate).__name__,
        "qf": gas.forward_rates_of_progress[i],
        "qr": gas.reverse_rates_of_progress[i],
    }
    print(json.dumps(out))
