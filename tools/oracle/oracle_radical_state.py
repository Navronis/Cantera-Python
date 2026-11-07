import cantera as ct
import json, sys

gas = ct.Solution('gri30.yaml')
gas.TPX = 1500.0, 101325.0, 'H2:2, O2:1, N2:4, OH:1e-6, H:1e-8'
out = {
    "T": gas.T, "P": gas.P,
    "forward_rate_constants": list(gas.forward_rate_constants),
    "reverse_rate_constants": list(gas.reverse_rate_constants),
    "net_production_rates": list(gas.net_production_rates),
    "creation_rates": list(gas.creation_rates),
    "destruction_rates": list(gas.destruction_rates),
    "standard_gibbs_RT": list(gas.standard_gibbs_RT),
    "partial_molar_enthalpies": list(gas.partial_molar_enthalpies),
}
json.dump(out, sys.stdout)
