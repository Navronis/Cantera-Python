import cantera as ct
import json, sys

gas = ct.Solution('gri30.yaml')
gas.TPX = 1500.0, 101325.0, 'H2:2, O2:1, N2:4, OH:1e-6, H:1e-8'
out = {
    "T": gas.T, "P": gas.P,
    "X_OH": gas.X[gas.species_index('OH')],
    "X_H": gas.X[gas.species_index('H')],
    "forward_rates_of_progress": list(gas.forward_rates_of_progress),
    "reverse_rates_of_progress": list(gas.reverse_rates_of_progress),
    "third_body_concentrations": list(gas.third_body_concentrations),
    "creation_rates": list(gas.creation_rates),
    "destruction_rates": list(gas.destruction_rates),
    "net_production_rates": list(gas.net_production_rates),
}
json.dump(out, sys.stdout)
