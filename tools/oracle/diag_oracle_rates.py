import cantera as ct

gas = ct.Solution('gri30.yaml')
gas.TPX = 1500.0, 101325.0, 'H2:2, O2:1, N2:4'
print('n_reactions:', gas.n_reactions)
kf0 = gas.forward_rate_constants[gas.species_index('H2O')] if False else gas.forward_rate_constants[0]
print('kf[0]:', kf0)
wdot = gas.net_production_rates
import math
print('max |wdot|:', max(abs(w) for w in wdot))
print('wdot[H2O]:', wdot[gas.species_index('H2O')])
print('multiplier check: reactions enabled by default')
