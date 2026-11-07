import cantera as ct

gas = ct.Solution('gri30.yaml')
gas.TPX = 1500.0, 101325.0, 'H2:2, O2:1, N2:4'
wdot = gas.net_production_rates
idx = {n: gas.species_index(n) for n in ('H', 'O', 'OH', 'H2O', 'H2', 'O2', 'N2')}
for n in ('H', 'O', 'OH', 'H2O'):
    print('wdot[%s] = %.6e' % (n, wdot[idx[n]]))
print('max |wdot| = %.6e' % max(abs(w) for w in wdot))
# forward/reverse ROP for first few reactions
ropf = gas.forward_rates_of_progress
ropr = gas.reverse_rates_of_progress
for i in range(4):
    print('rxn %d: qf=%.6e qr=%.6e' % (i, ropf[i], ropr[i]))
