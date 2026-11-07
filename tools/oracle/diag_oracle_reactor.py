import cantera as ct

gas = ct.Solution('gri30.yaml')
gas.TPX = 1500.0, 101325.0, 'H2:2, O2:1, N2:4'
r = ct.IdealGasReactor(gas)
net = ct.ReactorNet([r])
net.rtol = 1e-9
net.atol = 1e-15
net.advance(1e-6)
print('after 1us  T=%.6f H2O=%.6e OH=%.3e' % (gas.T, gas.X[gas.species_index('H2O')], gas.X[gas.species_index('OH')]))
net.advance(2e-5)
print('after 20us T=%.6f H2O=%.6e OH=%.3e' % (gas.T, gas.X[gas.species_index('H2O')], gas.X[gas.species_index('OH')]))
net.advance(5e-5)
print('after 50us T=%.6f H2O=%.6e' % (gas.T, gas.X[gas.species_index('H2O')]))
net.advance(2e-4)
print('after 200us T=%.6f H2O=%.6e' % (gas.T, gas.X[gas.species_index('H2O')]))
