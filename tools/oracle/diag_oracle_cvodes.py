import cantera as ct

# Does the reactor net integrate ANY derivative? Use a reactor with huge wdot.
gas = ct.Solution('gri30.yaml')
gas.TPX = 2500.0, 101325.0, 'H2:2, O2:1'
r = ct.IdealGasReactor(gas)
net = ct.ReactorNet([r])
print('integrator type available:', type(net).__name__)
net.advance(1e-3)
print('2500K after 1ms: T=%.3f H2O=%.4e' % (gas.T, gas.X[gas.species_index('H2O')]))

# Try explicit ReactorNet initialization order: construct net first
gas2 = ct.Solution('gri30.yaml')
net2 = ct.ReactorNet()
r2 = ct.IdealGasReactor(gas2)
net2.add_reactor(r2)
gas2.TPX = 2500.0, 101325.0, 'H2:2, O2:1'
net2.reinitialize()
net2.advance(1e-3)
print('alt order after 1ms: T=%.3f H2O=%.4e' % (gas2.T, gas2.X[gas2.species_index('H2O')]))
