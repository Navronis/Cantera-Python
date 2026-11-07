import cantera as ct

# Probe 1: undiluted, hotter mixture - should ignite fast if reactor works
gas = ct.Solution('gri30.yaml')
gas.TPX = 2000.0, 101325.0, 'H2:2, O2:1'
r = ct.IdealGasReactor(gas)
net = ct.ReactorNet([r])
net.advance(1e-5)
print('probe1 2000K H2/O2: T=%.3f H2O=%.3e nettime=%.3e' % (gas.T, gas.X[gas.species_index('H2O')], net.time))

# Probe 2: same 1500K case but check whether gas object changes at all
gas2 = ct.Solution('gri30.yaml')
gas2.TPX = 1500.0, 101325.0, 'H2:2, O2:1, N2:4'
r2 = ct.IdealGasReactor(gas2)
net2 = ct.ReactorNet([r2])
net2.advance(1e-4)
print('probe2 1500K dilute: T=%.6f H2O=%.3e nettime=%.3e' % (gas2.T, gas2.X[gas.species_index('H2O')], net2.time))

# Probe 3: standalone gas at 1500K advanced by wdot manually (sanity)
gas3 = ct.Solution('gri30.yaml')
gas3.TPX = 1500.0, 101325.0, 'H2:2, O2:1, N2:4'
wdot = gas3.net_production_rates
print('probe3 wdot[H2O] initial: %.3e  wdot[OH]: %.3e' % (
    wdot[gas3.species_index('H2O')], wdot[gas3.species_index('OH')]))
