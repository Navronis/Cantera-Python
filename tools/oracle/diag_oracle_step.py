import cantera as ct

# Isolate: does the ReactorNet's CVODE integrator run at all?
gas = ct.Solution('gri30.yaml')
gas.TPX = 2500.0, 101325.0, 'H2:2, O2:1'
r = ct.IdealGasReactor(gas)
net = ct.ReactorNet([r])

# Step mode instead of advance: does step() evolve?
t0 = net.time
net.step()
t1 = net.time
print('step: time %.3e -> %.3e, T %.3f -> %.3f' % (t0, t1, gas.T, gas.T))

# Check reactor state directly (does Reactor see the thermo object?)
print('reactor T:', r.T, 'gas T:', gas.T)
print('reactor mass:', r.mass, 'volume:', r.volume)

# Try const-pressure reactor variant
gas2 = ct.Solution('gri30.yaml')
gas2.TPX = 2500.0, 101325.0, 'H2:2, O2:1'
r2 = ct.IdealGasConstPressureReactor(gas2)
net2 = ct.ReactorNet([r2])
net2.advance(1e-3)
print('CP reactor after 1ms: T=%.3f H2O=%.4e' % (gas2.T, gas2.X[gas2.species_index('H2O')]))

# Check wdot seen *by the reactor's kinetics object* via ReactorNet diagnostics
net3 = ct.ReactorNet([r])
print('net integrator steps so far - advance again with longer horizon')
net3.advance(1.0)
print('2500K after 1s: T=%.3f H2O=%.4e time=%.3e' % (gas.T, gas.X[gas.species_index('H2O')], net3.time))
