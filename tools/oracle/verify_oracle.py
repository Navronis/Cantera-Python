"""Oracle verification: confirm pinned-commit identity and baseline values."""
import sys

sys.path.insert(0, '/tmp/cantera-oracle/install/lib/python3.12/site-packages')
import cantera as ct  # noqa: E402

print('cantera version:', ct.__version__)
print('git commit:', ct.__git_commit__)
gas = ct.Solution('gri30.yaml')
gas.TPX = 1200, 101325, 'CH4:1, O2:2, N2:7.52'
print('n_species', gas.n_species, 'n_reactions', gas.n_reactions)
print('H2O cp0 @1200K:', gas.species('H2O').thermo.cp(1200))
print('wdot OH:', gas.net_production_rates[gas.species_index('OH')])
print('kf fwd sample:', gas.forward_rate_constants[0])
print('standard_conc:', gas.standard_concentration)
