import cantera as ct
gas = ct.Solution('gri30.yaml')
names = [a for a in dir(gas) if 'chem_pot' in a or 'standard' in a or 'concen' in a]
print(names)
