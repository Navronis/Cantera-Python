with open('outputs/cantera-python/cantera_python/ideal_gas.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    def restoreState(self, state):
        if len(state) < 2 + self.n_species:
            raise IndexError("State vector too small for restoreState")
        T = float(state[0])
        rho = float(state[1])
        y = np.array(state[2:2 + self.n_species], dtype=float)
        self.setState_TD(T, rho)
        self.setMassFractions(y)
'''
assert target in text

charge_methods = '''
    def charge(self, k: int) -> float:
        if 0 <= k < len(self._state._species):
            return float(getattr(self._state._species[k], 'charge', 0.0))
        raise IndexError(f"Species index {k} out of range")

    def charges(self) -> list:
        return [float(getattr(s, 'charge', 0.0)) for s in self._state._species]

    def getCharges(self, charges=None):
        vals = self.charges()
        if charges is not None:
            charges[:len(vals)] = vals
            return charges
        return vals
'''

text = text.replace(target, target + charge_methods, 1)
with open('outputs/cantera-python/cantera_python/ideal_gas.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Successfully added charge methods to IdealGasPhase')
