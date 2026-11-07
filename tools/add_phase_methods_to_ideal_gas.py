with open('outputs/cantera-python/cantera_python/ideal_gas.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '    @HP.setter\n    def HP(self, val):\n        h, p = val\n        self.setState_HP(h, p)\n'
assert target in text

methods = '''
    def setMoleFractions(self, x):
        self.X = x

    def setMassFractions(self, y):
        self.Y = y

    def setConcentrations(self, c):
        c = np.asarray(c, dtype=float)
        total_c = float(np.sum(c))
        if total_c <= 0.0:
            raise CanteraError("Phase::setConcentrations", "Sum of concentrations must be positive")
        self.X = c / total_c
        self.setState_TD(self.temperature, total_c * self.mean_molecular_weight)

    def stateSize(self):
        return 2 + self.n_species

    def saveState(self, state=None):
        data = [float(self.temperature), float(self.density)] + [float(yk) for yk in self.Y]
        if state is not None:
            state[:len(data)] = data
            return state
        return data

    def restoreState(self, state):
        if len(state) < 2 + self.n_species:
            raise IndexError("State vector too small for restoreState")
        T = float(state[0])
        rho = float(state[1])
        y = np.array(state[2:2 + self.n_species], dtype=float)
        self.setState_TD(T, rho)
        self.setMassFractions(y)
'''

text = text.replace(target, target + methods, 1)
with open('outputs/cantera-python/cantera_python/ideal_gas.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Successfully added methods to IdealGasPhase')
