with open('outputs/cantera-python/cantera_python/thermo.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    def setRefPressure(self, pref):
        self.reference_pressure = float(pref)
        return self.reference_pressure
'''
assert target in text

methods = '''    def minTemp(self) -> float:
        return float(self.min_temp)

    def maxTemp(self) -> float:
        return float(self.max_temp)

    def refPressure(self) -> float:
        return float(self.reference_pressure)

    def updatePropertiesTemp(self, temp, cp_R=None, h_RT=None, s_R=None):
        cp, h, s = self.properties(temp)
        if cp_R is not None:
            cp_R[0] = cp
        if h_RT is not None:
            h_RT[0] = h
        if s_R is not None:
            s_R[0] = s
        return cp, h, s

    def updateProperties(self, tt, cp_R=None, h_RT=None, s_R=None):
        temp = tt[0] if hasattr(tt, '__getitem__') else float(tt)
        return self.updatePropertiesTemp(temp, cp_R, h_RT, s_R)

    def reportParameters(self):
        index = getattr(self, 'index', 0)
        type_id = self.reportType()
        min_t = self.minTemp()
        max_t = self.maxTemp()
        pref = self.refPressure()
        coeffs = self.dimensioned_coeffs()
        return index, type_id, min_t, max_t, pref, coeffs
'''

text = text.replace(target, target + '\n' + methods, 1)
with open('outputs/cantera-python/cantera_python/thermo.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Successfully added SpeciesThermoInterpType methods to thermo.py')
