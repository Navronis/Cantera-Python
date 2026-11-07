import re

path = 'outputs/cantera-python/cantera_python/thermo.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update _DimensionalProperties
old_base = """    def reportHf298(self, h298=None):
        val = self.h(298.15)
        if h298 is not None:
            h298[0] = val
            return h298
        return val

    def modifyOneHf298(self, k=0, Hf298New=0.0):
        current = self.h(298.15)
        diff = float(Hf298New) - current
        self._hf_offset = getattr(self, '_hf_offset', 0.0) + diff
        return diff

    def resetHf298(self, k=0):
        self._hf_offset = 0.0
        return 0.0"""

new_base = """    def reportHf298(self, h298=None):
        raise NotImplementedError("SpeciesThermoInterpType::reportHf298")

    def modifyOneHf298(self, k=0, Hf298New=0.0):
        raise NotImplementedError("SpeciesThermoInterpType::modifyOneHf298")

    def resetHf298(self, k=0):
        pass"""

# 2. Update NasaPoly1
old_nasa1 = """class NasaPoly1(_DimensionalProperties):
    \"\"\"Seven NASA coefficients ``[a0, ..., a6]`` for one temperature interval.\"\"\"

    def __init__(self, coeffs, *, min_temp=0, max_temp=inf,
                 reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.coeffs = _coefficients(coeffs, 7)"""

new_nasa1 = """class NasaPoly1(_DimensionalProperties):
    \"\"\"Seven NASA coefficients ``[a0, ..., a6]`` for one temperature interval.\"\"\"

    def __init__(self, coeffs, *, min_temp=0, max_temp=inf,
                 reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.coeffs = list(_coefficients(coeffs, 7))
        self._orig_a5 = self.coeffs[5]

    def reportHf298(self, h298=None):
        val = self.h(298.15)
        if h298 is not None:
            h298[0] = val
            return h298
        return val

    def modifyOneHf298(self, k=0, Hf298New=0.0):
        hcurr = self.reportHf298()
        delH = float(Hf298New) - hcurr
        self.coeffs[5] += delH / GAS_CONSTANT
        return delH

    def resetHf298(self, k=0):
        self.coeffs[5] = self._orig_a5
        return 0.0"""

# 3. Update NasaPoly2
old_nasa2 = """    def properties(self, temperature):
        temperature = _temperature(temperature)
        region = self.low if temperature <= self.mid_temp else self.high
        return region.properties(temperature)"""

new_nasa2 = """    def reportHf298(self, h298=None):
        val = self.low.reportHf298() if 298.15 <= self.mid_temp else self.high.reportHf298()
        if h298 is not None:
            h298[0] = val
            return h298
        return val

    def modifyOneHf298(self, k=0, Hf298New=0.0):
        h298now = self.reportHf298()
        delH = float(Hf298New) - h298now
        h_low = self.low.reportHf298()
        self.low.modifyOneHf298(k, h_low + delH)
        h_high = self.high.reportHf298()
        self.high.modifyOneHf298(k, h_high + delH)
        return delH

    def resetHf298(self, k=0):
        self.low.resetHf298(k)
        self.high.resetHf298(k)
        return 0.0

    def properties(self, temperature):
        temperature = _temperature(temperature)
        region = self.low if temperature <= self.mid_temp else self.high
        return region.properties(temperature)"""

# 4. Update ShomatePoly
old_shomate = """class ShomatePoly(_DimensionalProperties):
    \"\"\"Shomate coefficients ``[A, B, C, D, E, F, G]`` for one interval.

    Coefficients use the conventional NIST basis: the heat-capacity/entropy
    expressions give J/mol/K and the enthalpy expression gives kJ/mol, with
    reduced temperature t = T/1000. Results use the common module units.
    \"\"\"

    def __init__(self, coeffs, *, min_temp=0, max_temp=inf,
                 reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.coeffs = _coefficients(coeffs, 7)"""

new_shomate = """class ShomatePoly(_DimensionalProperties):
    \"\"\"Shomate coefficients ``[A, B, C, D, E, F, G]`` for one interval.

    Coefficients use the conventional NIST basis: the heat-capacity/entropy
    expressions give J/mol/K and the enthalpy expression gives kJ/mol, with
    reduced temperature t = T/1000. Results use the common module units.
    \"\"\"

    def __init__(self, coeffs, *, min_temp=0, max_temp=inf,
                 reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.coeffs = list(_coefficients(coeffs, 7))
        self._orig_a5 = self.coeffs[5]

    def reportHf298(self, h298=None):
        val = self.h(298.15)
        if h298 is not None:
            h298[0] = val
            return h298
        return val

    def modifyOneHf298(self, k=0, Hf298New=0.0):
        hnow = self.reportHf298()
        delH = float(Hf298New) - hnow
        self.coeffs[5] += delH / (1e3 * GAS_CONSTANT)
        return delH

    def resetHf298(self, k=0):
        self.coeffs[5] = self._orig_a5
        return 0.0"""

# 5. Update ShomatePoly2
old_shomate2 = """    def properties(self, temperature):
        temperature = _temperature(temperature)
        region = self.low if temperature <= self.mid_temp else self.high
        return region.properties(temperature)


class ConstCpPoly"""

new_shomate2 = """    def reportHf298(self, h298=None):
        val = self.low.reportHf298() if 298.15 <= self.mid_temp else self.high.reportHf298()
        if h298 is not None:
            h298[0] = val
            return h298
        return val

    def modifyOneHf298(self, k=0, Hf298New=0.0):
        h298now = self.reportHf298()
        delH = float(Hf298New) - h298now
        h_low = self.low.reportHf298()
        self.low.modifyOneHf298(k, h_low + delH)
        h_high = self.high.reportHf298()
        self.high.modifyOneHf298(k, h_high + delH)
        return delH

    def resetHf298(self, k=0):
        self.low.resetHf298(k)
        self.high.resetHf298(k)
        return 0.0

    def properties(self, temperature):
        temperature = _temperature(temperature)
        region = self.low if temperature <= self.mid_temp else self.high
        return region.properties(temperature)


class ConstCpPoly"""

# 6. Update ConstCpPoly
old_constcp = """class ConstCpPoly(_DimensionalProperties):
    \"\"\"Constant heat capacity, with h0/s0/cp0 in J/kmol-based module units.\"\"\"

    def __init__(self, t0=298.15, h0=0, s0=0, cp0=0, *, min_temp=0,
                 max_temp=inf, reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.t0 = _temperature(t0)
        self.h0, self.s0, self.cp0 = _coefficients((h0, s0, cp0), 3)"""

new_constcp = """class ConstCpPoly(_DimensionalProperties):
    \"\"\"Constant heat capacity, with h0/s0/cp0 in J/kmol-based module units.\"\"\"

    def __init__(self, t0=298.15, h0=0, s0=0, cp0=0, *, min_temp=0,
                 max_temp=inf, reference_pressure=101325):
        self._set_metadata(min_temp, max_temp, reference_pressure)
        self.t0 = _temperature(t0)
        self.h0, self.s0, self.cp0 = _coefficients((h0, s0, cp0), 3)
        self._orig_h0 = self.h0

    def reportHf298(self, h298=None):
        val = self.h(298.15)
        if h298 is not None:
            h298[0] = val
            return h298
        return val

    def modifyOneHf298(self, k=0, Hf298New=0.0):
        hnow = self.reportHf298()
        delH = float(Hf298New) - hnow
        self.h0 += delH
        return delH

    def resetHf298(self, k=0):
        self.h0 = self._orig_h0
        return 0.0"""

is_crlf = '\r\n' in text
text_norm = text.replace('\r\n', '\n')

for name, old, new in [
    ("base", old_base, new_base),
    ("nasa1", old_nasa1, new_nasa1),
    ("nasa2", old_nasa2, new_nasa2),
    ("shomate", old_shomate, new_shomate),
    ("shomate2", old_shomate2, new_shomate2),
    ("constcp", old_constcp, new_constcp),
]:
    if old in text_norm:
        text_norm = text_norm.replace(old, new)
        print(f"Replaced {name}")
    else:
        print(f"FAILED to find {name}")

if is_crlf:
    text_norm = text_norm.replace('\n', '\r\n')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text_norm)
print("Done writing updated thermo.py")
