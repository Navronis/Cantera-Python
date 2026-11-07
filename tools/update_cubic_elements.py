with open('cantera_python/cubic_eos.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    def nAtoms(self, k: int, m: int) -> float:
        sp_name = self.species_name(k) if callable(getattr(self, "species_name", None)) else self.species_names[k]
        el_name = self.elementName(m)
        comp = getattr(self, "_species_compositions", {})
        return float(comp.get(sp_name, {}).get(el_name, 0.0))'''

addition = '''    def nAtoms(self, k: int, m: int) -> float:
        sp_name = self.species_name(k) if callable(getattr(self, "species_name", None)) else self.species_names[k]
        el_name = self.elementName(m)
        comp = getattr(self, "_species_compositions", {})
        return float(comp.get(sp_name, {}).get(el_name, 0.0))

    @property
    def concentrations(self) -> np.ndarray:
        return self._mole_fractions * (self._density / self.mean_molecular_weight)

    def getConcentrations(self, c=None):
        conc = self.concentrations
        if c is not None:
            c[:] = conc
            return c
        return conc'''

assert target in text
updated = text.replace(target, addition, 1)
with open('cantera_python/cubic_eos.py', 'w', encoding='utf-8') as f:
    f.write(updated)
print('SUCCESS: added concentrations to MixtureFugacityTP in cubic_eos.py')
