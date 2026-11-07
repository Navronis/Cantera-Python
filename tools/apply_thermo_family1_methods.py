"""Add Family 1 core methods to class Phase in cantera_python/phases.py."""

with open("cantera_python/phases.py", "r", encoding="utf-8") as f:
    text = f.read()

# Target insertion point: right after setDensity / assignDensity
target = """    def assignDensity(self, rho: float) -> None:
        \"\"\"Directly assign mass density without recomputing state.\"\"\"
        self.setDensity(rho)"""

new_methods = """    def assignDensity(self, rho: float) -> None:
        \"\"\"Directly assign mass density without recomputing state.\"\"\"
        self.setDensity(rho)

    def setState_TD(self, T: float, rho: float) -> None:
        \"\"\"Set temperature and density simultaneously.\"\"\"
        self.setTemperature(T)
        self.setDensity(rho)

    def nAtoms(self, k: int, m: int) -> float:
        \"\"\"Number of atoms of element index m in species index k.\"\"\"
        self.checkSpeciesIndex(k)
        self.checkElementIndex(m)
        el_name = self._element_names[m]
        if hasattr(self, '_species') and k < len(self._species):
            sp = self._species[k]
            if hasattr(sp, 'composition') and isinstance(sp.composition, dict):
                return float(sp.composition.get(el_name, 0.0))
        return 0.0

    def getConcentrations(self, c=None) -> np.ndarray:
        \"\"\"Get molar concentrations of all species [kmol/m^3].\"\"\"
        conc = self.concentrations()
        if c is not None:
            n = min(len(c), len(conc))
            c[:n] = conc[:n]
            return c
        return conc

    def species(self, k_or_name):
        \"\"\"Return Species definition object for species index or name.\"\"\"
        if isinstance(k_or_name, int):
            self.checkSpeciesIndex(k_or_name)
            idx = k_or_name
        else:
            idx = self.speciesIndex(str(k_or_name))
            if idx < 0:
                raise IndexError(f"No such species: {k_or_name}")
        if hasattr(self, '_species') and idx < len(self._species):
            return self._species[idx]
        return self._species_names[idx]

    def stateSize(self) -> int:
        \"\"\"Number of state variables in the phase state vector (2 + n_species).\"\"\"
        return 2 + len(self._species_names)

    def saveState(self, state=None) -> list:
        \"\"\"Save internal thermodynamic state [T, rho, Y_0, ..., Y_{nsp-1}].\"\"\"
        data = [float(self.temperature()), float(self.density())] + list(self.massFractions())
        if state is not None:
            state[:len(data)] = data
            return state
        return data

    def restoreState(self, state) -> None:
        \"\"\"Restore internal thermodynamic state from vector [T, rho, Y_0, ..., Y_{nsp-1}].\"\"\"
        if len(state) < 2 + len(self._species_names):
            raise IndexError("State vector too small for restoreState")
        T = float(state[0])
        rho = float(state[1])
        y = np.array(state[2:2 + len(self._species_names)], dtype=float)
        self.setTemperature(T)
        self.setDensity(rho)
        self.setMassFractions_NoNorm(y)"""

if "\r\n" in text:
    target = target.replace("\n", "\r\n")
    new_methods = new_methods.replace("\n", "\r\n")

if target in text:
    text = text.replace(target, new_methods, 1)
    with open("cantera_python/phases.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("Added Family 1 core methods to Phase successfully.")
else:
    print("Warning: target not found in phases.py")
