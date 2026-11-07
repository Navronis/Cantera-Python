with open("cantera_python/ideal_gas.py", "r", encoding="utf-8") as f:
    text = f.read()

target = """    @property
    def element_names(self):
        return self._element_names"""

replace = """    @property
    def element_names(self):
        return self._element_names

    def atomic_weight(self, m: int) -> float:
        from .elements import getElementWeight
        if isinstance(m, int):
            el_name = self._element_names[m]
        else:
            el_name = str(m)
        return getElementWeight(el_name)

    @property
    def atomic_weights(self) -> np.ndarray:
        return np.array([self.atomic_weight(m) for m in range(len(self._element_names))], dtype=float)"""

if "\r\n" in text:
    target = target.replace("\n", "\r\n")
    replace = replace.replace("\n", "\r\n")

if target in text:
    text = text.replace(target, replace, 1)
    with open("cantera_python/ideal_gas.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("IdealGasPhase updated with atomic_weight and atomic_weights")
else:
    print("Warning: target not found in ideal_gas.py")

with open("cantera_python/solution.py", "r", encoding="utf-8") as f:
    sol_text = f.read()

target_sol = """    @property
    def element_names(self):
        return self.thermo.element_names"""

replace_sol = """    @property
    def element_names(self):
        return self.thermo.element_names

    def atomic_weight(self, m: int) -> float:
        return self.thermo.atomic_weight(m)

    @property
    def atomic_weights(self) -> np.ndarray:
        return self.thermo.atomic_weights"""

if "\r\n" in sol_text:
    target_sol = target_sol.replace("\n", "\r\n")
    replace_sol = replace_sol.replace("\n", "\r\n")

if target_sol in sol_text:
    sol_text = sol_text.replace(target_sol, replace_sol, 1)
    with open("cantera_python/solution.py", "w", encoding="utf-8") as f:
        f.write(sol_text)
    print("Solution updated with atomic_weight and atomic_weights")
else:
    print("Warning: target_sol not found in solution.py")
