with open("cantera_python/ideal_gas.py", "r", encoding="utf-8") as f:
    ig_text = f.read()

target_ig = """    elementIndex = element_index"""

replace_ig = """    elementIndex = element_index

    def atomic_weight(self, m) -> float:
        from .elements import getElementWeight
        name = self.element_name(m) if isinstance(m, int) else str(m)
        return getElementWeight(name)

    atomicWeight = atomic_weight

    @property
    def atomic_weights(self) -> np.ndarray:
        return np.array([self.atomic_weight(m) for m in range(len(self._element_names))], dtype=float)

    atomicWeights = atomic_weights"""

if "\r\n" in ig_text:
    target_ig = target_ig.replace("\n", "\r\n")
    replace_ig = replace_ig.replace("\n", "\r\n")

if target_ig in ig_text:
    ig_text = ig_text.replace(target_ig, replace_ig, 1)
    with open("cantera_python/ideal_gas.py", "w", encoding="utf-8") as f:
        f.write(ig_text)
    print("IdealGasPhase updated successfully")
else:
    print("Warning: target_ig not found in ideal_gas.py")

with open("cantera_python/solution.py", "r", encoding="utf-8") as f:
    sol_text = f.read()

target_sol = """    def element_index(self, name, error_if_not_found=True):
        try:
            return self.element_names.index(str(name))
        except ValueError:
            if error_if_not_found:
                raise ValueError(f"No such element '{name}'")
            return -1"""

replace_sol = """    def element_index(self, name, error_if_not_found=True):
        try:
            return self.element_names.index(str(name))
        except ValueError:
            if error_if_not_found:
                raise ValueError(f"No such element '{name}'")
            return -1

    def atomic_weight(self, m) -> float:
        if hasattr(self._thermo, "atomic_weight"):
            return float(self._thermo.atomic_weight(m))
        from .elements import getElementWeight
        name = self.element_name(m) if isinstance(m, int) else str(m)
        return getElementWeight(name)

    atomicWeight = atomic_weight

    @property
    def atomic_weights(self):
        if hasattr(self._thermo, "atomic_weights"):
            return self._thermo.atomic_weights
        return np.array([self.atomic_weight(m) for m in range(len(self.element_names))], dtype=float)

    atomicWeights = atomic_weights"""

if "\r\n" in sol_text:
    target_sol = target_sol.replace("\n", "\r\n")
    replace_sol = replace_sol.replace("\n", "\r\n")

if target_sol in sol_text:
    sol_text = sol_text.replace(target_sol, replace_sol, 1)
    with open("cantera_python/solution.py", "w", encoding="utf-8") as f:
        f.write(sol_text)
    print("Solution updated successfully")
else:
    print("Warning: target_sol not found in solution.py")
