with open("cantera_python/solution.py", "r", encoding="utf-8") as f:
    sol_text = f.read()

target_sol = """    elementIndex = element_index"""

replace_sol = """    elementIndex = element_index

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
