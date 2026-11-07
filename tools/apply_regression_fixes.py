"""Apply regression fixes across cantera-python.
1. rates.py: SriRate raises CanteraError on negative c/d; restore superelastic_rate on ElectronCollisionPlasmaRate.
2. surface.py: InterfaceKinetics.kineticsType returns "InterfaceKinetics".
3. kinetics.py: ReactionPathDiagram add threshold methods; _VectorizedKinetics add _general_stoich_products and fix zero default third body efficiency.
4. ideal_gas.py: IdealGasPhase and IdealGasMixture preserve elements order from YAML, and implement setMassFractions_NoNorm without normalizing.
"""
import re
import os

def fix_rates():
    path = "cantera_python/rates.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. SriRate
    content = content.replace(
        'raise ValueError("SRI c must be nonnegative")',
        'raise CanteraError("SRI c must be nonnegative")'
    ).replace(
        'raise ValueError("SRI d must be nonnegative")',
        'raise CanteraError("SRI d must be nonnegative")'
    )

    # 2. superelastic_rate
    if "def superelastic_rate" not in content:
        target = 'def type(self) -> str:\n        return "electron-collision-plasma"'
        superelastic_code = '''def superelastic_rate(self, shared_energy_levels, distribution):
        """Return the pinned super-elastic reverse coefficient.

        The source offsets the cross-section interpolation grid by the first
        tabulated energy (the collision threshold), then integrates
        ``(epsilon + threshold) * F0 * sigma_super``.
        """
        levels, eedf = self._shared_inputs(shared_energy_levels, distribution)
        threshold = self.energy_levels[0]
        offset_levels = (0.0,) + tuple(level - threshold for level in self.energy_levels[1:])
        offset_sections = tuple(linearInterp(level, offset_levels, self.cross_sections)
                                for level in levels)
        integrand = tuple((energy + threshold) * value * section
                          for energy, value, section in zip(levels, eedf, offset_sections))
        return self._prefactor * simpson(integrand, levels)

    '''
        content = content.replace(target, superelastic_code + target)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("rates.py patched.")

def fix_surface():
    path = "cantera_python/surface.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # InterfaceKinetics.kineticsType
    content = re.sub(
        r'def kineticsType\(self\)\s*->\s*str:\s*return\s*"surface"',
        'def kineticsType(self) -> str:\n        return "InterfaceKinetics"',
        content
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("surface.py patched.")

def fix_kinetics():
    path = "cantera_python/kinetics.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. ReactionPathDiagram methods
    if "def setThreshold" not in content:
        rpd_target = "    def maxFlow(self) -> float:\n        return float(self.m_flxmax)"
        rpd_methods = '''    def setThreshold(self, th: float):
        self.threshold = float(th)

    def threshold(self) -> float:
        return float(self.threshold) if hasattr(self, "threshold") else 0.005

    def setBoldThreshold(self, th: float):
        self.bold_threshold = float(th)

    def boldThreshold(self) -> float:
        return float(self.bold_threshold) if hasattr(self, "bold_threshold") else 0.2

    def export_dot(self) -> str:
        return self.get_dot()

    '''
        content = content.replace(rpd_target, rpd_methods + rpd_target)

    # 2. _VectorizedKinetics._general_stoich_products and _activity_products and tb_eff init
    # Fix tb_eff initialization
    content = content.replace(
        "            tb = r.third_body\n            if tb is not None:\n                tb_default[i] = tb.default_efficiency",
        "            tb = r.third_body\n            if tb is not None:\n                tb_default[i] = tb.default_efficiency\n                tb_eff[i, :] = tb.default_efficiency"
    )

    # Add _general_stoich_products
    if "_general_stoich_products" not in content:
        vk_target = "    def _activity_products(self, conc, orders):"
        vk_patch = '''    @staticmethod
    def _general_stoich_products(stoich_matrix, orders):
        stoich_matrix = np.asarray(stoich_matrix, dtype=float)
        orders = np.asarray(orders, dtype=float)
        n_rx = stoich_matrix.shape[0]
        general = np.zeros(n_rx, dtype=bool)
        for i in range(n_rx):
            s = stoich_matrix[i]
            o = orders[i]
            if not np.array_equal(s, o):
                general[i] = True
                continue
            nonzeros = o[o > 0]
            if len(nonzeros) != np.count_nonzero(o):
                general[i] = True
                continue
            if not np.all(np.equal(np.mod(nonzeros, 1), 0)):
                general[i] = True
                continue
            total_order = int(np.sum(nonzeros))
            if total_order not in (1, 2, 3):
                general[i] = True
                continue
            general[i] = False
        return general

    def _activity_products(self, conc, orders, general=None):
        conc = np.asarray(conc, dtype=float)
        orders = np.asarray(orders, dtype=float)
        if general is not None:
            general = np.asarray(general, dtype=bool)
            rates = np.zeros(orders.shape[0], dtype=float)
            for i in range(orders.shape[0]):
                o = orders[i]
                if not general[i]:
                    p = 1.0
                    for c_val, order_val in zip(conc, o):
                        if order_val == 1.0:
                            p *= c_val
                        elif order_val == 2.0:
                            p *= (c_val * c_val)
                        elif order_val == 3.0:
                            p *= (c_val * c_val * c_val)
                    rates[i] = p
                else:
                    active_idx = np.where(o != 0.0)[0]
                    if np.any(conc[active_idx] <= 0.0):
                        rates[i] = 0.0
                    else:
                        rates[i] = np.prod(conc[active_idx] ** o[active_idx])
            return rates
'''
        content = content.replace(vk_target, vk_patch + "    def _legacy_activity_products(self, conc, orders):")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("kinetics.py patched.")

def fix_ideal_gas():
    path = "cantera_python/ideal_gas.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. IdealGasMixture Y property supporting _y_nonorm
    content = content.replace(
        "    @property\n    def Y(self):\n        return tuple(x * s.molecular_weight / self.mean_molecular_weight\n                     for x, s in zip(self.X, self.species))",
        "    @property\n    def Y(self):\n        if getattr(self, '_y_nonorm', None) is not None:\n            return tuple(self._y_nonorm)\n        return tuple(x * s.molecular_weight / self.mean_molecular_weight\n                     for x, s in zip(self.X, self.species))"
    )

    # 2. IdealGasPhase __init__ storing _elements and _y_nonorm
    content = content.replace(
        "    def __init__(self, species, *, temperature=298.15, pressure=OneAtm, mole_fractions=None, species_compositions=None, name=\"\"):\n        self.name = name\n        self._state = IdealGasMixture(species, temperature=temperature, pressure=pressure, mole_fractions=mole_fractions)\n        self._species_compositions = dict(species_compositions or {})",
        "    def __init__(self, species, *, temperature=298.15, pressure=OneAtm, mole_fractions=None, species_compositions=None, name=\"\", elements=None):\n        self.name = name\n        self._state = IdealGasMixture(species, temperature=temperature, pressure=pressure, mole_fractions=mole_fractions)\n        self._species_compositions = dict(species_compositions or {})\n        self._elements = list(elements) if elements is not None else None\n        self._y_nonorm = None"
    )

    # 3. IdealGasPhase from_mechanism passing elements
    content = content.replace(
        "        p = cls(species, temperature=t, pressure=p, mole_fractions=x, species_compositions=species_compositions, name=mech.name)",
        "        elems = None\n        if p_obj is not None and getattr(p_obj, 'input_data', None) and 'elements' in p_obj.input_data:\n            elems = list(p_obj.input_data['elements'])\n        p = cls(species, temperature=t, pressure=p, mole_fractions=x, species_compositions=species_compositions, name=mech.name, elements=elems)"
    )

    # 4. IdealGasPhase element_names preserving YAML order or insertion order
    content = content.replace(
        "    @property\n    def element_names(self) -> list[str]:\n        elements = set()\n        for comp in self._species_compositions.values():\n            elements.update(comp.keys())\n        return sorted(elements)",
        "    @property\n    def element_names(self) -> list[str]:\n        if getattr(self, '_elements', None) is not None:\n            return list(self._elements)\n        elements = []\n        for comp in self._species_compositions.values():\n            for e in comp:\n                if e not in elements:\n                    elements.append(e)\n        return elements"
    )

    # 5. IdealGasPhase Y getter and setMassFractions_NoNorm
    content = content.replace(
        "    @property\n    def Y(self):\n        return self._state.Y",
        "    @property\n    def Y(self):\n        if getattr(self, '_y_nonorm', None) is not None:\n            return np.asarray(self._y_nonorm, dtype=float)\n        return np.asarray(self._state.Y, dtype=float)"
    )

    content = content.replace(
        "    def setMassFractions_NoNorm(self, y):\n        self.Y = y",
        """    def setMassFractions_NoNorm(self, y):
        y_arr = np.asarray(y, dtype=float)
        self._y_nonorm = y_arr
        inv_mw = np.array([1.0 / s.molecular_weight for s in self._state.species], dtype=float)
        ymol = y_arr * inv_mw
        sum_ymol = float(np.sum(ymol))
        if sum_ymol > 0:
            mmw = 1.0 / sum_ymol
            x = ymol * mmw
        else:
            x = np.ones(len(y_arr)) / len(y_arr)
        self._state = self._state.at(mole_fractions=x)
        self._state._y_nonorm = tuple(y_arr)"""
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("ideal_gas.py patched.")

if __name__ == "__main__":
    fix_rates()
    fix_surface()
    fix_kinetics()
    fix_ideal_gas()
    print("All regression patches applied.")
