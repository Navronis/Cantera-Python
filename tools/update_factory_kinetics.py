with open('cantera_python/kinetics.py', 'r', encoding='utf-8') as f:
    text_kin = f.read()

# 1. Update BulkKinetics.kineticsType
text_kin = text_kin.replace(
    '    def kineticsType(self) -> str:\n        return "BulkKinetics"',
    '    def kineticsType(self) -> str:\n        return "bulk"'
)

# 2. Update EdgeKinetics.kineticsType
text_kin = text_kin.replace(
    'class EdgeKinetics(InterfaceKinetics):\n    """Heterogeneous kinetics on 1D edge boundaries."""\n    def kineticsType(self) -> str:\n        return "Edge"',
    'class EdgeKinetics(InterfaceKinetics):\n    """Heterogeneous kinetics on 1D edge boundaries."""\n    def kineticsType(self) -> str:\n        return "edge"'
)

# 3. Update GasKinetics.__init__
old_gas_init = '''    def __init__(self, thermo, reactions=None):
        self._reactions = tuple(reactions or ())
        self._thermo = thermo
        self._fast = _VectorizedKinetics(thermo, self._reactions)
        self._multipliers = np.ones(len(self._reactions), dtype=float)'''

new_gas_init = '''    def __init__(self, thermo=None, reactions=None):
        self._reactions = tuple(reactions or ())
        self._thermo = thermo
        if thermo is not None:
            self._fast = _VectorizedKinetics(thermo, self._reactions)
        else:
            self._fast = None
        self._multipliers = np.ones(len(self._reactions), dtype=float)'''

assert old_gas_init in text_kin
text_kin = text_kin.replace(old_gas_init, new_gas_init, 1)

with open('cantera_python/kinetics.py', 'w', encoding='utf-8') as f:
    f.write(text_kin)
print("SUCCESS: updated kinetics.py")

with open('cantera_python/surface.py', 'r', encoding='utf-8') as f:
    text_surf = f.read()

# 4. Update InterfaceKinetics.kineticsType
text_surf = text_surf.replace(
    '    def kineticsType(self) -> str:\n        return "InterfaceKinetics"',
    '    def kineticsType(self) -> str:\n        return "surface"'
)

# 5. Support empty InterfaceKinetics()
old_ik_init = '''        else:
            super().__init__([], 1.0, reactions=reactions, **kwargs)'''

new_ik_init = '''        else:
            self.thermo_model = 'ideal-surface'
            self.kinetics_model = 'surface'
            self.transport_model = 'none'
            self.composite = ('ideal-surface', 'surface', 'none')
            self.adjacent = {}
            self._phases = ()
            self._species = []
            self._reactions = tuple(reactions or ())
            self._species_names = []
            self._global_species_names = []
            self._site_density = 1.0
            self._coverages = np.array([])
            self._multipliers = np.ones(len(self._reactions), dtype=float)'''

assert old_ik_init in text_surf
text_surf = text_surf.replace(old_ik_init, new_ik_init, 1)

with open('cantera_python/surface.py', 'w', encoding='utf-8') as f:
    f.write(text_surf)
print("SUCCESS: updated surface.py")
