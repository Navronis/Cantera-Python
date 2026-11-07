# Fix elements.py
with open('cantera_python/elements.py', 'r', encoding='utf-8') as f:
    c = f.read()

old_elem = '''def getElementSymbol(*args) -> str:
    if not args: return "H"
    if isinstance(args[0], int):
        idx = args[0]
        return str(COMMON_ELEMENTS[idx].symbol) if idx < len(COMMON_ELEMENTS) else "H"
    return str(args[0])

def getElementName(*args) -> str:
    if not args: return "Hydrogen"
    if isinstance(args[0], int):
        idx = args[0]
        return str(COMMON_ELEMENTS[idx].name or COMMON_ELEMENTS[idx].symbol) if idx < len(COMMON_ELEMENTS) else "Hydrogen"
    s = str(args[0])
    for e in COMMON_ELEMENTS:
        if e.symbol == s: return str(e.name or e.symbol)
    return s

def getAtomicNumber(sym: str) -> int:
    return 1'''

new_elem = '''def getElementSymbol(*args) -> str:
    if not args: return "H"
    if isinstance(args[0], int):
        idx = args[0]
        return str(COMMON_ELEMENTS[idx].symbol) if idx < len(COMMON_ELEMENTS) else "H"
    s = str(args[0])
    try:
        e = COMMON_ELEMENTS.get(s)
        return e.symbol
    except Exception:
        for e in COMMON_ELEMENTS:
            if e.symbol.lower() == s.lower() or (e.name and e.name.lower() == s.lower()):
                return e.symbol
        return s

def getElementName(*args) -> str:
    if not args: return "Hydrogen"
    if isinstance(args[0], int):
        idx = args[0]
        return str(COMMON_ELEMENTS[idx].name or COMMON_ELEMENTS[idx].symbol) if idx < len(COMMON_ELEMENTS) else "Hydrogen"
    s = str(args[0])
    try:
        e = COMMON_ELEMENTS.get(s)
        return str(e.name or e.symbol)
    except Exception:
        for e in COMMON_ELEMENTS:
            if e.symbol.lower() == s.lower() or (e.name and e.name.lower() == s.lower()):
                return str(e.name or e.symbol)
        return s

def getAtomicNumber(sym: str) -> int:
    s = str(sym)
    for i, e in enumerate(COMMON_ELEMENTS):
        if e.symbol.lower() == s.lower() or (e.name and e.name.lower() == s.lower()):
            return i + 1
    return 1'''

assert old_elem in c, "old_elem not found"
c = c.replace(old_elem, new_elem)
with open('cantera_python/elements.py', 'w', encoding='utf-8') as f:
    f.write(c)
print("elements.py patched.")

# Fix thermo.py (Nasa9Poly temperaturePolySize)
with open('cantera_python/thermo.py', 'r', encoding='utf-8') as f:
    c = f.read()

if "class Nasa9Poly1" in c and "def temperaturePolySize" not in c[c.find("class Nasa9Poly1"):c.find("class Nasa9PolyMultiTempRegion")]:
    c = c.replace(
        "    def properties(self, temperature):\n        \"\"\"Translate Nasa9Poly1::updateTemperaturePoly",
        "    def temperaturePolySize(self) -> int:\n        return 7\n\n    def properties(self, temperature):\n        \"\"\"Translate Nasa9Poly1::updateTemperaturePoly"
    )

if "class Nasa9PolyMultiTempRegion" in c and "def temperaturePolySize" not in c[c.find("class Nasa9PolyMultiTempRegion"):c.find("class ShomatePoly")]:
    c = c.replace(
        "    def properties(self, temperature):\n        temperature = _temperature(temperature)",
        "    def temperaturePolySize(self) -> int:\n        return 7\n\n    def properties(self, temperature):\n        temperature = _temperature(temperature)"
    )

with open('cantera_python/thermo.py', 'w', encoding='utf-8') as f:
    f.write(c)
print("thermo.py patched.")

# Fix multi_species_thermo.py (refPressure)
with open('cantera_python/multi_species_thermo.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace(
    "    def refPressure(self):\n        return self._p0",
    "    def refPressure(self):\n        return getattr(self, '_p0', 101325.0) if getattr(self, '_p0', 0.0) > 0.0 else 101325.0"
)

with open('cantera_python/multi_species_thermo.py', 'w', encoding='utf-8') as f:
    f.write(c)
print("multi_species_thermo.py patched.")

# Fix ideal_gas.py (_species_set_molecular_weight on frozen dataclass)
with open('cantera_python/ideal_gas.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace(
    "def _species_set_molecular_weight(self, val):\n    self.molecular_weight = val",
    "def _species_set_molecular_weight(self, val):\n    object.__setattr__(self, 'molecular_weight', float(val))"
)

with open('cantera_python/ideal_gas.py', 'w', encoding='utf-8') as f:
    f.write(c)
print("ideal_gas.py patched.")
