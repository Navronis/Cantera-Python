from pathlib import Path

fpath = Path("cantera_python/rates.py")
text = fpath.read_text(encoding="utf-8")

# 1. Add CanteraError import
text = text.replace(
    "from .constants import Avogadro, ElectronCharge, ElectronMass",
    "from .constants import Avogadro, ElectronCharge, ElectronMass, CanteraError"
)

# 2. Add type() method to CustomFunc1Rate
text = text.replace(
    '    type = "custom-rate-function"\n',
    '    type = "custom-rate-function"\n    def type(self) -> str:\n        return "custom-rate-function"\n'
)

fpath.write_text(text, encoding="utf-8")
print("SUCCESS: patched rates.py")
