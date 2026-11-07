from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
kinetics_file = ROOT / "cantera_python" / "kinetics.py"
rates_file = ROOT / "cantera_python" / "rates.py"

# 1. Update reaction(self, i) in kinetics.py
text = kinetics_file.read_text(encoding="utf-8")

target = """    def reaction(self, i):
        return self._reactions[i]"""

replacement = """    def reaction(self, i):
        idx = int(i)
        if idx < 0 or idx >= len(self._reactions):
            raise IndexError(f"Reaction index {i} out of bounds")
        rxn = self._reactions[idx]
        if isinstance(rxn, Reaction):
            return rxn
        r = Reaction()
        r.reactants = dict(rxn.reactants)
        r.products = dict(rxn.products)
        r.reversible = bool(rxn.reversible)
        r.duplicate = bool(getattr(rxn, "duplicate", False))
        r._third_body = getattr(rxn, "third_body", None)
        r._rate = getattr(rxn, "rate", None)
        r.orders = dict(getattr(rxn, "orders", None) or {})
        return r"""

if target in text:
    text = text.replace(target, replacement, 1)
    print("Updated reaction(self, i)")

# 2. Remove duplicate stub Group ... ReactionPathDiagram block
lines = text.splitlines(keepends=True)
second_group_idx = -1
stoich_idx = -1
for i, line in enumerate(lines):
    if line.startswith("class Group:") and i > 2000:
        second_group_idx = i
    if line.startswith("# Stoichiometry managers:"):
        stoich_idx = i

if second_group_idx != -1 and stoich_idx != -1:
    print(f"Removing duplicate block lines {second_group_idx+1} to {stoich_idx+1}")
    lines = lines[:second_group_idx] + lines[stoich_idx:]
    text = "".join(lines)

kinetics_file.write_text(text, encoding="utf-8")
print("Saved kinetics.py")

# 3. Update ReactionRateFactory and newReactionRate in rates.py
rates_text = rates_file.read_text(encoding="utf-8")

rates_target = """class ReactionRateFactory:
    \"\"\"Factory for reaction rates.\"\"\"
    _instance = None
    @classmethod
    def factory(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance
    @classmethod
    def deleteFactory(cls): cls._instance = None


def newReactionRate(*args, **kwargs) -> ReactionRate:
    \"\"\"Create a new reaction rate object.\"\"\"
    return ReactionRate()"""

rates_replacement = """class ReactionRateFactory:
    \"\"\"Factory for reaction rates.\"\"\"
    _instance = None
    @classmethod
    def factory(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance
    @classmethod
    def deleteFactory(cls): cls._instance = None

    def newReactionRate(self, rate_type="Arrhenius", *args, **kwargs):
        return newReactionRate(rate_type, *args, **kwargs)


def newReactionRate(rate_type="Arrhenius", *args, **kwargs) -> ReactionRate:
    \"\"\"Create a new reaction rate object.\"\"\"
    if isinstance(rate_type, dict):
        rate_type = rate_type.get("type", "Arrhenius")
    t = str(rate_type).lower().replace("_", "-")
    if t in ("", "arrhenius", "elementary", "three-body"):
        return ArrheniusRate(0.0, 0.0, 0.0)
    elif t in ("blowers-masel", "blowersmasel"):
        return BlowersMaselRate(0.0, 0.0, 0.0, 0.0)
    elif t in ("two-temperature-plasma", "twotempplasma"):
        return TwoTempPlasmaRate()
    elif t in ("electron-collision-plasma", "electroncollisionplasma"):
        return ElectronCollisionPlasmaRate()
    elif t in ("constant-vibrational-relaxation", "constantvibrationalrelaxation"):
        return ConstantVibrationalRelaxationRate()
    elif t in ("starikovskiy-vibrational-relaxation", "starikovskiyvibrationalrelaxation"):
        return StarikovskiyVibrationalRelaxationRate()
    elif t in ("castela-vibrational-relaxation", "castelavibrationalrelaxation"):
        return CastelaVibrationalRelaxationRate()
    elif t in ("multi-state-resolved-vibrational-relaxation", "multistateresolvedvibrationalrelaxation"):
        return MultiStateResolvedVibrationalRelaxationRate()
    elif t in ("custom-rate-function", "customfunc1"):
        return CustomFunc1Rate()
    elif t in ("lindemann", "falloff"):
        return LindemannRate()
    elif t == "troe":
        return TroeRate()
    elif t == "sri":
        return SriRate()
    elif t == "tsang":
        return TsangRate()
    elif t == "chebyshev":
        return ChebyshevRate()
    elif t in ("plog", "pressure-dependent-arrhenius"):
        return PlogRate()
    elif t == "linear-burke":
        return LinearBurkeRate()
    elif t == "interface-arrhenius":
        return InterfaceArrheniusRate()
    elif t == "sticking-arrhenius":
        return StickingArrheniusRate()
    elif t == "interface-blowers-masel":
        return InterfaceBlowersMaselRate()
    elif t == "sticking-blowers-masel":
        return StickingBlowersMaselRate()
    return ArrheniusRate(0.0, 0.0, 0.0)"""

if rates_target in rates_text:
    rates_text = rates_text.replace(rates_target, rates_replacement, 1)
    rates_file.write_text(rates_text, encoding="utf-8")
    print("Saved rates.py")
else:
    print("rates_target not found in rates.py")
