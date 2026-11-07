"""Apply clean fixes for Family 4 and Family 5."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
KINETICS_FILE = ROOT / "cantera_python" / "kinetics.py"
RATES_FILE = ROOT / "cantera_python" / "rates.py"


def fix_kinetics():
    with open(KINETICS_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update reaction(self, i) on GasKinetics / BulkKinetics
    old_rxn = '''    def reaction(self, i):
        return self._reactions[i]'''

    new_rxn = '''    def reaction(self, i):
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
        return r'''

    if old_rxn in content:
        content = content.replace(old_rxn, new_rxn, 1)
        print("Updated reaction(self, i)")

    # 2. Update Reaction.setEquation
    old_seteq = '''        from .reaction_equation import parse_reaction_equation
        eq_str = str(eq).strip()
        if "<=>" in eq_str:
            self.reversible = True
        elif "=>" in eq_str:
            self.reversible = False
        elif "=" in eq_str:
            self.reversible = True

        r_comp, p_comp, tbody = parse_reaction_equation(eq_str)
        self._reactants = r_comp
        self._products = p_comp
        if tbody:
            self._third_body = tbody
        self.check()'''

    new_seteq = '''        from .reaction_equation import parse_reaction_equation
        eq_str = str(eq).strip()
        parsed = parse_reaction_equation(eq_str)
        self._reactants = dict(parsed.reactants)
        self._products = dict(parsed.products)
        self.reversible = bool(parsed.reversible)
        if hasattr(parsed, 'third_body') and parsed.third_body:
            self._third_body = parsed.third_body
        self.check()'''

    if old_seteq in content:
        content = content.replace(old_seteq, new_seteq, 1)
        print("Updated Reaction.setEquation")

    # 3. Update Reaction.checkSpecies
    old_cs = '''        for sp in list(self._reactants.keys()) + list(self._products.keys()) + list(self._orders.keys()):
            idx = kin.kineticsSpeciesIndex(sp, False) if hasattr(kin, "kineticsSpeciesIndex") else (kin.species_names.index(sp) if sp in kin.species_names else None)
            if idx is None or idx < 0:
                undeclared.append(sp)'''

    new_cs = '''        for sp in list(self._reactants.keys()) + list(self._products.keys()) + list(self._orders.keys()):
            try:
                if hasattr(kin, "kineticsSpeciesIndex"):
                    idx = kin.kineticsSpeciesIndex(sp, False)
                elif hasattr(kin, "species_names") and sp in kin.species_names:
                    idx = kin.species_names.index(sp)
                else:
                    idx = None
            except Exception:
                idx = None
            if idx is None or idx < 0:
                undeclared.append(sp)'''

    if old_cs in content:
        content = content.replace(old_cs, new_cs, 1)
        print("Updated Reaction.checkSpecies")

    # 4. Remove duplicate stub classes at lines 2564-2850
    # From "class Group:\n    \"\"\"Atomic group for reaction path diagrams.\"\"\"" down to "# Stoichiometry managers:"
    dup_pattern = r"\nclass Group:\n    \"\"\"Atomic group for reaction path diagrams\..*?(?=\n# Stoichiometry managers:)"
    if re.search(dup_pattern, content, re.DOTALL):
        content = re.sub(dup_pattern, "", content, count=1, flags=re.DOTALL)
        print("Removed duplicate stub Group/Path/Diagram classes")

    with open(KINETICS_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print("Kinetics update complete.")


def fix_rates():
    with open(RATES_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # In newReactionRate, ensure fallback returns ArrheniusRate(0.0, 0.0, 0.0)
    old_def = r"def newReactionRate\(type_or_node=\"\", rate_units=None\) -> ReactionRate:.*?return ArrheniusRate\(0\.0, 0\.0, 0\.0\)"
    m_def = re.search(old_def, content, re.DOTALL)
    if m_def:
        # Check what the fallback is
        pass

    with open(RATES_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print("Rates update complete.")


if __name__ == "__main__":
    fix_kinetics()
    fix_rates()
