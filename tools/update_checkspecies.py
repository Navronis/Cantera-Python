with open('cantera_python/kinetics.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        for sp in list(self._reactants.keys()) + list(self._products.keys()) + list(self._orders.keys()):
            idx = kin.kineticsSpeciesIndex(sp, False) if hasattr(kin, "kineticsSpeciesIndex") else (kin.species_names.index(sp) if sp in kin.species_names else None)
            if idx is None or idx < 0:
                undeclared.append(sp)'''

replacement = '''        for sp in list(self._reactants.keys()) + list(self._products.keys()) + list(self._orders.keys()):
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

assert target in text
updated = text.replace(target, replacement, 1)
with open('cantera_python/kinetics.py', 'w', encoding='utf-8') as f:
    f.write(updated)
print("SUCCESS: updated checkSpecies in kinetics.py")
