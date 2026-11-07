with open('cantera_python/surface.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''        reactants = {}
        for item in irxn.reactants:
            gk = item[0]
            stoich = item[1]
            sp_name = self.total_species_names[gk] if gk < len(self.total_species_names) else str(gk)
            reactants[sp_name] = float(stoich)
        products = {}
        for item in irxn.products:
            gk = item[0]
            stoich = item[1]
            sp_name = self.total_species_names[gk] if gk < len(self.total_species_names) else str(gk)
            products[sp_name] = float(stoich)
        r.reactants = reactants
        r.products = products'''

replacement = '''        if not r.reactants:
            reactants = {}
            for item in irxn.reactants:
                gk = item[0]
                stoich = item[1]
                sp_name = self.total_species_names[gk] if gk < len(self.total_species_names) else str(gk)
                reactants[sp_name] = float(stoich)
            r.reactants = reactants
        if not r.products:
            products = {}
            for item in irxn.products:
                gk = item[0]
                stoich = item[1]
                sp_name = self.total_species_names[gk] if gk < len(self.total_species_names) else str(gk)
                products[sp_name] = float(stoich)
            r.products = products'''

assert target in text
updated = text.replace(target, replacement, 1)
with open('cantera_python/surface.py', 'w', encoding='utf-8') as f:
    f.write(updated)
print("SUCCESS: updated surface.py reaction preservation")
