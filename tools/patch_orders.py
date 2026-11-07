path = 'outputs/cantera-python/cantera_python/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_orders = """        rate_data = r_rec.rate_data or {}
        custom_orders = dict(rate_data.get('orders', {}) or {})

        # Map reactants and products
        reactants_list = []
        for sp_name, stoich in eq.reactants.items():
            gk = self.kinetics_species_index(sp_name)
            canon_name = self._global_species_names[gk]
            order = float(custom_orders.get(sp_name, custom_orders.get(canon_name, stoich)))
            reactants_list.append((gk, float(stoich), order))"""

new_orders = """        rate_data = r_rec.rate_data or {}
        raw_orders = dict(rate_data.get('orders', {}) or {})
        custom_orders = {}
        for k, v in raw_orders.items():
            gk = self.kinetics_species_index(k)
            canon = self._global_species_names[gk]
            custom_orders[canon] = v
            custom_orders[k] = v
            custom_orders[k.lower()] = v

        # Map reactants and products
        reactants_list = []
        for sp_name, stoich in eq.reactants.items():
            gk = self.kinetics_species_index(sp_name)
            canon_name = self._global_species_names[gk]
            order = float(custom_orders.get(canon_name, custom_orders.get(sp_name, custom_orders.get(sp_name.lower(), stoich))))
            reactants_list.append((gk, float(stoich), order))"""

if old_orders in content:
    content = content.replace(old_orders, new_orders, 1)
    print("Updated custom_orders parsing")
else:
    print("old_orders not found")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Finished patching orders.")
