import re

path = 'outputs/cantera-python/cantera_python/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update kinetics_species_index to allow case-insensitive fallback
old_ksi = """        if isinstance(species_name_or_k, str):
            try:
                return self._global_species_names.index(species_name_or_k)
            except ValueError:
                raise ValueError(f"Species {species_name_or_k!r} not found in interface phases")"""

new_ksi = """        if isinstance(species_name_or_k, str):
            try:
                return self._global_species_names.index(species_name_or_k)
            except ValueError:
                lower_names = [s.lower() for s in self._global_species_names]
                try:
                    return lower_names.index(species_name_or_k.lower())
                except ValueError:
                    raise ValueError(f"Species {species_name_or_k!r} not found in interface phases")"""

if old_ksi in content:
    content = content.replace(old_ksi, new_ksi, 1)
    print("Updated kinetics_species_index")
else:
    print("old_ksi not found")

# 2. Update custom_orders mapping in _parse_reaction_record
old_co = """        # Map reactants and products
        reactants_list = []
        for sp_name, stoich in eq.reactants.items():
            gk = self.kinetics_species_index(sp_name)
            order = float(custom_orders.get(sp_name, stoich))
            reactants_list.append((gk, float(stoich), order))"""

new_co = """        # Map reactants and products
        reactants_list = []
        for sp_name, stoich in eq.reactants.items():
            gk = self.kinetics_species_index(sp_name)
            canon_name = self._global_species_names[gk]
            order = float(custom_orders.get(sp_name, custom_orders.get(canon_name, stoich)))
            reactants_list.append((gk, float(stoich), order))"""

if old_co in content:
    content = content.replace(old_co, new_co, 1)
    print("Updated custom_orders")
else:
    print("old_co not found")

# 3. Update _ik_interfaceCurrent to accept iphase=0 matching native C++
old_ic = """def _ik_interfaceCurrent(self) -> float:
    rop = np.asarray(self.net_rates_of_progress(), dtype=float)
    charge_transferred = [0.0] * len(self._reactions)
    for i, r in enumerate(self._reactions):
        q = 0.0
        for gk, stoich in r.products:
            q += stoich * self.charge(gk)
        for gk, stoich, _ in r.reactants:
            q -= stoich * self.charge(gk)
        charge_transferred[i] = q
    curr = float(np.dot(rop, charge_transferred) * 96485332.12)
    return curr"""

new_ic = """def _ik_interfaceCurrent(self, iphase: int = 0) -> float:
    from .constants import Faraday
    p = self._phases[iphase]
    net_prod = np.asarray(self.net_production_rates(), dtype=float)
    start = sum(ph.n_species for ph in self._phases[:iphase])
    curr = 0.0
    for k in range(p.n_species):
        ch = float(p.species[k].charge if hasattr(p, 'species') and k < len(p.species) and hasattr(p.species[k], 'charge') else getattr(p, 'charge', lambda idx: 0.0)(k))
        curr += ch * net_prod[start + k]
    return float(curr * Faraday)"""

if old_ic in content:
    content = content.replace(old_ic, new_ic, 1)
    print("Updated _ik_interfaceCurrent")
else:
    print("old_ic not found")

# 4. Support ideal-condensed adjacent phases in from_mechanism
old_tm = """                elif t_model in {'ideal-surface', 'surface', 'surf'}:
                    adjacent_dict[adj_name] = cls.from_mechanism(mech, adj_name)
                else:
                    raise ValueError(f"Unsupported adjacent phase thermo model: {t_model}")"""

new_tm = """                elif t_model in {'ideal-surface', 'surface', 'surf'}:
                    adjacent_dict[adj_name] = cls.from_mechanism(mech, adj_name)
                elif t_model in {'ideal-condensed', 'ideal-solid-solution', 'ideal_solid'}:
                    from .ideal_solid import IdealSolidSolution
                    adj_p = mech.phase(adj_name)
                    mol_vols = [getattr(s, 'molar_volume', 1.0) for s in adj_p.species] if adj_p.species else []
                    adjacent_dict[adj_name] = IdealSolidSolution(adj_p.species, mol_vols)
                else:
                    raise ValueError(f"Unsupported adjacent phase thermo model: {t_model}")"""

if old_tm in content:
    content = content.replace(old_tm, new_tm, 1)
    print("Updated t_model support")
else:
    print("old_tm not found")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Finished patching surface.py")
