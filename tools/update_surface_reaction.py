with open('cantera_python/surface.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_block = '''                elif t_model in {'ideal-condensed', 'ideal-solid-solution', 'ideal_solid'}:
                    from .ideal_solid import IdealSolidSolution
                    adj_p = mech.phase(adj_name)
                    mol_vols = [getattr(s, 'molar_volume', 1.0) for s in adj_p.species] if adj_p.species else []
                    adjacent_dict[adj_name] = IdealSolidSolution(adj_p.species, mol_vols)'''

new_block = '''                elif t_model in {'ideal-condensed', 'ideal-solid-solution', 'ideal_solid'}:
                    from .ideal_solid import IdealSolidSolution
                    from .ideal_gas import Species
                    from .units import convert
                    adj_p = mech.phase(adj_name)
                    species_objs = []
                    mol_vols = []
                    for s in adj_p.species:
                        poly = create_species_thermo(s.thermo) if s.thermo else None
                        species_objs.append(Species(s.name, s.molecular_weight, poly, s.charge))
                        v = 1.0
                        if s.equation_of_state and 'molar-density' in s.equation_of_state:
                            val_str = str(s.equation_of_state['molar-density']).strip()
                            parts = val_str.split(maxsplit=1)
                            if len(parts) == 2:
                                dens = convert(float(parts[0]), parts[1], 'kmol/m^3')
                                v = 1.0 / dens if dens > 0 else 1.0
                            else:
                                v = 1.0 / float(parts[0]) if float(parts[0]) > 0 else 1.0
                        mol_vols.append(v)
                    adjacent_dict[adj_name] = IdealSolidSolution(species_objs, mol_vols)'''

if old_block not in text:
    print('ERROR: old_block not found in surface.py')
else:
    updated = text.replace(old_block, new_block, 1)
    with open('cantera_python/surface.py', 'w', encoding='utf-8') as f:
        f.write(updated)
    print('SUCCESS: updated ideal_solid adjacent handling in surface.py')
