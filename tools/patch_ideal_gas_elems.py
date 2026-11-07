with open('cantera_python/ideal_gas.py', 'r', encoding='utf-8') as f:
    c = f.read()

target = """        p_name = p_obj.name if (mech.phases and p_obj is not None) else (phase_name or "")
        phase = cls(gas_species, temperature=T, pressure=P,
                    mole_fractions=mole_fractions,
                    species_compositions=comp, name=p_name)"""

replacement = """        p_name = p_obj.name if (mech.phases and p_obj is not None) else (phase_name or "")
        elems = None
        if p_obj is not None and getattr(p_obj, 'input_data', None) and 'elements' in p_obj.input_data:
            elems = list(p_obj.input_data['elements'])
        phase = cls(gas_species, temperature=T, pressure=P,
                    mole_fractions=mole_fractions,
                    species_compositions=comp, name=p_name, elements=elems)"""

assert target in c, "target not found in ideal_gas.py"
c = c.replace(target, replacement)
with open('cantera_python/ideal_gas.py', 'w', encoding='utf-8') as f:
    f.write(c)
print("Successfully patched IdealGasPhase.from_mechanism.")
