path = 'outputs/cantera-python/cantera_python/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

patch_code = """
InterfaceKinetics.multiplier = _ik_multiplier
Interface.multiplier = _ik_multiplier
Interface.advanceCoverages = Interface.advance_coverages
Interface.solveCoverages = Interface.advance_coverages_to_steady_state
"""

if "Interface.advanceCoverages" not in content:
    content = content + "\n" + patch_code
    print("Added multiplier and advanceCoverages bindings")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
