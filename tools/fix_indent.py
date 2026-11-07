from pathlib import Path

fpath = Path("cantera_python/kinetics.py")
content = fpath.read_text(encoding="utf-8")

old_block = """            elif isinstance(r.rate, BlowersMaselRate):
                    r.rate.update_from_thermo(thermo, r.reactants, r.products)
                    out[i] = r.rate(T)
                elif type(r.rate).__name__ == "TwoTempPlasmaRate":
                    Te = getattr(thermo, 'electron_temperature', getattr(thermo, 'electronTemperature', T))
                    out[i] = r.rate(T, float(Te))
                elif type(r.rate).__name__ == "ElectronCollisionPlasmaRate":
                    levels = getattr(thermo, 'electron_energy_levels', ())
                    dist = getattr(thermo, 'electron_energy_distribution', ())
                    out[i] = r.rate(levels, dist)
                else:
                    out[i] = r.rate(T)"""

new_block = """            elif isinstance(r.rate, BlowersMaselRate):
                r.rate.update_from_thermo(thermo, r.reactants, r.products)
                out[i] = r.rate(T)
            elif type(r.rate).__name__ == "TwoTempPlasmaRate":
                Te = getattr(thermo, 'electron_temperature', getattr(thermo, 'electronTemperature', T))
                out[i] = r.rate(T, float(Te))
            elif type(r.rate).__name__ == "ElectronCollisionPlasmaRate":
                levels = getattr(thermo, 'electron_energy_levels', ())
                dist = getattr(thermo, 'electron_energy_distribution', ())
                out[i] = r.rate(levels, dist)
            else:
                out[i] = r.rate(T)"""

# normalize line endings
content_norm = content.replace("\r\n", "\n")
assert old_block in content_norm, "old_block not found"
content_norm = content_norm.replace(old_block, new_block, 1)
fpath.write_text(content_norm, encoding="utf-8")
print("SUCCESS: fixed indentation in kinetics.py")
