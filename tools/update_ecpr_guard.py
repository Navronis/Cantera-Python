from pathlib import Path

fpath = Path("cantera_python/kinetics.py")
content = fpath.read_text(encoding="utf-8")

old_1 = """        if isinstance(rate, ElectronCollisionPlasmaRate):
            state = self._live_equilibrium_state or self.rate_thermo or self.equilibrium_thermo
            levels = getattr(state, 'electron_energy_levels', ()) if state else ()
            dist = getattr(state, 'electron_energy_distribution', ()) if state else ()
            return rate(levels, dist)"""

new_1 = """        if isinstance(rate, ElectronCollisionPlasmaRate):
            state = self._live_equilibrium_state or self.rate_thermo or self.equilibrium_thermo
            levels = getattr(state, 'electron_energy_levels', ()) if state else ()
            dist = getattr(state, 'electron_energy_distribution', ()) if state else ()
            if not levels or not dist or len(levels) < 2 or len(dist) < 2:
                return 0.0
            return rate(levels, dist)"""

old_2 = """            elif type(r.rate).__name__ == "ElectronCollisionPlasmaRate":
                levels = getattr(thermo, 'electron_energy_levels', ())
                dist = getattr(thermo, 'electron_energy_distribution', ())
                out[i] = r.rate(levels, dist)"""

new_2 = """            elif type(r.rate).__name__ == "ElectronCollisionPlasmaRate":
                levels = getattr(thermo, 'electron_energy_levels', ())
                dist = getattr(thermo, 'electron_energy_distribution', ())
                if not levels or not dist or len(levels) < 2 or len(dist) < 2:
                    out[i] = 0.0
                else:
                    out[i] = r.rate(levels, dist)"""

content_norm = content.replace("\r\n", "\n")
assert old_1 in content_norm, "old_1 not found"
assert old_2 in content_norm, "old_2 not found"
content_norm = content_norm.replace(old_1, new_1, 1).replace(old_2, new_2, 1)
fpath.write_text(content_norm, encoding="utf-8")
print("SUCCESS: updated ElectronCollisionPlasmaRate guards in kinetics.py")
