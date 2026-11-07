from pathlib import Path

path = Path("outputs/cantera-python/cantera_python/transport.py")
content = path.read_text(encoding="utf-8")

old = 'sp = self._thermo.species(k) if hasattr(self._thermo, "species") else None'
new = '''sp = None
                if hasattr(self._thermo, "species"):
                    sp_attr = self._thermo.species
                    if callable(sp_attr):
                        sp = sp_attr(k)
                    elif isinstance(sp_attr, (list, tuple)) and k < len(sp_attr):
                        sp = sp_attr[k]'''

assert old in content, "old not found"
content = content.replace(old, new, 1)
path.write_text(content, encoding="utf-8")
print("Successfully patched species access in transport.py")
