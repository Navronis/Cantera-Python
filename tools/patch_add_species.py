with open("cantera_python/phases.py", "r", encoding="utf-8") as f:
    text = f.read()

target = """        self._mass_fractions = np.append(self._mass_fractions, 0.0)
        self.compositionChanged()
        return True"""

replace = """        self._mass_fractions = np.append(self._mass_fractions, 0.0)
        if not hasattr(self, "_species"):
            self._species = []
        self._species.append(spec)
        self.compositionChanged()
        return True"""

if "\r\n" in text:
    target = target.replace("\n", "\r\n")
    replace = replace.replace("\n", "\r\n")

if target in text:
    text = text.replace(target, replace, 1)
    with open("cantera_python/phases.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("addSpecies updated with self._species append")
else:
    print("Warning: target not found")
