with open("cantera_python/elements.py", "r", encoding="utf-8") as f:
    text = f.read()

target = """    @property
    def symbols(self):
        return tuple(self._elements)"""

replace = """    def __iter__(self):
        return iter(self._elements.values())

    def __len__(self):
        return len(self._elements)

    def __getitem__(self, item):
        if isinstance(item, int):
            return list(self._elements.values())[item]
        return self.get(str(item))

    @property
    def symbols(self):
        return tuple(self._elements)"""

if "\r\n" in text:
    target = target.replace("\n", "\r\n")
    replace = replace.replace("\n", "\r\n")

if target in text:
    text = text.replace(target, replace, 1)
    with open("cantera_python/elements.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("ElementRegistry updated with __iter__, __len__, and __getitem__")
else:
    print("Warning: target not found in elements.py")
