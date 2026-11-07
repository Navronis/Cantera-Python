with open('cantera_python/kinetics.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    def reaction(self, i):
        idx = int(i)
        if idx < 0 or idx >= len(self._reactions):
            raise IndexError(f"Reaction index {i} out of bounds")
        return self._reactions[idx]'''

addition = '''    def reaction(self, i):
        idx = int(i)
        if idx < 0 or idx >= len(self._reactions):
            raise IndexError(f"Reaction index {i} out of bounds")
        rxn = self._reactions[idx]
        if isinstance(rxn, Reaction):
            return rxn
        r = Reaction()
        r.reactants = dict(rxn.reactants)
        r.products = dict(rxn.products)
        r.reversible = bool(rxn.reversible)
        r.duplicate = bool(getattr(rxn, "duplicate", False))
        r._third_body = getattr(rxn, "third_body", None)
        r._rate = getattr(rxn, "rate", None)
        r.orders = dict(getattr(rxn, "orders", None) or {})
        return r'''

assert target in text
text = text.replace(target, addition, 1)

with open('cantera_python/kinetics.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('SUCCESS: updated BulkKinetics.reaction')
