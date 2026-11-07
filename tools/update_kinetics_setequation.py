with open('cantera_python/kinetics.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_block = '''        from .reaction_equation import parse_reaction_equation
        eq_str = str(eq).strip()
        if "<=>" in eq_str:
            self.reversible = True
        elif "=>" in eq_str:
            self.reversible = False
        elif "=" in eq_str:
            self.reversible = True

        r_comp, p_comp, tbody = parse_reaction_equation(eq_str)
        self._reactants = r_comp
        self._products = p_comp
        if tbody:
            self._third_body = tbody
        self.check()'''

new_block = '''        from .reaction_equation import parse_reaction_equation
        eq_str = str(eq).strip()
        parsed = parse_reaction_equation(eq_str)
        self._reactants = dict(parsed.reactants)
        self._products = dict(parsed.products)
        self.reversible = bool(parsed.reversible)
        if hasattr(parsed, 'third_body') and parsed.third_body:
            self._third_body = parsed.third_body
        self.check()'''

text_norm = text.replace('\r\n', '\n')
if old_block not in text_norm:
    print('ERROR: old_block not found in kinetics.py')
else:
    updated = text_norm.replace(old_block, new_block, 1)
    with open('cantera_python/kinetics.py', 'w', encoding='utf-8') as f:
        f.write(updated)
    print('SUCCESS: updated setEquation in kinetics.py')
