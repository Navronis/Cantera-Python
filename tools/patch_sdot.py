with open('outputs/cantera-python/cantera_python/oned.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """        sdot = np.zeros(self.m_kin.n_total_species, dtype=float)
        if hasattr(self.m_kin, "get_net_production_rates"):
            sdot = np.array(self.m_kin.get_net_production_rates(), dtype=float)
        elif hasattr(self.m_kin, "net_production_rates"):
            sdot = np.array(self.m_kin.net_production_rates, dtype=float)"""

replacement = """        if hasattr(self.m_kin, "net_production_rates"):
            sdot = np.array(self.m_kin.net_production_rates, dtype=float)
        elif hasattr(self.m_kin, "get_net_production_rates"):
            sdot = np.array(self.m_kin.get_net_production_rates(self.m_sphase), dtype=float)
        else:
            sdot = np.zeros(self.m_kin.n_total_species, dtype=float)"""

assert target in text, "target not found"
text = text.replace(target, replacement)
with open('outputs/cantera-python/cantera_python/oned.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Patched sdot in ReactingSurf1D.eval successfully")
