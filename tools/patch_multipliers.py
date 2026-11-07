path = 'outputs/cantera-python/cantera_python/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update _ik_setMultiplier
old_sm = """def _ik_setMultiplier(self, i: int, val: float):
    self._multipliers[i] = float(val)"""

new_sm = """def _ik_setMultiplier(self, i: int, val: float):
    if not hasattr(self, '_multipliers') or self._multipliers is None:
        self._multipliers = {}
    self._multipliers[int(i)] = float(val)

def _ik_multiplier(self, i: int) -> float:
    return getattr(self, '_multipliers', {}).get(int(i), 1.0)"""

if old_sm in content:
    content = content.replace(old_sm, new_sm, 1)
    print("Updated _ik_setMultiplier")
else:
    print("old_sm not found")

# 2. Update forward_rate_constants to apply multipliers
old_fwd = """    @property
    def forward_rate_constants(self):
        T = self.T
        gamma_tot = self.site_density
        rates = []
        for rxn in self._reactions:
            p = rxn.rate_params
            cov_factor = self._eval_coverage_factor(p.get('cov_deps'), T)
            if rxn.rate_type == 'interface-Arrhenius':
                k0 = p['A'] * (T ** p['b']) * exp(-p['Ea'] / (R * T))
                rates.append(k0 * cov_factor)
            elif rxn.rate_type == 'interface-Blowers-Masel':
                rates.append(self._blowers_masel_coefficient(rxn, T) * cov_factor)
            elif rxn.rate_type == 'sticking-Arrhenius':
                gamma = p['A'] * (T ** p['b']) * exp(-p['Ea'] / (R * T)) * cov_factor
                if p['motz_wise']:
                    gamma = gamma / (1.0 - 0.5 * gamma)
                kf = gamma * p['multiplier'] * (gamma_tot ** (-p['surface_order'])) * sqrt(T)
                rates.append(kf)
            elif rxn.rate_type == 'sticking-Blowers-Masel':
                gamma = self._blowers_masel_coefficient(rxn, T) * cov_factor
                if p['motz_wise']:
                    gamma = gamma / (1.0 - 0.5 * gamma)
                rates.append(gamma * p['multiplier']
                             * (gamma_tot ** (-p['surface_order'])) * sqrt(T))
        return tuple(rates)"""

new_fwd = """    @property
    def forward_rate_constants(self):
        T = self.T
        gamma_tot = self.site_density
        rates = []
        mults = getattr(self, '_multipliers', {}) or {}
        for j, rxn in enumerate(self._reactions):
            m = mults.get(j, 1.0)
            p = rxn.rate_params
            cov_factor = self._eval_coverage_factor(p.get('cov_deps'), T)
            if rxn.rate_type == 'interface-Arrhenius':
                k0 = p['A'] * (T ** p['b']) * exp(-p['Ea'] / (R * T))
                rates.append(k0 * cov_factor * m)
            elif rxn.rate_type == 'interface-Blowers-Masel':
                rates.append(self._blowers_masel_coefficient(rxn, T) * cov_factor * m)
            elif rxn.rate_type == 'sticking-Arrhenius':
                gamma = p['A'] * (T ** p['b']) * exp(-p['Ea'] / (R * T)) * cov_factor
                if p['motz_wise']:
                    gamma = gamma / (1.0 - 0.5 * gamma)
                kf = gamma * p['multiplier'] * (gamma_tot ** (-p['surface_order'])) * sqrt(T)
                rates.append(kf * m)
            elif rxn.rate_type == 'sticking-Blowers-Masel':
                gamma = self._blowers_masel_coefficient(rxn, T) * cov_factor
                if p['motz_wise']:
                    gamma = gamma / (1.0 - 0.5 * gamma)
                rates.append(gamma * p['multiplier']
                             * (gamma_tot ** (-p['surface_order'])) * sqrt(T) * m)
        return tuple(rates)"""

if old_fwd in content:
    content = content.replace(old_fwd, new_fwd, 1)
    print("Updated forward_rate_constants with multipliers")
else:
    print("old_fwd not found")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Finished patching multipliers")
