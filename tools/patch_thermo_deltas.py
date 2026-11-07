path = 'outputs/cantera-python/cantera_python/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update IdealSurfacePhase chemical_potentials and partial_molar_entropies
old_cp = """    @property
    def chemical_potentials(self):
        return tuple(mu + R * self.T * log(max(theta, SmallNumber))
                     for mu, theta in zip(self.standard_chemical_potentials, self.coverages))
    @property
    def partial_molar_entropies(self):
        return tuple(R * (p[2] - log(max(theta, SmallNumber)))
                     for p, theta in zip(self._reference(), self.coverages))"""

new_cp = """    @property
    def chemical_potentials(self):
        c = self.concentrations
        sd = self.site_density
        RT = R * self.T
        res = []
        for k, (mu0, sp) in enumerate(zip(self.standard_chemical_potentials, self.species)):
            c0 = sd / sp.size
            res.append(mu0 + RT * (log(max(c[k], SmallNumber)) - log(c0)))
        return tuple(res)
    @property
    def partial_molar_entropies(self):
        c = self.concentrations
        sd = self.site_density
        T = self.T
        res = []
        for k, (mu0, h, sp) in enumerate(zip(self.standard_chemical_potentials, self.partial_molar_enthalpies, self.species)):
            s0 = (h - mu0) / T
            c0 = sd / sp.size
            res.append(s0 - R * (log(max(c[k], SmallNumber)) - log(c0)))
        return tuple(res)"""

if old_cp in content:
    content = content.replace(old_cp, new_cp, 1)
    print("Updated chemical_potentials and partial_molar_entropies")
else:
    print("old_cp not found")

# 2. Add delta methods and helpers to Interface
delta_methods = """
    def _all_chemical_potentials(self):
        mu = []
        for p_idx, phase in enumerate(self._phases):
            if p_idx == 0:
                mu.extend(self.chemical_potentials)
            else:
                mu.extend(phase.chemical_potentials)
        return mu

    def _all_standard_chemical_potentials(self):
        mu0 = []
        for p_idx, phase in enumerate(self._phases):
            if p_idx == 0:
                mu0.extend(self.standard_chemical_potentials)
            else:
                mu0.extend(phase.standard_chemical_potentials)
        return mu0

    def _all_partial_molar_enthalpies(self):
        h = []
        for p_idx, phase in enumerate(self._phases):
            if p_idx == 0:
                h.extend(self.partial_molar_enthalpies)
            else:
                h.extend(phase.partial_molar_enthalpies)
        return h

    def _all_partial_molar_entropies(self):
        s = []
        for p_idx, phase in enumerate(self._phases):
            if p_idx == 0:
                s.extend(self.partial_molar_entropies)
            else:
                s.extend(phase.partial_molar_entropies)
        return s

    @property
    def delta_gibbs(self):
        mu = self._all_chemical_potentials()
        return tuple(sum(stoich * mu[gk] for gk, stoich in r.net_stoich.items()) for r in self._reactions)

    @property
    def delta_standard_gibbs(self):
        mu0 = self._all_standard_chemical_potentials()
        return tuple(sum(stoich * mu0[gk] for gk, stoich in r.net_stoich.items()) for r in self._reactions)

    @property
    def delta_enthalpy(self):
        h = self._all_partial_molar_enthalpies()
        return tuple(sum(stoich * h[gk] for gk, stoich in r.net_stoich.items()) for r in self._reactions)

    @property
    def delta_standard_enthalpy(self):
        return self.delta_enthalpy

    @property
    def delta_entropy(self):
        s = self._all_partial_molar_entropies()
        return tuple(sum(stoich * s[gk] for gk, stoich in r.net_stoich.items()) for r in self._reactions)

    @property
    def delta_standard_entropy(self):
        dh = self.delta_standard_enthalpy
        dg0 = self.delta_standard_gibbs
        T = self.T
        return tuple((dh[i] - dg0[i]) / T for i in range(len(dh)))

    def getDeltaGibbs(self, dg=None):
        res = np.asarray(self.delta_gibbs, dtype=float)
        if dg is not None:
            dg[:] = res
            return dg
        return res

    def getDeltaEnthalpy(self, dh=None):
        res = np.asarray(self.delta_enthalpy, dtype=float)
        if dh is not None:
            dh[:] = res
            return dh
        return res

    def getDeltaEntropy(self, ds=None):
        res = np.asarray(self.delta_entropy, dtype=float)
        if ds is not None:
            ds[:] = res
            return ds
        return res

    def getDeltaSSGibbs(self, dg0=None):
        res = np.asarray(self.delta_standard_gibbs, dtype=float)
        if dg0 is not None:
            dg0[:] = res
            return dg0
        return res

    def getDeltaSSEnthalpy(self, dh0=None):
        res = np.asarray(self.delta_standard_enthalpy, dtype=float)
        if dh0 is not None:
            dh0[:] = res
            return dh0
        return res

    def getDeltaSSEntropy(self, ds0=None):
        res = np.asarray(self.delta_standard_entropy, dtype=float)
        if ds0 is not None:
            ds0[:] = res
            return ds0
        return res

    def getDeltaElectrochemPotentials(self, dm=None):
        from .constants import Faraday
        dg = np.asarray(self.delta_gibbs, dtype=float)
        res = dg.copy()
        phis = [getattr(p, 'electric_potential', 0.0) for p in self._phases]
        for i, r in enumerate(self._reactions):
            d_phi_z = 0.0
            for gk, stoich in r.products:
                p_idx, loc_k = self._global_to_phase_local[gk]
                p = self._phases[p_idx]
                z = float(p.species[loc_k].charge if hasattr(p, 'species') and loc_k < len(p.species) and hasattr(p.species[loc_k], 'charge') else getattr(p, 'charge', lambda idx: 0.0)(loc_k))
                d_phi_z += stoich * z * phis[p_idx]
            for gk, stoich, _ in r.reactants:
                p_idx, loc_k = self._global_to_phase_local[gk]
                p = self._phases[p_idx]
                z = float(p.species[loc_k].charge if hasattr(p, 'species') and loc_k < len(p.species) and hasattr(p.species[loc_k], 'charge') else getattr(p, 'charge', lambda idx: 0.0)(loc_k))
                d_phi_z -= stoich * z * phis[p_idx]
            res[i] += Faraday * d_phi_z
        if dm is not None:
            dm[:] = res
            return dm
        return res
"""

target = "    def _blowers_masel_coefficient(self, reaction, T):"
if target in content:
    content = content.replace(target, delta_methods + "\n" + target, 1)
    print("Added delta methods to Interface")
else:
    print("target not found for delta methods")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Finished patching thermo deltas")
