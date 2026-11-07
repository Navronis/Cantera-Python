path = 'outputs/cantera-python/cantera_python/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_ac = """    def advanceCoverages(self, dt: float):
        \"\"\"Integrate surface coverages forward by timestep dt.\"\"\"
        dt = float(dt)
        if dt <= 0:
            return
        sdot = self.net_production_rates()
        surf_sp_count = self._phases[0].n_species
        covs = np.asarray(self._phases[0].coverages, dtype=float)
        sd = self.siteDensity()
        for i in range(surf_sp_count):
            covs[i] += (sdot[i] / sd) * dt
        covs = np.maximum(0.0, covs)
        total_cov = np.sum(covs)
        if total_cov > 0:
            covs /= total_cov
        self._phases[0].coverages = covs"""

new_ac = """    def advanceCoverages(self, dt: float, rtol=1e-7, atol=1e-14, max_step_size=0.0, max_steps=20000, max_error_test_failures=7):
        \"\"\"Integrate surface coverages forward by timestep dt.\"\"\"
        dt = float(dt)
        if dt <= 0:
            return
        if hasattr(self._phases[0], 'advance_coverages'):
            return self._phases[0].advance_coverages(
                dt, rtol=rtol, atol=atol, max_step_size=max_step_size,
                max_steps=max_steps, max_error_test_failures=max_error_test_failures
            )
        sdot = self.net_production_rates()
        surf_sp_count = self._phases[0].n_species
        covs = np.asarray(self._phases[0].coverages, dtype=float)
        sd = self.siteDensity()
        for i in range(surf_sp_count):
            covs[i] += (sdot[i] / sd) * dt
        covs = np.maximum(0.0, covs)
        total_cov = np.sum(covs)
        if total_cov > 0:
            covs /= total_cov
        self._phases[0].coverages = covs"""

if old_ac in content:
    content = content.replace(old_ac, new_ac, 1)
    print("Updated advanceCoverages")
else:
    print("old_ac not found")

old_pss = """def _ik_solvePseudoSteadyStateProblem(self, max_steps=1000, rtol=1e-5, atol=1e-10):
    self.advance_coverages_to_steady_state(max_steps=max_steps)
    return True"""

new_pss = """def _ik_solvePseudoSteadyStateProblem(self, loglevel: int = 0, max_steps=1000, rtol=1e-5, atol=1e-10):
    if hasattr(self._phases[0], 'advance_coverages_to_steady_state'):
        return self._phases[0].advance_coverages_to_steady_state(loglevel=loglevel, max_steps=max_steps)
    return True"""

if old_pss in content:
    content = content.replace(old_pss, new_pss, 1)
    print("Updated _ik_solvePseudoSteadyStateProblem")
else:
    print("old_pss not found")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Done patching.")
