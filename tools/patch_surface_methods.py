path = 'outputs/cantera-python/cantera_python/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_block = """def _ik_getDeltaGibbs(self, dg=None):
    res = np.asarray(self.delta_gibbs(), dtype=float)
    if dg is not None:
        dg[:] = res
        return dg
    return res

def _ik_getDeltaEnthalpy(self, dh=None):
    res = np.asarray(self.delta_enthalpy(), dtype=float)
    if dh is not None:
        dh[:] = res
        return dh
    return res

def _ik_getDeltaEntropy(self, ds=None):
    res = np.asarray(self.delta_entropy(), dtype=float)
    if ds is not None:
        ds[:] = res
        return ds
    return res

def _ik_getDeltaSSGibbs(self, dg0=None):
    return _ik_getDeltaGibbs(self, dg0)

def _ik_getDeltaSSEnthalpy(self, dh0=None):
    return _ik_getDeltaEnthalpy(self, dh0)

def _ik_getDeltaSSEntropy(self, ds0=None):
    return _ik_getDeltaEntropy(self, ds0)

def _ik_getDeltaElectrochemPotentials(self, dmu=None):
    return _ik_getDeltaGibbs(self, dmu)"""

new_block = """def _ik_getDeltaGibbs(self, dg=None):
    prop = getattr(self, 'delta_gibbs', None)
    res = np.asarray(prop() if callable(prop) else prop, dtype=float)
    if dg is not None:
        dg[:] = res
        return dg
    return res

def _ik_getDeltaEnthalpy(self, dh=None):
    prop = getattr(self, 'delta_enthalpy', None)
    res = np.asarray(prop() if callable(prop) else prop, dtype=float)
    if dh is not None:
        dh[:] = res
        return dh
    return res

def _ik_getDeltaEntropy(self, ds=None):
    prop = getattr(self, 'delta_entropy', None)
    res = np.asarray(prop() if callable(prop) else prop, dtype=float)
    if ds is not None:
        ds[:] = res
        return ds
    return res

def _ik_getDeltaSSGibbs(self, dg0=None):
    prop = getattr(self, 'delta_standard_gibbs', None)
    res = np.asarray(prop() if callable(prop) else prop, dtype=float)
    if dg0 is not None:
        dg0[:] = res
        return dg0
    return res

def _ik_getDeltaSSEnthalpy(self, dh0=None):
    prop = getattr(self, 'delta_standard_enthalpy', None)
    res = np.asarray(prop() if callable(prop) else prop, dtype=float)
    if dh0 is not None:
        dh0[:] = res
        return dh0
    return res

def _ik_getDeltaSSEntropy(self, ds0=None):
    prop = getattr(self, 'delta_standard_entropy', None)
    res = np.asarray(prop() if callable(prop) else prop, dtype=float)
    if ds0 is not None:
        ds0[:] = res
        return ds0
    return res

def _ik_getDeltaElectrochemPotentials(self, dmu=None):
    if hasattr(self, 'getDeltaElectrochemPotentials') and self.getDeltaElectrochemPotentials.__func__ is not _ik_getDeltaElectrochemPotentials:
        return self.getDeltaElectrochemPotentials(dmu)
    return _ik_getDeltaGibbs(self, dmu)"""

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    print("Replaced delta block")
else:
    print("old_block not found")

# Bind methods to Interface as well
append_code = """
for _name in [
    'updateMu0', 'setDerivativeSettings', 'getDerivativeSettings',
    'setPhaseStability', 'phaseStability', 'setPhaseExistence', 'phaseExistence',
    'buildNetwork', 'updateROP', 'updateKc', 'applyEquilibriumConstants',
    'getDeltaGibbs', 'getDeltaEnthalpy', 'getDeltaEntropy',
    'getDeltaSSGibbs', 'getDeltaSSEnthalpy', 'getDeltaSSEntropy',
    'getDeltaElectrochemPotentials', 'getEquilibriumConstants',
    'getFwdRateConstants', 'getRevRateConstants', 'interfaceCurrent',
    'solvePseudoSteadyStateProblem', 'modifyReaction', 'addReaction',
    'addThermo', 'setMultiplier', 'assertDerivativesValid',
    'resizeSpecies', 'resizeReactions', 'init',
    '_update_rates_C', '_update_rates_T', '_update_rates_phi',
    'revRatesOfProgress_ddCi', 'netRatesOfProgress_ddCi', 'fwdRatesOfProgress_ddCi'
]:
    if not hasattr(Interface, _name) or getattr(Interface, _name) is None:
        setattr(Interface, _name, getattr(InterfaceKinetics, _name, None))
"""

if "Interface.interfaceCurrent" not in content:
    content = content + "\n" + append_code
    print("Appended Interface method bindings")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Finished patching surface.py")
