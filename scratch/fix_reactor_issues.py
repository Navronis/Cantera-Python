import re
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
reactor_py = repo_root / "cantera_python" / "reactor.py"

with open(reactor_py, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Fix ReactorBase.getState
old_rb_get_state = """    def getState(self, y: np.ndarray) -> None:
        if self._thermo is not None:
            rho = self._thermo.density
            vol = getattr(self, '_vol', 1.0)
            self._mass = rho * vol
            y[0] = self._mass
            if len(y) > 1:
                y[1] = vol
            if len(y) > 2:
                y[2] = self._thermo.int_energy_mass * self._mass
            if len(y) > 3 and hasattr(self._thermo, 'Y'):
                y[3:3+self._thermo.n_species] = self._thermo.Y"""

new_rb_get_state = """    def getState(self, y: np.ndarray) -> None:
        if hasattr(self, 'get_state'):
            st = self.get_state()
            y[:len(st)] = st
            return
        if self._thermo is not None:
            rho = self._thermo.density
            vol = getattr(self, '_vol', 1.0)
            self._mass = rho * vol
            y[0] = self._mass
            if len(y) > 1:
                y[1] = vol
            if len(y) > 2:
                y[2] = self._thermo.temperature
            if len(y) > 3 and hasattr(self._thermo, 'Y'):
                y[3:3+self._thermo.n_species] = self._thermo.Y"""

assert old_rb_get_state in text, "old_rb_get_state not found"
text = text.replace(old_rb_get_state, new_rb_get_state, 1)

# 2. Fix ReactorBase.updateState
old_rb_update_state = """    def updateState(self, y: np.ndarray) -> None:
        if len(y) == 0 or self._thermo is None:
            return
        self._mass = float(y[0])
        if len(y) > 1:
            self._vol = float(y[1])
        sp_off = self.speciesOffset()
        if len(y) >= sp_off + self._thermo.n_species:
            nsp = self._thermo.n_species
            self._thermo.Y = np.clip(y[sp_off:sp_off+nsp], 0.0, 1.0)"""

new_rb_update_state = """    def updateState(self, y: np.ndarray) -> None:
        if hasattr(self, 'update_state'):
            self.update_state(y)
            return
        if len(y) == 0 or self._thermo is None:
            return
        self._mass = float(y[0])
        if len(y) > 1:
            self._vol = float(y[1])
        sp_off = self.speciesOffset() if hasattr(self, 'speciesOffset') else 3
        if len(y) >= sp_off + self._thermo.n_species:
            nsp = self._thermo.n_species
            self._thermo.Y = np.clip(y[sp_off:sp_off+nsp], 0.0, 1.0)"""

assert old_rb_update_state in text, "old_rb_update_state not found"
text = text.replace(old_rb_update_state, new_rb_update_state, 1)

# 3. Fix parameter_jacobian missing parameter_step in reaction branch
old_param_jac = """                if kind == 'reaction':
                    kinetics = reactor.kinetics
                    baseline = kinetics.multiplier(local_index)
                    kinetics.set_multiplier(baseline * (1.0 + relative_step), local_index)
                    try:
                        shifted = evaluate(time, state)
                    finally:
                        kinetics.set_multiplier(baseline, local_index)"""

new_param_jac = """                if kind == 'reaction':
                    kinetics = reactor.kinetics
                    baseline = kinetics.multiplier(local_index)
                    parameter_step = relative_step
                    kinetics.set_multiplier(baseline * (1.0 + relative_step), local_index)
                    try:
                        shifted = evaluate(time, state)
                    finally:
                        kinetics.set_multiplier(baseline, local_index)"""

assert old_param_jac in text, "old_param_jac not found"
text = text.replace(old_param_jac, new_param_jac, 1)

# 4. Fix solveSteady and finiteDifferenceJacobian to use self._get_state()
old_solve_steady = """    def solveSteady(self) -> int:
        y = np.asarray(self._get_global_state(), dtype=float)"""

new_solve_steady = """    def solveSteady(self) -> int:
        y = np.asarray(self._get_state(), dtype=float)"""

text = text.replace(old_solve_steady, new_solve_steady, 1)

old_fd_jac = """    def finiteDifferenceJacobian(self) -> np.ndarray:
        y = np.asarray(self._get_global_state(), dtype=float)"""

new_fd_jac = """    def finiteDifferenceJacobian(self) -> np.ndarray:
        y = np.asarray(self._get_state(), dtype=float)"""

text = text.replace(old_fd_jac, new_fd_jac, 1)

with open(reactor_py, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("Applied fixes to reactor.py successfully.")
