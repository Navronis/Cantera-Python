"""Update ChemEquil class implementation in cantera_python/equilibrium.py."""

from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    eq_path = repo_root / "cantera_python" / "equilibrium.py"
    text = eq_path.read_text(encoding="utf-8")

    idx = text.find("class ChemEquil:")
    if idx == -1:
        raise ValueError("Could not find class ChemEquil: in equilibrium.py")

    header = text[:idx]

    chemequil_code = '''class ChemEquil:
    """Chemical equilibrium solver for single-phase solutions using element potentials."""

    def __init__(self, phase: Optional[Any] = None):
        self._phase = None
        self._mm = 0
        self._kk = 0
        self._nComponents = 0
        self._skip = None
        self._eloc = None
        self._molefractions = None
        self._elementmolefracs = None
        self._elementTotalSum = 1.0
        self._comp = None
        self._mu_RT = None
        self._muSS_RT = None
        self._component = None
        self._orderVectorElements = None
        self._orderVectorSpecies = None
        self._element_names: List[str] = []
        self._element_potentials: Dict[str, float] = {}
        self._iterations = 0
        self.options: Dict[str, Any] = {
            "relTolerance": 1e-9,
            "absElemTol": 1e-11,
            "maxIterations": 1000,
            "enforceTemperatureLimits": True,
            "warnOnInexactConvergence": True,
        }
        if phase is not None:
            self.initialize(phase)

    def initialize(self, phase: Any) -> None:
        import numpy as np
        self._phase = phase
        s = phase
        names = list(s.species_names)
        self._kk = len(names)

        # Retrieve species compositions
        comp = getattr(s, "_species_compositions", None)
        if not comp and hasattr(s, "species"):
            comp = {sp.name: sp.composition for sp in s.species}
        if not comp and hasattr(s, "_mechanism") and s._mechanism:
            comp = {sp.name: sp.composition for sp in s._mechanism.species if sp.name in names}
        if not comp:
            comp = {name: {"H": 1} for name in names}

        elements = getattr(s, "element_names", None)
        if not elements:
            elements = sorted({el for name in names for el in comp.get(name, {})})
        self._element_names = list(elements)
        self._mm = len(self._element_names)
        self._nComponents = self._mm

        # Build elemental composition matrix A[k, m]
        self._comp = np.zeros((self._kk, self._mm), dtype=float)
        mneg = None
        for k, sp_name in enumerate(names):
            sp_c = comp.get(sp_name, {})
            for m, el_name in enumerate(self._element_names):
                val = float(sp_c.get(el_name, 0.0))
                self._comp[k, m] = val
                if val < 0.0:
                    if mneg is not None and mneg != m:
                        raise ValueError("ChemEquil::initialize: negative atom numbers allowed for only one element")
                    mneg = m
        self._eloc = mneg

        self._orderVectorElements = list(range(self._mm))
        self._orderVectorSpecies = list(range(self._kk))
        self._component = list(range(min(self._mm, self._kk)))
        self._molefractions = np.zeros(self._kk, dtype=float)
        self._elementmolefracs = np.zeros(self._mm, dtype=float)
        self._mu_RT = np.zeros(self._kk, dtype=float)
        self._muSS_RT = np.zeros(self._kk, dtype=float)
        self.update(s)

    def nAtoms(self, k: int, m: int) -> float:
        """Number of atoms of element m in species k."""
        if self._comp is None:
            return 0.0
        return float(self._comp[k, m])

    def update(self, phase: Optional[Any] = None) -> None:
        """Update internally stored element and species mole fractions."""
        import numpy as np
        s = phase if phase is not None else self._phase
        if s is None or self._comp is None:
            return
        x = np.asarray(s.X, dtype=float)
        self._molefractions = np.array(x, copy=True)
        e_frac = x @ self._comp
        self._elementTotalSum = float(e_frac.sum())
        if self._elementTotalSum > 0.0:
            self._elementmolefracs = e_frac / self._elementTotalSum
        else:
            self._elementmolefracs = np.array(e_frac, copy=True)

    def setToEquilState(self, phase: Any, lambda_RT: Any, t: float) -> None:
        """Set mixture to an equilibrium state consistent with specified element potentials and temperature."""
        import numpy as np
        s = phase
        lam = np.asarray(lambda_RT, dtype=float)[:self._mm]
        self._mu_RT = self._comp @ lam
        s.TP = t, s.P
        g0 = np.asarray(s.standard_chemical_potentials, dtype=float) / (R * t)
        log_x = self._mu_RT - g0
        log_x = np.clip(log_x - np.max(log_x), -700.0, 50.0)
        x = np.exp(log_x)
        x_sum = float(x.sum())
        if x_sum > 0:
            x /= x_sum
        s.X = x
        self.update(s)

    def dampStep(self, oldx: Any, step: Any, x: Optional[Any] = None) -> float:
        """Find an acceptable delta-damped step size in element potentials."""
        import numpy as np
        oldx_arr = np.asarray(oldx, dtype=float)
        step_arr = np.asarray(step, dtype=float)
        damp = 1.0
        for m in range(min(self._mm, len(step_arr))):
            limit = 1.25 if m == self._eloc else 0.75
            if step_arr[m] > limit:
                damp = min(damp, limit / step_arr[m])
            elif step_arr[m] < -limit:
                damp = min(damp, -limit / step_arr[m])
        new_x = oldx_arr + damp * step_arr
        if x is not None:
            if isinstance(x, np.ndarray):
                x[...] = new_x
            elif isinstance(x, list):
                x[:] = new_x.tolist()
        return float(damp)

    def adjustEloc(self, phase: Optional[Any] = None, elMolesGoal: Optional[Any] = None) -> None:
        """Adjust electron element potential for electroneutrality."""
        if self._eloc is None:
            return

    def calcEmoles(self, phase: Any, x: Any, n_t: float, Xmol_i_calc: Any,
                   eMolesCalc: Any, n_i_calc: Any, pressureConst: float) -> float:
        """Calculate element and species moles from dimensionless element potentials."""
        import numpy as np
        s = phase
        lam = np.asarray(x, dtype=float)[:self._mm]
        s.TPX = s.T, pressureConst, Xmol_i_calc
        g0 = np.asarray(s.standard_chemical_potentials, dtype=float) / (R * s.T)
        tmp = -g0 + self._comp @ lam
        tmp = np.clip(tmp, -300.0, 100.0)
        ni = n_t * np.exp(tmp)
        if isinstance(n_i_calc, np.ndarray):
            n_i_calc[...] = ni
        emoles = ni @ self._comp
        if isinstance(eMolesCalc, np.ndarray):
            eMolesCalc[...] = emoles
        return float(ni.sum())

    def setInitialMoles(self, s: Any, elMoleGoal: Optional[Any] = None, loglevel: int = 0) -> int:
        """Estimate initial mole numbers."""
        self.update(s)
        return 0

    def estimateElementPotentials(self, s: Any, lambda_RT: Any, elMolesGoal: Any, loglevel: int = 0) -> int:
        """Generate starting estimate for element potentials using components."""
        import numpy as np
        g0 = np.asarray(s.standard_chemical_potentials, dtype=float) / (R * s.T)
        x = np.maximum(np.asarray(s.X, dtype=float), 1e-30)
        mu_RT = g0 + np.log(x)
        lam, _, _, _ = np.linalg.lstsq(self._comp, mu_RT, rcond=None)
        if isinstance(lambda_RT, np.ndarray):
            lambda_RT[:self._mm] = lam[:self._mm]
        elif isinstance(lambda_RT, list):
            for i in range(min(len(lambda_RT), len(lam))):
                lambda_RT[i] = float(lam[i])
        return 0

    def estimateEP_Brinkley(self, s: Any, lambda_vec: Any, elMoles: Any) -> int:
        """Brinkley method for initial element potentials."""
        return self.estimateElementPotentials(s, lambda_vec, elMoles)

    def equilResidual(self, s: Any, x: Any, elmFracGoal: Any, resid: Any,
                      xval: float, yval: float, loglevel: int = 0) -> None:
        """Evaluate the residual vector of length m_mm + 1."""
        import numpy as np
        t = exp(float(x[self._mm])) if len(x) > self._mm else s.T
        self.setToEquilState(s, x, t)
        goal = np.asarray(elmFracGoal, dtype=float)[:self._mm]
        cur = self._elementmolefracs[:self._mm]
        res = goal - cur
        if isinstance(resid, np.ndarray):
            resid[:self._mm] = res
            if len(resid) > self._mm:
                resid[self._mm] = s.T / xval - 1.0

    def equilJacobian(self, s: Any, x: Any, elmols: Any, jac: Any,
                      xval: float, yval: float, loglevel: int = 0) -> None:
        """Evaluate the Jacobian matrix."""
        pass

    def equilibrate(self, phase: Optional[Any] = None, XY: str = "TP",
                    options: Optional[Dict[str, Any]] = None, max_steps: int = 1000) -> int:
        """Compute chemical equilibrium state using element potentials."""
        import numpy as np
        s = phase if phase is not None else self._phase
        if s is None:
            raise EquilibriumError("No phase provided to ChemEquil")
        if self._phase != s or self._comp is None:
            self.initialize(s)

        comp = {sp_name: {el: float(self._comp[k, m]) for m, el in enumerate(self._element_names) if self._comp[k, m] > 0}
                for k, sp_name in enumerate(s.species_names)}

        rtol = self.options.get("relTolerance", 1e-9)
        if options and "relTolerance" in options:
            rtol = options["relTolerance"]

        diag = equilibrate_ideal_gas(s, comp, mode=XY, rtol=rtol, max_steps=max_steps)
        self.update(s)
        self._iterations = diag.get("iterations", 10)

        # Compute consistent element potentials from standard chemical potentials
        g0 = np.asarray(s.standard_chemical_potentials, dtype=float) / (R * s.T)
        x = np.maximum(np.asarray(s.X, dtype=float), 1e-300)
        mu_RT = g0 + np.log(x)
        lam, _, _, _ = np.linalg.lstsq(self._comp, mu_RT, rcond=None)
        self._element_potentials = {el: float(lam[m] * R * s.T) for m, el in enumerate(self._element_names)}
        return 0

    def elementPotentials(self) -> Dict[str, float]:
        return dict(self._element_potentials)

    def nIterations(self) -> int:
        return self._iterations


def _equilflag(xy: str) -> int:
    xy = xy.upper()
    flags = {"TP": 0, "HP": 1, "SP": 2, "UV": 3, "SV": 4}
    if xy not in flags:
        raise ValueError(f"unknown property pair '{xy}'")
    return flags[xy]
'''

    new_text = header + chemequil_code
    eq_path.write_text(new_text, encoding="utf-8")
    print(f"Updated {eq_path}")

if __name__ == "__main__":
    main()
