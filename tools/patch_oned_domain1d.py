"""Cleanly patch Domain1D in cantera_python/oned.py to match native Cantera C++ semantics.
"""

from pathlib import Path


def patch_domain1d():
    repo_root = Path(__file__).resolve().parent.parent
    oned_py = repo_root / "cantera_python" / "oned.py"
    content = oned_py.read_text(encoding="utf-8")

    # Locate start of old Domain1D stubs (around line 324) and end of old Domain1D (before Boundary1D)
    start_marker = "class Domain1D:"
    boundary_marker = "class Boundary1D(Domain1D):"

    start_idx = content.find(start_marker)
    boundary_idx = content.find(boundary_marker)

    if start_idx == -1 or boundary_idx == -1:
        raise RuntimeError("Could not find Domain1D markers in oned.py")

    new_domain1d_code = '''class Domain1D:
    """Base class for one-dimensional domains matching native Cantera C++ Domain1D."""

    _domain_type: str = "domain"

    def __init__(self, nv: int = 1, points: int = 1, time: float = 0.0, name: str = ""):
        self.m_nv = 0
        self.m_points = 0
        self._name = str(name)
        self.m_id = str(name)
        self.m_container = None
        self.m_left = None
        self.m_right = None
        self.m_jstart = 0
        self.m_index = 0
        self.m_bw = -1
        self.m_iloc = 0
        self.m_loc = 0
        self.m_rdt = 0.0
        self.m_force_full_update = False
        self.m_state = None
        self.m_refiner = None
        self.m_jacobian_mode = "auto"
        self.have_user_tolerances = False
        self.resize(nv, points)

    _ALLOWED_ATTRS = {
        "gas", "phase", "name", "id", "grid", "time", "container",
        "m_flow_left", "m_flow_right", "have_user_tolerances",
        "nsp", "nv", "left_excess_species", "right_excess_species",
    }

    def __setattr__(self, name: str, value: Any):
        if (
            name.startswith("_")
            or name.startswith("m_")
            or name in self._ALLOWED_ATTRS
            or hasattr(type(self), name)
            or name in self.__dict__
        ):
            super().__setattr__(name, value)
            return
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def domain_type(self) -> str:
        return self.domainType()

    def domainType(self) -> str:
        return self._domain_type

    def domainIndex(self) -> int:
        return int(self.m_index)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: str):
        self._name = str(val)
        self.m_id = str(val)

    def id(self) -> str:
        if self.m_id != "":
            return self.m_id
        return f"domain {self.m_index}"

    def setID(self, val: str):
        self.m_id = str(val)
        self._name = str(val)

    @property
    def is_connector(self) -> bool:
        return self.isConnector()

    def isConnector(self) -> bool:
        return False

    def setTransportModel(self, model: str) -> None:
        raise NotImplementedError("Domain1D::setTransportModel")

    def _setKinetics(self, kin: Any) -> None:
        raise NotImplementedError("Domain1D::_setKinetics")

    def _setTransport(self, trans: Any) -> None:
        raise NotImplementedError("Domain1D::_setTransport")

    def container(self) -> Any:
        return self.m_container

    def setContainer(self, c: Any, index: int) -> None:
        self.m_container = c
        self.m_index = int(index)

    def bandwidth(self) -> int:
        return int(self.m_bw)

    def setBandwidth(self, bw: int = -1) -> None:
        self.m_bw = int(bw)

    def init(self) -> None:
        pass

    def resize(self, nv: int, np_pts: int) -> None:
        nv = int(nv)
        np_pts = int(np_pts)
        if nv != self.m_nv or not hasattr(self, "m_refiner") or self.m_refiner is None:
            self.m_nv = nv
            self.m_refiner = Refiner(self)
        self.m_nv = nv
        self.m_name = ["" for _ in range(self.m_nv)]
        self.m_max = np.zeros(self.m_nv, dtype=float)
        self.m_min = np.zeros(self.m_nv, dtype=float)
        self.m_rtol_ss = np.full(self.m_nv, 1.0e-4, dtype=float)
        self.m_atol_ss = np.full(self.m_nv, 1.0e-9, dtype=float)
        self.m_rtol_ts = np.full(self.m_nv, 1.0e-4, dtype=float)
        self.m_atol_ts = np.full(self.m_nv, 1.0e-11, dtype=float)
        self.m_points = np_pts
        self.m_z = np.zeros(self.m_points, dtype=float)
        self.m_slast = np.zeros(self.m_nv * self.m_points, dtype=float)
        self.locate()

    def refiner(self) -> Refiner:
        return self.m_refiner

    @property
    def n_components(self) -> int:
        return self.m_nv

    def nComponents(self) -> int:
        return self.m_nv

    @property
    def n_points(self) -> int:
        return self.m_points

    def nPoints(self) -> int:
        return self.m_points

    def componentName(self, n: int) -> str:
        n = int(n)
        if 0 <= n < self.m_nv:
            if self.m_name[n] != "":
                return self.m_name[n]
            return f"component {n}"
        raise IndexError(f"Domain1D::componentName: component {n} out of range [0, {self.m_nv})")

    component_name = componentName

    def setComponentName(self, n: int, name: str) -> None:
        n = int(n)
        if n < len(self.m_name):
            self.m_name[n] = str(name)

    set_component_name = setComponentName

    def componentIndex(self, name: str, check_alias: bool = True) -> int:
        for n in range(self.m_nv):
            if name == self.componentName(n):
                return n
        raise CanteraError("Domain1D::componentIndex", f"Component '{name}' not found")

    component_index = componentIndex

    def hasComponent(self, name: str, check_alias: bool = True) -> bool:
        for n in range(self.m_nv):
            if name == self.componentName(n):
                return True
        raise CanteraError("Domain1D::hasComponent", f"Component '{name}' not found")

    has_component = hasComponent

    def globalComponentIndex(self, n: int, j: int) -> int:
        return int(self.loc() + self.index(n, j))

    def updateState(self, loc: int = 0) -> None:
        pass

    def setBounds(self, n: int | None = None, lower: float = 0.0, upper: float = 0.0, **kwargs) -> None:
        if n is not None:
            n = int(n)
            self.m_min[n] = float(lower)
            self.m_max[n] = float(upper)
        for k, v in kwargs.items():
            idx = self.componentIndex(k)
            if isinstance(v, (tuple, list)) and len(v) == 2:
                self.m_min[idx] = float(v[0])
                self.m_max[idx] = float(v[1])
            else:
                self.m_min[idx] = float(v)

    set_bounds = setBounds

    def upperBound(self, n: int) -> float:
        return float(self.m_max[int(n)])

    upper_bound = upperBound

    def lowerBound(self, n: int) -> float:
        return float(self.m_min[int(n)])

    lower_bound = lowerBound

    def setTransientTolerances(self, rtol: float = 1e-4, atol: float = 1e-11, n: int = -1, **kwargs) -> None:
        self.have_user_tolerances = True
        n = int(n)
        if n == -1 or n >= self.m_nv:
            self.m_rtol_ts.fill(float(rtol))
            self.m_atol_ts.fill(float(atol))
        else:
            self.m_rtol_ts[n] = float(rtol)
            self.m_atol_ts[n] = float(atol)
        for k, v in kwargs.items():
            idx = self.componentIndex(k)
            if isinstance(v, (tuple, list)) and len(v) == 2:
                self.m_rtol_ts[idx] = float(v[0])
                self.m_atol_ts[idx] = float(v[1])

    set_transient_tolerances = setTransientTolerances

    def setSteadyTolerances(self, rtol: float = 1e-4, atol: float = 1e-9, n: int = -1, **kwargs) -> None:
        self.have_user_tolerances = True
        n = int(n)
        if n == -1 or n >= self.m_nv:
            self.m_rtol_ss.fill(float(rtol))
            self.m_atol_ss.fill(float(atol))
        else:
            self.m_rtol_ss[n] = float(rtol)
            self.m_atol_ss[n] = float(atol)
        for k, v in kwargs.items():
            idx = self.componentIndex(k)
            if isinstance(v, (tuple, list)) and len(v) == 2:
                self.m_rtol_ss[idx] = float(v[0])
                self.m_atol_ss[idx] = float(v[1])

    set_steady_tolerances = setSteadyTolerances

    def rtol(self, n: int = 0) -> float:
        return float(self.m_rtol_ss[int(n)] if self.m_rdt == 0.0 else self.m_rtol_ts[int(n)])

    def atol(self, n: int = 0) -> float:
        return float(self.m_atol_ss[int(n)] if self.m_rdt == 0.0 else self.m_atol_ts[int(n)])

    def steady_rtol(self, n: int = 0) -> float:
        return float(self.m_rtol_ss[int(n)])

    def steady_atol(self, n: int = 0) -> float:
        return float(self.m_atol_ss[int(n)])

    def transient_rtol(self, n: int = 0) -> float:
        return float(self.m_rtol_ts[int(n)])

    def transient_atol(self, n: int = 0) -> float:
        return float(self.m_atol_ts[int(n)])

    def setJacobianMode(self, mode: str) -> None:
        mode = str(mode)
        if mode not in ("auto", "analytic", "finite-difference"):
            raise CanteraError("Domain1D::setJacobianMode", f"Unknown Jacobian mode '{mode}'")
        self.m_jacobian_mode = mode

    set_jacobian_mode = setJacobianMode

    def jacobianMode(self) -> str:
        return str(self.m_jacobian_mode)

    jacobian_mode = property(jacobianMode, setJacobianMode)

    def hasAnalyticJacobian(self) -> bool:
        return False

    def evalJacobianAnalytic(self, x: np.ndarray, jac: Any = None) -> None:
        raise NotImplementedError("Domain1D::evalJacobianAnalytic")

    def checkAnalyticJacobian(self) -> bool:
        return True

    def setRefineCriteria(self, ratio: float = 10.0, slope: float = 0.8, curve: float = 0.8, prune: float = -0.1) -> None:
        self.m_refiner.setCriteria(ratio, slope, curve, prune)

    set_refine_criteria = setRefineCriteria

    def getRefineCriteria(self) -> list[float]:
        return list(self.m_refiner.getCriteria())

    get_refine_criteria = getRefineCriteria

    def initTimeInteg(self, dt: float, x: Any) -> None:
        self.m_rdt = 1.0 / float(dt)
        self.m_slast = np.array(x, dtype=float, copy=True)

    init_time_integ = initTimeInteg

    def setSteadyMode(self) -> None:
        self.m_rdt = 0.0

    set_steady_mode = setSteadyMode

    def steady(self) -> bool:
        return self.m_rdt == 0.0

    def transient(self) -> bool:
        return self.m_rdt != 0.0

    def needJacUpdate(self) -> None:
        if self.m_container is not None and hasattr(self.m_container, "linearSolver"):
            ls = self.m_container.linearSolver()
            if ls is not None and hasattr(ls, "setAge"):
                ls.setAge(10000)

    def eval(self, j: int | None, x: np.ndarray, r: np.ndarray, mask: np.ndarray, rdt: float = 0.0) -> None:
        raise NotImplementedError("Domain1D::eval")

    def index(self, n: int, j: int) -> int:
        return int(self.m_nv * int(j) + int(n))

    def value(self, x: Sequence[float], n: int, j: int) -> float:
        return float(x[self.index(n, j)])

    def setValue(self, n: int, j: int, val: float, x: Any = None) -> None:
        idx = self.index(n, j)
        if x is not None:
            x[idx] = float(val)
        elif self.m_state is not None:
            self.m_state[idx] = float(val)

    def values(self, component: str) -> list[float]:
        data = [0.0] * self.nPoints()
        self.getValues(component, data)
        return data

    def getValues(self, component: str, vals: list[float] | np.ndarray) -> None:
        raise NotImplementedError("Domain1D::getValues")

    def setValues(self, component: str, vals: Sequence[float] | np.ndarray) -> None:
        raise NotImplementedError("Domain1D::setValues")

    def residuals(self, component: str) -> list[float]:
        data = [0.0] * self.nPoints()
        self.getResiduals(component, data)
        return data

    def getResiduals(self, component: str, vals: list[float] | np.ndarray) -> None:
        raise NotImplementedError("Domain1D::getResiduals")

    def setFlatProfile(self, n: int, v: float) -> None:
        n = int(n)
        v = float(v)
        for j in range(self.m_points):
            self.setValue(n, j, v, self.m_state)

    def toArray(self) -> Any:
        return np.zeros(self.size(), dtype=float)

    def appendResiduals(self, arr: Any, res: Sequence[float] | np.ndarray) -> None:
        pass

    def fromArray(self, arr: Any) -> None:
        pass

    def info(self, keys: list[str] | None = None, rows: int = 0, width: int = 0) -> str:
        return ""

    def _info(self) -> dict[str, Any]:
        return {}

    def phase(self) -> Any:
        return getattr(self, "gas", getattr(self, "_phase", None))

    def size(self) -> int:
        return int(self.m_nv * self.m_points)

    def locate(self) -> None:
        if self.m_left is not None:
            left_last = self.m_left.lastPoint() if hasattr(self.m_left, "lastPoint") else (
                self.m_left.m_jstart + self.m_left.nPoints() - 1
            )
            left_loc = self.m_left.loc() if hasattr(self.m_left, "loc") else 0
            left_sz = self.m_left.size() if hasattr(self.m_left, "size") else 0
            self.m_jstart = left_last + 1
            self.m_iloc = left_loc + left_sz
        else:
            self.m_jstart = 0
            self.m_iloc = 0
        self.m_loc = self.m_iloc
        if self.m_right is not None and hasattr(self.m_right, "locate"):
            self.m_right.locate()

    def loc(self, j: int = 0) -> int:
        return int(self.m_iloc)

    def firstPoint(self) -> int:
        return int(self.m_jstart)

    def lastPoint(self) -> int:
        return int(self.m_jstart + self.m_points - 1)

    def linkLeft(self, left: Any) -> None:
        self.m_left = left
        self.locate()

    def linkRight(self, right: Any) -> None:
        self.m_right = right

    def append(self, right: Any) -> None:
        self.linkRight(right)
        right.linkLeft(self)

    def left(self) -> Any:
        return self.m_left

    def right(self) -> Any:
        return self.m_right

    def prevSoln(self, n: int, j: int) -> float:
        return float(self.m_slast[self.index(int(n), int(j))])

    def show(self, x: Sequence[float] | None = None) -> None:
        pass

    def z(self, j: int) -> float:
        return float(self.m_z[int(j)])

    def zmin(self) -> float:
        return float(self.m_z[0])

    def zmax(self) -> float:
        return float(self.m_z[-1])

    @property
    def grid(self) -> np.ndarray:
        return self.m_z.copy()

    @grid.setter
    def grid(self, z: Sequence[float] | np.ndarray) -> None:
        self.setupGrid(z)

    def setupGrid(self, z: Sequence[float] | np.ndarray) -> None:
        self.m_z = np.array(z, dtype=float, copy=True)
        self.m_points = len(self.m_z)
        self.m_slast = np.zeros(self.m_nv * self.m_points, dtype=float)

    def setupUniformGrid(self, points: int, length: float, start: float = 0.0) -> None:
        points = int(points)
        dz = float(length) / float(points - 1)
        grid = [float(start) + iz * dz for iz in range(points)]
        self.setupGrid(grid)

    def _getInitialSoln(self, x: Any) -> None:
        for j in range(self.m_points):
            for n in range(self.m_nv):
                x[self.index(n, j)] = self.initialValue(n, j)

    def initialValue(self, n: int, j: int) -> float:
        raise NotImplementedError("Domain1D::initialValue")

    def _finalize(self, x: Any) -> None:
        pass

    def forceFullUpdate(self, update: bool = True) -> None:
        self.m_force_full_update = bool(update)

    def setData(self, data: Any) -> None:
        self.m_state = data

    def getMeta(self) -> dict[str, Any]:
        def wrap_tols(tols: list[float] | np.ndarray) -> Any:
            unique_tols = {round(float(x), 12) for x in tols}
            if len(unique_tols) == 1:
                return float(tols[0])
            else:
                return {self.componentName(i): float(tols[i]) for i in range(len(tols))}

        state: dict[str, Any] = {
            "type": self.domainType(),
            "points": self.nPoints(),
        }
        if self.nComponents() and self.nPoints():
            state["tolerances"] = {
                "transient-abstol": wrap_tols(self.m_atol_ts),
                "steady-abstol": wrap_tols(self.m_atol_ss),
                "transient-reltol": wrap_tols(self.m_rtol_ts),
                "steady-reltol": wrap_tols(self.m_rtol_ss),
            }
        return state

    get_meta = getMeta

    def setMeta(self, meta: dict[str, Any]) -> None:
        def set_tols(tols_dict: Any, which: str, out: np.ndarray) -> None:
            if not isinstance(tols_dict, dict) or which not in tols_dict:
                return
            tol = tols_dict[which]
            if isinstance(tol, (int, float)):
                out.fill(float(tol))
            elif isinstance(tol, dict):
                for i in range(self.nComponents()):
                    name = self.componentName(i)
                    if name in tol:
                        out[i] = float(tol[name])

        if "tolerances" in meta:
            tols = meta["tolerances"]
            set_tols(tols, "transient-abstol", self.m_atol_ts)
            set_tols(tols, "transient-reltol", self.m_rtol_ts)
            set_tols(tols, "steady-abstol", self.m_atol_ss)
            set_tols(tols, "steady-reltol", self.m_rtol_ss)

    def __reduce__(self):
        raise NotImplementedError("Domain1D object is not picklable")

    def __copy__(self):
        raise NotImplementedError("Domain1D object is not copyable")
'''

    new_content = content[:start_idx] + new_domain1d_code + "\n\n\n" + content[boundary_idx:]
    oned_py.write_text(new_content, encoding="utf-8")
    print("Successfully patched Domain1D in oned.py")


if __name__ == "__main__":
    patch_domain1d()
