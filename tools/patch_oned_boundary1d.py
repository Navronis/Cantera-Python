"""Patch cantera_python/oned.py with exact native C++ Boundary1D subsystem."""

with open('outputs/cantera-python/cantera_python/oned.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Marker 1: class Boundary1D
start_marker = "class Boundary1D(Domain1D):"
assert start_marker in text, "start_marker not found"

# Marker 2: class Flow1D
end_marker = "class Flow1D(Domain1D):"
assert end_marker in text, "end_marker not found"

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

new_boundary_code = '''class Boundary1D(Domain1D):
    """Base class for boundary domains matching native Cantera C++ Boundary1D."""

    _domain_type = "boundary"

    def __init__(
        self,
        phase: Solution | None = None,
        name: str = "",
        **kwargs,
    ):
        if kwargs:
            first = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument '{first}'")
        self.gas = phase
        self.phase = phase
        self.m_solution = phase
        self.m_flow_left: Any = None
        self.m_flow_right: Any = None
        self.m_phase_left: Any = None
        self.m_phase_right: Any = None
        self.m_left_nv: int = 0
        self.m_right_nv: int = 0
        self.m_left_nsp: int = 0
        self.m_right_nsp: int = 0
        self.m_temp: float = float(phase.T) if phase is not None and hasattr(phase, "T") else 0.0
        self.m_mdot: float = 0.0
        self.m_V0: float = 0.0
        self.m_yin: np.ndarray = (
            np.array(phase.Y, dtype=float) if phase is not None and hasattr(phase, "Y") else np.array([])
        )
        super().__init__(nv=1, points=1, time=0.0)
        if name:
            self.m_id = str(name)
            self._name = str(name)

    def init(self) -> None:
        self._init(1)

    def domainType(self) -> str:
        return self._domain_type

    domain_type = property(domainType)

    def isConnector(self) -> bool:
        return True

    is_connector = property(isConnector)

    def setTemperature(self, t: float) -> None:
        self.m_temp = float(t)

    def temperature(self) -> float:
        return self.m_temp

    @property
    def T(self) -> float:
        return self.m_temp

    @T.setter
    def T(self, val: float):
        self.m_temp = float(val)

    def nSpecies(self) -> int:
        return 0

    @property
    def n_species(self) -> int:
        return self.nSpecies()

    def setMoleFractions(self, xin: Any) -> None:
        raise NotImplementedError("Boundary1D::setMoleFractions")

    def massFraction(self, k: int) -> float:
        raise NotImplementedError("Boundary1D::massFraction")

    def setMdot(self, mdot: float) -> None:
        self.m_mdot = float(mdot)

    def mdot(self) -> float:
        return self.m_mdot

    def setSpreadRate(self, V0: float) -> None:
        raise NotImplementedError("Boundary1D::setSpreadRate")

    def spreadRate(self) -> float:
        raise NotImplementedError("Boundary1D::spreadRate")

    @property
    def spread_rate(self) -> float:
        return self.spreadRate()

    @spread_rate.setter
    def spread_rate(self, val: float):
        self.setSpreadRate(val)

    def setupGrid(self, z: Any) -> None:
        pass

    def _init(self, n: int) -> None:
        if self.m_index is None or self.m_index == -1:
            raise CanteraError("Boundary1D::_init", "install in container before calling init.")
        self.resize(n, 1)
        self.m_left_nsp = 0
        self.m_right_nsp = 0
        self.m_flow_left = None
        self.m_flow_right = None
        self.m_phase_left = None
        self.m_phase_right = None

        if self.m_index > 0:
            r = self.container().domain(self.m_index - 1)
            if not r.isConnector():
                self.m_left_nv = r.nComponents()
                if self.m_left_nv > c_offset_Y:
                    self.m_left_nsp = self.m_left_nv - c_offset_Y
                else:
                    self.m_left_nsp = 0
                if isinstance(r, Flow1D):
                    self.m_flow_left = r
                    self.m_phase_left = r.phase
            else:
                raise CanteraError(
                    "Boundary1D::_init",
                    f"Boundary domains can only be connected on the left to flow domains, not '{r.domainType()}' domains.",
                )

        if self.m_index + 1 < self.container().nDomains():
            r = self.container().domain(self.m_index + 1)
            if not r.isConnector():
                self.m_right_nv = r.nComponents()
                if self.m_right_nv > c_offset_Y:
                    self.m_right_nsp = self.m_right_nv - c_offset_Y
                else:
                    self.m_right_nsp = 0
                if isinstance(r, Flow1D):
                    self.m_flow_right = r
                    self.m_phase_right = r.phase
            else:
                raise CanteraError(
                    "Boundary1D::_init",
                    f"Boundary domains can only be connected on the right to flow domains, not '{r.domainType()}' domains.",
                )

    @property
    def Y(self) -> np.ndarray:
        return self.m_yin.copy() if hasattr(self, "m_yin") and len(self.m_yin) > 0 else np.array([])

    @Y.setter
    def Y(self, val: Any):
        if self.phase is not None:
            orig_T, orig_P, orig_Y = self.phase.T, self.phase.P, self.phase.Y
            self.phase.Y = val
            self.m_yin = np.array(self.phase.Y, dtype=float)
            self.phase.set_state_tpy(orig_T, orig_P, orig_Y)
        else:
            self.m_yin = np.array(val, dtype=float)

    @property
    def X(self) -> np.ndarray:
        if self.phase is not None:
            orig_T, orig_P, orig_Y = self.phase.T, self.phase.P, self.phase.Y
            self.phase.set_state_tpy(self.m_temp, self.phase.P, self.m_yin)
            x_res = np.array(self.phase.X, dtype=float)
            self.phase.set_state_tpy(orig_T, orig_P, orig_Y)
            return x_res
        return np.array([])

    @X.setter
    def X(self, val: Any):
        self.setMoleFractions(val)

    def to_array(self, normalize: bool = False):
        from .solution_array import SolutionArray
        arr = SolutionArray(self.phase, shape=(1,))
        arr.T = np.array([self.T])
        p_val = getattr(self.m_flow_left or self.m_flow_right, "P", getattr(self.phase, "P", 101325.0))
        arr.P = np.array([p_val])
        arr.Y = np.array([self.Y])
        if hasattr(self, "mdot"):
            arr.mdot = np.array([self.mdot()])
            arr.meta["mass-flux"] = float(self.mdot())
        return arr

    def from_array(self, arr: Any) -> None:
        if hasattr(arr, "T"):
            t_val = np.asarray(arr.T).flatten()
            self.T = float(t_val[0])
        if hasattr(arr, "mdot") and hasattr(self, "setMdot"):
            m_val = np.asarray(arr.mdot).flatten()
            self.setMdot(float(m_val[0]))
        elif hasattr(arr, "meta") and "mass-flux" in arr.meta and hasattr(self, "setMdot"):
            self.setMdot(float(arr.meta["mass-flux"]))
        if hasattr(arr, "Y"):
            y_val = np.asarray(arr.Y)
            self.Y = y_val[0] if y_val.ndim > 1 else y_val


class Inlet1D(Boundary1D):
    """An inlet boundary."""

    _domain_type = "inlet"

    def __init__(self, phase: Solution | None = None, name: str = "inlet", **kwargs):
        super().__init__(phase=phase, name=name, **kwargs)
        self.m_ilr = 0
        self.m_V0 = 0.0
        self.m_nsp = 0
        self.m_yin = np.array([])
        self.m_xstr = ""
        self.m_flow = None
        if phase is not None and hasattr(phase, "n_species"):
            self.m_temp = float(phase.T)
            self.m_press = float(phase.P)
            self.m_nsp = phase.n_species
            self.m_yin = np.array(phase.Y, dtype=float)

    def setSpreadRate(self, V0: float) -> None:
        self.m_V0 = float(V0)
        self.needJacUpdate()

    def spreadRate(self) -> float:
        return self.m_V0

    def setTemperature(self, T: float) -> None:
        super().setTemperature(T)
        if self.m_flow is not None and self.m_flow.lowerBound(c_offset_T) >= self.m_temp:
            self.m_flow.setBounds(c_offset_T, self.m_temp - 5.0, self.m_flow.upperBound(c_offset_T))

    def nSpecies(self) -> int:
        return self.m_nsp

    def setMoleFractions(self, xin: Any) -> None:
        if isinstance(xin, str):
            if self.m_solution is not None and hasattr(self.m_solution, "set_mole_fractions_by_name"):
                self.m_solution.set_mole_fractions_by_name(xin)
                self.m_yin = np.array(self.m_solution.Y, dtype=float)
                self.needJacUpdate()
            elif self.m_solution is not None:
                self.m_solution.X = xin
                self.m_yin = np.array(self.m_solution.Y, dtype=float)
                self.needJacUpdate()
            else:
                self.m_xstr = str(xin)
        else:
            if self.m_solution is not None:
                self.m_solution.X = xin
                self.m_yin = np.array(self.m_solution.Y, dtype=float)
                self.needJacUpdate()
            else:
                self.m_yin = np.array(xin, dtype=float)

    def massFraction(self, k: int) -> float:
        return float(self.m_yin[k])

    def init(self) -> None:
        self._init(0)
        if self.m_flow_left and not self.m_flow_right:
            if not self.m_flow_left.isStrained():
                raise CanteraError(
                    "Inlet1D::init",
                    "Right inlets with right-to-left flow are only supported for strained flow configurations.",
                )
            self.m_ilr = -1  # RightInlet
            self.m_flow = self.m_flow_left
        elif self.m_flow_right:
            self.m_ilr = 1  # LeftInlet
            self.m_flow = self.m_flow_right
        else:
            raise CanteraError("Inlet1D::init", "Inlet1D is not properly connected.")

        self.m_press = self.m_flow.pressure()
        if not self.m_nsp:
            self.m_nsp = self.m_flow.phase.n_species
            self.m_yin = np.zeros(self.m_nsp, dtype=float)
            self.m_yin[0] = 1.0
        if self.m_temp > 0:
            self.setTemperature(self.m_temp)
        if self.m_xstr:
            self.setMoleFractions(self.m_xstr)

    def eval(
        self,
        jg: int,
        xg: np.ndarray,
        rg: np.ndarray,
        diagg: np.ndarray,
        rdt: float = 0.0,
    ) -> None:
        if jg is not None and jg != -1:
            if jg + 2 < self.firstPoint() or jg > self.lastPoint() + 2:
                return

        if self.m_ilr == 1:  # LeftInlet
            fl_loc = self.m_flow.loc()

            if self.m_flow.doEnergy(0):
                rg[fl_loc + c_offset_T] -= self.m_temp
            else:
                rg[fl_loc + c_offset_T] -= self.m_flow.T_fixed(0)

            if self.m_flow.isFree():
                self.m_mdot = self.m_flow.density(0) * xg[fl_loc + c_offset_U]
            elif self.m_flow.isStrained():
                if self.m_flow.twoPointControlEnabled():
                    self.m_mdot = self.m_flow.density(0) * xg[fl_loc + c_offset_U]
                else:
                    rg[fl_loc + c_offset_L] += self.m_mdot
                rg[fl_loc + c_offset_V] -= self.m_V0
            else:
                rg[fl_loc + c_offset_U] = self.m_flow.density(0) * xg[fl_loc + c_offset_U] - self.m_mdot

            left_excess = self.m_flow_right.leftExcessSpecies() if self.m_flow_right else -1
            for k in range(self.m_nsp):
                if k != left_excess:
                    rg[fl_loc + c_offset_Y + k] += self.m_mdot * self.m_yin[k]

        else:  # RightInlet
            last_j = self.m_flow.nPoints() - 1
            last_pt = self.loc() - self.m_flow.nComponents()

            rg[last_pt + c_offset_V] -= self.m_V0
            if self.m_flow.doEnergy(last_j):
                rg[last_pt + c_offset_T] -= self.m_temp
            else:
                rg[last_pt + c_offset_T] -= self.m_flow.T_fixed(last_j)

            if self.m_flow.twoPointControlEnabled():
                self.m_mdot = -self.m_flow.density(last_j) * xg[last_pt + c_offset_Uo]
            rg[last_pt + c_offset_U] += self.m_mdot

            right_excess = self.m_flow_left.rightExcessSpecies() if self.m_flow_left else -1
            for k in range(self.m_nsp):
                if k != right_excess:
                    rg[last_pt + c_offset_Y + k] += self.m_mdot * self.m_yin[k]


class Empty1D(Boundary1D):
    """An empty terminator domain."""

    _domain_type = "empty"

    def __init__(self, phase: Solution | None = None, name: str = "empty", **kwargs):
        super().__init__(phase=phase, name=name, **kwargs)

    def init(self) -> None:
        self._init(0)

    def eval(self, jg: int, xg: np.ndarray, rg: np.ndarray, diagg: np.ndarray, rdt: float = 0.0) -> None:
        pass


class Symm1D(Boundary1D):
    """A symmetry plane boundary."""

    _domain_type = "symmetry-plane"

    def __init__(self, phase: Solution | None = None, name: str = "symm", **kwargs):
        super().__init__(phase=phase, name=name, **kwargs)

    def init(self) -> None:
        self._init(0)

    def eval(self, jg: int, xg: np.ndarray, rg: np.ndarray, diagg: np.ndarray, rdt: float = 0.0) -> None:
        if jg is not None and jg != -1:
            if jg + 2 < self.firstPoint() or jg > self.lastPoint() + 2:
                return

        if self.m_flow_right:
            nc = self.m_flow_right.nComponents()
            loc = self.loc()
            diagg[loc + c_offset_V] = 0
            diagg[loc + c_offset_T] = 0
            rg[loc + c_offset_V] = xg[loc + c_offset_V] - xg[loc + c_offset_V + nc]
            if self.m_flow_right.doEnergy(0):
                rg[loc + c_offset_T] = xg[loc + c_offset_T] - xg[loc + c_offset_T + nc]

        if self.m_flow_left:
            nc = self.m_flow_left.nComponents()
            loc = self.loc()
            last = self.m_flow_left.nPoints() - 1
            last_pt = loc - nc
            prev_pt = loc - 2 * nc
            diagg[last_pt + c_offset_V] = 0
            diagg[last_pt + c_offset_T] = 0
            rg[last_pt + c_offset_V] = xg[last_pt + c_offset_V] - xg[prev_pt + c_offset_V]
            if self.m_flow_left.doEnergy(last):
                rg[last_pt + c_offset_T] = xg[last_pt + c_offset_T] - xg[prev_pt + c_offset_T]


SymmetryPlane1D = Symm1D


class Outlet1D(Boundary1D):
    """An outlet boundary with zero-gradient conditions."""

    _domain_type = "outlet"

    def __init__(self, phase: Solution | None = None, name: str = "outlet", **kwargs):
        super().__init__(phase=phase, name=name, **kwargs)

    def init(self) -> None:
        self._init(0)
        if self.m_flow_right:
            raise CanteraError("Outlet1D::init", "Left outlets with right-to-left flow are not supported.")
        if self.m_flow_left:
            self.m_flow_left.setViscosityFlag(False)
        else:
            raise CanteraError("Outlet1D::init", "Outlet1D is not connected.")

    def eval(self, jg: int, xg: np.ndarray, rg: np.ndarray, diagg: np.ndarray, rdt: float = 0.0) -> None:
        if jg is not None and jg != -1:
            if jg + 2 < self.firstPoint() or jg > self.lastPoint() + 2:
                return

        nc = self.m_flow_left.nComponents()
        loc = self.loc()
        last = self.m_flow_left.nPoints() - 1
        last_pt = loc - nc
        prev_pt = loc - 2 * nc

        if self.m_flow_left.doEnergy(last):
            rg[last_pt + c_offset_T] = xg[last_pt + c_offset_T] - xg[prev_pt + c_offset_T]
        else:
            rg[last_pt + c_offset_T] = xg[last_pt + c_offset_T] - self.m_flow_left.T_fixed(last)

        k_skip = c_offset_Y + self.m_flow_left.rightExcessSpecies()
        for k in range(c_offset_Y, nc):
            if k != k_skip:
                rg[last_pt + k] = xg[last_pt + k] - xg[prev_pt + k]
                diagg[last_pt + k] = 0


class OutletRes1D(Boundary1D):
    """An outlet boundary with reservoir composition."""

    _domain_type = "outlet-reservoir"

    def __init__(self, phase: Solution | None = None, name: str = "outlet_res", **kwargs):
        super().__init__(phase=phase, name=name, **kwargs)
        self.m_nsp = 0
        self.m_yres = np.array([])
        self.m_xstr = ""
        self.m_flow = None

    def nSpecies(self) -> int:
        return self.m_nsp

    def setMoleFractions(self, xin: Any) -> None:
        if isinstance(xin, str):
            self.m_xstr = str(xin)
            if self.m_flow is not None and hasattr(self.m_flow, "phase"):
                self.m_flow.phase.X = xin
                self.m_yres = np.array(self.m_flow.phase.Y, dtype=float)
                self.needJacUpdate()
        else:
            if self.m_flow is not None and hasattr(self.m_flow, "phase"):
                self.m_flow.phase.X = xin
                self.m_yres = np.array(self.m_flow.phase.Y, dtype=float)
                self.needJacUpdate()
            else:
                self.m_yres = np.array(xin, dtype=float)

    def massFraction(self, k: int) -> float:
        return float(self.m_yres[k])

    def init(self) -> None:
        self._init(0)
        if self.m_flow_right:
            raise CanteraError("OutletRes1D::init", "Left outlets with right-to-left flow are not supported.")
        if self.m_flow_left:
            self.m_flow = self.m_flow_left
        else:
            raise CanteraError("OutletRes1D::init", "no flow!")

        self.m_nsp = self.m_flow.phase.n_species
        self.m_yres = np.zeros(self.m_nsp, dtype=float)
        if self.m_xstr:
            self.setMoleFractions(self.m_xstr)
        else:
            self.m_yres[0] = 1.0

    def eval(self, jg: int, xg: np.ndarray, rg: np.ndarray, diagg: np.ndarray, rdt: float = 0.0) -> None:
        if jg is not None and jg != -1:
            if jg + 2 < self.firstPoint() or jg > self.lastPoint() + 2:
                return

        nc = self.m_flow_left.nComponents()
        loc = self.loc()
        last = self.m_flow_left.nPoints() - 1
        last_pt = loc - nc
        prev_pt = loc - 2 * nc

        if self.m_flow_left.doEnergy(last):
            rg[last_pt + c_offset_T] = xg[last_pt + c_offset_T] - xg[prev_pt + c_offset_T]
        else:
            rg[last_pt + c_offset_T] = xg[last_pt + c_offset_T] - self.m_flow_left.T_fixed(last)

        k_skip = self.m_flow_left.rightExcessSpecies()
        for k in range(c_offset_Y, nc):
            if k - c_offset_Y != k_skip:
                rg[last_pt + k] = xg[last_pt + k] - self.m_yres[k - c_offset_Y]
                diagg[last_pt + k] = 0


class Surf1D(Boundary1D):
    """A non-reacting surface boundary."""

    _domain_type = "surface"

    def __init__(self, phase: Any = None, name: str = "surface", **kwargs):
        super().__init__(phase=phase, name=name, **kwargs)

    def init(self) -> None:
        self._init(0)

    def eval(self, jg: int, xg: np.ndarray, rg: np.ndarray, diagg: np.ndarray, rdt: float = 0.0) -> None:
        if jg is not None and jg != -1:
            if jg + 2 < self.firstPoint() or jg > self.lastPoint() + 2:
                return

        if self.m_flow_right:
            loc = self.loc()
            rg[loc + c_offset_T] = xg[loc + c_offset_T] - self.m_temp

        if self.m_flow_left:
            nc = self.m_flow_left.nComponents()
            loc = self.loc()
            last_pt = loc - nc
            rg[last_pt + c_offset_T] = xg[last_pt + c_offset_T] - self.m_temp


Surface1D = Surf1D


class ReactingSurf1D(Boundary1D):
    """A chemically reacting surface boundary."""

    _domain_type = "reacting-surface"

    def __init__(self, phase: Any = None, name: str = "reacting-surface", **kwargs):
        if phase is not None:
            from .surface import Interface
            if not isinstance(phase, Interface):
                raise CanteraError(
                    "ReactingSurf1D::ReactingSurf1D",
                    f"Detected incompatible ThermoPhase type '{getattr(phase, 'thermo_model', type(phase).__name__)}'",
                )
            self.m_sphase = phase
            self.m_kin = getattr(phase, "kinetics", phase)
            self.m_nsp = phase.n_species
        else:
            self.m_sphase = None
            self.m_kin = None
            self.m_nsp = 0

        self.m_enabled = True
        self.m_work = np.array([])
        self.m_fixed_cov = np.array([])
        super().__init__(phase=phase, name=name, **kwargs)

    def enableCoverageEquations(self, docov: bool = True) -> None:
        self.m_enabled = bool(docov)

    def coverageEnabled(self) -> bool:
        return self.m_enabled

    def componentName(self, n: int) -> str:
        if n < self.m_nsp and self.m_sphase is not None:
            return self.m_sphase.species_names[n]
        raise IndexError(f"ReactingSurf1D::componentName: component {n} out of range ({self.m_nsp})")

    def componentIndex(self, name: str, checkAlias: bool = True) -> int:
        if self.m_sphase is not None:
            return self.m_sphase.species_names.index(name)
        raise CanteraError("ReactingSurf1D::componentIndex", f"Species '{name}' not found")

    def init(self) -> None:
        self.m_nv = self.m_nsp
        self._init(self.m_nsp)
        self.m_fixed_cov = np.zeros(self.m_nsp, dtype=float)
        self.m_fixed_cov[0] = 1.0
        n_tot = getattr(self.m_kin, "n_total_species", self.m_nsp)
        self.m_work = np.zeros(n_tot, dtype=float)
        for n in range(self.m_nsp):
            self.setBounds(n, -1.0e-5, 2.0)

    def resetBadValues(self, xg: Any) -> None:
        loc = self.loc()
        cov = np.array(xg[loc : loc + self.m_nsp], dtype=float)
        s = float(np.sum(cov))
        if s > 0:
            cov /= s
        xg[loc : loc + self.m_nsp] = cov

    def eval(
        self,
        jg: int,
        xg: np.ndarray,
        rg: np.ndarray,
        diagg: np.ndarray,
        rdt: float = 0.0,
    ) -> None:
        if jg is not None and jg != -1:
            if jg + 2 < self.firstPoint() or jg > self.lastPoint() + 2:
                return

        loc = self.loc()
        x = np.array(xg[loc : loc + self.m_nsp], dtype=float)
        sum_x = float(np.sum(x))

        self.m_sphase.temperature = self.m_temp
        if hasattr(self.m_sphase, "set_coverages_no_norm"):
            self.m_sphase.set_coverages_no_norm(x)
        elif hasattr(self.m_sphase, "coverages"):
            self.m_sphase.coverages = x

        if self.m_flow_left:
            leftloc = self.m_flow_left.loc()
            pnt = self.m_flow_left.nPoints() - 1
            self.m_flow_left.setGas(xg[leftloc : leftloc + self.m_flow_left.size()], pnt)
        if self.m_flow_right:
            rightloc = self.m_flow_right.loc()
            self.m_flow_right.setGas(xg[rightloc : rightloc + self.m_flow_right.size()], 0)

        sdot = np.zeros(self.m_kin.n_total_species, dtype=float)
        if hasattr(self.m_kin, "get_net_production_rates"):
            sdot = np.array(self.m_kin.get_net_production_rates(), dtype=float)
        elif hasattr(self.m_kin, "net_production_rates"):
            sdot = np.array(self.m_kin.net_production_rates, dtype=float)

        rs0 = 1.0 / self.m_sphase.site_density

        if self.m_enabled:
            for k in range(self.m_nsp):
                sz = self.m_sphase.species[k].size if hasattr(self.m_sphase.species[k], "size") else 1.0
                rg[loc + k] = sdot[k] * sz * rs0
                rg[loc + k] -= rdt * (x[k] - self.prevSoln(k, 0))
                diagg[loc + k] = 1
            rg[loc] = 1.0 - sum_x
            diagg[loc] = 0
        else:
            for k in range(self.m_nsp):
                rg[loc + k] = x[k] - self.m_fixed_cov[k]
                diagg[loc + k] = 0

        if self.m_flow_right:
            rloc = self.m_flow_right.loc()
            rg[rloc + c_offset_T] = xg[rloc + c_offset_T] - self.m_temp

        if self.m_flow_left:
            last_pt = self.m_flow_left.loc() + self.m_flow_left.index(0, self.m_flow_left.nPoints() - 1)
            mwleft = self.m_phase_left.molecular_weights
            rg[last_pt + c_offset_T] = xg[last_pt + c_offset_T] - self.m_temp
            n_skip = self.m_flow_left.rightExcessSpecies()
            l_offset = 0
            if hasattr(self.m_kin, "kinetics_species_index"):
                try:
                    l_offset = self.m_kin.kinetics_species_index(0, self.m_flow_left.phase)
                except Exception:
                    l_offset = 0
            for nl in range(self.m_left_nsp):
                if nl != n_skip:
                    rg[last_pt + c_offset_Y + nl] += sdot[nl + l_offset] * mwleft[nl]


ReactingSurface1D = ReactingSurf1D
'''

new_text = text[:start_idx] + new_boundary_code + "\n\n" + text[end_idx:]

# Also update Flow1D.setGas to support setGas(x, j)
old_set_gas = """    def setGas(self, gas) -> None:
        self._gas = gas"""
new_set_gas = """    def setGas(self, x_or_gas, j=None) -> None:
        if j is None:
            self._gas = x_or_gas
        else:
            x = np.asarray(x_or_gas)
            T_val = x[self.index(c_offset_T, j)]
            y_start = self.index(c_offset_Y, j)
            Y_val = x[y_start:y_start + self.nsp]
            self.phase.set_state_tpy(T_val, self.pressure(), Y_val)"""

assert old_set_gas in new_text, "old_set_gas not found"
new_text = new_text.replace(old_set_gas, new_set_gas)

with open('outputs/cantera-python/cantera_python/oned.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print("Successfully patched oned.py with complete Boundary1D subsystem!")
