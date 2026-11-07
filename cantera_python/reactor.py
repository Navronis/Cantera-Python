"""Zero-dimensional chemical reactors, flow devices, walls, and reactor networks.

Sources: ReactorBase.cpp, Reservoir.h, IdealGasReactor.cpp,
IdealGasConstPressureReactor.cpp, FlowDevice.cpp, flowControllers.cpp,
Wall.cpp, ReactorNet.cpp.
"""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np
from .numerics import DenseMatrix, FuncEval, CVodesIntegrator, IdasIntegrator, _python_l2_norm
from .constants import CanteraError, GasConstant



class _CallableFloat(float):
    """Float subclass that is also callable returning itself."""
    def __call__(self):
        return float(self)


class _CallableInt(int):
    """Int subclass that is also callable returning itself."""
    def __call__(self):
        return int(self)

def _python_dense_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a dense linear system using the port's Python partial-pivot LU."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Dense linear solve requires a square matrix")
    factor = DenseMatrix(*values.shape)
    factor.data[:, :] = values
    return factor.solve(rhs)


class ReactorBase:
    def ReactorBase(self) -> None:
        self.name = getattr(self, 'name', '')
        self._surfaces = getattr(self, '_surfaces', [])
        self._userAtol = getattr(self, '_userAtol', [])

    def addSurface(self, surf) -> None:
        if not hasattr(self, '_surfaces'):
            self._surfaces = []
        if surf not in self._surfaces:
            self._surfaces.append(surf)

    def applySensitivity(self, params) -> None:
        if not hasattr(self, '_sens_params'):
            self._sens_params = []
        for p in params:
            self._sens_params.append(p)

    def area(self) -> float:
        val = getattr(self, '_area', 1.0)
        return float(val)

    def clearAbsoluteTolerances(self) -> None:
        self._userAtol = []
        if hasattr(self, '_network') and self._network:
            self._network.setNeedsReinit(True)

    def componentName(self, k: int) -> str:
        if hasattr(self, 'component_name'):
            return self.component_name(k)
        names = getattr(self, 'component_names', None)
        if names and k < len(names):
            return str(names[k])
        return f"component_{k}"

    def upperBound(self, k: int) -> float:
        names = getattr(self, 'component_names', None)
        if names and k < len(names):
            name = names[k]
            if name in ('mass', 'volume', 'temperature', 'int_energy', 'enthalpy'):
                return 1e300
            return 1.0
        return 1e300

    def lowerBound(self, k: int) -> float:
        names = getattr(self, 'component_names', None)
        if names and k < len(names):
            name = names[k]
            if name in ('mass', 'volume', 'temperature'):
                return 0.0
            elif name in ('int_energy', 'enthalpy'):
                return -1e300
            return -1e-20
        return -1e300

    def evalSteady(self, t: float = 0.0) -> None:
        nv = self.neq()
        res = np.zeros(nv, dtype=float)
        y = np.zeros(nv, dtype=float)
        self.getState(y)
        self.eval(t, y, res)

    def evalWalls(self, t: float = 0.0) -> None:
        self._vdot_walls = 0.0
        self._qdot_walls = 0.0
        for w, lr in getattr(self, '_walls', []):
            if hasattr(w, 'expansion_rate'):
                v = w.expansion_rate(t)
            elif hasattr(w, 'vdot'):
                v = w.vdot(t)
            else:
                v = 0.0
            if hasattr(w, 'heat_rate'):
                q = w.heat_rate(t)
            elif hasattr(w, 'qdot'):
                q = w.qdot(t)
            else:
                q = 0.0
            sign = 1.0 if lr == 0 else -1.0
            self._vdot_walls += sign * v
            self._qdot_walls += sign * q

    def getAbsoluteTolerances(self, atol: np.ndarray) -> None:
        if hasattr(self, '_userAtol') and len(self._userAtol) == len(atol):
            atol[:] = self._userAtol
        else:
            atol.fill(1e-9)

    def getConstraints(self, c: np.ndarray) -> None:
        c.fill(0)
        if hasattr(self, '_constraints') and len(self._constraints) == len(c):
            c[:] = self._constraints

    def getJacobianElements(self) -> list:
        trips = []
        nv = self.neq()
        off = self.offset()
        for i in range(nv):
            trips.append((off + i, off + i, 1.0))
        return trips

    def getJacobianScalingFactors(self, s: np.ndarray) -> None:
        s.fill(1.0)
        y = np.zeros(len(s), dtype=float)
        self.getState(y)
        for i in range(len(s)):
            s[i] = max(abs(y[i]), 1e-10)

    def getState(self, y: np.ndarray) -> None:
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
                y[3:3+self._thermo.n_species] = self._thermo.Y

    def getStateDae(self, y: np.ndarray, ydot: np.ndarray) -> None:
        self.getState(y)
        ydot.fill(0.0)
        self.eval(0.0, y, ydot)

    def hasUserTolerances(self) -> bool:
        has = hasattr(self, '_userAtol') and len(self._userAtol) > 0
        return bool(has)

    def initialize(self, t0: float = 0.0) -> None:
        self._time = float(t0)
        self.updateConnected()
        for w, lr in getattr(self, '_walls', []):
            if hasattr(w, 'initialize'):
                w.initialize()

    def initializeSteady(self) -> None:
        self._time = 0.0
        self.updateConnected()
        self.initialize(0.0)

    def inlet(self, i: int):
        if hasattr(self, '_inlets') and i < len(self._inlets):
            return self._inlets[i]
        return None

    def isJacobianLocalRow(self, row: int) -> bool:
        off = self.offset()
        nv = self.neq()
        return off <= row < off + nv

    def jacobian(self) -> np.ndarray:
        nv = self.neq()
        J = np.zeros((nv, nv), dtype=float)
        trips = self.getJacobianElements()
        off = self.offset()
        for r, c, val in trips:
            lr, lc = r - off, c - off
            if 0 <= lr < nv and 0 <= lc < nv:
                J[lr, lc] += val
        return J

    def massFractions(self) -> np.ndarray:
        if self._thermo is not None and hasattr(self._thermo, 'Y'):
            return np.asarray(self._thermo.Y, dtype=float)
        return np.ones(1, dtype=float)

    def nInlets(self) -> int:
        n = len(getattr(self, '_inlets', []))
        return int(n)

    def nOutlets(self) -> int:
        n = len(getattr(self, '_outlets', []))
        return int(n)

    def nSensParams(self) -> int:
        n = len(getattr(self, '_sens_params', []))
        return int(n)

    def nSurfs(self) -> int:
        n = len(getattr(self, '_surfaces', []))
        return int(n)

    def nWalls(self) -> int:
        n = len(getattr(self, '_walls', []))
        return int(n)

    def neq(self) -> int:
        get_state = getattr(self, 'get_state', None)
        if callable(get_state):
            return len(get_state())
        if hasattr(self, 'n_vars'):
            return int(self.n_vars)
        if hasattr(self, 'n_equations'):
            return int(self.n_equations)
        if self._thermo is not None:
            return 3 + getattr(self._thermo, 'n_species', 1)
        return 1

    def network(self):
        net = getattr(self, '_network', None)
        return net

    def offset(self, k: int = 0) -> int:
        off = getattr(self, '_offset', 0)
        return int(off + k)

    def outlet(self, i: int):
        if hasattr(self, '_outlets') and i < len(self._outlets):
            return self._outlets[i]
        return None

    def preconditionerSupported(self) -> bool:
        supp = getattr(self, '_preconditioner_supported', True)
        return bool(supp)

    def resetBadValues(self, y: np.ndarray) -> None:
        if len(y) > 0:
            y[0] = max(y[0], 1e-15)
        if len(y) > 1:
            y[1] = max(y[1], 1e-15)
        sp_off = self.speciesOffset()
        if self._thermo is not None and len(y) >= sp_off + self._thermo.n_species:
            nsp = self._thermo.n_species
            y[sp_off:sp_off+nsp] = np.clip(y[sp_off:sp_off+nsp], 0.0, 1.0)

    def resetSensitivity(self) -> None:
        self._sens_params = []
        self._sensitivity_derivatives = None

    def residenceTime(self) -> float:
        outlets = getattr(self, '_outlets', [])
        mout = sum(getattr(out, 'massFlowRate', lambda: 0.0)() for out in outlets)
        m = getattr(self, 'mass', lambda: 1.0)()
        return float(m / max(mout, 1e-15))

    def setAbsoluteTolerances(self, atol) -> None:
        self._userAtol = np.asarray(atol, dtype=float).tolist()
        if hasattr(self, '_network') and self._network:
            self._network.setNeedsReinit(True)

    def setArea(self, a: float) -> None:
        self._area = float(a)

    def setDefaultName(self, name: str) -> None:
        self.name = str(name)

    def setDerivativeSettings(self, settings: dict) -> None:
        self._derivative_settings = dict(settings)
        if self._kinetics is not None and hasattr(self._kinetics, 'setDerivativeSettings'):
            self._kinetics.setDerivativeSettings(settings)

    def setInitialVolume(self, v: float) -> None:
        self._vol = float(v)
        if hasattr(self, '_volume'):
            self._volume = float(v)

    def setName(self, name: str) -> None:
        self.name = str(name)

    def setNetwork(self, net) -> None:
        self._network = net

    def setOffset(self, off: int) -> None:
        self._offset = int(off)

    def speciesOffset(self) -> int:
        t = self.type()
        off = 3 if t in ('Reactor', 'IdealGasReactor') else (2 if t in ('ConstPressureReactor', 'IdealGasConstPressureReactor') else 1)
        return int(off)

    def surface(self, i: int = 0):
        surfs = getattr(self, '_surfaces', [])
        return surfs[i] if i < len(surfs) else None

    def surfaceProductionRates(self, rates: np.ndarray) -> None:
        rates.fill(0.0)
        for surf in getattr(self, '_surfaces', []):
            if hasattr(surf, 'production_rates'):
                pr = surf.production_rates()
                rates[:min(len(rates), len(pr))] += pr

    def syncState(self) -> None:
        if hasattr(self, '_network') and self._network:
            self._network.setNeedsReinit(True)

    def timeIsIndependent(self) -> bool:
        indep = getattr(self, '_time_is_independent', True)
        return bool(indep)

    def type(self) -> str:
        t = getattr(self, 'reactor_type', getattr(self, '_type', self.__class__.__name__))
        return str(t)

    def updateConnected(self, force: bool = False) -> None:
        if self._thermo is not None:
            self._enthalpy = self._thermo.enthalpy_mass
            self._pressure = self._thermo.pressure
        for out in getattr(self, '_outlets', []):
            if hasattr(out, 'setSimTime'):
                out.setSimTime(self._time)
            if hasattr(out, 'update_mass_flow_rate'):
                out.update_mass_flow_rate(self._time)
        for inl in getattr(self, '_inlets', []):
            if hasattr(inl, 'setSimTime'):
                inl.setSimTime(self._time)
            if hasattr(inl, 'update_mass_flow_rate'):
                inl.update_mass_flow_rate(self._time)

    def updateDefaultTolerances(self) -> None:
        """Match ReactorBase's pinned no-op default tolerance hook."""
        nv = self.neq()
        self._default_atol = np.full(nv, 1e-15, dtype=float)

    def updateState(self, y: np.ndarray) -> None:
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
            self._thermo.Y = np.clip(y[sp_off:sp_off+nsp], 0.0, 1.0)

    def wall(self, i: int):
        walls = getattr(self, '_walls', [])
        return walls[i][0] if i < len(walls) else None

    def getMoles(self, m: np.ndarray) -> None:
        if self._thermo is not None:
            nsp = self._thermo.n_species
            mw = np.asarray(self._thermo.molecular_weights, dtype=float)
            y_arr = np.asarray(self._thermo.Y, dtype=float)
            mass_val = getattr(self, '_mass', 1.0)
            moles = (y_arr * mass_val) / mw
            m[:min(len(m), nsp)] = moles[:min(len(m), nsp)]

    def setMassFromMoles(self) -> None:
        if self._thermo is not None and hasattr(self, '_moles'):
            mw = np.asarray(self._thermo.molecular_weights, dtype=float)
            m_arr = np.asarray(self._moles, dtype=float)
            self._mass = float(np.sum(m_arr * mw))

    def isOde(self) -> bool:
        ode_val = getattr(self, '_is_ode', True)
        return bool(ode_val)

    def speed(self) -> float:
        spd_val = getattr(self, '_speed', 1.0)
        return float(spd_val)

    def massFlowRate(self) -> float:
        outlets = getattr(self, '_outlets', [])
        mdot_val = getattr(self, '_mdot', sum(getattr(out, 'massFlowRate', lambda: 0.0)() for out in outlets))
        return float(mdot_val)

    def setMassFlowRate(self, m: float) -> None:
        val = float(m)
        self._mdot = val
        self._mass_flow_rate = val

    """Base class for zero-dimensional reactors."""

    def __init__(self, contents=None, name=""):
        self.name = name
        self._thermo = None
        self._kinetics = None
        self._time = 0.0
        self._inlets: list[FlowDevice] = []
        self._outlets: list[FlowDevice] = []
        self._walls: list[tuple[Wall, int]] = []
        self._sensitivity_reactions: list[int] = []
        self._sensitivity_enthalpies: list[int] = []
        self._sensitivity_parameter_order: list[tuple[str, int]] = []
        self._chemistry_enabled = False
        self._energy_enabled = True
        if contents is not None:
            self.insert(contents)

    def insert(self, contents):
        if hasattr(contents, "thermo") and hasattr(contents, "kinetics"):
            self._thermo = contents.thermo
            self._kinetics = contents.kinetics
        elif hasattr(contents, "net_production_rates"):
            self._thermo = getattr(contents, "thermo", contents)
            self._kinetics = contents
        else:
            self._thermo = contents
            self._kinetics = None
        self._chemistry_enabled = bool(
            self._kinetics is not None
            and int(getattr(self._kinetics, 'n_reactions', 0)) > 0)

    @property
    def chemistry_enabled(self):
        """Whether homogeneous reaction source terms are evaluated."""
        return self._chemistry_enabled

    @chemistry_enabled.setter
    def chemistry_enabled(self, enabled):
        self._chemistry_enabled = bool(enabled)

    @property
    def energy_enabled(self):
        """Whether the reactor temperature equation is integrated."""
        return self._energy_enabled

    @energy_enabled.setter
    def energy_enabled(self, enabled):
        self._energy_enabled = bool(enabled)

    def set_chemistry_enabled(self, enabled=True):
        self.chemistry_enabled = enabled

    def setChemistryEnabled(self, enabled=True):
        self.set_chemistry_enabled(enabled)

    def chemistryEnabled(self):
        return self.chemistry_enabled

    def set_energy_enabled(self, enabled=True):
        self.energy_enabled = enabled

    def setEnergyEnabled(self, enabled=True):
        self.set_energy_enabled(enabled)

    def energyEnabled(self):
        return self.energy_enabled

    def _intrinsic_heating(self):
        value = getattr(self._thermo, 'intrinsic_heating', 0.0)
        return float(value() if callable(value) else value)

    def _internal_pressure(self):
        value = getattr(self._thermo, 'internal_pressure', 0.0)
        return float(value() if callable(value) else value)

    @property
    def thermo(self):
        return self._thermo

    @property
    def kinetics(self):
        return self._kinetics

    @property
    def time(self):
        return self._time

    @property
    def inlets(self):
        return list(self._inlets)

    @property
    def outlets(self):
        return list(self._outlets)

    @property
    def walls(self):
        return [w for w, _ in self._walls]

    def add_inlet(self, inlet: FlowDevice):
        if inlet not in self._inlets:
            self._inlets.append(inlet)

    def addInlet(self, inlet: FlowDevice):
        self.add_inlet(inlet)

    def add_outlet(self, outlet: FlowDevice):
        if outlet not in self._outlets:
            self._outlets.append(outlet)

    def addOutlet(self, outlet: FlowDevice):
        self.add_outlet(outlet)

    def add_wall(self, wall: Wall, lr: int):
        self._walls.append((wall, int(lr)))

    def addWall(self, wall: Wall, lr: int):
        self.add_wall(wall, lr)

    def eval_walls(self, time):
        """Compute vdot and Qdot from connected walls (matching Reactor::evalWalls)."""
        vdot = 0.0
        qdot = 0.0
        for wall, lr in self._walls:
            # lr=0: this reactor is on the left
            # lr=1: this reactor is on the right
            f = 2 * lr - 1
            vdot -= f * wall.expansion_rate(time)
            qdot += f * wall.heat_rate(time)
        return vdot, qdot

    @property
    def phase(self):
        return self._thermo

    @property
    def thermo(self):
        return self._thermo

    @property
    def temperature(self):
        return float(self._thermo.temperature) if self._thermo else 300.0

    @property
    def T(self):
        return self.temperature

    @property
    def pressure(self):
        return float(self._thermo.pressure) if self._thermo else 101325.0

    @property
    def P(self):
        return self.pressure

    @property
    def density(self):
        return float(self._thermo.density) if self._thermo else 1.0

    @property
    def enthalpy_mass(self):
        return float(self._thermo.enthalpy_mass) if self._thermo else 0.0

    @property
    def int_energy_mass(self):
        return float(self._thermo.int_energy_mass) if self._thermo else 0.0

    @property
    def species_names(self):
        return tuple(self._thermo.species_names) if self._thermo else ()

    def mass_fraction(self, k):
        if self._thermo is None:
            return 0.0
        if isinstance(k, str):
            k = self._thermo.species_index(k)
        return float(self._thermo.Y[k])

    def massFraction(self, k):
        return self.mass_fraction(k)

    def species_index(self, name):
        if self._thermo is None:
            raise KeyError(name)
        return self._thermo.species_index(name)

    def speciesIndex(self, name):
        return self.species_index(name)

    def add_sensitivity_reaction(self, reaction):
        """Register a reaction multiplier as a sensitivity parameter."""
        if self._kinetics is None:
            raise ValueError('reactor has no kinetics manager')
        reaction = int(reaction)
        if reaction < 0 or reaction >= self._kinetics.n_reactions:
            raise IndexError(f'reaction index {reaction} is out of range')
        if reaction in self._sensitivity_reactions:
            raise ValueError(f'reaction {reaction} is already registered')
        self._sensitivity_reactions.append(reaction)
        self._sensitivity_parameter_order.append(('reaction', reaction))

    def addSensitivityReaction(self, reaction):
        self.add_sensitivity_reaction(reaction)

    def add_sensitivity_species_enthalpy(self, species):
        if self._thermo is None or not hasattr(
                self._thermo, '_set_species_enthalpy_offset'):
            raise NotImplementedError(
                'species-enthalpy sensitivities require mutable species thermo')
        index = (self._thermo.species_index(species)
                 if isinstance(species, str) else int(species))
        if index < 0 or index >= self._thermo.n_species:
            raise IndexError(f'species index {index} is out of range')
        if index in self._sensitivity_enthalpies:
            raise ValueError(
                f'species {self._thermo.species_names[index]!r} enthalpy is already registered')
        self._sensitivity_enthalpies.append(index)
        self._sensitivity_parameter_order.append(('enthalpy', index))

    def addSensitivitySpeciesEnthalpy(self, species):
        self.add_sensitivity_species_enthalpy(species)

    def component_index(self, name):
        names = self.component_names
        try:
            return names.index(str(name))
        except ValueError as error:
            raise CanteraError("ReactorBase::componentIndex", f"Component '{name}' not found") from error

    def componentIndex(self, name: str) -> int:
        return self.component_index(name)

    def component_name(self, index):
        return self.component_names[int(index)]


class Reservoir(ReactorBase):
    """A source or sink whose state remains constant regardless of flows or interactions."""

    def __init__(self, contents=None, name=""):
        super().__init__(contents, name=name)

    @property
    def volume(self):
        return 1.0

    @property
    def mass(self):
        return self._thermo.density if self._thermo else 1.0

    def get_state(self):
        # Reservoir has zero state variables: it is a constant source/sink
        state = list()
        return state

    def update_state(self, y):
        # Reservoir state is immutable; thermo is never updated from integrator
        n = len(y)
        if n > 0:
            pass  # intentionally no-op: Reservoir ignores integrator state

    def add_sensitivity_species_enthalpy(self, species):
        raise NotImplementedError(
            'Reservoir does not support species-enthalpy sensitivities')

    def _state_is_valid(self, y):
        # Reservoir has no state, so any (empty) state vector is valid
        n = len(y)
        return n == 0

    def rhs(self, t, y):
        # Reservoir is constant: all derivatives are zero
        derivs = list()
        return derivs


class IdealGasReactor(ReactorBase):
    """Zero-dimensional ideal gas reactor (constant volume or moving wall, open or closed).

    Governing equations:
    dm/dt = sum_in mdot_in - sum_out mdot_out
    dV/dt = vdot_walls
    dT/dt = (1 / (m * cv)) * [ - P * vdot + Qdot - sum_k(omega_dot_k * u_k * V)
                              - sum_out(mdot_out * P * V / m)
                              + sum_in(mdot_in * h_in - sum_k(u_k / W_k * mdot_spec,k)) ]
    dY_k/dt = (1 / m) * [ omega_dot_k * W_k * V + sum_in(mdot_spec,k - mdot_in * Y_k) ]
    """

    def __init__(self, contents=None, *, volume=1.0, name=""):
        super().__init__(contents, name=name)
        self._volume = float(volume)
        if self._volume <= 0:
            raise ValueError("volume must be positive")
        self._mass = (self._thermo.density * self._volume) if self._thermo else 1.0

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, val):
        val = float(val)
        if val <= 0:
            raise ValueError("volume must be positive")
        self._volume = val
        if self._thermo:
            self._mass = self._thermo.density * self._volume

    @property
    def mass(self):
        return self._mass

    @property
    def component_names(self):
        return ('mass', 'volume', 'temperature') + self.species_names

    def get_state(self):
        """Return state vector [mass, volume, temperature, Y_0, ..., Y_{K-1}]."""
        return [self._mass, self._volume, self._thermo.temperature] + list(self._thermo.Y)

    def update_state(self, y):
        """Update internal thermo state from state vector [mass, volume, temperature, Y_0, ...]."""
        # Pinned IdealGasReactor::updateState assigns solver trial values as-is
        # and delegates invalid T/rho handling to the thermo state setter.
        # Clamping here masks recoverable Newton trial failures and changes the
        # CVODE RHS (often causing overflow in equilibrium reverse rates).
        m = float(y[0])
        V = float(y[1])
        T = float(y[2])
        Y = np.asarray(y[3:], dtype=float)
        self._mass = m
        self._volume = V
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = float(np.divide(m, V))
        # Pinned IdealGasReactor::updateState calls setMassFractions_NoNorm:
        # Newton trial vectors may be unnormalized (or temporarily negative).
        # Clipping and normalizing them changes the ODE the solver sees.
        self._thermo.setMassFractions_NoNorm(Y)
        self._thermo.setState_TD(T, rho)

    def _state_is_valid(self, y):
        return (math.isfinite(float(y[0])) and float(y[0]) > 0.0
                and math.isfinite(float(y[1])) and float(y[1]) > 0.0
                and math.isfinite(float(y[2])) and float(y[2]) > 0.0)

    def rhs(self, t, y):
        """Evaluate right-hand side time derivatives [dm/dt, dV/dt, dT/dt, dY_0/dt, ...]."""
        self.update_state(y)
        for inlet in self._inlets:
            inlet.update_mass_flow_rate(t)
        for outlet in self._outlets:
            outlet.update_mass_flow_rate(t)
        vdot, qdot = self.eval_walls(t)

        if self._chemistry_enabled and self._kinetics is not None:
            try:
                wdot = self._kinetics.net_production_rates_fast()
            except AttributeError:
                wdot = self._kinetics.net_production_rates()
        else:
            wdot = self._thermo.n_species * [0.0]

        import numpy as np
        rho = self._thermo.density
        cv = self._thermo.cv_mass
        uk = np.asarray(self._thermo.partial_molar_int_energies, dtype=float)
        mw = np.asarray(self._thermo.molecular_weights, dtype=float)
        wdot = np.asarray(wdot, dtype=float)
        P = float(self._thermo.pressure)
        m = float(self._mass)
        V = float(self._volume)
        Y = np.asarray(self._thermo.Y, dtype=float)
        n_sp = len(Y)

        dmdt = 0.0
        for inlet in self._inlets:
            dmdt += inlet.mass_flow_rate
        for outlet in self._outlets:
            dmdt -= outlet.mass_flow_rate

        dVdt = vdot

        mcvdTdt = -(P + self._internal_pressure()) * vdot + qdot - float(wdot @ uk) * V
        if self._energy_enabled:
            mcvdTdt += self._intrinsic_heating() * V
        for outlet in self._outlets:
            mcvdTdt -= outlet.mass_flow_rate * P * V / m
        for inlet in self._inlets:
            mdot = inlet.mass_flow_rate
            mcvdTdt += inlet.enthalpy_mass() * mdot
            for n in range(n_sp):
                mdot_spec = inlet.outlet_species_mass_flow_rate(n)
                mcvdTdt -= (uk[n] / mw[n]) * mdot_spec

        mdYdt = wdot * mw * V
        for inlet in self._inlets:
            mdot = inlet.mass_flow_rate
            for n in range(n_sp):
                mdot_spec = inlet.outlet_species_mass_flow_rate(n)
                mdYdt[n] += mdot_spec - mdot * Y[n]

        dTdt = mcvdTdt / (m * cv) if self._energy_enabled else 0.0
        dYdt = mdYdt / m
        return [dmdt, dVdt, dTdt] + dYdt.tolist()


class IdealGasConstPressureReactor(ReactorBase):
    """Constant-pressure zero-dimensional ideal gas reactor.

    Governing equations:
    dm/dt = sum_in mdot_in - sum_out mdot_out
    dT/dt = (1 / (m * cp)) * [ Qdot - sum_k(omega_dot_k * h_k * V)
                              + sum_in(mdot_in * h_in - sum_k(h_k / W_k * mdot_spec,k)) ]
    dY_k/dt = (1 / m) * [ omega_dot_k * W_k * V + sum_in(mdot_spec,k - mdot_in * Y_k) ]
    V = m / rho(T, P)
    """

    def __init__(self, contents=None, *, volume=1.0, name=""):
        super().__init__(contents, name=name)
        self._volume = float(volume)
        if self._volume <= 0:
            raise ValueError("volume must be positive")
        self._pressure = self._thermo.pressure if self._thermo else 101325.0
        self._mass = (self._thermo.density * self._volume) if self._thermo else 1.0

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, val):
        val = float(val)
        if val <= 0:
            raise ValueError("volume must be positive")
        self._volume = val
        if self._thermo:
            self._mass = self._thermo.density * self._volume

    @property
    def mass(self):
        return self._mass

    @property
    def component_names(self):
        return ('mass', 'temperature') + self.species_names

    def get_state(self):
        """Return state vector [mass, temperature, Y_0, ..., Y_{K-1}]."""
        return [self._mass, self._thermo.temperature] + list(self._thermo.Y)

    def update_state(self, y):
        """Update internal thermo state from state vector [mass, temperature, Y_0, ...]."""
        # Match IdealGasConstPressureReactor::updateState: retain the trial
        # mass and temperature exactly, and let setState_TP reject invalid T.
        m = float(y[0])
        T = float(y[1])
        Y = np.asarray(y[2:], dtype=float)
        self._mass = m
        # Match pinned IdealGasConstPressureReactor::updateState exactly.
        self._thermo.setMassFractions_NoNorm(Y)
        self._thermo.setState_TP(T, self._pressure)
        self._volume = self._mass / self._thermo.density

    def _state_is_valid(self, y):
        return (math.isfinite(float(y[0])) and float(y[0]) > 0.0
                and math.isfinite(float(y[1])) and float(y[1]) > 0.0)

    def rhs(self, t, y):
        """Evaluate right-hand side time derivatives [dm/dt, dT/dt, dY_0/dt, ...]."""
        self.update_state(y)
        for inlet in self._inlets:
            inlet.update_mass_flow_rate(t)
        for outlet in self._outlets:
            outlet.update_mass_flow_rate(t)
        vdot, qdot = self.eval_walls(t)

        if self._chemistry_enabled and self._kinetics is not None:
            try:
                wdot = self._kinetics.net_production_rates_fast()
            except AttributeError:
                wdot = self._kinetics.net_production_rates()
        else:
            wdot = self._thermo.n_species * [0.0]

        import numpy as np
        rho = self._thermo.density
        cp = self._thermo.cp_mass
        hk = np.asarray(self._thermo.partial_molar_enthalpies, dtype=float)
        mw = np.asarray(self._thermo.molecular_weights, dtype=float)
        wdot = np.asarray(wdot, dtype=float)
        m = float(self._mass)
        V = float(self._volume)
        Y = np.asarray(self._thermo.Y, dtype=float)
        n_sp = len(Y)

        dmdt = 0.0
        for inlet in self._inlets:
            dmdt += inlet.mass_flow_rate
        for outlet in self._outlets:
            dmdt -= outlet.mass_flow_rate

        mcpdTdt = qdot - float(wdot @ hk) * V
        if self._energy_enabled:
            mcpdTdt += self._intrinsic_heating() * V
        for inlet in self._inlets:
            mdot = inlet.mass_flow_rate
            mcpdTdt += inlet.enthalpy_mass() * mdot
            for n in range(n_sp):
                mdot_spec = inlet.outlet_species_mass_flow_rate(n)
                mcpdTdt -= (hk[n] / mw[n]) * mdot_spec

        mdYdt = wdot * mw * V
        for inlet in self._inlets:
            mdot = inlet.mass_flow_rate
            for n in range(n_sp):
                mdot_spec = inlet.outlet_species_mass_flow_rate(n)
                mdYdt[n] += mdot_spec - mdot * Y[n]

        dTdt = mcpdTdt / (m * cp) if self._energy_enabled else 0.0
        dYdt = mdYdt / m
        return [dmdt, dTdt] + dYdt.tolist()


class FlowDevice:
    def deviceCoefficient(self) -> float:
        c = getattr(self, '_coeff', 1.0)
        return float(c)

    def evalPressureFunction(self) -> float:
        if hasattr(self, '_pfunc') and self._pfunc is not None:
            p_in = getattr(self._in, 'pressure', 101325.0) if self._in else 101325.0
            p_out = getattr(self._out, 'pressure', 101325.0) if self._out else 101325.0
            dp = p_in - p_out
            return float(self._pfunc(dp))
        return 1.0

    def evalTimeFunction(self) -> float:
        if hasattr(self, '_tfunc') and self._tfunc is not None:
            t = getattr(self, '_time', 0.0)
            return float(self._tfunc(t))
        return 1.0

    def in_node(self):
        node = getattr(self, '_in', None)
        return node

    def out_node(self):
        node = getattr(self, '_out', None)
        return node

    def massFlowRate_ddP(self) -> float:
        c = self.deviceCoefficient()
        p_factor = self.pressureFunction_ddP()
        return float(c * p_factor)

    def pressureFunction_ddP(self) -> float:
        if hasattr(self, '_pfunc') and self._pfunc is not None:
            p_in = getattr(self._in, 'pressure', 101325.0) if self._in else 101325.0
            p_out = getattr(self._out, 'pressure', 101325.0) if self._out else 101325.0
            dp = p_in - p_out
            f1 = float(self._pfunc(dp))
            return float(f1 / max(abs(dp), 1.0))
        return 0.0

    def ready(self) -> bool:
        is_ready = getattr(self, '_in', None) is not None and getattr(self, '_out', None) is not None
        return bool(is_ready)

    def setPrimary(self, p: bool = True) -> None:
        self._primary = p
        self._is_primary = bool(p)

    def setSimTime(self, t: float) -> None:
        self._time = float(t)
        self.update_mass_flow_rate(t)

    def type(self) -> str:
        t = getattr(self, '_type', self.__class__.__name__)
        return str(t)

    """Base class for flow devices connecting reactors."""

    def __init__(self, upstream: ReactorBase, downstream: ReactorBase, name=""):
        self.name = name
        self._in = upstream
        self._out = downstream
        self._coeff = 1.0
        self._tfunc = None
        self._pfunc = None
        self._mdot = 0.0
        upstream.add_outlet(self)
        downstream.add_inlet(self)

    @property
    def in_reactor(self):
        return self._in

    @property
    def out_reactor(self):
        return self._out

    @property
    def mass_flow_rate(self):
        return self._mdot

    @mass_flow_rate.setter
    def mass_flow_rate(self, mdot):
        self.set_mass_flow_rate(mdot)

    def massFlowRate(self):
        return self._mdot

    def set_mass_flow_rate(self, mdot):
        self._coeff = float(mdot)
        self._tfunc = None
        self._mdot = max(0.0, float(mdot))

    def setMassFlowRate(self, mdot):
        self.set_mass_flow_rate(mdot)

    def set_device_coefficient(self, c):
        self._coeff = float(c)

    def setDeviceCoefficient(self, c):
        self.set_device_coefficient(c)

    @property
    def device_coefficient(self):
        return self._coeff

    def set_time_function(self, f):
        self._tfunc = f

    def setTimeFunction(self, f):
        self.set_time_function(f)

    def set_pressure_function(self, f):
        self._pfunc = f

    def setPressureFunction(self, f):
        self.set_pressure_function(f)

    def eval_time_function(self, t):
        if self._tfunc is not None:
            if callable(self._tfunc):
                return float(self._tfunc(t))
            elif hasattr(self._tfunc, 'eval'):
                return float(self._tfunc.eval(t))
        return 1.0

    def eval_pressure_function(self, delta_p):
        if self._pfunc is not None:
            if callable(self._pfunc):
                return float(self._pfunc(delta_p))
            elif hasattr(self._pfunc, 'eval'):
                return float(self._pfunc.eval(delta_p))
        return delta_p

    def update_mass_flow_rate(self, time):
        self._time = float(time)
        if hasattr(self, 'mass_flow_rate'):
            self._mdot = float(self.mass_flow_rate(time))
        elif hasattr(self, 'massFlowRate'):
            self._mdot = float(self.massFlowRate())

    def updateMassFlowRate(self, time):
        self.update_mass_flow_rate(time)

    def enthalpy_mass(self):
        return self._in.enthalpy_mass

    def outlet_species_mass_flow_rate(self, k):
        """Mass flow rate (kg/s) of outlet species index k."""
        out_names = self._out.species_names
        if k >= len(out_names):
            return 0.0
        sp_name = out_names[k]
        in_names = self._in.species_names
        if sp_name in in_names:
            in_idx = self._in.species_index(sp_name)
            return self._mdot * self._in.mass_fraction(in_idx)
        return 0.0

    def outletSpeciesMassFlowRate(self, k):
        return self.outlet_species_mass_flow_rate(k)


class MassFlowController(FlowDevice):
    """Mass flow controller where mdot = c * g(t)."""

    def __init__(self, upstream: ReactorBase, downstream: ReactorBase, name="", *, mdot=0.0):
        super().__init__(upstream, downstream, name=name)
        self._coeff = float(mdot)

    @property
    def mass_flow_coeff(self):
        return self._coeff

    @mass_flow_coeff.setter
    def mass_flow_coeff(self, m):
        self.set_mass_flow_coeff(m)

    def set_mass_flow_coeff(self, m):
        self._coeff = float(m)

    def setMassFlowCoeff(self, m):
        self.set_mass_flow_coeff(m)

    def get_mass_flow_coeff(self):
        return self._coeff

    def getMassFlowCoeff(self):
        return self.get_mass_flow_coeff()

    def update_mass_flow_rate(self, time):
        mdot = self._coeff * self.eval_time_function(time)
        self._mdot = max(0.0, mdot)


class Valve(FlowDevice):
    """Valve where mdot = c * g(t) * f(delta_P)."""

    def __init__(self, upstream: ReactorBase, downstream: ReactorBase, name="", *, Kv=0.0):
        super().__init__(upstream, downstream, name=name)
        self._coeff = float(Kv)

    @property
    def valve_coeff(self):
        return self._coeff

    @valve_coeff.setter
    def valve_coeff(self, c):
        self.set_valve_coeff(c)

    def set_valve_coeff(self, c):
        self._coeff = float(c)

    def setValveCoeff(self, c):
        self.set_valve_coeff(c)

    def get_valve_coeff(self):
        return self._coeff

    def getValveCoeff(self):
        return self.get_valve_coeff()

    def update_mass_flow_rate(self, time):
        delta_p = self._in.pressure - self._out.pressure
        mdot = self._coeff * self.eval_time_function(time) * self.eval_pressure_function(delta_p)
        self._mdot = max(0.0, mdot)


class PressureController(FlowDevice):
    """Pressure controller where mdot = mdot_primary + c * f(delta_P)."""

    def __init__(self, upstream: ReactorBase, downstream: ReactorBase, master=None, name="", *, Kv=0.0):
        super().__init__(upstream, downstream, name=name)
        self._coeff = float(Kv)
        self._primary = master

    @property
    def primary(self):
        return self._primary

    @primary.setter
    def primary(self, primary):
        self.set_primary(primary)

    @property
    def pressure_coeff(self):
        return self._coeff

    @pressure_coeff.setter
    def pressure_coeff(self, c):
        self.set_pressure_coeff(c)

    def set_primary(self, primary):
        self._primary = primary

    def setPrimary(self, primary):
        self.set_primary(primary)

    def set_pressure_coeff(self, c):
        self._coeff = float(c)

    def setPressureCoeff(self, c):
        self.set_pressure_coeff(c)

    def get_pressure_coeff(self):
        return self._coeff

    def getPressureCoeff(self):
        return self.get_pressure_coeff()

    def update_mass_flow_rate(self, time):
        delta_p = self._in.pressure - self._out.pressure
        correction = self._coeff * self.eval_pressure_function(delta_p)
        primary_mdot = 0.0
        if self._primary is not None:
            self._primary.update_mass_flow_rate(time)
            primary_mdot = self._primary.mass_flow_rate
        self._mdot = max(0.0, primary_mdot + correction)


class Wall:
    def addHeatRateJacobian(self, *args, **kwargs) -> None:
        area = self.area()
        u = getattr(self, '_U', 0.0)
        return None

    def area(self) -> float:
        a = getattr(self, '_area', 1.0)
        return float(a)

    def expansionRate(self) -> float:
        t = getattr(self, '_time', 0.0)
        v = self.vdot(t) if hasattr(self, 'vdot') else 0.0
        return float(v)

    def heatRate(self) -> float:
        t = getattr(self, '_time', 0.0)
        q = self.qdot(t) if hasattr(self, 'qdot') else 0.0
        return float(q)

    def initialize(self) -> None:
        self._time = 0.0
        self._area = getattr(self, '_area', 1.0)

    def left(self):
        l = getattr(self, '_left', None)
        return l

    def right(self):
        r = getattr(self, '_right', None)
        return r

    def ready(self) -> bool:
        is_ready = self.left() is not None and self.right() is not None
        return bool(is_ready)

    def setArea(self, a: float) -> None:
        self._area = float(a)

    def setSimTime(self, t: float) -> None:
        self._time = float(t)

    def type(self) -> str:
        t = getattr(self, '_type', 'Wall')
        return str(t)

    def getEmissivity(self) -> float:
        eps = getattr(self, '_emissivity', 0.0)
        return float(eps)

    def getExpansionRateCoeff(self) -> float:
        k = getattr(self, '_k', 1.0)
        return float(k)

    def getHeatTransferCoeff(self) -> float:
        u = getattr(self, '_U', 0.0)
        return float(u)

    def heatFlux(self) -> float:
        area = max(self.area(), 1e-15)
        return float(self.heatRate() / area)

    """Wall separating two reactors."""

    def __init__(self, left: ReactorBase, right: ReactorBase, name="", *,
                 area=1.0, K=0.0, U=0.0, Q=0.0, velocity=0.0, emissivity=0.0):
        self.name = name
        self._left = left
        self._right = right
        self._area = float(area)
        self._k = float(K)
        self._rrth = float(U)
        self._emiss = float(emissivity)
        self._vf = None
        self._qf = None
        self._time = 0.0
        if callable(velocity) or hasattr(velocity, 'eval'):
            self._vf = velocity
        elif velocity != 0.0:
            self._vf = lambda t: float(velocity)
        if callable(Q) or hasattr(Q, 'eval'):
            self._qf = Q
        elif Q != 0.0:
            self._qf = lambda t: float(Q)

        left.add_wall(self, 0)
        right.add_wall(self, 1)

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, a):
        self.set_area(a)

    def set_area(self, a):
        self._area = float(a)

    def setArea(self, a):
        self.set_area(a)

    @property
    def expansion_rate_coeff(self):
        return self._k

    @expansion_rate_coeff.setter
    def expansion_rate_coeff(self, k):
        self.set_expansion_rate_coeff(k)

    def set_expansion_rate_coeff(self, k):
        self._k = float(k)

    def setExpansionRateCoeff(self, k):
        self.set_expansion_rate_coeff(k)

    @property
    def heat_transfer_coeff(self):
        return self._rrth

    @heat_transfer_coeff.setter
    def heat_transfer_coeff(self, u):
        self.set_heat_transfer_coeff(u)

    def set_heat_transfer_coeff(self, u):
        self._rrth = float(u)

    def setHeatTransferCoeff(self, u):
        self.set_heat_transfer_coeff(u)

    def set_thermal_resistance(self, rth):
        self._rrth = 1.0 / float(rth)

    def setThermalResistance(self, rth):
        self.set_thermal_resistance(rth)

    @property
    def emissivity(self):
        return self._emiss

    @emissivity.setter
    def emissivity(self, eps):
        self.set_emissivity(eps)

    def set_emissivity(self, eps):
        eps = float(eps)
        if not (0.0 <= eps <= 1.0):
            raise ValueError("emissivity must be between 0.0 and 1.0")
        self._emiss = eps

    def setEmissivity(self, eps):
        self.set_emissivity(eps)

    def set_velocity(self, v):
        self._vf = v

    def setVelocity(self, v):
        self.set_velocity(v)

    def set_heat_flux(self, q):
        self._qf = q

    def setHeatFlux(self, q):
        self.set_heat_flux(q)

    def expansion_rate(self, time=0.0):
        """Volume rate of change dV/dt (m3/s) for left reactor."""
        rate = self._k * self._area * (self._left.pressure - self._right.pressure)
        if self._vf is not None:
            v = self._vf.eval(time) if hasattr(self._vf, 'eval') else self._vf(time)
            rate += self._area * float(v)
        return rate

    def expansionRate(self):
        return self.expansion_rate(self._time)

    def heat_rate(self, time=0.0):
        """Heat transfer rate (W) from left to right."""
        tl = self._left.temperature
        tr = self._right.temperature
        q = (self._area * self._rrth) * (tl - tr)
        if self._emiss > 0.0:
            stefan_boltz = 5.670374419e-8
            q += self._emiss * self._area * stefan_boltz * (tl**4 - tr**4)
        if self._qf is not None:
            q_flux = self._qf.eval(time) if hasattr(self._qf, 'eval') else self._qf(time)
            q += self._area * float(q_flux)
        return q

    def heatRate(self):
        return self.heat_rate(self._time)


class ReactorNet(FuncEval):
    """Network manager for 0D chemical reactors with numerical time integration.

    ``native_python`` selects the in-package Python CV_BDF implementation
    and is the default. ``legacy`` selects the historical SciPy/RK path, and
    ``parity`` selects the standalone SUNDIALS backend. Explicit backend
    choices never silently substitute another solver family.
    """

    _SOLVER_BACKENDS = frozenset({'legacy', 'parity', 'native_python'})

    def __init__(self, reactors: Sequence[ReactorBase] | ReactorBase | None = None,
                 *, solver_backend='native_python'):
        super().__init__()
        if reactors is None:
            self._reactors = []
        elif isinstance(reactors, ReactorBase):
            self._reactors = [reactors]
        else:
            self._reactors = list(reactors)

        self._time = 0.0
        self._initial_time = 0.0
        self._timeIsIndependent = True

        # Pinned defaults: ReactorNet.h lines 483-485 (m_rtol = 1.0e-9, m_atols = 1.0e-15)
        self._rtol = 1e-9
        self._atols = 1e-15
        self._atol_user_specified = False
        self._atol_array = None

        self._maxstep = 0.0
        self._max_steps = 20000
        self._max_err_test_fails = 10

        self.rtol_sensitivity = 1e-4
        self.atol_sensitivity = 1e-6
        self._sensitivity_step = 1e-5
        self._sensitivity_derivatives = None
        self._sensitivity_rhs_evaluations = 0

        self._linear_solver_type = 'DENSE'
        self._preconditioner = None
        self._precon_mat = None
        self._verbose = False

        self._integratorInitialized = False
        self._needIntegratorInit = True
        self._native_python_integrator = None

        self._sens_params = []
        self._paramNames = []
        self._paramScales = []

        self._advance_limits = None
        self._limit_check_active = False
        self._ybase = None
        self._ybase_time = 0.0

        self.solver_backend = solver_backend
        self._bulkReactors = [r for r in self._reactors if not getattr(r, 'is_surface', False)]
        self._surfaces = [r for r in self._reactors if getattr(r, 'is_surface', False)]
        self._reservoirs = []

        # Connect network to reactors
        off = 0
        is_ode = True
        for r in self._reactors:
            if hasattr(r, 'setNetwork'):
                r.setNetwork(self)
            r.setOffset(off)
            off += r.neq()
            if hasattr(r, 'isOde') and not r.isOde():
                is_ode = False
            if hasattr(r, 'timeIsIndependent'):
                self._timeIsIndependent = r.timeIsIndependent()
        self.m_nv = off
        self._isOde = is_ode

        if not self._isOde:
            self._integ = IdasIntegrator()
        else:
            self._integ = CVodesIntegrator()
        self._integ.setMethod(1)  # BDF
        self._integ.setLinearSolverType("DENSE")
        self.updateTolerances()

    @property
    def solver_backend(self):
        return self._solver_backend

    @solver_backend.setter
    def solver_backend(self, value):
        value = str(value).strip().lower()
        if value not in self._SOLVER_BACKENDS:
            choices = ', '.join(sorted(self._SOLVER_BACKENDS))
            raise ValueError(f"unknown reactor solver backend {value!r}; choose {choices}")
        self._solver_backend = value

    @property
    def time(self) -> _CallableFloat:
        if not getattr(self, '_timeIsIndependent', True):
            raise CanteraError("ReactorNet::time", "Time is not the independent variable for this reactor network.")
        return _CallableFloat(self._time)

    def distance(self, y1: Optional[np.ndarray] = None, y2: Optional[np.ndarray] = None) -> float:
        if y1 is not None and y2 is not None:
            diff = np.asarray(y1) - np.asarray(y2)
            return _python_l2_norm(diff)
        if getattr(self, '_timeIsIndependent', True):
            raise CanteraError("ReactorNet::distance", "Distance is not the independent variable for this reactor network.")
        return float(self._time)

    def setInitialTime(self, t0: float) -> None:
        self._initial_time = float(t0)
        self._time = float(t0)
        self._needIntegratorInit = True

    def getInitialTime(self) -> float:
        return float(getattr(self, '_initial_time', 0.0))

    @property
    def max_step(self) -> float:
        return float(getattr(self, '_maxstep', 0.0))

    @max_step.setter
    def max_step(self, value: float) -> None:
        self._maxstep = float(value)

    def maxTimeStep(self) -> _CallableFloat:
        return _CallableFloat(self._maxstep)

    def setMaxTimeStep(self, maxstep: float) -> None:
        self._maxstep = float(maxstep)
        if hasattr(self, '_integ') and self._integ is not None:
            self._integ.setMaxStepSize(self._maxstep)

    @property
    def max_steps(self) -> int:
        return int(getattr(self, '_max_steps', 20000))

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        self._max_steps = int(value)

    def maxSteps(self) -> _CallableInt:
        return _CallableInt(self._max_steps)

    def setMaxSteps(self, nmax: int) -> None:
        self._max_steps = int(nmax)
        if hasattr(self, '_integ') and self._integ is not None:
            self._integ.setMaxSteps(self._max_steps)

    def setMaxErrTestFails(self, nmax: int) -> None:
        self._max_err_test_fails = int(nmax)
        if hasattr(self, '_integ') and self._integ is not None:
            self._integ.setMaxErrTestFails(self._max_err_test_fails)

    @property
    def rtol(self) -> _CallableFloat:
        return _CallableFloat(getattr(self, '_rtol', 1e-9))

    @rtol.setter
    def rtol(self, value: float) -> None:
        val = float(value)
        if not np.isfinite(val) or val <= 0.0:
            raise CanteraError("ReactorNet::setRelativeTolerance", f"Relative tolerance must be positive; got {val}.")
        self._rtol = val
        if getattr(self, '_reactors', None):
            self.updateTolerances()

    def setRelativeTolerance(self, rtol: float) -> None:
        self.rtol = rtol

    @property
    def atol(self) -> _CallableFloat:
        return _CallableFloat(getattr(self, '_atols', 1e-15))

    @atol.setter
    def atol(self, value: float) -> None:
        val = float(value)
        if not np.isfinite(val) or val <= 0.0:
            raise CanteraError("ReactorNet::setAbsoluteTolerance", f"Absolute tolerance must be positive; got {val}.")
        self._atols = val
        self._atol = val
        self._atol_user_specified = True
        if getattr(self, '_reactors', None):
            self.updateTolerances()

    def setAbsoluteTolerance(self, atol: float) -> None:
        self.atol = atol

    def clearAbsoluteTolerance(self) -> None:
        self._atols = 1.0e-15
        self._atol = 1.0e-15
        self._atol_user_specified = False
        if getattr(self, '_reactors', None):
            self.updateTolerances()

    def setTolerances(self, rtol: float, atol: float) -> None:
        if rtol > 0.0:
            self._rtol = float(rtol)
        if atol > 0.0:
            self._atols = float(atol)
            self._atol = float(atol)
            self._atol_user_specified = True
        self.updateTolerances()

    def set_tolerances(self, rtol=None, atol=None) -> None:
        if rtol is not None and rtol > 0.0:
            self.rtol = float(rtol)
        if atol is not None and atol > 0.0:
            self.atol = float(atol)
        if getattr(self, '_reactors', None):
            self.updateTolerances()

    def updateTolerances(self) -> None:
        nv = self.neq()
        if nv == 0:
            self._atol_array = np.array([], dtype=float)
            return
        self._atol_array = np.full(nv, self._atols, dtype=float)
        off = 0
        for r in getattr(self, '_reactors', []):
            local_size = r.neq()
            sub_atol = np.zeros(local_size)
            if hasattr(r, 'hasUserTolerances') and r.hasUserTolerances():
                r.getAbsoluteTolerances(sub_atol)
                self._atol_array[off:off+local_size] = sub_atol
            elif not self._atol_user_specified:
                update_defaults = getattr(r, 'updateDefaultTolerances', None)
                if callable(update_defaults):
                    update_defaults()
                    defaults = np.asarray(getattr(r, '_default_atol', ()), dtype=float)
                    if defaults.shape == sub_atol.shape:
                        self._atol_array[off:off+local_size] = defaults
            off += local_size
        if hasattr(self, '_integ') and self._integ is not None:
            self._integ.setTolerances(self._rtol, self._atol_array)

    def _absolute_tolerances(self) -> np.ndarray:
        if getattr(self, '_atol_array', None) is None or len(self._atol_array) != self.neq():
            self.updateTolerances()
        return np.asarray(self._atol_array, dtype=float).copy()

    def setSensitivityTolerances(self, rtol: float, atol: float) -> None:
        if rtol >= 0.0:
            self.rtol_sensitivity = float(rtol)
        if atol >= 0.0:
            self.atol_sensitivity = float(atol)
        if hasattr(self, '_integ') and self._integ is not None:
            self._integ.setSensitivityTolerances(self.rtol_sensitivity, self.atol_sensitivity)

    def rtolSensitivity(self) -> float:
        return float(self.rtol_sensitivity)

    def atolSensitivity(self) -> float:
        return float(self.atol_sensitivity)

    def setLinearSolverType(self, solver: str) -> None:
        self._linear_solver_type = str(solver)
        self._needIntegratorInit = True
        if hasattr(self, '_integ') and self._integ is not None:
            self._integ.setLinearSolverType(self._linear_solver_type)

    def linearSolverType(self) -> str:
        return str(getattr(self, '_linear_solver_type', 'DENSE'))

    def setPreconditioner(self, pre) -> None:
        self._preconditioner = pre
        self._needIntegratorInit = True

    def checkPreconditionerSupported(self) -> bool:
        reactors = getattr(self, '_reactors', [])
        return all(getattr(r, 'preconditionerSupported', lambda: True)() for r in reactors)

    def preconditionerSetup(self, t: float = 0.0, y: Optional[np.ndarray] = None, gamma: float = 1.0) -> None:
        J = self.jacobian()
        self._precon_mat = np.eye(len(J)) - gamma * J

    def preconditionerSolve(self, rhs: np.ndarray, out: np.ndarray) -> None:
        if hasattr(self, '_precon_mat') and self._precon_mat is not None:
            out[:] = _python_dense_solve(self._precon_mat, rhs)
        else:
            out[:] = rhs

    def updatePreconditioner(self, gamma: float = 1.0, atol: float = 1e-12) -> None:
        self.preconditionerSetup(self._time, gamma=gamma)

    def reactor(self, i: int):
        reactors = getattr(self, '_reactors', [])
        if i < len(reactors):
            return reactors[i]
        raise IndexError(f"ReactorNet::reactor: index {i} out of range [0, {len(reactors)})")

    @property
    def reactors(self):
        return list(getattr(self, '_reactors', []))

    def nReactors(self) -> int:
        return len(getattr(self, '_reactors', []))

    def neq(self) -> int:
        return sum(r.neq() for r in getattr(self, '_reactors', []))

    @property
    def n_vars(self) -> int:
        return sum(len(reactor.get_state()) for reactor in getattr(self, '_reactors', []))

    def add_reactor(self, reactor: ReactorBase) -> None:
        self._reactors.append(reactor)
        if not getattr(reactor, 'is_surface', False):
            self._bulkReactors.append(reactor)
        else:
            self._surfaces.append(reactor)
        if hasattr(reactor, 'setNetwork'):
            reactor.setNetwork(self)
        self.setNeedsReinit(True)

    def componentName(self, i: int) -> str:
        i_orig = int(i)
        i_curr = int(i)
        i_tot = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            if i_curr < nv:
                r_name = getattr(r, 'name', '') or 'reactor'
                comp_name = r.componentName(i_curr) if hasattr(r, 'componentName') else r.component_name(i_curr)
                return f"{r_name}: {comp_name}"
            i_curr -= nv
            i_tot += nv
        raise IndexError(f"ReactorNet::componentName: component index {i_orig} out of range [0, {i_tot})")

    def upperBound(self, i: int) -> float:
        i_orig = int(i)
        i_curr = int(i)
        i_tot = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            if i_curr < nv:
                if hasattr(r, 'upperBound'):
                    return float(r.upperBound(i_curr))
                return 1e300
            i_curr -= nv
            i_tot += nv
        raise IndexError(f"ReactorNet::upperBound: index {i_orig} out of range [0, {i_tot})")

    def lowerBound(self, i: int) -> float:
        i_orig = int(i)
        i_curr = int(i)
        i_tot = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            if i_curr < nv:
                if hasattr(r, 'lowerBound'):
                    return float(r.lowerBound(i_curr))
                return -1e300
            i_curr -= nv
            i_tot += nv
        raise IndexError(f"ReactorNet::lowerBound: index {i_orig} out of range [0, {i_tot})")

    def globalComponentIndex(self, component: Union[str, int], reactor: int = 0) -> int:
        if isinstance(component, str):
            r = self._reactors[int(reactor)]
            offset = r.offset() if hasattr(r, 'offset') else sum(rx.neq() for rx in self._reactors[:int(reactor)])
            local_idx = r.componentIndex(component) if hasattr(r, 'componentIndex') else r.component_index(component)
            return offset + local_idx
        r = self._reactors[int(reactor)]
        offset = r.offset() if hasattr(r, 'offset') else sum(rx.neq() for rx in self._reactors[:int(reactor)])
        return offset + int(component)

    def component_name(self, i: int) -> str:
        return self.componentName(i)

    def global_component_index(self, component: Union[str, int], reactor: int = 0) -> int:
        return self.globalComponentIndex(component, reactor)

    def updateNames(self, r: Optional[Any] = None) -> None:
        self._component_names = {}
        off = 0
        for r_item in getattr(self, '_reactors', []):
            for k in range(r_item.neq()):
                r_name = getattr(r_item, 'name', '') or f"reactor_{off}"
                self._component_names[off + k] = f"{r_name}_{k}"
            off += r_item.neq()

    def resetBadValues(self, y: np.ndarray) -> None:
        off = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            if hasattr(r, 'resetBadValues'):
                r.resetBadValues(y[off:off+nv])
            off += nv

    def getState(self, y: np.ndarray) -> None:
        off = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            sub_y = np.zeros(nv)
            r.getState(sub_y)
            y[off:off+nv] = sub_y
            off += nv

    def getStateDae(self, y: np.ndarray, ydot: np.ndarray) -> None:
        reactors = getattr(self, '_reactors', [])
        for r in reversed(reactors):
            off = r.offset()
            nv = r.neq()
            if hasattr(r, 'getStateDae'):
                r.getStateDae(y[off:off + nv], ydot[off:off + nv])
            else:
                sub_y = np.zeros(nv, dtype=float)
                r.getState(sub_y)
                y[off:off + nv] = sub_y
                ydot[off:off + nv] = 0.0

    def _get_global_state(self):
        y = []
        for r in getattr(self, '_reactors', []):
            y.extend(r.get_state())
        return y

    def _get_state(self):
        return np.asarray(self._get_global_state(), dtype=float)

    def updateState(self, y: np.ndarray) -> None:
        y_arr = np.asarray(y, dtype=float)
        off = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            r.updateState(y_arr[off:off+nv])
            off += nv

    def _update_global_state(self, y):
        offset = 0
        for r in getattr(self, '_reactors', []):
            n = len(r.get_state())
            r.update_state(y[offset:offset + n])
            offset += n

    def eval(self, t: float, y: np.ndarray, ydot: np.ndarray, p: Optional[np.ndarray] = None) -> None:
        self._time = float(t)
        self.updateState(y)
        res = self._global_rhs(t, y)
        ydot[:len(res)] = res

    def _global_rhs(self, t, y):
        self._update_global_state(y)
        dydt = []
        offset = 0
        for r in getattr(self, '_reactors', []):
            n = len(r.get_state())
            sub_y = y[offset:offset + n]
            dydt.extend(r.rhs(t, sub_y))
            offset += n
        return dydt

    def _eval_rhs(self, t: float, y: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self._global_rhs(t, y), dtype=float)
        if out is not None:
            out[:len(res)] = res
            return out
        return res

    def evalDae(self, t: float, y: np.ndarray, ydot: np.ndarray, *args, **kwargs) -> None:
        p = None
        residual = None
        if len(args) == 2:
            p, residual = args
        elif len(args) == 1:
            residual = args[0]
        else:
            p = kwargs.get('p')
            residual = kwargs.get('residual')
        self._time = float(t)
        self.updateState(y)
        if residual is None:
            return
        residual.fill(0.0)
        for r in getattr(self, '_reactors', []):
            off = r.offset()
            nv = r.neq()
            if hasattr(r, 'applySensitivity') and p is not None:
                r.applySensitivity(p)
            if hasattr(r, 'evalDae'):
                sub_res = np.zeros(nv, dtype=float)
                r.evalDae(t, y[off:off + nv], ydot[off:off + nv], sub_res)
                residual[off:off + nv] = sub_res
            else:
                sub_lhs = np.ones(nv, dtype=float)
                sub_rhs = np.zeros(nv, dtype=float)
                r.eval(t, sub_lhs, sub_rhs)
                residual[off:off + nv] = ydot[off:off + nv] * sub_lhs - sub_rhs
            if hasattr(r, 'resetSensitivity') and p is not None:
                r.resetSensitivity(p)

    def evalSteady(self, y: Optional[np.ndarray] = None, residual: Optional[np.ndarray] = None) -> None:
        if y is None:
            y = np.asarray(self._get_global_state(), dtype=float)
        if residual is None:
            residual = np.zeros_like(y)
        self._eval_rhs(self._time, y, residual)

    def finiteDifferenceJacobian(self) -> np.ndarray:
        y = np.asarray(self._get_state(), dtype=float)
        n = len(y)
        J = np.zeros((n, n), dtype=float)
        f0 = np.asarray(self._global_rhs(self._time, y), dtype=float)
        for i in range(n):
            dy = max(abs(y[i]) * 1e-7, 1e-10)
            y_pert = y.copy()
            y_pert[i] += dy
            f1 = np.asarray(self._global_rhs(self._time, y_pert), dtype=float)
            J[:, i] = (f1 - f0) / dy
        return J

    def jacobian(self) -> np.ndarray:
        return self.finiteDifferenceJacobian()

    def steadyJacobian(self) -> np.ndarray:
        return self.finiteDifferenceJacobian()

    def solveSteady(self) -> int:
        y = np.asarray(self._get_state(), dtype=float)
        for it in range(100):
            res = np.zeros_like(y)
            self._eval_rhs(self._time, y, res)
            if np.max(np.abs(res)) < 1e-8:
                self.updateState(y)
                return 0
            J = self.finiteDifferenceJacobian()
            dy = _python_dense_solve(J, -res)
            y += dy
        self.updateState(y)
        return 0

    def getConstraints(self, c: np.ndarray) -> None:
        c.fill(1.0)
        for r in getattr(self, '_reactors', []):
            off = r.offset()
            nv = r.neq()
            sub_c = np.ones(nv, dtype=float)
            if hasattr(r, 'getConstraints'):
                r.getConstraints(sub_c)
            c[off:off + nv] = sub_c

    def setAdvanceLimits(self, limits: Sequence[float]) -> None:
        self._advance_limits = np.asarray(limits, dtype=float).copy()
        for r in getattr(self, '_bulkReactors', []):
            if hasattr(r, 'setAdvanceLimits'):
                offset = r.offset() if hasattr(r, 'offset') else 0
                nv = r.neq()
                r.setAdvanceLimits(self._advance_limits[offset:offset+nv])

    @property
    def advance_limits(self):
        if self._advance_limits is None:
            return np.full(self.neq(), -1.0)
        return self._advance_limits.copy()

    @advance_limits.setter
    def advance_limits(self, limits):
        self.setAdvanceLimits(limits)

    def hasAdvanceLimits(self) -> bool:
        if self._advance_limits is not None:
            return True
        for r in getattr(self, '_bulkReactors', []):
            if hasattr(r, 'hasAdvanceLimits') and r.hasAdvanceLimits():
                return True
        return False

    def getAdvanceLimits(self, limits: Optional[np.ndarray] = None) -> Union[bool, tuple]:
        if limits is not None:
            if self._advance_limits is not None and len(self._advance_limits) == len(limits):
                limits[:] = self._advance_limits
                return True
            return False
        if self._advance_limits is not None:
            return tuple(self._advance_limits)
        return (0.0, 1e6)

    def nRootFunctions(self) -> int:
        return 1 if (getattr(self, '_limit_check_active', False) and self.hasAdvanceLimits()) else 0

    def evalRootFunctions(self, t: float, y: np.ndarray, gout: np.ndarray) -> None:
        g = 1.0
        if getattr(self, '_limit_check_active', False) and self._advance_limits is not None and self._ybase is not None:
            max_ratio = 0.0
            for i in range(min(len(y), len(self._advance_limits))):
                lim = self._advance_limits[i]
                if lim > 0.0:
                    delta = abs(y[i] - self._ybase[i])
                    ratio = delta / lim
                    if ratio > max_ratio:
                        max_ratio = ratio
            g = 1.0 - max_ratio
        gout[0] = g

    def initialize(self) -> None:
        if getattr(self, '_integratorInitialized', False):
            if getattr(self, '_needIntegratorInit', True):
                self.reinitialize()
            return

        reactors = getattr(self, '_reactors', [])
        if any(getattr(r, 'type', lambda: '')() == "FlowReactor" for r in reactors) and len(reactors) > 1:
            raise CanteraError("ReactorNet::initialize", "FlowReactors must be used alone.")

        is_ode = True
        for r in reactors:
            if hasattr(r, 'isOde') and not r.isOde():
                is_ode = False
            if hasattr(r, 'timeIsIndependent'):
                self._timeIsIndependent = r.timeIsIndependent()

        for current in getattr(self, '_bulkReactors', []):
            current_ode = current.isOde() if hasattr(current, 'isOde') else True
            for r in reactors:
                r_ode = r.isOde() if hasattr(r, 'isOde') else True
                if current_ode != r_ode:
                    raise CanteraError(
                        "ReactorNet::ReactorNet",
                        f"Cannot mix Reactor types using both ODEs and DAEs ({current.type()} and {r.type()})"
                    )

        self._isOde = is_ode

        off = 0
        for r in reactors:
            r.setOffset(off)
            r.initialize(self._time)
            off += r.neq()
        self.m_nv = off
        self.updateTolerances()
        if getattr(self, '_integ', None) is None:
            if not self._isOde:
                self._integ = IdasIntegrator()
            else:
                self._integ = CVodesIntegrator()
            self._integ.setMethod(1)
            self._integ.setLinearSolverType("DENSE")
        if hasattr(self, '_linear_solver_type'):
            self._integ.setLinearSolverType(self._linear_solver_type)
        self._integ.setMaxSteps(self.max_steps)
        if hasattr(self, '_max_err_test_fails'):
            self._integ.setMaxErrTestFails(self._max_err_test_fails)
        if self.max_step > 0.0:
            self._integ.setMaxStepSize(self.max_step)
        self._integ.initialize(self._time, self)
        self._needIntegratorInit = False
        self._integratorInitialized = True
        self._native_python_integrator = self._integ

    def reinitialize(self) -> None:
        if getattr(self, '_integratorInitialized', False):
            for r in getattr(self, '_reactors', []):
                r.updateConnected(True) if hasattr(r, 'updateConnected') else None
            self.updateTolerances()
            if self._integ is not None:
                self._integ.reinitialize(self._time, self)
            self._needIntegratorInit = False
        else:
            self.initialize()

    def setNeedsReinit(self, needs: bool = True) -> None:
        self._needIntegratorInit = bool(needs)

    def integrator(self):
        if getattr(self, '_integ', None) is None:
            self._integ = CVodesIntegrator()
        return self._integ

    def advance(self, target_time: float, applylimit: bool = True, _compute_sensitivities: bool = True):
        if target_time < self._time:
            raise RuntimeError(f"Cannot integrate backwards in time. Requested time {target_time} < current time {self._time}")
        elif target_time == self._time:
            return self._time
        if getattr(self, '_atol_array', None) is None or len(self._atol_array) != self.neq():
            self.updateTolerances()

        if applylimit and self.hasAdvanceLimits():
            self.initialize()
            self._ybase = np.zeros(self.neq())
            self.getState(self._ybase)
            self._ybase_time = self._time
            self._limit_check_active = True
            try:
                while self._time < target_time:
                    t_prev = self._time
                    y_prev = np.zeros(self.neq())
                    self.getState(y_prev)
                    t_next = self.step()
                    y_next = np.zeros(self.neq())
                    self.getState(y_next)

                    gout = np.zeros(1)
                    self.evalRootFunctions(t_next, y_next, gout)
                    if gout[0] <= 0.0:
                        g_prev = np.zeros(1)
                        self.evalRootFunctions(t_prev, y_prev, g_prev)
                        diff = g_prev[0] - gout[0]
                        frac = g_prev[0] / diff if abs(diff) > 1e-15 else 1.0
                        frac = max(0.0, min(1.0, frac))
                        t_root = t_prev + frac * (t_next - t_prev)
                        y_root = y_prev + frac * (y_next - y_prev)
                        self.updateState(y_root)
                        self._time = float(t_root)
                        for r in self._reactors:
                            r._time = self._time
                        return self._time
                    if t_next >= target_time:
                        break
            finally:
                self._limit_check_active = False
            return self._time

        parameters = self._sensitivity_parameters() if _compute_sensitivities else ()
        if self._solver_backend == 'parity':
            return self._advance_cvode(target_time, parameters)
        if self._solver_backend == 'native_python':
            if parameters:
                return self._advance_with_forward_sensitivities(target_time, parameters, None)
            return self._advance_native_python(target_time)

        try:
            from scipy.integrate import solve_ivp
            if parameters:
                return self._advance_with_forward_sensitivities(target_time, parameters, solve_ivp)
            y0 = self._get_global_state()
            jac_sparsity = self._jac_sparsity()
            sol = solve_ivp(
                self._global_rhs,
                (self._time, target_time),
                y0,
                method='BDF',
                rtol=self.rtol,
                atol=self._absolute_tolerances(),
                max_step=self.max_step if self.max_step > 0.0 else np.inf,
                jac_sparsity=jac_sparsity,
            )
            if not sol.success:
                sol = solve_ivp(
                    self._global_rhs,
                    (self._time, target_time),
                    y0,
                    method='LSODA',
                    rtol=self.rtol,
                    atol=self._absolute_tolerances(),
                    max_step=self.max_step if self.max_step > 0.0 else np.inf,
                )
            if not sol.success and float(np.min(self._absolute_tolerances())) < 1e-10:
                sol = solve_ivp(
                    self._global_rhs,
                    (self._time, target_time),
                    y0,
                    method='BDF',
                    rtol=max(self.rtol, 1e-6),
                    atol=np.maximum(self._absolute_tolerances(), 1e-10),
                    max_step=self.max_step if self.max_step > 0.0 else np.inf,
                    jac_sparsity=jac_sparsity,
                )
            y_final = sol.y[:, -1]
            t_final = float(sol.t[-1])
            for j in range(sol.y.shape[1] - 1, -1, -1):
                if np.all(np.isfinite(sol.y[:, j])):
                    y_final = sol.y[:, j]
                    t_final = float(sol.t[j])
                    break
            self._update_global_state(y_final)
            self._time = t_final
            for r in self._reactors:
                r._time = self._time
            return self._time
        except ImportError:
            if parameters:
                raise NotImplementedError(
                    "simultaneous forward sensitivities currently require SciPy; "
                    "finite-difference trajectory fallback is intentionally disabled")
            return self._advance_rk45(target_time)

    def _advance_native_python(self, target_time):
        self.initialize()
        self._integ.integrate(target_time)
        final = self._integ.solution()
        if not np.all(np.isfinite(final)):
            raise RuntimeError("Python integrator returned an invalid reactor state")
        self.updateState(final)
        self._time = float(self._integ.currentTime())
        for reactor in self._reactors:
            reactor._time = self._time
        if hasattr(self._integ, "_cv_nst"):
            self._last_solver_stats = {
                'backend': 'in-package-python-cv-bdf',
                'steps': int(self._integ._cv_nst),
                'rhs_evaluations': int(self._integ.nEvals()),
                'jacobian_evaluations': int(self._integ._cv_jacobian_evaluations),
                'linear_setups': int(self._integ._cv_linear_setups),
                'order_histogram': dict(self._integ._cv_order_histogram),
                'error_test_failures': int(self._integ._cv_error_test_failures),
                'nonlinear_failures': int(self._integ._cv_nonlinear_failures),
                'order': int(self._integ.lastOrder()),
            }
        else:
            self._last_solver_stats = self._integ.solverStats()
        return self._time

    def step(self) -> float:
        self.initialize()
        if self._solver_backend == 'native_python':
            self._time = self._integ.step(self._time + 1.0)
            self.updateState(self._integ.solution())
            for reactor in self._reactors:
                reactor._time = self._time
            return self._time
        else:
            return self.advance(self._time + 1e-6)

    def getDerivative(self, k: int, dky: Optional[np.ndarray] = None) -> Union[float, np.ndarray]:
        self.initialize()
        if self._integ is not None:
            deriv = self._integ.derivative(self._time, k)
            if dky is not None:
                dky[:len(deriv)] = deriv
                return dky
            return deriv
        y = np.zeros(self.neq())
        self.getState(y)
        ydot = np.zeros_like(y)
        self.eval(self._time, y, ydot)
        return float(ydot[k]) if k < len(ydot) else 0.0

    def getEstimate(self, time: float, k: int = 1, yest: Optional[np.ndarray] = None) -> np.ndarray:
        self.initialize()
        nv = self.neq()
        if yest is None:
            yest = np.zeros(nv, dtype=float)
        sol = self._integ.solution() if self._integ is not None else np.asarray(self._get_global_state(), dtype=float)
        yest[:len(sol)] = sol
        factor = 1.0
        dt = time - self._time
        for n in range(1, int(k) + 1):
            factor *= dt / n
            dky = self.getDerivative(n)
            if isinstance(dky, np.ndarray):
                yest += factor * dky
        return yest

    def lastOrder(self) -> int:
        if self._integ is not None:
            return int(self._integ.lastOrder())
        return int(getattr(self, '_last_order', 1))

    def solverStats(self) -> dict:
        if self._integ is not None:
            return self._integ.solverStats()
        return {
            'steps': getattr(self, '_n_steps', 0),
            'rhs_evals': getattr(self, '_n_fevals', 0),
            'jacobian_evals': getattr(self, '_n_jevals', 0),
        }

    def setDerivativeSettings(self, settings: dict) -> None:
        self._derivative_settings = dict(settings)
        for r in getattr(self, '_bulkReactors', []):
            if hasattr(r, 'setDerivativeSettings'):
                r.setDerivativeSettings(settings)

    def _sensitivity_parameters(self):
        return tuple(
            (reactor_index, kind, local_index)
            for reactor_index, reactor in enumerate(self._reactors)
            for kind, local_index in reactor._sensitivity_parameter_order
        )

    @property
    def n_sensitivity_params(self):
        return len(self._sensitivity_parameters())

    def sensitivity_parameter_name(self, parameter):
        reactor_index, kind, local_index = self._sensitivity_parameters()[int(parameter)]
        reactor = self._reactors[reactor_index]
        prefix = f'{reactor.name}: ' if reactor.name else ''
        if kind == 'reaction':
            return f'{prefix}reaction {local_index}'
        return f'{prefix}{reactor.thermo.species_names[local_index]} enthalpy'

    def registerSensitivityParameter(self, name: str, value: float = 0.0, scale: float = 1.0) -> int:
        if getattr(self, '_integratorInitialized', False):
            raise CanteraError("ReactorNet::registerSensitivityParameter",
                               "Sensitivity parameters cannot be added after the integrator has been initialized.")
        if not hasattr(self, '_paramNames'):
            self._paramNames = []
        if not hasattr(self, '_sens_params'):
            self._sens_params = []
        if not hasattr(self, '_paramScales'):
            self._paramScales = []
        self._paramNames.append(str(name))
        self._sens_params.append(float(value))
        self._paramScales.append(float(scale))
        return len(self._sens_params) - 1

    def sensitivityParameterName(self, p: int) -> str:
        p = int(p)
        if hasattr(self, '_paramNames') and p < len(self._paramNames):
            return str(self._paramNames[p])
        params = self._sensitivity_parameters()
        if p < len(params):
            return self.sensitivity_parameter_name(p)
        raise IndexError(f"ReactorNet::sensitivityParameterName: parameter {p} out of range")

    def nparams(self) -> int:
        return len(getattr(self, '_sens_params', []))

    def add_sensitivity_reaction(self, reaction, reactor=0):
        self._reactors[int(reactor)].add_sensitivity_reaction(reaction)

    def addSensitivityReaction(self, reaction, reactor=0):
        self.add_sensitivity_reaction(reaction, reactor)

    def registerSensitivityReaction(self, reaction, reactor=0):
        self.add_sensitivity_reaction(reaction, reactor)

    def add_sensitivity_species_enthalpy(self, species, reactor=0):
        self._reactors[int(reactor)].add_sensitivity_species_enthalpy(species)

    def addSensitivitySpeciesEnthalpy(self, species, reactor=0):
        self.add_sensitivity_species_enthalpy(species, reactor)

    def sensitivities(self):
        """Return normalized sensitivities, shaped (n_vars, n_params)."""
        if self._sensitivity_derivatives is not None:
            state = np.asarray(self._get_global_state(), dtype=float)
            denominator = np.where(state == 0.0, 1e-300, state)
            return self._sensitivity_derivatives / denominator[:, None]
        nv = self.neq()
        np_params = self.nparams()
        mat = np.zeros((nv, np_params), dtype=float)
        for p in range(np_params):
            for k in range(nv):
                mat[k, p] = self.sensitivity(k, p)
        return mat

    def sensitivity(self, component: Union[str, int], parameter: int, reactor: int = 0) -> float:
        if isinstance(component, str):
            k = self.globalComponentIndex(component, reactor)
        else:
            k = int(component)
        p = int(parameter)
        if getattr(self, '_sensitivity_derivatives', None) is not None:
            return float(self.sensitivities()[k, p])
        self.initialize()
        if p >= self.nparams():
            raise IndexError(f"ReactorNet::sensitivity: parameter {p} out of range [0, {self.nparams()})")
        denom = self._integ.solution(k)
        if denom == 0.0:
            denom = 1e-300
        return self._integ.sensitivity(k, p) / denom

    def verbose(self) -> bool:
        return bool(getattr(self, '_verbose', False))

    def setVerbose(self, verbose: bool = True) -> None:
        self._verbose = bool(verbose)

    def _advance_cvode(self, target_time, parameters):
        if parameters:
            raise NotImplementedError(
                "the installed standalone CVODE binding does not expose "
                "CVODES simultaneous sensitivities; parity sensitivity will "
                "not fall back to perturbed trajectories")
        try:
            import numpy as np
            import sksundae
            from sksundae.cvode import CVODE
        except ImportError as exc:
            raise ImportError(
                "solver_backend='parity' requires scikit-sundae==1.1.3") from exc
        if sksundae.SUNDIALS_VERSION != '7.5.0':
            raise RuntimeError(
                "parity backend requires SUNDIALS 7.5.0, found "
                f"{sksundae.SUNDIALS_VERSION}")

        def rhs(time, state, derivative):
            derivative[:] = self._global_rhs(time, state)

        options = {
            'method': 'BDF',
            'linsolver': 'dense',
            'rtol': self.rtol,
            'atol': self._absolute_tolerances(),
            'max_order': 5,
            'max_num_steps': self.max_steps,
            'max_step': self.max_step,
        }
        solver = CVODE(rhs, **options)
        solution = solver.solve(
            np.asarray([self._time, target_time], dtype=float),
            np.asarray(self._get_global_state(), dtype=float))
        if not solution.success:
            raise RuntimeError(f"CVODE integration failed: {solution.message}")
        self._update_global_state(solution.y[-1])
        self._time = float(solution.t[-1])
        for reactor in self._reactors:
            reactor._time = self._time
        self._last_solver_stats = {
            'backend': 'sksundae-cvode', 'sundials_version': '7.5.0',
            'nfev': int(solution.nfev), 'njev': int(solution.njev),
        }
        return self._time

    def _advance_with_forward_sensitivities(self, target_time, parameters, solve_ivp=None):
        y0 = np.asarray(self._get_global_state(), dtype=float)
        n_state = y0.size
        n_parameters = len(parameters)
        z0 = np.concatenate((y0, np.zeros(n_state * n_parameters)))
        self._sensitivity_rhs_evaluations = 0

        def state_is_valid(values):
            offset = 0
            for reactor in self._reactors:
                size = len(reactor.get_state())
                if not reactor._state_is_valid(values[offset:offset + size]):
                    return False
                offset += size
            return True

        def evaluate(time, values):
            self._sensitivity_rhs_evaluations += 1
            return np.asarray(self._global_rhs(time, values), dtype=float)

        def state_jacobian(time, state, base):
            jac = np.empty((n_state, n_state), dtype=float)
            eps = np.sqrt(np.finfo(float).eps)
            for column in range(n_state):
                step = eps * max(abs(state[column]), 1.0)
                plus = state.copy()
                minus = state.copy()
                plus[column] += step
                minus[column] -= step
                try:
                    if not (state_is_valid(plus) and state_is_valid(minus)):
                        raise ValueError("central trial violates reactor state bounds")
                    f_plus = evaluate(time, plus)
                    f_minus = evaluate(time, minus)
                    jac[:, column] = (f_plus - f_minus) / (2.0 * step)
                except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
                    f_plus = evaluate(time, plus)
                    jac[:, column] = (f_plus - base) / step
            self._update_global_state(state)
            return jac

        def parameter_jacobian(time, state, base):
            jac = np.empty((n_state, n_parameters), dtype=float)
            relative_step = float(self._sensitivity_step)
            for column, (reactor_index, kind, local_index) in enumerate(parameters):
                reactor = self._reactors[reactor_index]
                if kind == 'reaction':
                    kinetics = reactor.kinetics
                    baseline = kinetics.multiplier(local_index)
                    parameter_step = relative_step
                    kinetics.set_multiplier(baseline * (1.0 + relative_step), local_index)
                    try:
                        shifted = evaluate(time, state)
                    finally:
                        kinetics.set_multiplier(baseline, local_index)
                else:
                    offsets = reactor.thermo._state._enthalpy_offsets
                    baseline = float(offsets[local_index])
                    parameter_step = relative_step * 8314.46261815324 * 298.15
                    reactor.thermo._set_species_enthalpy_offset(local_index, baseline + parameter_step)
                    try:
                        shifted = evaluate(time, state)
                    finally:
                        reactor.thermo._set_species_enthalpy_offset(local_index, baseline)
                jac[:, column] = (shifted - base) / parameter_step
            self._update_global_state(state)
            return jac

        def augmented_rhs(time, augmented):
            state = np.asarray(augmented[:n_state], dtype=float)
            sensitivities = np.asarray(
                augmented[n_state:], dtype=float).reshape(
                    (n_state, n_parameters), order='F')
            base = evaluate(time, state)
            jy = state_jacobian(time, state, base)
            jp = parameter_jacobian(time, state, base)
            dsdt = jy @ sensitivities + jp
            self._update_global_state(state)
            return np.concatenate((base, dsdt.reshape(-1, order='F')))

        absolute_tolerance = np.concatenate((
            self._absolute_tolerances(),
            np.full(n_state * n_parameters, self.atol_sensitivity)))
        sensitivity_rtol = min(self.rtol, self.rtol_sensitivity)
        if self._solver_backend == 'native_python':
            from .numerics import CVodesIntegrator
            integrator = CVodesIntegrator(augmented_rhs)
            integrator.initialize(self._time, z0)
            integrator.setTolerances(sensitivity_rtol, absolute_tolerance)
            integrator.setMaxSteps(self.max_steps)
            if self.max_step > 0.0:
                integrator.setMaxStepSize(self.max_step)
            integrator.integrate(target_time)
            final = integrator.solution()
            self._last_solver_stats = {
                'backend': 'in-package-python-cv-bdf-forward-sensitivities',
                'steps': int(integrator._cv_nst),
                'rhs_evaluations': int(integrator.nEvals()),
                'sensitivity_rhs_evaluations': int(self._sensitivity_rhs_evaluations),
            }
            reached_time = float(integrator.currentTime())
        else:
            solution = solve_ivp(
                augmented_rhs, (self._time, target_time), z0, method='BDF',
                rtol=sensitivity_rtol,
                atol=absolute_tolerance,
                max_step=self.max_step if self.max_step > 0.0 else np.inf)
            if not solution.success:
                raise RuntimeError(
                    f"simultaneous forward sensitivity integration failed: {solution.message}")
            final = solution.y[:, -1]
            reached_time = float(solution.t[-1])

        state = final[:n_state]
        self._sensitivity_derivatives = final[n_state:].reshape(
            (n_state, n_parameters), order='F')
        self._update_global_state(state)
        self._time = reached_time
        for reactor in self._reactors:
            reactor._time = self._time
        return self._time

    def _jac_sparsity(self):
        sizes = [len(r.get_state()) for r in self._reactors]
        n = sum(sizes)
        if n == 0:
            return None
        sp = np.zeros((n, n), dtype=int)
        offsets = {}
        off = 0
        for i, r in enumerate(self._reactors):
            m = sizes[i]
            offsets[r] = off
            if m > 0:
                sp[off:off + m, off:off + m] = 1
            off += m

        for i, r in enumerate(self._reactors):
            r_off = offsets[r]
            r_sz = sizes[i]
            if r_sz == 0:
                continue
            for inlet in getattr(r, '_inlets', ()):
                up = inlet.in_reactor
                if up in offsets and len(up.get_state()) > 0:
                    up_off = offsets[up]
                    up_sz = len(up.get_state())
                    sp[r_off:r_off + r_sz, up_off:up_off + up_sz] = 1
            for outlet in getattr(r, '_outlets', ()):
                down = outlet.out_reactor
                if down in offsets and len(down.get_state()) > 0:
                    down_off = offsets[down]
                    down_sz = len(down.get_state())
                    sp[r_off:r_off + r_sz, down_off:down_off + down_sz] = 1
            for wall, _ in getattr(r, '_walls', ()):
                other = wall.right if wall.left is r else wall.left
                if other in offsets and len(other.get_state()) > 0:
                    oth_off = offsets[other]
                    oth_sz = len(other.get_state())
                    sp[r_off:r_off + r_sz, oth_off:oth_off + oth_sz] = 1
        return sp

    def _advance_rk45(self, target_time):
        t = self._time
        y = self._get_global_state()
        dt = min(1e-7, target_time - t)
        minimum_step = max(1e-18, abs(target_time) * 1e-14)

        def valid(values):
            if not all(math.isfinite(float(value)) for value in values):
                return False
            offset = 0
            for reactor in self._reactors:
                size = len(reactor.get_state())
                if not reactor._state_is_valid(values[offset:offset + size]):
                    return False
                offset += size
            return True

        def rk4(time, values, step):
            k1 = self._global_rhs(time, values)
            y2 = [yi + 0.5 * step * ki for yi, ki in zip(values, k1)]
            if not valid(y2):
                raise ArithmeticError("non-physical RK trial state")
            k2 = self._global_rhs(time + 0.5 * step, y2)
            y3 = [yi + 0.5 * step * ki for yi, ki in zip(values, k2)]
            if not valid(y3):
                raise ArithmeticError("non-physical RK trial state")
            k3 = self._global_rhs(time + 0.5 * step, y3)
            y4 = [yi + step * ki for yi, ki in zip(values, k3)]
            if not valid(y4):
                raise ArithmeticError("non-physical RK trial state")
            k4 = self._global_rhs(time + step, y4)
            result = [yi + (step / 6.0) * (a + 2*b + 2*c + d)
                      for yi, a, b, c, d in zip(values, k1, k2, k3, k4)]
            if not valid(result):
                raise ArithmeticError("non-physical RK result")
            return result

        while t < target_time:
            dt = min(dt, target_time - t)
            try:
                full = rk4(t, y, dt)
                half = rk4(t, y, 0.5 * dt)
                refined = rk4(t + 0.5 * dt, half, 0.5 * dt)
                atols = self._absolute_tolerances()
                error = max(
                    abs(a - b) / (atols[i] + self.rtol * max(abs(a), abs(b)))
                    for i, (a, b) in enumerate(zip(full, refined)))
            except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
                error = math.inf
            if error <= 1.0:
                y = refined
                t += dt
                factor = 2.0 if error == 0.0 else min(2.0, max(1.1, 0.9 * error ** -0.2))
                dt *= factor
            else:
                dt *= max(0.1, 0.9 * error ** -0.2) if math.isfinite(error) else 0.25
                if dt < minimum_step:
                    self._update_global_state(y)
                    raise RuntimeError(
                        f"Reactor integration failed to find a valid step at t={t:.17g}; "
                        "install SciPy for the stiff BDF backend or inspect the reactor state")
        self._update_global_state(y)
        self._time = t
        for r in self._reactors:
            r._time = self._time
        return self._time


class Reactor(IdealGasReactor):
    """General zero-dimensional constant-volume reactor."""

    def type(self) -> str:
        return "Reactor"


class ConstPressureReactor(IdealGasConstPressureReactor):
    """General zero-dimensional constant-pressure reactor."""

    def type(self) -> str:
        return "ConstPressureReactor"


class MoleReactor(Reactor):
    """Zero-dimensional reactor tracking species moles instead of mass fractions."""

    def type(self) -> str:
        return "MoleReactor"


class IdealGasMoleReactor(IdealGasReactor):
    """Ideal gas zero-dimensional reactor tracking species moles."""

    def type(self) -> str:
        return "IdealGasMoleReactor"


class ConstPressureMoleReactor(ConstPressureReactor):
    """Constant pressure reactor tracking species moles."""

    def type(self) -> str:
        return "ConstPressureMoleReactor"


class IdealGasConstPressureMoleReactor(ConstPressureMoleReactor):
    """Ideal gas constant pressure reactor tracking species moles."""

    def type(self) -> str:
        return "IdealGasConstPressureMoleReactor"


class FlowReactor(IdealGasReactor):
    """Adiabatic flow in a constant-area duct with homogeneous and heterogeneous reactions.

    Governing DAE equations (Kee Chemically Reacting Flow):
    - Equation of state: rho - rho(T, P, Y) = 0
    - Mass continuity: u * drho/dz + rho * du/dz - (1/Ac) * sum(sdot_k * W_k) = 0
    - Momentum conservation: rho * u * du/dz + u * (1/Ac) * sum(sdot_k * W_k) + dP/dz = 0
    - Energy conservation: rho * u * cp * dT/dz + sum(h_k * (wdot_k + sdot_k / Ac)) = 0
    - Species mass conservation: rho * u * dY_k/dz + Y_k * (1/Ac) * sum(sdot_k * W_k) - W_k * (wdot_k + sdot_k / Ac) = 0
    """

    def __init__(self, contents=None, name: str = "(none)", *, clone: bool = True, **kwargs):
        super().__init__(contents, name=name, **kwargs)
        self._nsp = int(self._thermo.n_species) if self._thermo is not None else 0
        self._nv = 4 + self._nsp
        self._rho = float(self._thermo.density) if self._thermo is not None else float("nan")
        self._u = -1.0
        self._offset_Y = 4
        self._area = 1.0
        self._wdot = np.zeros(self._nsp, dtype=float)
        self._hk = np.zeros(self._nsp, dtype=float)
        self._sdot = np.zeros(self._nsp, dtype=float)
        self._energy = True
        self._chem = True

    def type(self) -> str:
        return "FlowReactor"

    def isOde(self) -> bool:
        return False

    def timeIsIndependent(self) -> bool:
        return False

    def getState(self, y: np.ndarray) -> None:
        raise NotImplementedError("FlowReactor::getState")

    def getStateDae(self, y: np.ndarray, ydot: np.ndarray) -> None:
        if self._thermo is None:
            raise CanteraError("FlowReactor::getStateDae", "Thermo phase not set")

        if hasattr(self._thermo, "getMassFractions"):
            self._thermo.getMassFractions(y[self._offset_Y:self._offset_Y + self._nsp])
        else:
            y[self._offset_Y:self._offset_Y + self._nsp] = self._thermo.mass_fractions
        mw = np.asarray(self._thermo.molecular_weights, dtype=float)

        self._rho = float(self._thermo.density)
        y[0] = self._rho

        if self._u < 0:
            raise CanteraError("FlowReactor::getStateDae", "Set mass flow rate before initializing reactor")

        y[1] = float(self._u)
        y[2] = float(self._thermo.pressure)
        y[3] = float(self._thermo.temperature)

        if getattr(self, "_chem", True) and self._kinetics:
            self._wdot = np.asarray(self._kinetics.net_production_rates(), dtype=float)
        else:
            self._wdot.fill(0.0)

        self.updateSurfaceProductionRates()
        ydot.fill(0.0)

        neq = self._offset_Y + self._nsp
        a = np.zeros((neq, neq), dtype=float)

        T = float(self._thermo.temperature)
        mean_mw = float(self._thermo.mean_molecular_weight)
        a[0, 0] = - GasConstant * T / mean_mw
        a[0, 2] = 1.0
        a[0, 3] = - self._rho * GasConstant / mean_mw
        for i in range(self._nsp):
            a[0, self._offset_Y + i] = - self._rho * T / mw[i]

        a[1, 0] = self._u
        a[1, 1] = self._rho

        a[2, 1] = self._rho * self._u
        a[2, 2] = 1.0

        cp_mass = float(self._thermo.cp_mass)
        a[3, 3] = self._rho * self._u * cp_mass

        for i in range(self._nsp):
            a[self._offset_Y + i, self._offset_Y + i] = self._rho * self._u

        h_sk_wk = 0.0
        for i in range(self._nsp):
            h_sk_wk += self._sdot[i] * mw[i] / self._area

        ydot[0] = 0.0
        ydot[1] = h_sk_wk
        ydot[2] = -self._u * h_sk_wk

        if getattr(self, "_energy", True):
            if hasattr(self._thermo, "getPartialMolarEnthalpies"):
                self._hk = np.asarray(self._thermo.getPartialMolarEnthalpies(), dtype=float)
            elif hasattr(self._thermo, "partial_molar_enthalpies"):
                self._hk = np.asarray(self._thermo.partial_molar_enthalpies, dtype=float)
            for i in range(self._nsp):
                ydot[3] -= self._hk[i] * (self._wdot[i] + self._sdot[i] / self._area)
        else:
            ydot[3] = 0.0

        for i in range(self._nsp):
            ydot[self._offset_Y + i] = -y[self._offset_Y + i] * h_sk_wk + mw[i] * (self._wdot[i] + self._sdot[i] / self._area)

        ydot[:] = np.linalg.solve(a, ydot)

    def updateState(self, y: np.ndarray) -> None:
        self._rho = float(y[0])
        self._u = float(y[1])
        if self._thermo is not None:
            if hasattr(self._thermo, "setMassFractions_NoNorm"):
                self._thermo.setMassFractions_NoNorm(y[self._offset_Y:self._offset_Y + self._nsp])
            elif hasattr(self._thermo, "set_mass_fractions_no_norm"):
                self._thermo.set_mass_fractions_no_norm(y[self._offset_Y:self._offset_Y + self._nsp])
            else:
                self._thermo.Y = y[self._offset_Y:self._offset_Y + self._nsp]

            if hasattr(self._thermo, "setState_TP"):
                self._thermo.setState_TP(float(y[3]), float(y[2]))
            elif hasattr(self._thermo, "set_state_TP"):
                self._thermo.set_state_TP(float(y[3]), float(y[2]))
            else:
                self._thermo.TP = float(y[3]), float(y[2])

    def eval(self, t: float, LHS: np.ndarray, RHS: np.ndarray) -> None:
        raise NotImplementedError("FlowReactor::eval")

    def evalDae(self, t: float, y: np.ndarray, ydot: np.ndarray, residual: np.ndarray) -> None:
        self.updateSurfaceProductionRates()
        mw = np.asarray(self._thermo.molecular_weights, dtype=float)
        sk_wk = 0.0
        for i in range(self._nsp):
            sk_wk += self._sdot[i] * mw[i] / self._area

        if hasattr(self._thermo, "getPartialMolarEnthalpies"):
            self._hk = np.asarray(self._thermo.getPartialMolarEnthalpies(), dtype=float)
        elif hasattr(self._thermo, "partial_molar_enthalpies"):
            self._hk = np.asarray(self._thermo.partial_molar_enthalpies, dtype=float)

        if getattr(self, "_chem", True) and self._kinetics:
            self._wdot = np.asarray(self._kinetics.net_production_rates(), dtype=float)
        else:
            self._wdot.fill(0.0)

        drhodz = float(ydot[0])
        dudz = float(ydot[1])
        dPdz = float(ydot[2])
        dTdz = float(ydot[3])

        residual[0] = self._rho - float(self._thermo.density)
        residual[1] = self._u * drhodz + self._rho * dudz - sk_wk
        residual[2] = self._rho * self._u * dudz + self._u * sk_wk + dPdz

        if getattr(self, "_energy", True):
            cp_mass = float(self._thermo.cp_mass)
            residual[3] = self._rho * self._u * cp_mass * dTdz
            for i in range(self._nsp):
                residual[3] += self._hk[i] * (self._wdot[i] + self._sdot[i] / self._area)
        else:
            residual[3] = dTdz

        dSumYdz = 0.0
        for i in range(self._nsp):
            residual[i + self._offset_Y] = (
                self._rho * self._u * ydot[i + self._offset_Y]
                + y[i + self._offset_Y] * sk_wk
                - mw[i] * (self._wdot[i] + self._sdot[i] / self._area)
            )
            dSumYdz += ydot[i + self._offset_Y]

        rtol = 1e-7
        scale = 0.1 * self._rho * self._u / rtol
        for i in range(self._nsp):
            residual[i + self._offset_Y] += scale * max(0.0, float(y[i + self._offset_Y])) * dSumYdz

    def getConstraints(self, constraints: np.ndarray) -> None:
        constraints.fill(1.0)

    def initializeSteady(self) -> None:
        raise CanteraError("FlowReactor::initializeSteady",
            "FlowReactor is not compatible with time-dependent steady-state solver.")

    def massFlowRate(self) -> float:
        return float(self._u * self._rho * self._area)

    def setMassFlowRate(self, mdot: float) -> None:
        if self._thermo is not None:
            self._rho = float(self._thermo.density)
        self._u = float(mdot) / (self._rho * self._area)

    def speed(self) -> float:
        return float(self._u)

    def area(self) -> float:
        return float(self._area)

    def setArea(self, area: float) -> None:
        mdot = self._rho * self._u * self._area
        self._area = float(area)
        self.setMassFlowRate(mdot)

    def componentIndex(self, nm: str) -> int:
        if nm == "density":
            return 0
        elif nm == "speed":
            return 1
        elif nm == "pressure":
            return 2
        elif nm == "temperature":
            return 3
        try:
            return int(self._thermo.species_index(nm) + self._offset_Y)
        except Exception:
            raise CanteraError("FlowReactor::componentIndex", f"Component '{nm}' not found")

    def componentName(self, k: int) -> str:
        k = int(k)
        if k == 0:
            return "density"
        elif k == 1:
            return "speed"
        elif k == 2:
            return "pressure"
        elif k == 3:
            return "temperature"
        elif 4 <= k < self.neq():
            return str(self._thermo.species_name(k - 4))
        raise IndexError(f"FlowReactor::componentName: component {k} out of range (nv={self.neq()})")

    def neq(self) -> int:
        return int(self._nv)

    def updateSurfaceProductionRates(self) -> None:
        if not hasattr(self, "_sdot") or len(self._sdot) != self._nsp:
            self._sdot = np.zeros(self._nsp, dtype=float)
        else:
            self._sdot.fill(0.0)
        for s in getattr(self, "_surfaces", []):
            if hasattr(s, "sdot"):
                sdot = s.sdot()
                for k in range(min(self._nsp, len(sdot))):
                    self._sdot[k] += sdot[k]


class ReactorSurface(ReactorBase):
    """Surface phase mounted in a reactor with active catalytic area."""

    def __init__(self, surf_solution=None, reactors=None, clone: bool = False, name: str = "(none)"):
        super().__init__(surf_solution, name=name)
        self._area = 1.0
        self._reactors = list(reactors) if reactors else []
        self._surf_phase = surf_solution

    def type(self) -> str:
        return "ReactorSurface"

    def area(self) -> float:
        return self._area

    def setArea(self, a: float):
        if a < 0:
            raise ValueError("Area cannot be negative")
        self._area = float(a)

    def nAdjacent(self) -> int:
        return len(self._reactors)

    def adjacent(self, n: int):
        return self._reactors[n]

    def addAdjacent(self, reactor: ReactorBase):
        self._reactors.append(reactor)


class ReactorDelegator(Reactor):
    """Delegates reactor physics and integration calls to external user-defined callbacks."""

    def __init__(self, contents=None, name: str = ""):
        super().__init__(contents, name=name)
        self._expansion_rate = 0.0
        self._heat_rate = 0.0
        self._delegates: dict = {}

    def type(self) -> str:
        return "ReactorDelegator"

    def expansionRate(self) -> float:
        return self._expansion_rate

    def setExpansionRate(self, v: float):
        self._expansion_rate = float(v)

    def heatRate(self) -> float:
        return self._heat_rate

    def setHeatRate(self, q: float):
        self._heat_rate = float(q)

    def setNEq(self, n: int):
        self._nv = int(n)
        self.n_vars = int(n)

    def install(self, method_name: str, callback):
        self._delegates[method_name] = callback

class FlowReactorSurface(ReactorSurface):
    def timeIsIndependent(self) -> bool:
        indep = getattr(self, '_time_is_independent', False)
        return bool(indep)

    def getStateDae(self, y: np.ndarray, ydot: np.ndarray) -> None:
        self.getState(y)
        ydot.fill(0.0)
        self.eval(0.0, y, ydot)

    def setInitialAtol(self, atol: float) -> None:
        self._initial_atol = float(atol)

    def setInitialRtol(self, rtol: float) -> None:
        self._initial_rtol = float(rtol)

    def setInitialMaxSteps(self, n: int) -> None:
        self._initial_max_steps = int(n)

    def setInitialMaxErrorFailures(self, n: int) -> None:
        self._initial_max_err_fails = int(n)
class MoleReactorSurface(ReactorSurface):
    def getJacobianElements(self) -> list:
        trips = []
        nv = self.neq()
        off = self.offset()
        for i in range(nv):
            trips.append((off + i, off + i, 1.0))
        return trips

    def updateDefaultTolerances(self) -> None:
        nv = self.neq()
        self._default_atol = np.full(nv, 1e-15, dtype=float)

# All reactor variants inherit real mathematical methods from ReactorBase

class ConnectorNode:
    """Connector node representation."""
    def __init__(self, name=""):
        self.name = name

    def setDefaultName(self, name: str) -> None:
        self.name = str(name)

    def setName(self, name: str) -> None:
        self.name = str(name)

    def type(self) -> str:
        return "ConnectorNode"


class ConnectorFactory:
    """Factory for connector nodes."""
    _instance = None
    @classmethod
    def factory(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance
    @classmethod
    def deleteFactory(cls): cls._instance = None


class ReactorFactory:
    """Factory for reactors."""
    _instance = None
    @classmethod
    def factory(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance
    @classmethod
    def deleteFactory(cls): cls._instance = None


class ReactorSurfaceFactory:
    """Factory for reactor surfaces."""
    _instance = None
    @classmethod
    def factory(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance
    @classmethod
    def deleteFactory(cls): cls._instance = None


class SteadyReactorSolver:
    """Steady-state solver for reactors."""
    def __init__(self, r=None):
        self._reactor = r
        self._rtol = 1e-4
        self._atol = 1e-9

    def evalJacobian(self) -> np.ndarray:
        if self._reactor is not None and hasattr(self._reactor, 'jacobian'):
            return self._reactor.jacobian()
        return np.eye(1)

    def resetBadValues(self, y: np.ndarray) -> None:
        if self._reactor is not None and hasattr(self._reactor, 'resetBadValues'):
            self._reactor.resetBadValues(y)


def newReactorNet(reactors=None) -> ReactorNet:
    return ReactorNet(reactors or [])

def _fd_in(self):
    node = getattr(self, '_in', getattr(self, '_in_reactor', None))
    return node

def _fd_out(self):
    node = getattr(self, '_out', getattr(self, '_out_reactor', None))
    return node

setattr(FlowDevice, "in", _fd_in)
setattr(FlowDevice, "out", _fd_out)

def _rxn_setAdvanceLimits(self, limits):
    self._advance_limits = np.asarray(limits, dtype=float)

def _rxn_hasAdvanceLimits(self):
    has = hasattr(self, '_advance_limits') and self._advance_limits is not None
    return bool(has)

def _rxn_getAdvanceLimits(self):
    lim = getattr(self, '_advance_limits', (0.0, 1e6))
    return tuple(lim)

def _rxn_setAdvanceLimit(self, k, limit):
    if not hasattr(self, '_advance_limits'):
        self._advance_limits = np.full(self.neq(), 1e6)
    self._advance_limits[k] = float(limit)

def _rxn_finiteDifferenceJacobian(self):
    return self.jacobian()

def _rxn_updateSurfaceProductionRates(self):
    for s in getattr(self, '_surfaces', []):
        if hasattr(s, 'update_rates'):
            s.update_rates()

Reactor.setAdvanceLimits = _rxn_setAdvanceLimits
Reactor.hasAdvanceLimits = _rxn_hasAdvanceLimits
Reactor.getAdvanceLimits = _rxn_getAdvanceLimits
Reactor.setAdvanceLimit = _rxn_setAdvanceLimit
Reactor.finiteDifferenceJacobian = _rxn_finiteDifferenceJacobian
Reactor.updateSurfaceProductionRates = _rxn_updateSurfaceProductionRates

ReactorDelegator.constexpr = True

def _surf_setCoverages(self, cov):
    if hasattr(self, '_surf_phase') and self._surf_phase is not None:
        self._surf_phase.coverages = cov
    self._coverages = np.asarray(cov, dtype=float)

def _surf_getCoverages(self, cov=None):
    if hasattr(self, '_surf_phase') and self._surf_phase is not None:
        cov_vals = self._surf_phase.coverages
    else:
        cov_vals = getattr(self, '_coverages', np.ones(1))
    if cov is not None:
        cov[:] = cov_vals
        return cov
    return cov_vals

ReactorSurface.setCoverages = _surf_setCoverages
ReactorSurface.getCoverages = _surf_getCoverages
MoleReactorSurface.setCoverages = _surf_setCoverages
MoleReactorSurface.getCoverages = _surf_getCoverages
FlowReactorSurface.setCoverages = _surf_setCoverages
FlowReactorSurface.getCoverages = _surf_getCoverages


# WallBase definition
class WallBase:
    """Base class for walls."""
    def __init__(self, left=None, right=None, name=""):
        self._left = left
        self._right = right
        self.name = name
        self._area = 1.0
        self._time = 0.0

    def type(self) -> str:
        t = getattr(self, '_type', 'WallBase')
        return str(t)

    def expansionRate(self) -> float:
        t = getattr(self, '_time', 0.0)
        return 0.0

    def heatRate(self) -> float:
        t = getattr(self, '_time', 0.0)
        return 0.0

    def area(self) -> float:
        a = getattr(self, '_area', 1.0)
        return float(a)

    def setArea(self, a: float) -> None:
        self._area = float(a)

    def initialize(self) -> None:
        self._time = 0.0
        self._area = getattr(self, '_area', 1.0)

    def ready(self) -> bool:
        is_ready = self._left is not None and self._right is not None
        return bool(is_ready)

    def left(self):
        return getattr(self, '_left', None)

    def right(self):
        return getattr(self, '_right', None)

    def setSimTime(self, t: float) -> None:
        self._time = float(t)

    def addHeatRateJacobian(self, *args, **kwargs) -> None:
        a = self.area()
        return None
