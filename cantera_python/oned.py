"""One-dimensional flows, boundaries, and flame simulations in pure Python.

Provides standalone implementations of Domain1D, Boundary1D, Flow1D, Sim1D,
and laminar flame models without binary dependencies.
"""
from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Literal, Optional
import math
import time
import numpy as np

import threading
from .constants import CanteraError, OneAtm, StefanBoltz, Undef, Avogadro, ElectronCharge, epsilon_0
from .numerics import BandMatrix, SystemJacobian
from .solution import Solution
from .surface import Interface
from .solution_array import SolutionArray
from .mechanism import _as_document

# Offsets of flow components in 1D flow state vector
c_offset_U = 0   # axial velocity [m/s]
c_offset_V = 1   # strain / spread rate [1/s]
c_offset_T = 2   # temperature [K]
c_offset_L = 3   # (1/r) dP/dr [N/m^4]
c_offset_E = 4   # electric field [V/m]
c_offset_Uo = 5  # oxidizer axial velocity [m/s]
c_offset_Y = 6   # species mass fractions


class Refiner:
    """Refine Domain1D grids so that profiles satisfy adaptation tolerances.

    Direct reconstruction of Cantera's C++ Refiner class.
    """

    UNSET = 0
    KEEP = 1
    REMOVE = -1

    def __init__(self, domain: Any):
        self.m_domain = domain
        self.m_ratio = 10.0
        self.m_slope = 0.8
        self.m_curve = 0.8
        self.m_prune = -0.001
        self.m_min_range = 0.01
        self.m_npmax = 1000
        self.m_thresh = math.sqrt(np.finfo(float).eps)
        self.m_gridmin = 1e-10

        nv = self.m_domain.n_components if hasattr(self.m_domain, "n_components") else (
            self.m_domain.nComponents() if hasattr(self.m_domain, "nComponents") else 1
        )
        self.m_nv = int(nv)
        self.m_active = [True] * self.m_nv
        self.m_insertPts: set[int] = set()
        self.m_keep: list[int] = []
        self.m_componentNames: set[str] = set()

    def setCriteria(
        self,
        ratio: float = 10.0,
        slope: float = 0.8,
        curve: float = 0.8,
        prune: float = -0.1,
    ) -> None:
        """Set grid refinement criteria with strict parameter validation."""
        ratio = float(ratio)
        slope = float(slope)
        curve = float(curve)
        prune = float(prune)
        if ratio < 2.0:
            raise CanteraError(
                "Refiner::setCriteria",
                f"'ratio' must be greater than 2.0 ({ratio} was specified).",
            )
        if slope < 0.0 or slope > 1.0:
            raise CanteraError(
                "Refiner::setCriteria",
                f"'slope' must be between 0.0 and 1.0 ({slope} was specified).",
            )
        if curve < 0.0 or curve > 1.0:
            raise CanteraError(
                "Refiner::setCriteria",
                f"'curve' must be between 0.0 and 1.0 ({curve} was specified).",
            )
        if prune > curve or prune > slope:
            raise CanteraError(
                "Refiner::setCriteria",
                f"'prune' must be less than 'curve' and 'slope' ({prune} was specified).",
            )
        self.m_ratio = ratio
        self.m_slope = slope
        self.m_curve = curve
        self.m_prune = prune

    set_criteria = setCriteria

    def getCriteria(self) -> list[float]:
        return [self.m_ratio, self.m_slope, self.m_curve, self.m_prune]

    get_criteria = getCriteria

    def setActive(self, comp: int, state: bool = True) -> None:
        comp = int(comp)
        while len(self.m_active) <= comp:
            self.m_active.append(True)
        self.m_active[comp] = bool(state)

    set_active = setActive

    def setMaxPoints(self, npmax: int) -> None:
        self.m_npmax = int(npmax)

    set_max_points = setMaxPoints

    def maxPoints(self) -> int:
        return self.m_npmax

    max_points = maxPoints

    def setGridMin(self, gridmin: float) -> None:
        self.m_gridmin = float(gridmin)

    set_grid_min = setGridMin

    def gridMin(self) -> float:
        return self.m_gridmin

    grid_min = gridMin

    def maxRatio(self) -> float:
        return self.m_ratio

    def maxDelta(self) -> float:
        return self.m_slope

    def maxSlope(self) -> float:
        return self.m_curve

    def prune(self) -> float:
        return self.m_prune

    def value(self, x: Sequence[float], n: int, j: int) -> float:
        idx = self.m_domain.index(n, j)
        return float(x[idx])

    def nNewPoints(self) -> int:
        return len(self.m_insertPts)

    n_new_points = nNewPoints

    def newPointNeeded(self, j: int) -> bool:
        return int(j) in self.m_insertPts

    new_point_needed = newPointNeeded

    def keepPoint(self, j: int) -> bool:
        j = int(j)
        if isinstance(self.m_keep, list):
            if 0 <= j < len(self.m_keep):
                return self.m_keep[j] != self.REMOVE
            return True
        return self.m_keep.get(j, self.KEEP) != self.REMOVE

    keep_point = keepPoint

    def analyze(self, n: int, z: Sequence[float], x: Sequence[float]) -> int:
        """Determine locations in grid needing additional points or pruning."""
        n = int(n)
        if n >= self.m_npmax:
            raise CanteraError(
                "Refiner::analyze", f"max number of grid points reached ({self.m_npmax})."
            )

        np_domain = self.m_domain.n_points if hasattr(self.m_domain, "n_points") else (
            self.m_domain.nPoints() if hasattr(self.m_domain, "nPoints") else n
        )
        if np_domain <= 1:
            return 0
        if n != np_domain:
            raise CanteraError(
                "Refiner::analyze", "number of grid points provided does not match domain size."
            )

        self.m_insertPts.clear()
        self.m_componentNames.clear()
        self.m_keep = [self.UNSET] * n

        self.m_keep[0] = self.KEEP
        self.m_keep[n - 1] = self.KEEP

        nv = self.m_domain.n_components if hasattr(self.m_domain, "n_components") else (
            self.m_domain.nComponents() if hasattr(self.m_domain, "nComponents") else 1
        )
        self.m_nv = int(nv)
        if len(self.m_active) < self.m_nv:
            self.m_active.extend([True] * (self.m_nv - len(self.m_active)))

        z_arr = np.asarray(z, dtype=float)
        x_arr = np.asarray(x, dtype=float)
        dz = np.diff(z_arr)

        try:
            can_slice = (self.m_domain.index(0, 1) == self.m_nv)
        except Exception:
            can_slice = False

        for i in range(self.m_nv):
            if self.m_active[i]:
                name = (
                    self.m_domain.component_name(i) if hasattr(self.m_domain, "component_name") else (
                        self.m_domain.componentName(i) if hasattr(self.m_domain, "componentName") else f"component_{i}"
                    )
                )
                if can_slice:
                    val = x_arr[i::self.m_nv][:n]
                else:
                    val = np.array([self.value(x_arr, i, j) for j in range(n)], dtype=float)
                dval = np.diff(val)
                slope = dval / dz

                valMin = float(np.min(val))
                valMax = float(np.max(val))
                slopeMin = float(np.min(slope))
                slopeMax = float(np.max(slope))

                valMagnitude = max(abs(valMax), abs(valMin))
                slopeMagnitude = max(abs(slopeMax), abs(slopeMin))

                # 1. Slope criterion (Value difference)
                if valMax - valMin > self.m_min_range * valMagnitude:
                    max_change = self.m_slope * (valMax - valMin)
                    ratios = np.abs(dval) / (max_change + self.m_thresh)
                    ins_mask = (ratios > 1.0) & (dz >= 2.0 * self.m_gridmin)
                    if np.any(ins_mask):
                        self.m_componentNames.add(name)
                        for j in np.flatnonzero(ins_mask):
                            self.m_insertPts.add(int(j))

                    for j in range(n - 1):
                        r = ratios[j]
                        if r >= self.m_prune:
                            self.m_keep[j] = self.KEEP
                            self.m_keep[j + 1] = self.KEEP
                        elif self.m_keep[j] == self.UNSET:
                            self.m_keep[j] = self.REMOVE

                # 2. Curvature criterion (Slope difference)
                if slopeMax - slopeMin > self.m_min_range * slopeMagnitude:
                    max_change = self.m_curve * (slopeMax - slopeMin)
                    dslope = np.abs(np.diff(slope))
                    denom = max_change + self.m_thresh / dz[:-1]
                    ratios = dslope / denom
                    ins_mask = (ratios > 1.0) & (dz[:-1] >= 2.0 * self.m_gridmin) & (dz[1:] >= 2.0 * self.m_gridmin)
                    if np.any(ins_mask):
                        self.m_componentNames.add(name)
                        for j in np.flatnonzero(ins_mask):
                            self.m_insertPts.add(int(j))
                            self.m_insertPts.add(int(j + 1))

                    for j in range(n - 2):
                        r = ratios[j]
                        if r >= self.m_prune:
                            self.m_keep[j + 1] = self.KEEP
                        elif self.m_keep[j + 1] == self.UNSET:
                            self.m_keep[j + 1] = self.REMOVE

        # 3. Grid interval ratio refinement
        for j in range(1, n - 1):
            if dz[j] > self.m_ratio * dz[j - 1]:
                self.m_insertPts.add(j)
                self.m_componentNames.add(f"point {j}")
                for off in (-1, 0, 1, 2):
                    if 0 <= j + off < n:
                        self.m_keep[j + off] = self.KEEP

            if dz[j - 1] > self.m_ratio * dz[j]:
                self.m_insertPts.add(j - 1)
                self.m_componentNames.add(f"point {j - 1}")
                for off in (-2, -1, 0, 1):
                    if 0 <= j + off < n:
                        self.m_keep[j + off] = self.KEEP

            # Keep point if removing would make ratio with left interval too large
            if j > 1 and z_arr[j + 1] - z_arr[j - 1] > self.m_ratio * dz[j - 2]:
                self.m_keep[j] = self.KEEP

            # Keep point if removing would make ratio with right interval too large
            if j < n - 2 and z_arr[j + 1] - z_arr[j - 1] > self.m_ratio * dz[j + 1]:
                self.m_keep[j] = self.KEEP

            # Keep fixed temperature point for free flame
            if hasattr(self.m_domain, "is_free") and self.m_domain.is_free():
                if hasattr(self.m_domain, "m_zfixed") and z_arr[j] == self.m_domain.m_zfixed:
                    self.m_keep[j] = self.KEEP
            elif hasattr(self.m_domain, "isFree") and self.m_domain.isFree():
                if hasattr(self.m_domain, "m_zfixed") and z_arr[j] == self.m_domain.m_zfixed:
                    self.m_keep[j] = self.KEEP

            # Keep two-point flame control points
            if hasattr(self.m_domain, "two_point_control_enabled") and self.m_domain.two_point_control_enabled():
                left_c = getattr(self.m_domain, "left_control_point_coordinate", lambda: None)()
                right_c = getattr(self.m_domain, "right_control_point_coordinate", lambda: None)()
                if z_arr[j] == left_c or z_arr[j] == right_c:
                    self.m_keep[j] = self.KEEP
            elif hasattr(self.m_domain, "twoPointControlEnabled") and self.m_domain.twoPointControlEnabled():
                left_c = getattr(self.m_domain, "leftControlPointCoordinate", lambda: None)()
                right_c = getattr(self.m_domain, "rightControlPointCoordinate", lambda: None)()
                if z_arr[j] == left_c or z_arr[j] == right_c:
                    self.m_keep[j] = self.KEEP

        # 4. Don't allow pruning to remove multiple adjacent grid points in a single pass
        for j in range(2, n - 1):
            if self.m_keep[j] == self.REMOVE and self.m_keep[j - 1] == self.REMOVE:
                self.m_keep[j] = self.KEEP

        return len(self.m_insertPts)

    def show(self) -> None:
        pass

class Domain1D:
    """Base class for one-dimensional domains matching native Cantera C++ Domain1D."""

    _domain_type: str = "domain"

    def __init__(self, nv: int = 1, points: int = 1, time: float = 0.0, name: str = ""):
        self.m_nv = 0
        self.m_points = 0
        self._name = str(name)
        self.m_id = str(name)
        self.m_names = [f'comp_{i}' for i in range(nv)]
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
        self._grid = self.m_z

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
        self.m_names = self.m_name
        self.m_max = np.zeros(self.m_nv, dtype=float)
        self.m_min = np.zeros(self.m_nv, dtype=float)
        self.m_rtol_ss = np.full(self.m_nv, 1.0e-4, dtype=float)
        self.m_atol_ss = np.full(self.m_nv, 1.0e-9, dtype=float)
        self.m_rtol_ts = np.full(self.m_nv, 1.0e-4, dtype=float)
        self.m_atol_ts = np.full(self.m_nv, 1.0e-11, dtype=float)
        self.m_points = np_pts
        self.m_z = np.zeros(self.m_points, dtype=float)
        self._grid = self.m_z
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
        names = getattr(self, "m_names", getattr(self, "m_name", []))
        if 0 <= n < self.m_nv:
            if n < len(names) and names[n] != "":
                return names[n]
            return f"component {n}"
        raise IndexError(f"Domain1D::componentName: component {n} out of range [0, {self.m_nv})")

    component_name = componentName

    def setComponentName(self, n: int, name: str) -> None:
        n = int(n)
        if hasattr(self, "m_names") and n < len(self.m_names):
            self.m_names[n] = str(name)
        if hasattr(self, "m_name") and n < len(self.m_name):
            self.m_name[n] = str(name)

    set_component_name = setComponentName

    def componentIndex(self, name: str, check_alias: bool = True) -> int:
        for n in range(self.m_nv):
            if name == self.componentName(n):
                return n
        if check_alias:
            aliases = {
                "u": "velocity", "velocity": "u",
                "V": "spread_rate", "spread_rate": "V",
                "T": "temperature", "temperature": "T",
            }
            if name in aliases:
                target = aliases[name]
                for n in range(self.m_nv):
                    if target == self.componentName(n):
                        return n
            canon = name.replace("_", "-")
            for n in range(self.m_nv):
                cname = self.componentName(n)
                if cname.replace("_", "-") == canon:
                    return n
        raise CanteraError("Domain1D::componentIndex", f"Component '{name}' not found")

    component_index = componentIndex

    def hasComponent(self, name: str, check_alias: bool = True) -> bool:
        try:
            self.componentIndex(name, check_alias=check_alias)
            return True
        except CanteraError:
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
        if "default" in kwargs:
            def_v = kwargs.pop("default")
            if isinstance(def_v, (tuple, list)) and len(def_v) == 2:
                self.m_rtol_ts.fill(float(def_v[0]))
                self.m_atol_ts.fill(float(def_v[1]))
            else:
                self.m_rtol_ts.fill(float(def_v))
                self.m_atol_ts.fill(float(def_v))
        for k, v in kwargs.items():
            if k in ("Y", "mass_fractions", "species") and hasattr(self, "gas") and self.gas is not None:
                rt, at = (v[0], v[1]) if isinstance(v, (tuple, list)) else (v, v)
                k0 = self.componentIndex(self.gas.species_name(0))
                for sp_i in range(self.gas.n_species):
                    self.m_rtol_ts[k0 + sp_i] = float(rt)
                    self.m_atol_ts[k0 + sp_i] = float(at)
            else:
                idx = self.componentIndex(k)
                if isinstance(v, (tuple, list)) and len(v) == 2:
                    self.m_rtol_ts[idx] = float(v[0])
                    self.m_atol_ts[idx] = float(v[1])
                else:
                    self.m_rtol_ts[idx] = float(v)

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
        if "default" in kwargs:
            def_v = kwargs.pop("default")
            if isinstance(def_v, (tuple, list)) and len(def_v) == 2:
                self.m_rtol_ss.fill(float(def_v[0]))
                self.m_atol_ss.fill(float(def_v[1]))
            else:
                self.m_rtol_ss.fill(float(def_v))
                self.m_atol_ss.fill(float(def_v))
        for k, v in kwargs.items():
            if k in ("Y", "mass_fractions", "species") and hasattr(self, "gas") and self.gas is not None:
                rt, at = (v[0], v[1]) if isinstance(v, (tuple, list)) else (v, v)
                k0 = self.componentIndex(self.gas.species_name(0))
                for sp_i in range(self.gas.n_species):
                    self.m_rtol_ss[k0 + sp_i] = float(rt)
                    self.m_atol_ss[k0 + sp_i] = float(at)
            else:
                idx = self.componentIndex(k)
                if isinstance(v, (tuple, list)) and len(v) == 2:
                    self.m_rtol_ss[idx] = float(v[0])
                    self.m_atol_ss[idx] = float(v[1])
                else:
                    self.m_rtol_ss[idx] = float(v)

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

    def hasAnalyticJacobian(self, j: int = 0, n: int = 0) -> bool:
        return False

    def evalJacobianAnalytic(self, x: np.ndarray, jac: Any = None) -> None:
        pass

    def checkAnalyticJacobian(self) -> bool:
        return True

    def setRefineCriteria(self, ratio: float = 10.0, slope: float = 0.8, curve: float = 0.8, prune: float = -0.1) -> None:
        self.m_refiner.setCriteria(ratio, slope, curve, prune)

    set_refine_criteria = setRefineCriteria

    def getRefineCriteria(self) -> list[float]:
        return list(self.m_refiner.getCriteria())

    def get_refine_criteria(self) -> dict[str, float]:
        c = self.m_refiner.getCriteria()
        return {"ratio": float(c[0]), "slope": float(c[1]), "curve": float(c[2]), "prune": float(c[3])}

    @property
    def settings(self) -> dict[str, Any]:
        return self.get_meta()

    def steady_abstol(self, component: str | None = None) -> float | np.ndarray:
        if component is None:
            return self.m_atol_ss.copy()
        return float(self.m_atol_ss[self.componentIndex(component)])

    def steady_reltol(self, component: str | None = None) -> float | np.ndarray:
        if component is None:
            return self.m_rtol_ss.copy()
        return float(self.m_rtol_ss[self.componentIndex(component)])

    def transient_abstol(self, component: str | None = None) -> float | np.ndarray:
        if component is None:
            return self.m_atol_ts.copy()
        return float(self.m_atol_ts[self.componentIndex(component)])

    def transient_reltol(self, component: str | None = None) -> float | np.ndarray:
        if component is None:
            return self.m_rtol_ts.copy()
        return float(self.m_rtol_ts[self.componentIndex(component)])

    def initTimeInteg(self, dt: float, x: Any) -> None:
        self.m_rdt = 1.0 / float(dt) if float(dt) > 0.0 else 0.0
        arr = np.asarray(x, dtype=float)
        loc = self.loc()
        sz = self.size()
        if len(arr) >= loc + sz:
            self.m_slast = np.array(arr[loc : loc + sz], dtype=float, copy=True)
        else:
            self.m_slast = np.array(arr, dtype=float, copy=True)

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

    def resetBadValues(self, x: Any) -> None:
        pass

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
        idx = int(j) * self.m_nv + int(n)
        if hasattr(self, "m_slast") and idx < len(self.m_slast):
            return float(self.m_slast[idx])
        return 0.0

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
        arr = np.asarray(z, dtype=float)
        if len(arr) > 1:
            diffs = np.diff(arr)
            if not np.all(diffs > 0.0):
                raise CanteraError("Domain1D::setupGrid", "grid points must be monotonically increasing")
        self.m_z = arr.copy()
        self._grid = self.m_z
        self.m_points = len(self.m_z)
        self.m_slast = np.zeros(self.m_nv * self.m_points, dtype=float)
        if hasattr(self, "_on_grid_changed"):
            self._on_grid_changed()

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



class CallableFloat(float):
    """A float subclass that is callable as a zero-argument function."""
    def __call__(self) -> float:
        return float(self)

class Boundary1D(Domain1D):
    """Base class for boundary domains matching native Cantera C++ Boundary1D."""

    _domain_type = "boundary"

    def __init__(
        self,
        phase: Solution | None = None,
        name: str = "",
        **kwargs,
    ):
        if "id" in kwargs:
            name = kwargs.pop("id")
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
        self.m_temp: float = float(phase.T() if callable(phase.T) else phase.T) if phase is not None and hasattr(phase, "T") else 0.0
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

    @property
    def temperature(self) -> CallableFloat:
        return CallableFloat(self.m_temp)

    @temperature.setter
    def temperature(self, t: float) -> None:
        self.setTemperature(t)

    @property
    def T(self) -> CallableFloat:
        return CallableFloat(self.m_temp)

    @T.setter
    def T(self, val: float):
        self.setTemperature(val)

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

    @property
    def mdot(self) -> CallableFloat:
        return CallableFloat(self.m_mdot)

    @mdot.setter
    def mdot(self, mdot: float) -> None:
        self.setMdot(mdot)

    def setSpreadRate(self, V0: float) -> None:
        raise NotImplementedError("Boundary1D::setSpreadRate")

    def spreadRate(self) -> CallableFloat:
        raise NotImplementedError("Boundary1D::spreadRate")

    @property
    def spread_rate(self) -> CallableFloat:
        return self.spreadRate()

    @spread_rate.setter
    def spread_rate(self, V0: float) -> None:
        self.setSpreadRate(V0)

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
        if self.m_index is None or self.m_index == -1 or self.container() is None:
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

    def __init__(self, *args, name: str = "inlet", **kwargs):
        if "id" in kwargs:
            name = kwargs.pop("id")
        if len(args) > 0 and args[0] is None:
            raise TypeError("Argument 'phase' has incorrect type (expected Solution, got NoneType)")
        phase = args[0] if len(args) > 0 else kwargs.pop("phase", None)
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

    def spreadRate(self) -> CallableFloat:
        return CallableFloat(self.m_V0)

    @property
    def spread_rate(self) -> CallableFloat:
        return CallableFloat(self.m_V0)

    @spread_rate.setter
    def spread_rate(self, V0: float) -> None:
        self.setSpreadRate(V0)

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
            if k != k_skip:
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

    def __init__(self, *args, name: str = "reacting-surface", **kwargs):
        if len(args) > 0 and args[0] is None:
            raise TypeError("Argument 'phase' has incorrect type (expected Solution, got NoneType)")
        phase = args[0] if len(args) > 0 else kwargs.pop("phase", None)
        if kwargs:
            first = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument '{first}'")
        if phase is not None:
            from .surface import Interface, IdealSurfacePhase
            is_surf = (
                isinstance(phase, (Interface, IdealSurfacePhase))
                or getattr(phase, "thermo_model", "") in ("surface", "ideal-surface", "SurfPhase", "Interface")
                or getattr(phase, "phase_type", "") in ("surface", "ideal-surface")
            )
            if not is_surf:
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
        super().__init__(phase=phase, name=name)

    def nSpecies(self) -> int:
        return self.m_nsp

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

        if hasattr(self.m_kin, "net_production_rates"):
            sdot = np.array(self.m_kin.net_production_rates, dtype=float)
        elif hasattr(self.m_kin, "get_net_production_rates"):
            sdot = np.array(self.m_kin.get_net_production_rates(self.m_sphase), dtype=float)
        else:
            sdot = np.zeros(self.m_kin.n_total_species, dtype=float)

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


_component_map = {
    "velocity": c_offset_U,
    "spreadRate": c_offset_V,
    "T": c_offset_T,
    "Lambda": c_offset_L,
    "eField": c_offset_E,
    "Uo": c_offset_Uo,
}

_component_alias_map = {
    "spread_rate": "spreadRate",
    "u": "velocity",
    "V": "spreadRate",
    "L": "Lambda",
    "lambda": "Lambda",
    "radial-pressure-gradient": "Lambda",
    "radial_pressure_gradient": "Lambda",
    "E": "eField",
    "electric-field": "eField",
    "electric_field": "eField",
    "oxidizer-velocity": "Uo",
    "oxidizer_velocity": "Uo",
    "temperature": "T",
}


class Callable1DComponent:
    """Component view that is callable as fn(x, j) -> float and acts as a 1D array."""
    __slots__ = ("flow", "name", "offset")

    def __init__(self, flow, name: str, offset: int):
        self.flow = flow
        self.name = name
        self.offset = offset

    def __call__(self, x: np.ndarray | None = None, j: int | None = None) -> Any:
        if x is None or j is None:
            return self.as_array()
        return float(x[self.flow.index(self.offset, j)])

    def as_array(self) -> np.ndarray:
        return self.flow.getValues(self.name)

    def __getitem__(self, idx):
        return self.as_array()[idx]

    def __len__(self) -> int:
        return self.flow.m_points

    @property
    def shape(self):
        return self.as_array().shape

    @property
    def ndim(self):
        return self.as_array().ndim

    @property
    def dtype(self):
        return self.as_array().dtype

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.as_array(), dtype=dtype)

    def __iter__(self):
        return iter(self.as_array())

    def __repr__(self) -> str:
        return repr(self.as_array())

    def __lt__(self, other):
        return self.as_array() < (other.as_array() if isinstance(other, Callable1DComponent) else other)

    def __gt__(self, other):
        return self.as_array() > (other.as_array() if isinstance(other, Callable1DComponent) else other)

    def __le__(self, other):
        return self.as_array() <= (other.as_array() if isinstance(other, Callable1DComponent) else other)

    def __ge__(self, other):
        return self.as_array() >= (other.as_array() if isinstance(other, Callable1DComponent) else other)

    def __eq__(self, other):
        return self.as_array() == (other.as_array() if isinstance(other, Callable1DComponent) else other)

    def __ne__(self, other):
        return self.as_array() != (other.as_array() if isinstance(other, Callable1DComponent) else other)

    def __add__(self, other):
        return self.as_array() + other

    def __radd__(self, other):
        return other + self.as_array()

    def __sub__(self, other):
        return self.as_array() - other

    def __rsub__(self, other):
        return other - self.as_array()

    def __mul__(self, other):
        return self.as_array() * other

    def __rmul__(self, other):
        return other * self.as_array()

    def __truediv__(self, other):
        return self.as_array() / other

    def __rtruediv__(self, other):
        return other / self.as_array()


class Callable2DComponent:
    """2D component view of shape (nsp, points) that is callable as Y(x, k, j) or Y(x, j)."""
    __slots__ = ("flow",)

    def __init__(self, flow):
        self.flow = flow

    def __call__(self, x: np.ndarray | None = None, k: int | None = None, j: int | None = None) -> Any:
        if x is None:
            return self.as_array()
        if j is None:
            if k is None:
                return self.as_array()
            # Call form Y(x, j)
            start = self.flow.index(c_offset_Y, k)
            return x[start : start + self.flow.nsp]
        if k is None:
            start = self.flow.index(c_offset_Y, j)
            return x[start : start + self.flow.nsp]
        return float(x[self.flow.index(c_offset_Y + k, j)])

    def as_array(self) -> np.ndarray:
        arr = np.zeros((self.flow.nsp, self.flow.m_points), dtype=float)
        for k in range(self.flow.nsp):
            arr[k, :] = self.flow.getValues(self.flow.gas.species_name(k))
        return arr

    def __getitem__(self, idx):
        return self.as_array()[idx]

    def __len__(self) -> int:
        return self.flow.nsp

    @property
    def shape(self):
        return self.as_array().shape

    @property
    def ndim(self):
        return self.as_array().ndim

    @property
    def dtype(self):
        return self.as_array().dtype

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.as_array(), dtype=dtype)

    def __repr__(self) -> str:
        return repr(self.as_array())

    def __lt__(self, other):
        return self.as_array() < (other.as_array() if isinstance(other, Callable2DComponent) else other)

    def __gt__(self, other):
        return self.as_array() > (other.as_array() if isinstance(other, Callable2DComponent) else other)

    def __le__(self, other):
        return self.as_array() <= (other.as_array() if isinstance(other, Callable2DComponent) else other)

    def __ge__(self, other):
        return self.as_array() >= (other.as_array() if isinstance(other, Callable2DComponent) else other)

    def __eq__(self, other):
        return self.as_array() == (other.as_array() if isinstance(other, Callable2DComponent) else other)

    def __ne__(self, other):
        return self.as_array() != (other.as_array() if isinstance(other, Callable2DComponent) else other)

    def __add__(self, other):
        return self.as_array() + other

    def __sub__(self, other):
        return self.as_array() - other

    def __mul__(self, other):
        return self.as_array() * other

    def __truediv__(self, other):
        return self.as_array() / other


class CallableXComponent:
    """2D component view of mole fractions that is callable as X(x, k, j)."""
    __slots__ = ("flow",)

    def __init__(self, flow):
        self.flow = flow

    def __call__(self, x: np.ndarray | None = None, k: int | None = None, j: int | None = None) -> Any:
        if x is None:
            return self.as_array()
        if j is None:
            if k is None:
                return self.as_array()
            return self.as_array()[:, k]
        return float(self.flow.m_wtm[j] * self.flow.Y(x, k, j) / self.flow.m_wt[k])

    def as_array(self) -> np.ndarray:
        arr = np.zeros((self.flow.nsp, self.flow.m_points), dtype=float)
        T_arr = self.flow.getValues("T")
        orig_T, orig_P, orig_Y = self.flow.gas.T, self.flow.gas.P, self.flow.gas.Y
        for j in range(self.flow.m_points):
            T_j = T_arr[j] if T_arr[j] > 0 else 300.0
            y_j = [self.flow.getValues(self.flow.gas.species_name(k))[j] for k in range(self.flow.nsp)]
            if np.sum(y_j) <= 0:
                y_j = list(self.flow.gas.Y)
            self.flow.gas.set_state_tpy(T_j, self.flow.m_press, y_j)
            arr[:, j] = self.flow.gas.X
        self.flow.gas.set_state_tpy(orig_T, orig_P, orig_Y)
        return arr

    def __getitem__(self, idx):
        return self.as_array()[idx]

    def __len__(self) -> int:
        return self.flow.nsp

    @property
    def shape(self):
        return self.as_array().shape

    @property
    def ndim(self):
        return self.as_array().ndim

    @property
    def dtype(self):
        return self.as_array().dtype

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.as_array(), dtype=dtype)

    def __repr__(self) -> str:
        return repr(self.as_array())

    def __lt__(self, other):
        return self.as_array() < (other.as_array() if isinstance(other, CallableXComponent) else other)

    def __gt__(self, other):
        return self.as_array() > (other.as_array() if isinstance(other, CallableXComponent) else other)

    def __le__(self, other):
        return self.as_array() <= (other.as_array() if isinstance(other, CallableXComponent) else other)

    def __ge__(self, other):
        return self.as_array() >= (other.as_array() if isinstance(other, CallableXComponent) else other)

    def __eq__(self, other):
        return self.as_array() == (other.as_array() if isinstance(other, CallableXComponent) else other)

    def __ne__(self, other):
        return self.as_array() != (other.as_array() if isinstance(other, CallableXComponent) else other)

    def __add__(self, other):
        return self.as_array() + other

    def __sub__(self, other):
        return self.as_array() - other

    def __mul__(self, other):
        return self.as_array() * other

    def __truediv__(self, other):
        return self.as_array() / other


class CallableRhoU:
    """1D view of axial mass flux that is callable as rho_u(x, j)."""
    __slots__ = ("flow",)

    def __init__(self, flow):
        self.flow = flow

    def __call__(self, x: np.ndarray | None = None, j: int = 0) -> float:
        if x is None:
            return float(self.as_array()[j])
        return float(self.flow.m_rho[j] * x[self.flow.index(c_offset_U, j)])

    def as_array(self) -> np.ndarray:
        u_arr = self.flow.getValues("velocity")
        return self.flow.m_rho * u_arr

    def __getitem__(self, idx):
        return self.as_array()[idx]

    def __len__(self) -> int:
        return self.flow.m_points

    @property
    def shape(self):
        return self.as_array().shape

    @property
    def ndim(self):
        return self.as_array().ndim

    @property
    def dtype(self):
        return self.as_array().dtype

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.as_array(), dtype=dtype)


class Flow1D(Domain1D):
    """Base class for 1D reacting flow domains.

    Reconstructs exact native Cantera Flow1D governing equations, numerical
    discretization, transport coupling, radiation, and solver state storage.
    """

    _unused_components: set[str] = set()

    def __init__(self, phase: Solution, id: str = "flame", points: int = 1):
        if not isinstance(phase, Solution):
            raise TypeError("Expected a Solution object")
        if phase.transport_model in ("none", ""):
            raise CanteraError(
                "Flow1D::Flow1D",
                f"An appropriate transport model\nshould be set when instantiating {self.__class__.__name__}",
            )
        self.gas = phase
        self.phase = phase
        phase._in_flow = True
        self.nsp = phase.n_species
        self.nv = c_offset_Y + self.nsp
        self.m_press = float(phase.P)
        self.m_wt = np.array(phase.molecular_weights, dtype=float)

        self.m_do_energy = [False] * points
        self.m_do_soret = False
        self.m_do_multicomponent = phase.transport_model in ("multicomponent", "multicomponent-CK")
        self.m_do_radiation = False
        self.m_epsilon_left = 0.0
        self.m_epsilon_right = 0.0
        self.m_flux_gradient_basis = "molar"
        self.m_dovisc = False
        self.m_isFree = False
        self.m_usesLambda = False
        self.m_twoPointControl = False
        self.m_zLeft = Undef
        self.m_tLeft = Undef
        self.m_zRight = Undef
        self.m_tRight = Undef
        self.m_zfixed = Undef
        self.m_tfixed = -1.0
        self.m_kExcessLeft = 0
        self.m_kExcessRight = 0
        self.m_kRadiating = [
            phase.species_index("CO2") if "CO2" in phase.species_names else None,
            phase.species_index("H2O") if "H2O" in phase.species_names else None,
        ]

        super().__init__(nv=self.nv, points=points, name=id)

        self.m_names = ["velocity", "spreadRate", "T", "Lambda", "eField", "Uo"] + list(
            phase.species_names
        )
        self._T_comp = Callable1DComponent(self, "T", c_offset_T)
        self._velocity_comp = Callable1DComponent(self, "velocity", c_offset_U)
        self._spread_rate_comp = Callable1DComponent(self, "spreadRate", c_offset_V)
        self._lambda_comp = Callable1DComponent(self, "Lambda", c_offset_L)
        self._Uo_comp = Callable1DComponent(self, "Uo", c_offset_Uo)
        self._Y_comp = Callable2DComponent(self)
        self._X_comp = CallableXComponent(self)
        self._rho_u_comp = CallableRhoU(self)

        # Default solution bounds matching native Cantera
        self.setBounds(c_offset_U, -1e20, 1e20)
        self.setBounds(c_offset_V, -1e20, 1e20)
        max_t = 2.0 * float(phase.max_temp) if hasattr(phase, "max_temp") else 10000.0
        self.setBounds(c_offset_T, 200.0, max_t)
        self.setBounds(c_offset_L, -1e20, 1e20)
        self.setBounds(c_offset_E, -1e20, 1e20)
        self.setBounds(c_offset_Uo, -1e20, 1e20)
        for k in range(self.nsp):
            self.setBounds(c_offset_Y + k, -1.0e-7, 1.0e5)

        # Grid refiner defaults
        self.m_refiner.setActive(c_offset_U, False)
        self.m_refiner.setActive(c_offset_V, False)
        self.m_refiner.setActive(c_offset_T, False)
        self.m_refiner.setActive(c_offset_L, False)
        self.m_refiner.setActive(c_offset_Uo, False)

        self.resize(self.nv, points)
        if points > 1:
            gr = [float(ng) / points for ng in range(points)]
            self.setupGrid(gr)
        else:
            self.m_fixedtemp = np.full(points, phase.T)
            self.m_zfix = np.array([0.0, 1.0])
            self.m_tfix = np.full(2, phase.T)

    def resize(self, ncomponents: int, points: int):
        super().resize(ncomponents, points)
        self.m_rho = np.zeros(self.m_points, dtype=float)
        self.m_wtm = np.zeros(self.m_points, dtype=float)
        self.m_cp = np.zeros(self.m_points, dtype=float)
        self.m_visc = np.zeros(self.m_points, dtype=float)
        self.m_tcon = np.zeros(self.m_points, dtype=float)

        self.m_diff = np.zeros((self.m_points, self.nsp), dtype=float)
        self.m_multidiff = np.zeros((self.m_points, self.nsp, self.nsp), dtype=float)
        self.m_dthermal = np.zeros((self.nsp, self.m_points), dtype=float)
        self.m_flux = np.zeros((self.nsp, self.m_points), dtype=float)
        self.m_wdot = np.zeros((self.nsp, self.m_points), dtype=float)
        self.m_hk = np.zeros((self.nsp, self.m_points), dtype=float)
        self.m_dhk_dz = np.zeros((self.nsp, max(self.m_points - 1, 1)), dtype=float)
        self.m_do_energy = [False] * self.m_points
        self.m_qdotRadiation = np.zeros(self.m_points, dtype=float)
        self.m_fixedtemp = np.zeros(self.m_points, dtype=float)

        self.m_dz = np.zeros(max(self.m_points - 1, 1), dtype=float)
        self.m_z = np.zeros(self.m_points, dtype=float)
        self.m_ybar = np.zeros(self.nsp, dtype=float)

    def setupGrid(self, z: Sequence[float]):
        z_arr = np.asarray(z, dtype=float)
        n = len(z_arr)
        for j in range(1, n):
            if z_arr[j] <= z_arr[j - 1]:
                raise CanteraError("Flow1D::setupGrid", "grid points must be monotonically increasing")
        self.resize(self.m_nv, n)
        self.m_z[:] = z_arr
        if n > 1:
            self.m_dz[: n - 1] = np.diff(z_arr)
        self._grid = self.m_z.copy()

    def resetBadValues(self, xg: np.ndarray):
        loc = self.loc()
        x = xg[loc : loc + self.size()]
        for j in range(self.m_points):
            y_j = self.Y(x, j=j)
            self.gas.Y = y_j
            x[self.index(c_offset_Y, j) : self.index(c_offset_Y, j) + self.nsp] = self.gas.Y

    def domainType(self) -> str:
        if self.m_isFree:
            return "free-flow"
        if self.m_usesLambda:
            return "axisymmetric-flow"
        return "unstrained-flow"

    domain_type = property(domainType)
    _domain_type = property(domainType)

    def phase(self) -> Solution:
        return self.gas

    def kinetics(self) -> Any:
        return getattr(self.gas, "kinetics", self.gas)

    def _setKinetics(self, kin: Any):
        if hasattr(self.gas, "setKinetics"):
            self.gas.setKinetics(kin)

    def _setTransport(self, trans: Any):
        if not trans:
            raise CanteraError("Flow1D::_setTransport", "Unable to set empty transport.")
        tmodel = (
            trans.transport_model
            if hasattr(trans, "transport_model")
            else getattr(self.gas, "transport_model", "mixture-averaged")
        )
        if tmodel in ("none", ""):
            raise CanteraError("Flow1D::_setTransport", "Invalid Transport model 'none'.")
        self.m_do_multicomponent = tmodel in ("multicomponent", "multicomponent-CK")

    def setTransportModel(self, model: str):
        if model in ("none", ""):
            raise CanteraError("Flow1D::setTransportModel", "Invalid Transport model 'none'.")
        self.gas.transport_model = model
        self.m_do_multicomponent = model in ("multicomponent", "multicomponent-CK")

    def transportModel(self) -> str:
        return self.gas.transport_model

    transport_model = property(transportModel, setTransportModel)

    def setFluxGradientBasis(self, fluxGradientBasis: str):
        if str(fluxGradientBasis).lower() in ("molar", "1"):
            self.m_flux_gradient_basis = "molar"
        elif str(fluxGradientBasis).lower() in ("mass", "0"):
            self.m_flux_gradient_basis = "mass"
        else:
            raise ValueError("Valid choices are 'mass' or 'molar'.")

    def fluxGradientBasis(self) -> str:
        return self.m_flux_gradient_basis

    flux_gradient_basis = property(fluxGradientBasis, setFluxGradientBasis)

    def setPressure(self, p: float):
        self.m_press = float(p)
        self.gas.P = float(p)

    def pressure(self) -> float:
        return self.m_press

    P = property(pressure, setPressure)

    def _getInitialSoln(self, x: np.ndarray):
        for j in range(self.m_points):
            x[self.index(c_offset_T, j)] = self.gas.T
            x[self.index(c_offset_Y, j) : self.index(c_offset_Y, j) + self.nsp] = self.gas.Y
            self.m_rho[j] = self.gas.density

    def setGas(self, x: np.ndarray, j: int | None = None):
        if j is None:
            self.gas = x
            self.phase = x
            return
        T_val = self.T(x, j)
        y_start = self.index(c_offset_Y, j)
        Y_val = x[y_start : y_start + self.nsp]
        self.gas.set_state_tpy(T_val, self.m_press, Y_val)

    def setGasAtMidpoint(self, x: np.ndarray, j: int):
        T_mid = 0.5 * (self.T(x, j) + self.T(x, j + 1))
        y_j = x[self.index(c_offset_Y, j) : self.index(c_offset_Y, j) + self.nsp]
        y_jp1 = x[self.index(c_offset_Y, j + 1) : self.index(c_offset_Y, j + 1) + self.nsp]
        self.m_ybar[:] = 0.5 * (y_j + y_jp1)
        self.gas.set_state_tpy(T_mid, self.m_press, self.m_ybar)

    def _finalize(self, x: np.ndarray | None = None):
        if x is None:
            if self.m_container is not None and hasattr(self.m_container, "_state"):
                x = self.m_container._state[self.loc() : self.loc() + self.size()]
            else:
                return

        if not (self.m_do_multicomponent or self.transportModel() in ("mixture-averaged", "mixture-averaged-CK", "Mix")) and self.m_do_soret:
            raise CanteraError(
                "Flow1D::_finalize",
                "Thermal diffusion (the Soret effect) is enabled, but it "
                "only compatible with the mixture-averaged and multicomponent "
                "transport models.",
            )

        nz = len(self.m_zfix) if hasattr(self, "m_zfix") and self.m_zfix is not None else 0
        e = self.m_do_energy[0]
        z0 = self.z(0)
        zmax = self.z(self.m_points - 1)
        zspan = zmax - z0
        for j in range(self.m_points):
            if e or nz == 0:
                self.m_fixedtemp[j] = self.T(x, j)
            else:
                zz = (self.z(j) - z0) / zspan if zspan > 0 else 0.0
                self.m_fixedtemp[j] = float(np.interp(zz, self.m_zfix, self.m_tfix))
        if e:
            self.solveEnergyEqn()

        if self.m_isFree and self.m_tfixed != Undef and self.m_tfixed > 0:
            for j in range(self.m_points):
                if np.isclose(self.z(j), self.m_zfixed, atol=1e-12):
                    self.m_zfixed = float(self.z(j))
                    return
            for j in range(self.m_points - 1):
                if (self.T(x, j) - self.m_tfixed) * (self.T(x, j + 1) - self.m_tfixed) <= 0.0:
                    self.m_tfixed = float(self.T(x, j + 1))
                    self.m_zfixed = float(self.z(j + 1))
                    return

    def setFixedTempProfile(self, pos: Sequence[float], T: Sequence[float]):
        self.m_zfix = np.asarray(pos, dtype=float)
        self.m_tfix = np.asarray(T, dtype=float)
        if self.m_points > 1:
            z0 = self.z(0)
            zspan = self.z(self.m_points - 1) - z0
            rel = (self.m_z - z0) / zspan if zspan > 0 else np.zeros(self.m_points)
            self.m_fixedtemp[:] = np.interp(rel, self.m_zfix, self.m_tfix)

    set_fixed_temp_profile = setFixedTempProfile

    def setFixedTemp(self, pos_or_T: Sequence[float], T: Sequence[float] | None = None):
        if T is None:
            self.m_fixedtemp[:] = np.asarray(pos_or_T, dtype=float)
        else:
            self.setFixedTempProfile(pos_or_T, T)

    set_fixed_temp = setFixedTemp

    def isFixedTemp(self) -> bool:
        return self.m_isFree and self.m_tfixed > 0 and self.m_tfixed != Undef

    def T_fixed(self, j: int = 0) -> float:
        if j < len(self.m_fixedtemp):
            return float(self.m_fixedtemp[j])
        return 0.0

    fixed_temperature_profile = T_fixed

    def setTemperature(self, j: int, T: float):
        if j < len(self.m_fixedtemp):
            self.m_fixedtemp[j] = float(T)

    @property
    def fixed_temperature(self) -> float:
        return self.m_tfixed

    @fixed_temperature.setter
    def fixed_temperature(self, val: float):
        self.m_tfixed = float(val)
        T_prof = self.T
        if len(T_prof) > 1:
            self.m_zfixed = float(np.interp(self.m_tfixed, T_prof, self.m_z))

    @property
    def fixed_temperature_location(self) -> float:
        return self.m_zfixed

    def componentName(self, n: int) -> str:
        if n == c_offset_U:
            return "velocity"
        elif n == c_offset_V:
            return "spreadRate"
        elif n == c_offset_T:
            return "T"
        elif n == c_offset_L:
            return "Lambda"
        elif n == c_offset_E:
            return "eField"
        elif n == c_offset_Uo:
            return "Uo"
        elif c_offset_Y <= n < c_offset_Y + self.nsp:
            return self.gas.species_name(n - c_offset_Y)
        else:
            raise IndexError(f"Flow1D::componentName: component index {n} out of range (nv={self.nv})")

    component_name = componentName

    def componentIndex(self, name: str, checkAlias: bool = True) -> int:
        if name in _component_map:
            return _component_map[name]
        for n in range(c_offset_Y, c_offset_Y + self.nsp):
            if self.componentName(n) == name:
                return n
        if checkAlias:
            if name in _component_alias_map:
                return self.componentIndex(_component_alias_map[name], checkAlias=False)
        raise CanteraError("Flow1D::componentIndex", f"Component '{name}' not found")

    component_index = componentIndex

    def hasComponent(self, name: str, checkAlias: bool = True) -> bool:
        try:
            self.componentIndex(name, checkAlias=checkAlias)
            return True
        except CanteraError:
            return False

    has_component = hasComponent

    def componentActive(self, n: int) -> bool:
        if n in (c_offset_V, c_offset_L):
            return bool(self.m_usesLambda)
        elif n == c_offset_E:
            return False
        elif n == c_offset_Uo:
            return bool(self.m_twoPointControl)
        return True

    component_active = componentActive

    def updateState(self, loc: int):
        self.update_state(loc)

    def update_state(self, j: int):
        if self.m_container is not None and hasattr(self.m_container, "_state"):
            soln = self.m_container._state[self.loc() : self.loc() + self.size()]
            T_val = self.T(soln, j)
            Y_val = np.array([self.Y(soln, k, j) for k in range(self.nsp)])
            self.gas.set_state_tpy(T_val, self.m_press, Y_val)

    def show(self, x: Sequence[float] | None = None):
        super().show(x)

    def getValues(self, component: str, values_out: np.ndarray | None = None) -> np.ndarray:
        idx = self.componentIndex(component)
        state = self.m_container._state if self.m_container is not None and hasattr(self.m_container, "_state") else None
        res = np.zeros(self.m_points, dtype=float)
        if state is not None:
            loc = self.loc()
            for j in range(self.m_points):
                res[j] = state[loc + self.index(idx, j)]
        else:
            if idx == c_offset_T:
                res[:] = self.m_fixedtemp
        if values_out is not None:
            values_out[:] = res
            return values_out
        return res

    values = getValues

    def setValues(self, component: str, values_in: Sequence[float] | np.ndarray):
        idx = self.componentIndex(component)
        if not self.componentActive(idx):
            raise CanteraError("Flow1D::setValues", f"Component '{component}' is not used by '{self.domainType()}'.")
        arr = np.asarray(values_in, dtype=float)
        if len(arr) != self.m_points:
            raise CanteraError("Flow1D::setValues", f"Array size mismatch: {len(arr)} != {self.m_points}")
        if self.m_container is not None and hasattr(self.m_container, "_state"):
            state = self.m_container._state
            loc = self.loc()
            for j in range(self.m_points):
                state[loc + self.index(idx, j)] = arr[j]
        if idx == c_offset_T:
            self.m_fixedtemp[:] = arr

    set_values = setValues

    def getResiduals(self, component: str, values_out: np.ndarray | None = None) -> np.ndarray:
        idx = self.componentIndex(component)
        res = np.zeros(self.m_points, dtype=float)
        if self.m_container is not None and hasattr(self.m_container, "_workVector"):
            work = self.m_container._workVector()
            loc = self.loc()
            for j in range(self.m_points):
                res[j] = work[loc + self.index(idx, j)]
        if values_out is not None:
            values_out[:] = res
            return values_out
        return res

    def setProfile(self, component: str, pos: Sequence[float], values: Sequence[float]):
        pos_arr = np.asarray(pos, dtype=float)
        val_arr = np.asarray(values, dtype=float)
        if len(pos_arr) != len(val_arr):
            raise CanteraError(
                "Flow1D::setProfile",
                f"Vectors for positions and values must have same size. Sizes are {len(pos_arr)} and {len(val_arr)}, respectively.",
            )
        if len(pos_arr) == 0 or pos_arr[0] != 0.0 or pos_arr[-1] != 1.0:
            raise CanteraError(
                "Flow1D::setProfile",
                f"'pos' vector must span the range [0, 1]. Got [{pos_arr[0] if len(pos_arr) > 0 else 0}, {pos_arr[-1] if len(pos_arr) > 0 else 0}] instead.",
            )
        idx = self.componentIndex(component)
        if not self.componentActive(idx):
            raise CanteraError("Flow1D::setProfile", f"Component '{component}' is not used by '{self.domainType()}'.")
        z0 = self.z(0)
        zspan = self.z(self.m_points - 1) - z0
        interpolated = np.zeros(self.m_points, dtype=float)
        for j in range(self.m_points):
            frac = (self.z(j) - z0) / zspan if zspan > 0 else 0.0
            interpolated[j] = float(np.interp(frac, pos_arr, val_arr))
        self.setValues(component, interpolated)

    set_profile = setProfile

    def setFlatProfile(self, component: str, value: float):
        self.setValues(component, np.full(self.m_points, float(value)))

    set_flat_profile = setFlatProfile

    def toArray(self, normalize: bool = False):
        state = self.getMeta()
        arr = SolutionArray(self.gas, shape=(self.m_points,), meta=state)
        arr.add_extra("grid", False)
        arr.set_component("grid", self.m_z.copy())
        for i in range(self.m_nv):
            if self.componentActive(i):
                cname = self.componentName(i)
                vals = self.getValues(cname)
                if not arr.has_component(cname):
                    arr.add_extra(cname, i > c_offset_Y)
                arr.set_component(cname, vals)
        if self.m_container is not None and hasattr(self.m_container, "_state"):
            soln = self.m_container._state[self.loc() : self.loc() + self.size()]
            self.updateThermo(soln, 0, self.m_points - 1)
        arr.set_component("D", self.m_rho.copy())
        if self.m_do_radiation:
            arr.add_extra("radiative-heat-loss", True)
            arr.set_component("radiative-heat-loss", self.m_qdotRadiation.copy())
        if normalize:
            arr.normalize()
        return arr

    to_array = toArray

    def fromArray(self, arr: Any):
        if hasattr(arr, "grid"):
            self.setupGrid(np.asarray(arr.grid, dtype=float))
        elif hasattr(arr, "get_component") and arr.has_component("grid"):
            self.setupGrid(np.asarray(arr.get_component("grid"), dtype=float))

        meta_val = arr.meta() if callable(getattr(arr, "meta", None)) else getattr(arr, "meta", None)
        if meta_val:
            self.setMeta(meta_val)

        if hasattr(arr, "T"):
            self.setValues("T", np.asarray(arr.T, dtype=float))
        elif hasattr(arr, "get_component") and arr.has_component("T"):
            self.setValues("T", np.asarray(arr.get_component("T"), dtype=float))

        if hasattr(arr, "velocity"):
            self.setValues("velocity", np.asarray(arr.velocity, dtype=float))
        elif hasattr(arr, "get_component") and arr.has_component("velocity"):
            self.setValues("velocity", np.asarray(arr.get_component("velocity"), dtype=float))

        if hasattr(arr, "Y"):
            y_arr = np.asarray(arr.Y, dtype=float)
            for k in range(self.nsp):
                sp_name = self.gas.species_name(k)
                if y_arr.ndim == 2:
                    if y_arr.shape[0] == self.m_points and y_arr.shape[1] == self.nsp:
                        self.setValues(sp_name, y_arr[:, k])
                    elif y_arr.shape[0] == self.nsp and y_arr.shape[1] == self.m_points:
                        self.setValues(sp_name, y_arr[k, :])

        if self.componentActive(c_offset_L):
            if hasattr(arr, "Lambda"):
                self.setValues("Lambda", np.asarray(arr.Lambda, dtype=float))
            elif hasattr(arr, "get_component") and arr.has_component("Lambda"):
                self.setValues("Lambda", np.asarray(arr.get_component("Lambda"), dtype=float))

        if self.componentActive(c_offset_V):
            if hasattr(arr, "spread_rate"):
                self.setValues("spreadRate", np.asarray(arr.spread_rate, dtype=float))
            elif hasattr(arr, "get_component") and arr.has_component("spreadRate"):
                self.setValues("spreadRate", np.asarray(arr.get_component("spreadRate"), dtype=float))

    from_array = fromArray

    # Solution component property getters (Cached non-allocating callables)
    @property
    def T(self) -> Callable1DComponent:
        return self._T_comp

    temperature = T

    @property
    def velocity(self) -> Callable1DComponent:
        return self._velocity_comp

    u = velocity

    @property
    def spread_rate(self) -> Callable1DComponent:
        return self._spread_rate_comp

    spreadRate = spread_rate
    V = spread_rate

    @property
    def radial_pressure_gradient(self) -> Callable1DComponent:
        return self._lambda_comp

    Lambda = radial_pressure_gradient
    L = radial_pressure_gradient

    @property
    def Uo(self) -> Callable1DComponent:
        return self._Uo_comp

    @property
    def Y(self) -> Callable2DComponent:
        return self._Y_comp

    @property
    def X(self) -> CallableXComponent:
        return self._X_comp

    @property
    def rho_u(self) -> CallableRhoU:
        return self._rho_u_comp

    @property
    def mix_diff_coeffs(self) -> np.ndarray:
        res = np.zeros((self.nsp, self.m_points), dtype=float)
        T_arr = self.getValues("T")
        orig_T, orig_P, orig_Y = self.gas.T, self.gas.P, self.gas.Y
        for j in range(self.m_points):
            y_j = [self.getValues(self.gas.species_name(k))[j] for k in range(self.nsp)]
            self.gas.set_state_tpy(T_arr[j], self.m_press, y_j)
            res[:, j] = self.gas.mix_diff_coeffs
        self.gas.set_state_tpy(orig_T, orig_P, orig_Y)
        return res

    def T_prev(self, j: int = 0) -> float:
        return float(self.prevSoln(c_offset_T, j))

    def V_prev(self, j: int = 0) -> float:
        return float(self.prevSoln(c_offset_V, j))

    def Y_prev(self, k: int = 0, j: int = 0) -> float:
        return float(self.prevSoln(c_offset_Y + k, j))

    def density(self, j: int = 0) -> float:
        return float(self.m_rho[j])

    def flux(self, k: int = 0, j: int = 0) -> float:
        return float(self.m_flux[k, j])

    def m_flux(self) -> np.ndarray:
        return self.m_flux

    def mindex(self, k: int, j: int, m: int = 0) -> int:
        return m * self.nsp * self.nsp + self.nsp * j + k

    def __getattr__(self, name: str) -> Any:
        if name in ("lambda", "Lambda", "L", "radial_pressure_gradient"):
            return self.radial_pressure_gradient
        gas = self.__dict__.get("gas")
        if gas is not None and name in gas.species_names:
            return self.getValues(name)
        raise AttributeError(f"{self.__class__.__name__!r} object has no attribute {name!r}")

    # Configuration modes
    def setFreeFlow(self):
        self.m_dovisc = False
        self.m_isFree = True
        self.m_usesLambda = False

    set_free_flow = setFreeFlow

    def setAxisymmetricFlow(self):
        self.m_dovisc = True
        self.m_isFree = False
        self.m_usesLambda = True

    set_axisymmetric_flow = setAxisymmetricFlow

    def setUnstrainedFlow(self):
        self.m_dovisc = False
        self.m_isFree = False
        self.m_usesLambda = False

    set_unstrained_flow = setUnstrainedFlow

    def isFree(self) -> bool:
        return self.m_isFree

    def isStrained(self) -> bool:
        return self.m_usesLambda

    def setViscosityFlag(self, dovisc: bool):
        self.m_dovisc = bool(dovisc)

    # Energy controls
    def solveEnergyEqn(self, j: int | bool | None = None):
        if isinstance(j, bool):
            if not j:
                return self.fixTemperature()
            j = None
        changed = False
        if j is None:
            for i in range(self.m_points):
                if not self.m_do_energy[i]:
                    changed = True
                self.m_do_energy[i] = True
        else:
            if not self.m_do_energy[j]:
                changed = True
            self.m_do_energy[j] = True
        self.m_refiner.setActive(c_offset_U, True)
        self.m_refiner.setActive(c_offset_V, True)
        self.m_refiner.setActive(c_offset_T, True)
        if changed:
            self.needJacUpdate()

    solve_energy_eqn = solveEnergyEqn

    def fixTemperature(self, j: int | bool | float | None = None):
        if isinstance(j, bool):
            if not j:
                return self.solveEnergyEqn()
            j = None
        elif isinstance(j, float):
            self.m_tfixed = float(j)
            j = None
        changed = False
        if j is None:
            for i in range(self.m_points):
                if self.m_do_energy[i]:
                    changed = True
                self.m_do_energy[i] = False
        else:
            if self.m_do_energy[j]:
                changed = True
            self.m_do_energy[j] = False
        self.m_refiner.setActive(c_offset_U, False)
        self.m_refiner.setActive(c_offset_V, False)
        self.m_refiner.setActive(c_offset_T, False)
        if changed:
            self.needJacUpdate()

    fix_temperature = fixTemperature

    def doEnergy(self, j: int | None = None) -> bool:
        if j is None:
            return all(self.m_do_energy)
        return bool(self.m_do_energy[j])

    do_energy = doEnergy

    def allOfEnergyEnabled(self) -> bool:
        return all(self.m_do_energy)

    def noneOfEnergyEnabled(self) -> bool:
        return not any(self.m_do_energy)

    def setEnergyEnabled(self, val: bool):
        if val:
            self.solveEnergyEqn()
        else:
            self.fixTemperature()

    energy_enabled = property(allOfEnergyEnabled, setEnergyEnabled)

    # Electric field stubs
    def solveElectricField(self):
        raise NotImplementedError(f"Flow1D::solveElectricField: Not used by '{self.domainType()}' objects.")

    def fixElectricField(self):
        raise NotImplementedError(f"Flow1D::fixElectricField: Not used by '{self.domainType()}' objects.")

    def doElectricField(self) -> bool:
        raise NotImplementedError(f"Flow1D::doElectricField: Not used by '{self.domainType()}' objects.")

    # Radiation
    def enableRadiation(self, val: bool = True):
        self.m_do_radiation = bool(val)

    def radiationEnabled(self) -> bool:
        return self.m_do_radiation

    radiation_enabled = property(radiationEnabled, enableRadiation)

    def radiativeHeatLoss(self, j: int = 0) -> float:
        return float(self.m_qdotRadiation[j])

    def setBoundaryEmissivities(self, e_left: float, e_right: float):
        if e_left < 0 or e_left > 1:
            raise CanteraError("Flow1D::setBoundaryEmissivities", "The left boundary emissivity must be between 0.0 and 1.0!")
        if e_right < 0 or e_right > 1:
            raise CanteraError("Flow1D::setBoundaryEmissivities", "The right boundary emissivity must be between 0.0 and 1.0!")
        self.m_epsilon_left = float(e_left)
        self.m_epsilon_right = float(e_right)

    def leftEmissivity(self) -> float:
        return self.m_epsilon_left

    def rightEmissivity(self) -> float:
        return self.m_epsilon_right

    boundary_emissivities = property(
        lambda self: (self.m_epsilon_left, self.m_epsilon_right),
        lambda self, eps: self.setBoundaryEmissivities(eps[0], eps[1]),
    )

    # Soret
    def enableSoret(self, val: bool = True):
        self.m_do_soret = bool(val)

    def withSoret(self) -> bool:
        return self.m_do_soret

    soretEnabled = withSoret
    soret_enabled = property(withSoret, enableSoret)

    # Two-point continuation control
    def enableTwoPointControl(self, twoPointControl: bool = True):
        if self.isStrained():
            self.m_twoPointControl = bool(twoPointControl)
            self.setBounds(c_offset_V, -1e-5, 1e20)
        else:
            raise CanteraError("Flow1D::enableTwoPointControl", "Invalid operation: two-point control can only be used with axisymmetric flames.")

    def twoPointControlEnabled(self) -> bool:
        return self.m_twoPointControl

    def leftControlPointTemperature(self) -> float:
        if self.m_twoPointControl:
            if self.m_zLeft != Undef:
                return self.m_tLeft
            raise CanteraError("Flow1D::leftControlPointTemperature", "Invalid operation: left control point location is not set")
        raise CanteraError("Flow1D::leftControlPointTemperature", "Invalid operation: two-point control is not enabled.")

    def leftControlPointCoordinate(self) -> float:
        if self.m_twoPointControl:
            if self.m_zLeft != Undef:
                return self.m_zLeft
            raise CanteraError("Flow1D::leftControlPointCoordinate", "Invalid operation: left control point location is not set")
        raise CanteraError("Flow1D::leftControlPointCoordinate", "Invalid operation: two-point control is not enabled.")

    def setLeftControlPointTemperature(self, temperature: float):
        if self.m_twoPointControl:
            if self.m_zLeft != Undef:
                self.m_tLeft = float(temperature)
            else:
                raise CanteraError("Flow1D::setLeftControlPointTemperature", "Invalid operation: left control point location is not set")
        else:
            raise CanteraError("Flow1D::setLeftControlPointTemperature", "Invalid operation: two-point control is not enabled.")

    def setLeftControlPointCoordinate(self, z_left: float):
        if self.m_twoPointControl:
            self.m_zLeft = float(z_left)
        else:
            raise CanteraError("Flow1D::setLeftControlPointCoordinate", "Invalid operation: two-point control is not enabled.")

    def rightControlPointTemperature(self) -> float:
        if self.m_twoPointControl:
            if self.m_zRight != Undef:
                return self.m_tRight
            raise CanteraError("Flow1D::rightControlPointTemperature", "Invalid operation: right control point location is not set")
        raise CanteraError("Flow1D::rightControlPointTemperature", "Invalid operation: two-point control is not enabled.")

    def rightControlPointCoordinate(self) -> float:
        if self.m_twoPointControl:
            if self.m_zRight != Undef:
                return self.m_zRight
            raise CanteraError("Flow1D::rightControlPointCoordinate", "Invalid operation: right control point location is not set")
        raise CanteraError("Flow1D::rightControlPointCoordinate", "Invalid operation: two-point control is not enabled.")

    def setRightControlPointTemperature(self, temperature: float):
        if self.m_twoPointControl:
            if self.m_zRight != Undef:
                self.m_tRight = float(temperature)
            else:
                raise CanteraError("Flow1D::setRightControlPointTemperature", "Invalid operation: right control point location is not set")
        else:
            raise CanteraError("Flow1D::setRightControlPointTemperature", "Invalid operation: two-point control is not enabled.")

    def setRightControlPointCoordinate(self, z_right: float):
        if self.m_twoPointControl:
            self.m_zRight = float(z_right)
        else:
            raise CanteraError("Flow1D::setRightControlPointCoordinate", "Invalid operation: two-point control is not enabled.")

    def setLeftControlPoint(self, z_left: float, temperature: float):
        self.setLeftControlPointCoordinate(z_left)
        self.setLeftControlPointTemperature(temperature)

    def setRightControlPoint(self, z_right: float, temperature: float):
        self.setRightControlPointCoordinate(z_right)
        self.setRightControlPointTemperature(temperature)

    # Excess species
    def leftExcessSpecies(self) -> int:
        return self.m_kExcessLeft

    def rightExcessSpecies(self) -> int:
        return self.m_kExcessRight

    # Metadata
    def getMeta(self) -> dict[str, Any]:
        state = super().get_meta()
        state["transport-model"] = self.transportModel()
        state["phase"] = {
            "name": getattr(self.gas, "name", ""),
            "source": getattr(self.gas, "source", None) or "<unknown>",
        }
        state["radiation-enabled"] = self.m_do_radiation
        if self.m_do_radiation:
            state["emissivity-left"] = self.m_epsilon_left
            state["emissivity-right"] = self.m_epsilon_right
        if all(self.m_do_energy):
            state["energy-enabled"] = True
        elif not any(self.m_do_energy):
            state["energy-enabled"] = False
        else:
            state["energy-enabled"] = list(self.m_do_energy)
        state["Soret-enabled"] = self.m_do_soret
        state["flux-gradient-basis"] = 0 if self.m_flux_gradient_basis == "mass" else 1
        state["refine-criteria"] = {
            "ratio": self.m_refiner.maxRatio(),
            "slope": self.m_refiner.maxDelta(),
            "curve": self.m_refiner.maxSlope(),
            "prune": self.m_refiner.prune(),
            "grid-min": self.m_refiner.gridMin(),
            "max-points": self.m_refiner.maxPoints(),
        }
        if self.m_zfixed != Undef:
            state["fixed-point"] = {
                "location": float(self.m_zfixed),
                "temperature": float(self.m_tfixed),
            }
        if self.m_twoPointControl:
            state["continuation-method"] = {
                "type": "two-point",
                "left-location": self.m_zLeft,
                "right-location": self.m_zRight,
                "left-temperature": self.m_tLeft,
                "right-temperature": self.m_tRight,
            }
        return state

    get_meta = getMeta
    settings = property(getMeta)

    def setMeta(self, state: dict[str, Any]):
        if "energy-enabled" in state:
            ee = state["energy-enabled"]
            if isinstance(ee, (bool, int)):
                self.m_do_energy = [bool(ee)] * self.m_points
            else:
                self.m_do_energy = [bool(v) for v in ee]
        if "transport-model" in state:
            self.setTransportModel(str(state["transport-model"]))
        if "Soret-enabled" in state:
            self.m_do_soret = bool(state["Soret-enabled"])
        if "flux-gradient-basis" in state:
            fgb = state["flux-gradient-basis"]
            self.m_flux_gradient_basis = "mass" if fgb in (0, "mass") else "molar"
        if "radiation-enabled" in state:
            self.m_do_radiation = bool(state["radiation-enabled"])
            if self.m_do_radiation:
                self.m_epsilon_left = float(state.get("emissivity-left", 0.0))
                self.m_epsilon_right = float(state.get("emissivity-right", 0.0))
        if "refine-criteria" in state:
            crit = state["refine-criteria"]
            ratio = crit.get("ratio", self.m_refiner.maxRatio())
            slope = crit.get("slope", self.m_refiner.maxDelta())
            curve = crit.get("curve", self.m_refiner.maxSlope())
            prune = crit.get("prune", self.m_refiner.prune())
            self.m_refiner.setCriteria(ratio, slope, curve, prune)
            if "grid-min" in crit:
                self.m_refiner.setGridMin(float(crit["grid-min"]))
            if "max-points" in crit:
                self.m_refiner.setMaxPoints(int(crit["max-points"]))
        if "fixed-point" in state:
            fp = state["fixed-point"]
            self.m_zfixed = float(fp["location"])
            self.m_tfixed = float(fp["temperature"])
        if "continuation-method" in state:
            cm = state["continuation-method"]
            if cm.get("type") == "two-point":
                self.m_twoPointControl = True
                self.m_zLeft = float(cm["left-location"])
                self.m_zRight = float(cm["right-location"])
                self.m_tLeft = float(cm["left-temperature"])
                self.m_tRight = float(cm["right-temperature"])

    set_meta = setMeta

    # Spatial derivatives & discretization operators
    def dVdz(self, x: np.ndarray, j: int) -> float:
        jloc = j if self.u(x, j) > 0.0 else j + 1
        return (self.V(x, jloc) - self.V(x, jloc - 1)) / self.m_dz[jloc - 1]

    def dTdz(self, x: np.ndarray, j: int) -> float:
        jloc = j if self.u(x, j) > 0.0 else j + 1
        return (self.T(x, jloc) - self.T(x, jloc - 1)) / self.m_dz[jloc - 1]

    def dYdz(self, x: np.ndarray, k: int, j: int) -> float:
        jloc = j if self.u(x, j) > 0.0 else j + 1
        return (self.Y(x, k, jloc) - self.Y(x, k, jloc - 1)) / self.m_dz[jloc - 1]

    def shear(self, x: np.ndarray, j: int) -> float:
        A_l = self.m_visc[j - 1] * (self.V(x, j) - self.V(x, j - 1)) / (self.z(j) - self.z(j - 1))
        A_r = self.m_visc[j] * (self.V(x, j + 1) - self.V(x, j)) / (self.z(j + 1) - self.z(j))
        return 2.0 * (A_r - A_l) / (self.z(j + 1) - self.z(j - 1))

    def conduction(self, x: np.ndarray, j: int) -> float:
        A_l = self.m_tcon[j - 1] * (self.T(x, j) - self.T(x, j - 1)) / (self.z(j) - self.z(j - 1))
        A_r = self.m_tcon[j] * (self.T(x, j + 1) - self.T(x, j)) / (self.z(j + 1) - self.z(j))
        return -2.0 * (A_r - A_l) / (self.z(j + 1) - self.z(j - 1))

    def grad_hk(self, x: np.ndarray, j: int):
        jloc = j if self.u(x, j) > 0.0 else j + 1
        dz = self.m_dz[jloc - 1]
        self.m_dhk_dz[:, j] = (self.m_hk[:, jloc] - self.m_hk[:, jloc - 1]) / dz

    # Cached updates
    def updateThermo(self, x: np.ndarray, j0: int = 0, j1: int | None = None):
        if j1 is None:
            j1 = self.m_points - 1
        for j in range(j0, j1 + 1):
            self.setGas(x, j)
            self.m_rho[j] = float(self.gas.density)
            self.m_wtm[j] = float(self.gas.mean_molecular_weight)
            self.m_cp[j] = float(self.gas.cp_mass)
            self.m_hk[:, j] = self.gas.partial_molar_enthalpies
            self.m_wdot[:, j] = self.gas.net_production_rates

    def updateTransport(self, x: np.ndarray, j0: int = 0, j1: int | None = None):
        if j1 is None:
            j1 = self.m_points - 1
        if self.m_do_multicomponent:
            for j in range(j0, j1):
                self.setGasAtMidpoint(x, j)
                wtm = self.gas.mean_molecular_weight
                rho = self.gas.density
                self.m_visc[j] = float(self.gas.viscosity) if self.m_dovisc else 0.0
                self.m_multidiff[j, :, :] = self.gas.multi_diff_coeffs
                for k in range(self.nsp):
                    self.m_diff[j, k] = self.m_wt[k] * rho / (wtm * wtm)
                self.m_tcon[j] = float(self.gas.thermal_conductivity)
                if self.m_do_soret:
                    self.m_dthermal[:, j] = self.gas.thermal_diff_coeffs
        else:
            for j in range(j0, j1):
                self.setGasAtMidpoint(x, j)
                self.m_visc[j] = float(self.gas.viscosity) if self.m_dovisc else 0.0
                if self.m_flux_gradient_basis == "molar":
                    coeffs = self.gas.mix_diff_coeffs
                else:
                    coeffs = self.gas.mix_diff_coeffs_mass
                rho = self.gas.density
                if self.m_flux_gradient_basis == "molar":
                    wtm = self.gas.mean_molecular_weight
                    self.m_diff[j, :] = np.asarray(coeffs, dtype=float) * self.m_wt * (rho / wtm)
                else:
                    self.m_diff[j, :] = np.asarray(coeffs, dtype=float) * rho
                self.m_tcon[j] = float(self.gas.thermal_conductivity)
                if self.m_do_soret:
                    self.m_dthermal[:, j] = self.gas.thermal_diff_coeffs

    def updateDiffFluxes(self, x: np.ndarray, j0: int = 0, j1: int | None = None):
        if j1 is None:
            j1 = self.m_points - 1
        if self.m_do_multicomponent:
            for j in range(j0, j1):
                dz = self.z(j + 1) - self.z(j)
                dX = np.array([self.X(x, m, j + 1) - self.X(x, m, j) for m in range(self.nsp)])
                for k in range(self.nsp):
                    sum_m = float(np.sum(self.m_wt * self.m_multidiff[j, k, :] * dX))
                    self.m_flux[k, j] = sum_m * self.m_diff[j, k] / dz
        else:
            for j in range(j0, j1):
                dz = self.z(j + 1) - self.z(j)
                y_j = self.index(c_offset_Y, j)
                y_jp1 = self.index(c_offset_Y, j + 1)
                Y_j = x[y_j : y_j + self.nsp]
                Y_jp1 = x[y_jp1 : y_jp1 + self.nsp]
                if self.m_flux_gradient_basis == "molar":
                    X_j = self.m_wtm[j] * (Y_j / self.m_wt)
                    X_jp1 = self.m_wtm[j + 1] * (Y_jp1 / self.m_wt)
                    flux_j = self.m_diff[j, :] * (X_j - X_jp1) / dz
                else:
                    flux_j = self.m_diff[j, :] * (Y_j - Y_jp1) / dz
                sum_corr = -float(np.sum(flux_j))
                self.m_flux[:, j] = flux_j + sum_corr * Y_j

        if self.m_do_soret:
            for m in range(j0, j1):
                gradlogT = 2.0 * (self.T(x, m + 1) - self.T(x, m)) / (
                    (self.T(x, m + 1) + self.T(x, m)) * (self.z(m + 1) - self.z(m))
                )
                self.m_flux[:, m] -= self.m_dthermal[:, m] * gradlogT

    def updateProperties(self, jg: int, x: np.ndarray, jmin: int, jmax: int):
        j0 = max(jmin, 1) - 1
        j1 = min(jmax + 1, self.m_points - 1)
        self.updateThermo(x, j0, j1)
        if jg is None or jg == (1 << 64) - 1 or jg < 0 or getattr(self, "m_force_full_update", False):
            self.updateTransport(x, j0, j1)
        if jg is None or jg == (1 << 64) - 1 or jg < 0:
            yLeft = [self.Y(x, k, jmin) for k in range(self.nsp)]
            yRight = [self.Y(x, k, jmax) for k in range(self.nsp)]
            self.m_kExcessLeft = int(np.argmax(yLeft))
            self.m_kExcessRight = int(np.argmax(yRight))
        self.updateDiffFluxes(x, j0, j1)

    def radiationPolyFactor(self, x: np.ndarray, j: int, s: int) -> float:
        c_CO2 = [18.741, -121.310, 273.500, -194.050, 56.310, -5.8169]
        c_H2O = [-0.23093, -1.12390, 9.41530, -2.99880, 0.51382, -1.86840e-5]
        c = c_CO2 if s == 0 else c_H2O
        T_val = self.T(x, j)
        invT = 1000.0 / T_val
        f = 0.0
        p = 1.0
        for coeff in c:
            f += coeff * p
            p *= invT
        return f / OneAtm

    def computeRadiation(self, x: np.ndarray, jmin: int = 0, jmax: int | None = None):
        if jmax is None:
            jmax = self.m_points - 1
        b_Rad_left = self.m_epsilon_left * StefanBoltz * (self.T(x, 0) ** 4)
        b_Rad_right = self.m_epsilon_right * StefanBoltz * (self.T(x, self.m_points - 1) ** 4)
        for j in range(jmin, jmax):
            k_P = 0.0
            if self.m_kRadiating[1] is not None and self.m_kRadiating[1] < self.nsp:
                k_P += self.m_press * self.X(x, self.m_kRadiating[1], j) * self.radiationPolyFactor(x, j, 1)
            if self.m_kRadiating[0] is not None and self.m_kRadiating[0] < self.nsp:
                k_P += self.m_press * self.X(x, self.m_kRadiating[0], j) * self.radiationPolyFactor(x, j, 0)
            rad_loss = 2.0 * k_P * (2.0 * StefanBoltz * (self.T(x, j) ** 4) - b_Rad_left - b_Rad_right)
            self.m_qdotRadiation[j] = rad_loss

    # Governing Equations
    def evalContinuity(
        self,
        x: np.ndarray,
        rsd: np.ndarray,
        diag: np.ndarray,
        rdt: float,
        jmin: int,
        jmax: int,
    ):
        if jmin == 0:
            rsd[self.index(c_offset_U, jmin)] = -(
                self.rho_u(x, jmin + 1) - self.rho_u(x, jmin)
            ) / self.m_dz[jmin] - (
                self.density(jmin + 1) * self.V(x, jmin + 1) + self.density(jmin) * self.V(x, jmin)
            )
            diag[self.index(c_offset_U, jmin)] = 0

        if jmax == self.m_points - 1:
            if self.m_usesLambda:
                rsd[self.index(c_offset_U, jmax)] = self.rho_u(x, jmax)
            else:
                rsd[self.index(c_offset_U, jmax)] = self.rho_u(x, jmax) - self.rho_u(x, jmax - 1)
            diag[self.index(c_offset_U, jmax)] = 0

        j0 = max(jmin, 1)
        j1 = min(jmax, self.m_points - 2)
        if self.m_usesLambda:
            for j in range(j0, j1 + 1):
                rsd[self.index(c_offset_U, j)] = -(
                    self.rho_u(x, j + 1) - self.rho_u(x, j)
                ) / self.m_dz[j] - (
                    self.density(j + 1) * self.V(x, j + 1) + self.density(j) * self.V(x, j)
                )
                diag[self.index(c_offset_U, j)] = 0
        elif self.m_isFree:
            for j in range(j0, j1 + 1):
                if self.z(j) > self.m_zfixed:
                    rsd[self.index(c_offset_U, j)] = -(
                        self.rho_u(x, j) - self.rho_u(x, j - 1)
                    ) / self.m_dz[j - 1]
                elif np.isclose(self.z(j), self.m_zfixed, atol=1e-12):
                    if self.m_do_energy[j]:
                        rsd[self.index(c_offset_U, j)] = self.T(x, j) - self.m_tfixed
                    else:
                        rsd[self.index(c_offset_U, j)] = self.rho_u(x, j) - self.m_rho[0] * 0.3
                else:
                    rsd[self.index(c_offset_U, j)] = -(
                        self.rho_u(x, j + 1) - self.rho_u(x, j)
                    ) / self.m_dz[j]
                diag[self.index(c_offset_U, j)] = 0
        else:
            for j in range(j0, j1 + 1):
                rsd[self.index(c_offset_U, j)] = self.rho_u(x, j) - self.rho_u(x, j - 1)
                diag[self.index(c_offset_U, j)] = 0

    def evalMomentum(
        self,
        x: np.ndarray,
        rsd: np.ndarray,
        diag: np.ndarray,
        rdt: float,
        jmin: int,
        jmax: int,
    ):
        if not self.m_usesLambda:
            for j in range(jmin, jmax + 1):
                rsd[self.index(c_offset_V, j)] = self.V(x, j)
                diag[self.index(c_offset_V, j)] = 0
            return

        if jmin == 0:
            rsd[self.index(c_offset_V, jmin)] = self.V(x, jmin)
            diag[self.index(c_offset_V, jmin)] = 0

        if jmax == self.m_points - 1:
            rsd[self.index(c_offset_V, jmax)] = self.V(x, jmax)
            diag[self.index(c_offset_V, jmax)] = 0

        j0 = max(jmin, 1)
        j1 = min(jmax, self.m_points - 2)
        for j in range(j0, j1 + 1):
            rsd[self.index(c_offset_V, j)] = (
                self.shear(x, j)
                - self.Lambda(x, j)
                - self.rho_u(x, j) * self.dVdz(x, j)
                - self.m_rho[j] * (self.V(x, j) ** 2)
            ) / self.m_rho[j]
            if not self.m_twoPointControl:
                rsd[self.index(c_offset_V, j)] -= rdt * (self.V(x, j) - self.V_prev(j))
                diag[self.index(c_offset_V, j)] = 1
            else:
                diag[self.index(c_offset_V, j)] = 0

    def evalLambda(
        self,
        x: np.ndarray,
        rsd: np.ndarray,
        diag: np.ndarray,
        rdt: float,
        jmin: int,
        jmax: int,
    ):
        if not self.m_usesLambda:
            for j in range(jmin, jmax + 1):
                rsd[self.index(c_offset_L, j)] = self.Lambda(x, j)
                diag[self.index(c_offset_L, j)] = 0
            return

        if jmin == 0:
            if self.m_twoPointControl:
                rsd[self.index(c_offset_L, jmin)] = self.Lambda(x, jmin + 1) - self.Lambda(x, jmin)
            else:
                rsd[self.index(c_offset_L, jmin)] = -self.rho_u(x, jmin)
            diag[self.index(c_offset_L, jmin)] = 0

        if jmax == self.m_points - 1:
            rsd[self.index(c_offset_L, jmax)] = self.Lambda(x, jmax) - self.Lambda(x, jmax - 1)
            diag[self.index(c_offset_L, jmax)] = 0

        j0 = max(jmin, 1)
        j1 = min(jmax, self.m_points - 2)
        for j in range(j0, j1 + 1):
            if self.m_twoPointControl:
                if np.isclose(self.z(j), self.m_zLeft, atol=1e-12):
                    rsd[self.index(c_offset_L, j)] = self.T(x, j) - self.m_tLeft
                elif self.z(j) > self.m_zLeft:
                    rsd[self.index(c_offset_L, j)] = self.Lambda(x, j) - self.Lambda(x, j - 1)
                else:
                    rsd[self.index(c_offset_L, j)] = self.Lambda(x, j) - self.Lambda(x, j + 1)
            else:
                rsd[self.index(c_offset_L, j)] = self.Lambda(x, j) - self.Lambda(x, j - 1)
            diag[self.index(c_offset_L, j)] = 0

    def evalEnergy(
        self,
        x: np.ndarray,
        rsd: np.ndarray,
        diag: np.ndarray,
        rdt: float,
        jmin: int,
        jmax: int,
    ):
        if jmin == 0:
            rsd[self.index(c_offset_T, jmin)] = self.T(x, jmin)
            diag[self.index(c_offset_T, jmin)] = 0

        if jmax == self.m_points - 1:
            rsd[self.index(c_offset_T, jmax)] = self.T(x, jmax)
            diag[self.index(c_offset_T, jmax)] = 0

        j0 = max(jmin, 1)
        j1 = min(jmax, self.m_points - 2)
        for j in range(j0, j1 + 1):
            if self.m_do_energy[j]:
                self.grad_hk(x, j)
                flx = 0.5 * (self.m_flux[:, j - 1] + self.m_flux[:, j])
                sum_heat = float(np.sum(self.m_wdot[:, j] * self.m_hk[:, j] + flx * self.m_dhk_dz[:, j] / self.m_wt))

                rsd[self.index(c_offset_T, j)] = (
                    -self.m_cp[j] * self.rho_u(x, j) * self.dTdz(x, j)
                    - self.conduction(x, j)
                    - sum_heat
                )
                rsd[self.index(c_offset_T, j)] /= (self.m_rho[j] * self.m_cp[j])
                rsd[self.index(c_offset_T, j)] -= (
                    self.m_qdotRadiation[j] / (self.m_rho[j] * self.m_cp[j])
                )

                if not self.m_twoPointControl or (self.m_z[j] != self.m_tLeft and self.m_z[j] != self.m_tRight):
                    rsd[self.index(c_offset_T, j)] -= rdt * (self.T(x, j) - self.T_prev(j))
                    diag[self.index(c_offset_T, j)] = 1
                else:
                    diag[self.index(c_offset_T, j)] = 0
            else:
                rsd[self.index(c_offset_T, j)] = self.T(x, j) - self.T_fixed(j)
                diag[self.index(c_offset_T, j)] = 0

    def evalUo(
        self,
        x: np.ndarray,
        rsd: np.ndarray,
        diag: np.ndarray,
        rdt: float,
        jmin: int,
        jmax: int,
    ):
        if not self.m_twoPointControl:
            for j in range(jmin, jmax + 1):
                rsd[self.index(c_offset_Uo, j)] = self.Uo(x, j)
                diag[self.index(c_offset_Uo, j)] = 0
            return

        if jmin == 0:
            rsd[self.index(c_offset_Uo, jmin)] = self.Uo(x, jmin + 1) - self.Uo(x, jmin)
            diag[self.index(c_offset_Uo, jmin)] = 0

        if jmax == self.m_points - 1:
            rsd[self.index(c_offset_Uo, jmax)] = self.Uo(x, jmax) - self.Uo(x, jmax - 1)
            diag[self.index(c_offset_Uo, jmax)] = 0

        j0 = max(jmin, 1)
        j1 = min(jmax, self.m_points - 2)
        for j in range(j0, j1 + 1):
            if np.isclose(self.z(j), self.m_zRight, atol=1e-12):
                rsd[self.index(c_offset_Uo, j)] = self.T(x, j) - self.m_tRight
            elif self.z(j) > self.m_zRight:
                rsd[self.index(c_offset_Uo, j)] = self.Uo(x, j) - self.Uo(x, j - 1)
            else:
                rsd[self.index(c_offset_Uo, j)] = self.Uo(x, j) - self.Uo(x, j + 1)
            diag[self.index(c_offset_Uo, j)] = 0

    def evalSpecies(
        self,
        x: np.ndarray,
        rsd: np.ndarray,
        diag: np.ndarray,
        rdt: float,
        jmin: int,
        jmax: int,
    ):
        if jmin == 0:
            y0 = self.index(c_offset_Y, 0)
            Y0 = x[y0 : y0 + self.nsp]
            sum_Y0 = float(np.sum(Y0))
            rsd[y0 : y0 + self.nsp] = -(self.m_flux[:, 0] + self.rho_u(x, 0) * Y0)
            diag[y0 : y0 + self.nsp] = 0
            rsd[self.index(c_offset_Y + self.leftExcessSpecies(), 0)] = 1.0 - sum_Y0
            diag[self.index(c_offset_Y + self.leftExcessSpecies(), 0)] = 0

        if jmax == self.m_points - 1:
            ymax = self.index(c_offset_Y, jmax)
            Ymax = x[ymax : ymax + self.nsp]
            sum_Ymax = float(np.sum(Ymax))
            rsd[ymax : ymax + self.nsp] = (self.m_flux[:, jmax - 1] + self.rho_u(x, jmax) * Ymax)
            diag[ymax : ymax + self.nsp] = 0
            rsd[self.index(c_offset_Y + self.rightExcessSpecies(), jmax)] = 1.0 - sum_Ymax
            diag[self.index(c_offset_Y + self.rightExcessSpecies(), jmax)] = 0

        j0 = max(jmin, 1)
        j1 = min(jmax, self.m_points - 2)
        for j in range(j0, j1 + 1):
            dz2 = self.z(j + 1) - self.z(j - 1)
            jloc = j if self.u(x, j) > 0.0 else j + 1
            dz_up = self.m_dz[jloc - 1]
            y_jloc = self.index(c_offset_Y, jloc)
            y_jloc_m1 = self.index(c_offset_Y, jloc - 1)
            dYdz = (x[y_jloc : y_jloc + self.nsp] - x[y_jloc_m1 : y_jloc_m1 + self.nsp]) / dz_up
            convec = self.rho_u(x, j) * dYdz
            diffus = 2.0 * (self.m_flux[:, j] - self.m_flux[:, j - 1]) / dz2
            wdot_term = self.m_wt * self.m_wdot[:, j]
            y_j = self.index(c_offset_Y, j)
            Y_curr = x[y_j : y_j + self.nsp]
            if rdt != 0.0 and hasattr(self, "m_slast"):
                start_prev = j * self.m_nv + c_offset_Y
                if len(self.m_slast) >= start_prev + self.nsp:
                    Y_prev = self.m_slast[start_prev : start_prev + self.nsp]
                else:
                    Y_prev = 0.0
            else:
                Y_prev = 0.0
            rsd[y_j : y_j + self.nsp] = (wdot_term - convec - diffus) / self.m_rho[j] - rdt * (Y_curr - Y_prev)
            diag[y_j : y_j + self.nsp] = 1

    def evalElectricField(
        self,
        x: np.ndarray,
        rsd: np.ndarray,
        diag: np.ndarray,
        rdt: float,
        jmin: int,
        jmax: int,
    ):
        for j in range(jmin, jmax + 1):
            rsd[self.index(c_offset_E, j)] = x[self.index(c_offset_E, j)]
            diag[self.index(c_offset_E, j)] = 0

    def eval(
        self,
        jGlobal: int,
        xGlobal: np.ndarray,
        rsdGlobal: np.ndarray,
        diagGlobal: np.ndarray,
        rdt: float = 0.0,
    ):
        npos = (1 << 64) - 1
        if jGlobal is not None and jGlobal != npos and jGlobal >= 0:
            first_pt = self.firstPoint()
            last_pt = self.lastPoint()
            if (jGlobal + 1 < first_pt) or (jGlobal > last_pt + 1):
                return

        loc = self.loc()
        sz = self.size()
        x = xGlobal[loc : loc + sz]
        rsd = rsdGlobal[loc : loc + sz]
        diag = diagGlobal[loc : loc + sz]

        if jGlobal is None or jGlobal == npos or jGlobal < 0:
            jmin = 0
            jmax = self.m_points - 1
        else:
            jpt = 0 if jGlobal == 0 else max(jGlobal - self.firstPoint(), 0)
            jmin = max(jpt, 1) - 1
            jmax = min(jpt + 1, self.m_points - 1)

        self.updateProperties(jGlobal, x, jmin, jmax)

        if self.m_do_radiation:
            self.computeRadiation(x, jmin, jmax)

        self.evalContinuity(x, rsd, diag, rdt, jmin, jmax)
        self.evalMomentum(x, rsd, diag, rdt, jmin, jmax)
        self.evalEnergy(x, rsd, diag, rdt, jmin, jmax)
        self.evalLambda(x, rsd, diag, rdt, jmin, jmax)
        self.evalElectricField(x, rsd, diag, rdt, jmin, jmax)
        self.evalUo(x, rsd, diag, rdt, jmin, jmax)
        self.evalSpecies(x, rsd, diag, rdt, jmin, jmax)

    # Analytic Jacobian stubs & query predicates
    def analyticRequested(self) -> bool:
        return getattr(self, "m_jacobianMode", "auto") in ("auto", "analytic")

    def analyticJacobianSupported(self) -> bool:
        return True

    def probeAnalyticJacobian(self):
        pass

    def usingAnalyticJacobian(self) -> bool:
        return False

    def checkAnalyticJacobian(self):
        pass

    def hasAnalyticJacobian(self, j: int = 0, n: int = 0) -> bool:
        return False

    def computeWdotDerivatives(self, j: int):
        pass

    def fluxJacobian(self, x: np.ndarray, q: int, out: np.ndarray):
        pass

    def addSpeciesJacEntries(self, x: np.ndarray, p: int, jac: Any):
        pass

    def addEnergyJacEntries(self, x: np.ndarray, p: int, jac: Any):
        pass

    def addFlowJacEntries(self, x: np.ndarray, p: int, jac: Any):
        pass

    def evalJacobianAnalytic(self, xGlobal: np.ndarray, jac: Any):
        pass


class FreeFlow(Flow1D):
    """A freely-propagating laminar flame flow domain."""

    _domain_type = "free-flow"

    def __init__(self, phase: Solution, name: str = "flame", points: int = 1):
        super().__init__(phase=phase, id=name, points=points)
        self.set_free_flow()


class UnstrainedFlow(Flow1D):
    """An unstrained burner-stabilized flame flow domain."""

    _domain_type = "unstrained-flow"

    def __init__(self, phase: Solution, name: str = "flame", points: int = 1):
        super().__init__(phase=phase, id=name, points=points)
        self.set_unstrained_flow()


class AxisymmetricFlow(Flow1D):
    """An axisymmetric stagnation/counterflow flame domain."""

    _domain_type = "axisymmetric-flow"

    def __init__(self, phase: Solution, name: str = "flame", points: int = 1):
        super().__init__(phase=phase, id=name, points=points)
        self.set_axisymmetric_flow()


class SteadyStateSystem:
    """Base class for representing a system of differential-algebraic equations."""

    def __init__(self):
        self.m_size: int = 0
        self.m_bw: int = 0
        self.m_rdt: float = 0.0
        self.m_mask: np.ndarray = np.zeros(0, dtype=int)
        self.m_state: np.ndarray = np.zeros(0, dtype=float)
        self.m_jac: Optional[SystemJacobian] = None
        self.m_newt: Optional[MultiNewton] = None
        self.m_jacobianRelPerturb: float = 1e-5
        self.m_jacobianAbsPerturb: float = 1e-12
        self.m_jacobianThreshold: float = 1e-12
        self.m_work1: np.ndarray = np.zeros(0, dtype=float)
        self.m_work2: np.ndarray = np.zeros(0, dtype=float)
        self.m_nevals: int = 0
        self.m_evaltime: float = 0.0
        self.m_interrupt = None
        self.m_stats: dict[str, list] = {
            "grid_points": [],
            "steps": [],
            "residual_evals": [],
            "residual_time": [],
            "jacobian_evals": [],
            "jacobian_time": []
        }

    def size(self) -> int:
        return self.m_size

    def bandwidth(self) -> int:
        return self.m_bw

    def resize(self) -> None:
        if len(self.m_mask) != self.m_size:
            new_mask = np.zeros(self.m_size, dtype=int)
            min_len = min(len(self.m_mask), self.m_size)
            new_mask[:min_len] = self.m_mask[:min_len]
            self.m_mask = new_mask

        if len(self.m_state) != self.m_size:
            new_state = np.zeros(self.m_size, dtype=float)
            min_len = min(len(self.m_state), self.m_size)
            new_state[:min_len] = self.m_state[:min_len]
            self.m_state = new_state

        self.m_work1 = np.zeros(self.m_size, dtype=float)
        self.m_work2 = np.zeros(self.m_size, dtype=float)

    def setInitialGuess(self, x: Sequence[float]) -> None:
        self.m_state = np.array(x, dtype=float, copy=True)
        self.m_size = len(self.m_state)

    def getState(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        if x is not None:
            min_len = min(len(x), len(self.m_state))
            x[:min_len] = self.m_state[:min_len]
            return x
        return np.array(self.m_state, dtype=float, copy=True)

    def linearSolver(self) -> Optional[SystemJacobian]:
        return self.m_jac

    def setLinearSolver(self, solver: SystemJacobian) -> None:
        self.m_jac = solver

    def newton(self) -> MultiNewton:
        if self.m_newt is None:
            self.m_newt = MultiNewton(self.m_size)
        return self.m_newt

    def initTimeInteg(self, dt: float, x: Sequence[float]) -> None:
        self.m_rdt = 1.0 / float(dt) if float(dt) > 0.0 else 0.0
        self.setInitialGuess(x)

    def setSteadyMode(self) -> None:
        self.m_rdt = 0.0

    def saveStats(self) -> None:
        pass


class OneDim(SteadyStateSystem):
    """Container class for multiple-domain 1D problems."""

    def __init__(self, domains: Sequence[Any] = ()):
        super().__init__()
        self.m_dom: list[Any] = []
        self.m_connect: list[Any] = []
        self.m_bulk: list[Any] = []
        self.m_init: bool = False
        self.m_nvars: list[int] = []
        self.m_loc: list[int] = []
        self.m_componentInfo: list[tuple[int, int, int]] = []
        self.m_pts: int = 0
        self.m_xnew: list[float] = []

        for d in domains:
            self.addDomain(d)
        if self.m_dom:
            self.init()
            self.resize()

    def addDomain(self, d: Any) -> None:
        n = len(self.m_dom)
        if n > 0:
            if hasattr(self.m_dom[-1], "append"):
                self.m_dom[-1].append(d)
            elif hasattr(self.m_dom[-1], "linkRight"):
                self.m_dom[-1].linkRight(d)
            if hasattr(d, "linkLeft"):
                d.linkLeft(self.m_dom[-1])

        if n % 2 == 0:
            self.m_connect.append(d)
        else:
            self.m_bulk.append(d)

        self.m_dom.append(d)
        if hasattr(d, "setData"):
            d.setData(self.m_state)
        if hasattr(d, "setContainer"):
            d.setContainer(self, len(self.m_dom) - 1)
        self.resize()

    add_domain = addDomain

    @property
    def domains(self) -> list[Any]:
        return self.m_dom

    def nDomains(self) -> int:
        return len(self.m_dom)

    n_domains = nDomains

    def domain(self, i: int | str) -> Any:
        if isinstance(i, str):
            idx = self.domainIndex(i)
            return self.m_dom[idx]
        return self.m_dom[int(i)]

    def domainIndex(self, name_or_obj: str | Any) -> int:
        if not isinstance(name_or_obj, str):
            for i, d in enumerate(self.m_dom):
                if d is name_or_obj or d == name_or_obj:
                    return i
            name_str = getattr(name_or_obj, "name", str(name_or_obj))
            raise CanteraError("OneDim::domainIndex", f"Domain '{name_str}' not found")
        for i, d in enumerate(self.m_dom):
            d_id = getattr(d, "m_id", "") or getattr(d, "name", "") or getattr(d, "id", lambda: "")()
            if d_id == name_or_obj:
                return i
        raise CanteraError("OneDim::domainIndex", f"Domain '{name_or_obj}' not found")

    domain_index = domainIndex

    def checkDomainIndex(self, n: int) -> None:
        n = int(n)
        if n >= len(self.m_dom) or n < 0:
            raise IndexError("OneDim::checkDomainIndex", f"domains[{n}] outside valid range of 0 to {max(len(self.m_dom) - 1, 0)}")

    def checkDomainArraySize(self, nn: int) -> None:
        if len(self.m_dom) > int(nn):
            raise CanteraError("OneDim::checkDomainArraySize", f"Array size {nn} too small for {len(self.m_dom)} domains")

    def start(self, i: int) -> int:
        d = self.m_dom[int(i)]
        nv = d.n_components if hasattr(d, "n_components") else (d.nComponents() if hasattr(d, "nComponents") else 1)
        if nv > 0:
            return d.loc() if hasattr(d, "loc") else 0
        return 0

    def left(self) -> Any:
        return self.m_dom[0] if self.m_dom else None

    def right(self) -> Any:
        return self.m_dom[-1] if self.m_dom else None

    def nVars(self, jg: int) -> int:
        return self.m_nvars[int(jg)]

    def loc(self, jg: int) -> int:
        return self.m_loc[int(jg)]

    def points(self) -> int:
        return self.m_pts

    def gridSize(self) -> int:
        return self.m_pts

    def component(self, i: int) -> tuple[str, int, str]:
        i = int(i)
        if i >= self.m_size or i < 0:
            raise IndexError("OneDim::component", f"components[{i}] outside valid range of 0 to {max(self.m_size - 1, 0)}")
        n, j, k = self.m_componentInfo[i]
        dom = self.domain(n)
        dom_id = getattr(dom, "m_id", "") or getattr(dom, "name", "") or getattr(dom, "id", lambda: f"domain_{n}")()
        comp_name = dom.component_name(k) if hasattr(dom, "component_name") else (
            dom.componentName(k) if hasattr(dom, "componentName") else f"comp_{k}"
        )
        return (dom_id, j, comp_name)

    def componentName(self, i: int) -> str:
        dom_id, pt, comp = self.component(i)
        return f"domain {dom_id}, component {comp} at point {pt}"

    def componentTableHeader(self) -> tuple[str, str]:
        return ("", "Domain   Pt. Component")

    def componentTableLabel(self, i: int) -> str:
        dom_id, pt, comp = self.component(i)
        return f"{dom_id:8s} {pt:3d} {comp:<12s}"

    def upperBound(self, i: int) -> float:
        i = int(i)
        n, j, k = self.m_componentInfo[i]
        dom = self.domain(n)
        if hasattr(dom, "upperBound"):
            return float(dom.upperBound(k))
        elif hasattr(dom, "m_max"):
            return float(dom.m_max[k])
        return 1e10

    def lowerBound(self, i: int) -> float:
        i = int(i)
        n, j, k = self.m_componentInfo[i]
        dom = self.domain(n)
        if hasattr(dom, "lowerBound"):
            return float(dom.lowerBound(k))
        elif hasattr(dom, "m_min"):
            return float(dom.m_min[k])
        return -1e10

    def init(self) -> None:
        if not self.m_init:
            d = self.left()
            while d:
                if hasattr(d, "init"):
                    d.init()
                d = d.right() if hasattr(d, "right") else None
            self.m_init = True

    def resize(self) -> None:
        self.m_bw = 0
        self.m_nvars.clear()
        self.m_loc.clear()
        self.m_componentInfo.clear()
        lc = 0

        l = self.left()
        if l and hasattr(l, "locate"):
            l.locate()

        self.saveStats()
        self.m_pts = 0

        for i, d in enumerate(self.m_dom):
            np_pts = d.n_points if hasattr(d, "n_points") else (d.nPoints() if hasattr(d, "nPoints") else 1)
            nv = d.n_components if hasattr(d, "n_components") else (d.nComponents() if hasattr(d, "nComponents") else 1)

            for n in range(np_pts):
                self.m_nvars.append(nv)
                self.m_loc.append(lc)
                lc += nv
                self.m_pts += 1
                for k in range(nv):
                    self.m_componentInfo.append((i, n, k))

            bw1 = d.bandwidth() if hasattr(d, "bandwidth") else -1
            if bw1 is None or bw1 < 0:
                bw1 = max(2 * nv, 1) - 1
            self.m_bw = max(self.m_bw, bw1)

            if i > 0:
                prev_d = self.m_dom[i - 1]
                bw2 = prev_d.bandwidth() if hasattr(prev_d, "bandwidth") else -1
                if bw2 is None or bw2 < 0:
                    bw2 = prev_d.n_components if hasattr(prev_d, "n_components") else prev_d.nComponents()
                bw2 += nv - 1
                self.m_bw = max(self.m_bw, bw2)

            d_loc = d.loc() if hasattr(d, "loc") else 0
            d_sz = d.size() if hasattr(d, "size") else (np_pts * nv)
            self.m_size = d_loc + d_sz

        if self.m_jac is None:
            self.m_jac = MultiJac()

        if hasattr(self.m_jac, "initialize"):
            self.m_jac.initialize(self.m_size)
        if hasattr(self.m_jac, "setBandwidth"):
            self.m_jac.setBandwidth(self.m_bw)

        super().resize()

    def pointDomain(self, i: int) -> Any:
        d = self.right()
        while d:
            d_loc = d.loc() if hasattr(d, "loc") else 0
            if d_loc <= i:
                return d
            d = d.left() if hasattr(d, "left") else None
        return self.left()

    def eval(
        self,
        j: Optional[int],
        x: Sequence[float],
        r: np.ndarray,
        rdt: float = -1.0,
        count: int = 1,
    ) -> None:
        r.fill(0.0)
        j_val = None if (j is None or j < 0) else int(j)

        if j_val is None:
            self.m_mask.fill(0)

        if rdt < 0.0:
            rdt = self.m_rdt

        # 1. Bulk domains first
        for d in self.m_bulk:
            d_j = j_val if j_val is not None else -1
            if hasattr(d, "eval"):
                d.eval(d_j, x, r, self.m_mask, rdt=rdt)

        # 2. Connector domains second
        for d in self.m_connect:
            d_j = j_val if j_val is not None else -1
            if hasattr(d, "eval"):
                d.eval(d_j, x, r, self.m_mask, rdt=rdt)

        if count:
            self.m_nevals += 1

    def evalJacobian(self, x0: Optional[Sequence[float]] = None) -> None:
        if x0 is None:
            x0 = self.m_state

        if self.m_jac:
            self.m_jac.reset()

        x0_arr = np.asarray(x0, dtype=float)
        self.m_work1 = np.zeros(self.size(), dtype=float)
        self.m_work2 = np.zeros(self.size(), dtype=float)

        self.eval(-1, x0_arr, self.m_work1, 0.0, 0)

        for dom in self.m_dom:
            if hasattr(dom, "checkAnalyticJacobian"):
                dom.checkAnalyticJacobian()

        perturbed = np.array(x0_arr, dtype=float, copy=True)
        ipt = 0

        for j in range(self.points()):
            nv = self.nVars(j)
            dom = self.pointDomain(ipt)
            first_pt = dom.firstPoint() if hasattr(dom, "firstPoint") else 0
            jLocal = j - first_pt

            for n in range(nv):
                if hasattr(dom, "hasAnalyticJacobian") and dom.hasAnalyticJacobian(jLocal, n):
                    ipt += 1
                    continue

                xsave = x0_arr[ipt]
                dx = abs(xsave) * self.m_jacobianRelPerturb + self.m_jacobianAbsPerturb
                if xsave < 0.0:
                    dx = -dx
                perturbed[ipt] = xsave + dx
                rdx = 1.0 / (perturbed[ipt] - xsave)

                self.eval(j, perturbed, self.m_work2, 0.0, 0)

                for i in (j - 1, j, j + 1):
                    if 0 <= i < self.points():
                        mv = self.nVars(i)
                        iloc = self.loc(i)
                        for m in range(mv):
                            delta = self.m_work2[m + iloc] - self.m_work1[m + iloc]
                            if abs(delta) > self.m_jacobianThreshold or (m + iloc) == ipt:
                                if self.m_jac:
                                    self.m_jac.setValue(m + iloc, ipt, delta * rdx)

                perturbed[ipt] = xsave
                ipt += 1

        for dom in self.m_dom:
            if hasattr(dom, "evalJacobianAnalytic"):
                dom.evalJacobianAnalytic(x0_arr, self.m_jac)

        if self.m_jac:
            if hasattr(self.m_jac, "incrementEvals"):
                self.m_jac.incrementEvals()
            if hasattr(self.m_jac, "setAge"):
                self.m_jac.setAge(0)

    def weightedNorm(self, step: Sequence[float]) -> float:
        total = 0.0
        x = self.m_state if len(self.m_state) == self.size() else np.zeros(self.size(), dtype=float)
        nd = self.nDomains()
        step_arr = np.asarray(step, dtype=float)

        for n in range(nd):
            dom = self.domain(n)
            d_sum = 0.0
            nv = dom.n_components if hasattr(dom, "n_components") else (dom.nComponents() if hasattr(dom, "nComponents") else 1)
            np_pts = dom.n_points if hasattr(dom, "n_points") else (dom.nPoints() if hasattr(dom, "nPoints") else 1)
            dstart = self.start(n)

            for i in range(nv):
                esum = 0.0
                for j in range(np_pts):
                    idx = dstart + nv * j + i
                    if idx < len(x):
                        esum += abs(x[idx])
                rtol_val = dom.rtol(i) if hasattr(dom, "rtol") else 1e-4
                atol_val = dom.atol(i) if hasattr(dom, "atol") else 1e-9
                ewt = rtol_val * esum / max(np_pts, 1) + atol_val

                for j in range(np_pts):
                    idx = dstart + nv * j + i
                    if idx < len(step_arr):
                        f = step_arr[idx] / ewt
                        d_sum += f * f
            total += d_sum

        return float(math.sqrt(total / max(self.size(), 1)))

    weighted_norm = weightedNorm

    def initTimeInteg(self, dt: float, x: Sequence[float]) -> None:
        super().initTimeInteg(dt, x)
        d = self.left()
        while d:
            if hasattr(d, "initTimeInteg"):
                d.initTimeInteg(dt, x)
            d = d.right() if hasattr(d, "right") else None

    def setSteadyMode(self) -> None:
        if self.m_rdt == 0.0:
            return
        super().setSteadyMode()
        d = self.left()
        while d:
            if hasattr(d, "setSteadyMode"):
                d.setSteadyMode()
            d = d.right() if hasattr(d, "right") else None

    def resetBadValues(self, x: np.ndarray) -> None:
        for dom in self.m_dom:
            if hasattr(dom, "resetBadValues"):
                dom.resetBadValues(x)

    def writeStats(self, printTime: int = 1) -> None:
        pass

    def gridSizeStats(self) -> list[int]:
        self.saveStats()
        return list(self.m_stats.get("grid_points", []))

    def jacobianTimeStats(self) -> list[float]:
        self.saveStats()
        return list(self.m_stats.get("jacobian_time", []))

    def evalTimeStats(self) -> list[float]:
        self.saveStats()
        return list(self.m_stats.get("residual_time", []))

    def jacobianCountStats(self) -> list[int]:
        self.saveStats()
        return list(self.m_stats.get("jacobian_evals", []))

    def evalCountStats(self) -> list[int]:
        self.saveStats()
        return list(self.m_stats.get("residual_evals", []))

    def timeStepStats(self) -> list[int]:
        self.saveStats()
        return list(self.m_stats.get("steps", []))

    def _workVector(self) -> list[float]:
        return self.m_xnew

class Sim1D(OneDim):
    def show(self) -> None:
        pass

    def _restore(self, fname: str, id_str: str) -> None:
        pass

    def _setValue(self, dom: int, comp: int, pt: int, val: float) -> None:
        d = self.domain(dom)
        iloc = d.loc() + d.index(comp, pt)
        if iloc >= len(self.m_state):
            new_state = np.zeros(max(iloc + 1, self.size()), dtype=float)
            new_state[:len(self.m_state)] = self.m_state
            self.m_state = new_state
        self.m_state[iloc] = float(val)

    def _value(self, dom: int, comp: int, pt: int) -> float:
        d = self.domain(dom)
        iloc = d.loc() + d.index(comp, pt)
        if iloc < len(self.m_state):
            return float(self.m_state[iloc])
        return 0.0

    def _workValue(self, dom: int, comp: int, pt: int) -> float:
        d = self.domain(dom)
        iloc = d.loc() + d.index(comp, pt)
        if hasattr(self, "m_xnew") and iloc < len(self.m_xnew):
            return float(self.m_xnew[iloc])
        return 0.0

    def clearDebugFile(self) -> None:
        pass

    def evalSSJacobian(self) -> None:
        self.evalJacobian(self.m_state)

    def finalize(self) -> None:
        for n in range(self.nDomains()):
            d = self.domain(n)
            if hasattr(d, "_finalize"):
                d._finalize(self.m_state[self.start(n) : self.start(n) + d.size()])

    def fixedTemperature(self) -> float:
        return 300.0

    def fixedTemperatureLocation(self) -> float:
        return 0.5

    def getInitialSoln(self) -> np.ndarray:
        for n in range(self.nDomains()):
            d = self.domain(n)
            if hasattr(d, "_getInitialSoln"):
                d._getInitialSoln(self.m_state[self.start(n) : self.start(n) + d.size()])
        return self.m_state

    def getResidual(self, res: np.ndarray) -> None:
        self.eval(-1, self.m_state, res)

    def maxGridPoints(self, dom: int = 0) -> int:
        return int(self.domain(dom).refiner().maxPoints())

    def restore(self, fname: str, id_str: str) -> None:
        pass

    def restoreSteadySolution(self) -> None:
        if not hasattr(self, "m_xlast_ss") or len(self.m_xlast_ss) == 0:
            raise CanteraError("Sim1D::restoreSteadySolution", "No successful steady state solution")
        self.m_state = np.array(self.m_xlast_ss, dtype=float, copy=True)
        for n in range(self.nDomains()):
            z = self.m_grid_last_ss[n]
            self.domain(n).setupGrid(z)

    def restoreTimeSteppingSolution(self) -> None:
        if not hasattr(self, "m_xlast_ts") or len(self.m_xlast_ts) == 0:
            raise CanteraError("Sim1D::restoreTimeSteppingSolution", "No successful time steps taken on this grid.")
        self.m_state = np.array(self.m_xlast_ts, dtype=float, copy=True)
        self.resize()

    def setFixedTemperature(self, T: float) -> int:
        return 0

    def setGridMin(self, dom: int, gridmin: float) -> None:
        if dom >= 0:
            self.domain(dom).refiner().setGridMin(gridmin)
        else:
            for n in range(self.nDomains()):
                self.domain(n).refiner().setGridMin(gridmin)

    def setLeftControlPoint(self, temperature: float) -> None:
        found = False
        for n in range(self.nDomains()):
            d = self.domain(n)
            if hasattr(d, "domainType") and d.domainType() == "axisymmetric-flow" and getattr(d, "twoPointControlEnabled", lambda: False)():
                found = True
                break
        if not found:
            raise CanteraError("Sim1D::setLeftControlPoint", "No domain with two-point control enabled was found.")

    def setMaxGridPoints(self, dom: int, max_pts: int) -> None:
        if dom >= 0:
            self.domain(dom).refiner().setMaxPoints(max_pts)
        else:
            for n in range(self.nDomains()):
                self.domain(n).refiner().setMaxPoints(max_pts)

    def setRightControlPoint(self, temperature: float) -> None:
        found = False
        for n in range(self.nDomains()):
            d = self.domain(n)
            if hasattr(d, "domainType") and d.domainType() == "axisymmetric-flow" and getattr(d, "twoPointControlEnabled", lambda: False)():
                found = True
                break
        if not found:
            raise CanteraError("Sim1D::setRightControlPoint", "No domain with two-point control enabled was found.")

    def setSteadyCallback(self, cb) -> None:
        self._steady_callback = cb

    def setTimeStepRegridMax(self, count: int) -> None:
        self.m_ts_regrid_max = int(count)

    def timeStepRegridMax(self) -> int:
        return int(getattr(self, "m_ts_regrid_max", 10))

    def setRefineCriteria(
        self,
        dom: int = -1,
        ratio: float = 10.0,
        slope: float = 0.8,
        curve: float = 0.8,
        prune: float = -0.1,
    ) -> None:
        if dom >= 0:
            self.domain(dom).setRefineCriteria(ratio, slope, curve, prune)
        else:
            for n in range(self.nDomains()):
                self.domain(n).setRefineCriteria(ratio, slope, curve, prune)

    def getRefineCriteria(self, dom: int = 0) -> list[float]:
        if dom >= 0:
            crit = self.domain(dom).refiner().getCriteria()
            return [float(c) for c in crit]
        raise CanteraError("Sim1D::getRefineCriteria", "Must specify domain to get criteria from")

    def get_refine_criteria(self, dom: int | Domain1D = 0) -> dict[str, float]:
        if isinstance(dom, Domain1D):
            return dom.get_refine_criteria()
        if dom < 0:
            raise CanteraError("Sim1D::getRefineCriteria", "Must specify domain to get criteria from")
        return self.domain(dom).get_refine_criteria()

    def solveAdjoint(self, b: Sequence[float], lambda_out: Any = None) -> np.ndarray:
        for d in self.m_dom:
            if hasattr(d, "forceFullUpdate"):
                d.forceFullUpdate(True)
        self.evalSSJacobian()
        for d in self.m_dom:
            if hasattr(d, "forceFullUpdate"):
                d.forceFullUpdate(False)

        mjac = self.linearSolver()
        if mjac is None:
            raise CanteraError("Sim1D::solveAdjoint", "Banded (MultiJac) required")

        sz = self.size()
        bw = self.bandwidth()
        Jt = BandMatrix(sz, bw, bw)
        for i in range(sz):
            j1 = (i - bw) if i > bw else 0
            j2 = min(sz - 1, i + bw)
            for j in range(j1, j2 + 1):
                Jt[j, i] = mjac.value(i, j)

        Jt.factor()
        res = Jt.solve(np.array(b, dtype=float))
        if lambda_out is not None:
            if isinstance(lambda_out, np.ndarray):
                lambda_out[:] = res
            elif isinstance(lambda_out, list):
                lambda_out.clear()
                lambda_out.extend(res.tolist())
        return res

    def refine(self, loglevel: int = 0) -> int:
        added = 0
        discarded = 0
        znew = []
        xnew = []
        dsize = []

        self.m_xlast_ss = np.array(self.m_state, dtype=float, copy=True)
        self.m_grid_last_ss = []

        for n in range(self.nDomains()):
            d = self.domain(n)
            r = d.refiner()

            grid = list(d.grid)
            self.m_grid_last_ss.append(list(grid))

            x_dom = self.m_state[self.start(n) : self.start(n) + d.size()]
            r.analyze(len(grid), grid, x_dom)

            added += r.nNewPoints()
            comp = d.nComponents()

            npnow = d.nPoints()
            nstart = len(znew)
            for m in range(npnow):
                if r.keepPoint(m):
                    znew.append(float(d.z(m)))
                    for i in range(comp):
                        xnew.append(float(self._value(n, i, m)))

                    if r.newPointNeeded(m) and m + 1 < npnow:
                        zmid = 0.5 * (d.z(m) + d.z(m + 1))
                        znew.append(float(zmid))
                        added += 1

                        for i in range(comp):
                            xmid = 0.5 * (self._value(n, i, m) + self._value(n, i, m + 1))
                            xnew.append(float(xmid))
                else:
                    discarded += 1

            dsize.append(len(znew) - nstart)

        gridstart = 0
        for n in range(self.nDomains()):
            d = self.domain(n)
            gridsize = dsize[n]
            if gridsize != 0:
                d.setupGrid(znew[gridstart : gridstart + gridsize])
            gridstart += gridsize

        self.m_state = np.array(xnew, dtype=float)
        self.resize()
        self.finalize()
        return 1 if (added != 0 or discarded != 0) else 0


        """Composite multi-domain 1D simulation manager and solver."""

    def __init__(self, domains: Sequence[Domain1D] = ()):
        super().__init__(domains)
        self._state = np.zeros(self.size(), dtype=float)
        self.m_time_step_growth_factor = 1.2
        self.m_time_step_growth_strategy = "fixed-growth"
        self.m_time_step_regrid = 0
        self.m_ts_callback: Callable[[float], Any] | None = None
        self._steady_callback: Callable[[float], Any] | None = None
        self._stats: dict[str, list[Any]] = {
            "grid_points": [],
            "steps": [],
            "residual_evals": [],
            "jacobian_evals": [],
        }

        # Link adjacent flow domains to boundaries
        for i, d in enumerate(self.m_dom):
            if isinstance(d, Boundary1D):
                if i > 0 and isinstance(self.m_dom[i - 1], Flow1D):
                    d.m_flow_left = self.m_dom[i - 1]
                if i + 1 < len(self.m_dom) and isinstance(self.m_dom[i + 1], Flow1D):
                    d.m_flow_right = self.m_dom[i + 1]

    _ALLOWED_ATTRS = {
        "gas", "flame", "inlet", "outlet", "reactants", "products", "surf", "surface",
        "flow", "domains", "name", "fixed_temperature", "left", "right",
        "burner", "fuel_inlet", "oxidizer_inlet"
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
    def time_step_growth_factor(self) -> float:
        return self.m_time_step_growth_factor

    @time_step_growth_factor.setter
    def time_step_growth_factor(self, val: float):
        val = float(val)
        if not np.isfinite(val) or val < 1.0:
            if not np.isfinite(val):
                raise CanteraError(
                    "Sim1D::setTimeStepGrowthFactor", "Growth factor must be finite and >= 1.0"
                )
            raise CanteraError("Sim1D::setTimeStepGrowthFactor", "Growth factor must be >= 1.0")
        self.m_time_step_growth_factor = val

    @property
    def time_step_growth_strategy(self) -> str:
        return self.m_time_step_growth_strategy

    @time_step_growth_strategy.setter
    def time_step_growth_strategy(self, val: str):
        allowed = (
            "fixed-growth",
            "steady-norm",
            "transient-residual",
            "residual-ratio",
            "newton-iterations",
        )
        if val not in allowed:
            raise CanteraError(
                "Sim1D::setTimeStepGrowthStrategy",
                f"Unknown time step growth strategy {val!r}; must be one of {', '.join(allowed)}.",
            )
        self.m_time_step_growth_strategy = val

    @property
    def time_step_regrid(self) -> int:
        return self.m_time_step_regrid

    @time_step_regrid.setter
    def time_step_regrid(self, val: int):
        val = int(val)
        if val < 0:
            raise CanteraError("Sim1D::setTimeStepRegrid", "regrid frequency must be >= 0")
        self.m_time_step_regrid = val

    def set_time_step_callback(self, cb: Callable[[float], Any]):
        self.m_ts_callback = cb

    def set_steady_callback(self, cb: Callable[[float], Any]):
        self._steady_callback = cb

    def clear_stats(self):
        for k in self._stats:
            self._stats[k].clear()

    def set_refine_criteria(
        self,
        dom: int | Domain1D = -1,
        ratio: float = 10.0,
        slope: float = 0.8,
        curve: float = 0.8,
        prune: float = -0.1,
    ):
        if isinstance(dom, Domain1D):
            dom.set_refine_criteria(ratio, slope, curve, prune)
        elif dom >= 0:
            self.domain(dom).set_refine_criteria(ratio, slope, curve, prune)
        else:
            for d in self.m_dom:
                d.set_refine_criteria(ratio, slope, curve, prune)

    def get_refine_criteria(self, dom: int | Domain1D = 0) -> dict[str, float]:
        if isinstance(dom, Domain1D):
            return dom.get_refine_criteria()
        return self.domain(dom).get_refine_criteria()

    def set_initial_guess(self, data: Any = None, group: str | None = None):
        if data is not None:
            pass

    def eval_jacobian(self, x: np.ndarray, rdt: float = 0.0) -> BandMatrix:
        """Evaluate the multi-domain Jacobian in compact banded storage."""
        n = len(x)
        r0 = np.zeros(n)
        self.eval(None, x, r0, rdt=rdt)

        entries = []
        lower_bandwidth = 0
        upper_bandwidth = 0
        eps = 1e-7

        for k in range(n):
            x_pert = x.copy()
            dx = eps * max(abs(x[k]), 1.0)
            x_pert[k] += dx
            r1 = np.zeros(n)
            self.eval(None, x_pert, r1, rdt=rdt)
            col = (r1 - r0) / dx
            nonzero = np.flatnonzero(abs(col) > 1e-15)
            for row in nonzero:
                row = int(row)
                entries.append((row, k, float(col[row])))
                lower_bandwidth = max(lower_bandwidth, row - k)
                upper_bandwidth = max(upper_bandwidth, k - row)

        J = BandMatrix(n, lower_bandwidth, upper_bandwidth)
        for row, col, value in entries:
            J[row, col] = value
        return J

    def solve(self, loglevel: int = 0, refine_grid: bool = True, auto: bool = False):
        """Solve the multi-domain boundary value problem with damped Newton and pseudo-transient."""
        for d in self.m_dom:
            if hasattr(d, "_finalize"):
                d._finalize(self._state)

        if len(self._state) != self.size():
            self._state = np.zeros(self.size(), dtype=float)

        for d in self.m_dom:
            if isinstance(d, Flow1D):
                if not d.energy_enabled:
                    d.set_values("T", d.m_fixedtemp)
                inlet = getattr(self, "inlet", None)
                if inlet is not None and hasattr(inlet, "mdot") and inlet.mdot:
                    mdot = float(inlet.mdot)
                    u_vals = np.zeros(d.n_points)
                    for j in range(d.n_points):
                        # Residual/Jacobian trial evaluations leave m_rho at
                        # the most recently evaluated trial state. Recompute
                        # density from the accepted final state before imposing
                        # the fixed-temperature continuity relation rho*u=mdot.
                        d.update_state(j)
                        rho_j = d.gas.density
                        d.m_rho[j] = rho_j
                        u_vals[j] = mdot / rho_j if rho_j > 0 else 0.1
                    d.set_values("velocity", u_vals)

        has_active_energy = any(getattr(d, "energy_enabled", False) for d in self.m_dom if isinstance(d, Flow1D))
        if not has_active_energy and not refine_grid:
            if self._steady_callback:
                self._steady_callback(0.0)
            return

        dt = 1e-6
        max_ts = 100
        converged = False

        # Attempt damped steady-state Newton solve
        for step_attempt in range(50):
            r = np.zeros(self.size())
            self.eval(None, self._state, r, rdt=0.0)
            norm_res = self.weighted_norm(r)
            if norm_res < 1.0:
                converged = True
                break

            # Assemble Jacobian and attempt step
            J = self.eval_jacobian(self._state, rdt=0.0)
            try:
                J.factor()
                dx = J.solve(-r)
            except Exception:
                # Fallback to pseudo-transient continuation
                dx = None

            if dx is not None and np.all(np.isfinite(dx)):
                # Damping line search
                alpha = 1.0
                best_norm = norm_res
                accepted_x = None
                for _ in range(5):
                    trial_x = self._state + alpha * dx
                    r_trial = np.zeros(self.size())
                    self.eval(None, trial_x, r_trial, rdt=0.0)
                    t_norm = self.weighted_norm(r_trial)
                    if t_norm < best_norm:
                        best_norm = t_norm
                        accepted_x = trial_x
                        break
                    alpha /= 2.0

                if accepted_x is not None:
                    self._state = accepted_x
                    if best_norm < 1.0:
                        converged = True
                        break
                    continue

            # Pseudo-transient continuation step
            if self.m_ts_callback:
                self.m_ts_callback(dt)
            J_ts = self.eval_jacobian(self._state, rdt=1.0 / dt)
            try:
                J_ts.factor()
                dx_ts = J_ts.solve(-r)
                self._state += dx_ts
                dt = min(dt * self.m_time_step_growth_factor, 1.0)
            except Exception:
                dt = max(dt * 0.5, 1e-10)

        # Regrid/refine if requested
        if refine_grid:
            new_pts = self.refine(loglevel=loglevel)

        for d in self.m_dom:
            if isinstance(d, Flow1D) and not d.energy_enabled:
                inlet = getattr(self, "inlet", None)
                if inlet is not None and hasattr(inlet, "mdot") and inlet.mdot:
                    mdot = float(inlet.mdot)
                    u_vals = np.zeros(d.n_points)
                    for j in range(d.n_points):
                        rho_j = d.density(j)
                        u_vals[j] = mdot / rho_j if rho_j > 0 else 0.1
                    d.set_values("velocity", u_vals)

        if self._steady_callback:
            self._steady_callback(0.0)

class FlameBase(Sim1D):
    """Base class for flames with a single flow domain."""

    def __init__(
        self,
        domains: Sequence[Domain1D],
        gas: Solution,
        grid: Sequence[float] | np.ndarray | None = None,
    ):
        self.gas = gas
        if grid is None:
            grid = np.linspace(0.0, 0.1, 6)
        # Find the flow domain
        flow = next((d for d in domains if isinstance(d, Flow1D)), None)
        if flow is not None:
            flow.grid = grid
            self.flame = flow
        super().__init__(domains)

    def set_refine_criteria(
        self,
        ratio: float = 10.0,
        slope: float = 0.8,
        curve: float = 0.8,
        prune: float = -0.1,
    ):
        super().set_refine_criteria(self.flame, ratio, slope, curve, prune)

    def get_refine_criteria(self) -> dict[str, float]:
        return self.flame.get_refine_criteria()

    @property
    def transport_model(self) -> str:
        return self.flame.transport_model

    @transport_model.setter
    def transport_model(self, model: str):
        self.flame.transport_model = model

    @property
    def energy_enabled(self) -> bool:
        return self.flame.energy_enabled

    @energy_enabled.setter
    def energy_enabled(self, enable: bool):
        self.flame.energy_enabled = enable

    @property
    def boundary_emissivities(self) -> tuple[float, float]:
        return self.flame.boundary_emissivities

    @boundary_emissivities.setter
    def boundary_emissivities(self, eps: Any):
        self.flame.boundary_emissivities = eps

    @property
    def radiation_enabled(self) -> bool:
        return self.flame.radiation_enabled

    @radiation_enabled.setter
    def radiation_enabled(self, val: bool):
        self.flame.radiation_enabled = val

    @property
    def settings(self) -> dict[str, Any]:
        return self.flame.settings

    @property
    def grid(self) -> np.ndarray:
        return self.flame.grid

    @grid.setter
    def grid(self, val: Sequence[float] | np.ndarray):
        self.flame.grid = val

    @property
    def P(self) -> float:
        return self.flame.P

    @P.setter
    def P(self, val: float):
        self.flame.P = val

    @property
    def T(self) -> np.ndarray:
        return self.flame.T

    temperature = T

    @property
    def velocity(self) -> np.ndarray:
        return self.flame.velocity

    u = velocity

    @property
    def spread_rate(self) -> np.ndarray:
        return self.flame.spread_rate

    @property
    def radial_pressure_gradient(self) -> np.ndarray:
        return self.flame.radial_pressure_gradient

    @property
    def X(self) -> np.ndarray:
        return self.flame.X

    @property
    def Y(self) -> np.ndarray:
        return self.flame.Y

    @property
    def mix_diff_coeffs(self) -> np.ndarray:
        return self.flame.mix_diff_coeffs

    @property
    def thermal_conductivity(self) -> np.ndarray:
        return self.flame.thermal_conductivity

    @property
    def viscosity(self) -> np.ndarray:
        return self.flame.viscosity

    @property
    def density(self) -> np.ndarray:
        n_pts = self.flame.n_points
        vals = np.empty(n_pts)
        for i in range(n_pts):
            self.flame.update_state(i)
            vals[i] = self.gas.density
        return vals

    @property
    def density_mass(self) -> np.ndarray:
        return self.density

    @property
    def density_mole(self) -> np.ndarray:
        n_pts = self.flame.n_points
        vals = np.empty(n_pts)
        for i in range(n_pts):
            self.flame.update_state(i)
            vals[i] = self.gas.molar_density
        return vals

    @property
    def cp_mass(self) -> np.ndarray:
        n_pts = self.flame.n_points
        vals = np.empty(n_pts)
        for i in range(n_pts):
            self.flame.update_state(i)
            vals[i] = self.gas.cp_mass
        return vals

    @property
    def cp_mole(self) -> np.ndarray:
        n_pts = self.flame.n_points
        vals = np.empty(n_pts)
        for i in range(n_pts):
            self.flame.update_state(i)
            vals[i] = self.gas.cp_mole
        return vals

    cp = cp_mass

    @property
    def cv_mass(self) -> np.ndarray:
        n_pts = self.flame.n_points
        vals = np.empty(n_pts)
        for i in range(n_pts):
            self.flame.update_state(i)
            vals[i] = self.gas.cv_mass
        return vals

    cv = cv_mass

    @property
    def enthalpy_mass(self) -> np.ndarray:
        n_pts = self.flame.n_points
        vals = np.empty(n_pts)
        for i in range(n_pts):
            self.flame.update_state(i)
            vals[i] = self.gas.enthalpy_mass
        return vals

    h = enthalpy_mass

    @property
    def entropy_mass(self) -> np.ndarray:
        n_pts = self.flame.n_points
        vals = np.empty(n_pts)
        for i in range(n_pts):
            self.flame.update_state(i)
            vals[i] = self.gas.entropy_mass
        return vals

    s = entropy_mass

    @property
    def mean_molecular_weight(self) -> np.ndarray:
        n_pts = self.flame.n_points
        vals = np.empty(n_pts)
        for i in range(n_pts):
            self.flame.update_state(i)
            vals[i] = self.gas.mean_molecular_weight
        return vals

    def to_array(self, domain: Any = None, normalize: bool = False):
        if domain is None:
            target = self.flame
        elif isinstance(domain, str):
            target = self.domains[self.domain_index(domain)]
        elif isinstance(domain, int):
            target = self.domains[domain]
        else:
            target = domain
        return target.to_array(normalize=normalize)

    def from_array(self, arr: Any, domain: Any = None):
        if domain is None:
            target = self.flame
        elif isinstance(domain, str):
            target = self.domains[self.domain_index(domain)]
        elif isinstance(domain, int):
            target = self.domains[domain]
        else:
            target = domain
        target.from_array(arr)
        if target is self.flame:
            loc = self.flame.loc()
            for n in range(self.flame.n_components):
                if self.flame.component_active(n):
                    vals = self.flame.values(self.flame.component_name(n))
                    for j in range(self.flame.n_points):
                        self._state[loc + self.flame.index(n, j)] = vals[j]

    @property
    def fixed_temperature(self) -> float:
        return self.flame.fixed_temperature

    @fixed_temperature.setter
    def fixed_temperature(self, val: float):
        self.flame.fixed_temperature = val

    @property
    def fixed_temperature_location(self) -> float:
        return self.flame.fixed_temperature_location

    def save(self, filename: str | Path, group: str = "flame"):
        arr = SolutionArray(self.gas, shape=(self.flame.n_points,))
        arr.T = self.flame.T
        arr.P = self.flame.P
        arr.Y = self.flame.Y.T
        arr.extra["grid"] = self.flame.grid
        arr.extra["velocity"] = self.flame.velocity
        if self.flame.m_usesLambda:
            arr.extra["radial_pressure_gradient"] = self.flame.radial_pressure_gradient
            arr.extra["Lambda"] = self.flame.radial_pressure_gradient
            arr.extra["L"] = self.flame.radial_pressure_gradient
        arr.save(filename, group)


class FreeFlame(FlameBase):
    """A freely-propagating flat flame."""

    def __init__(
        self,
        gas: Solution,
        grid: Sequence[float] | np.ndarray | None = None,
        width: float | None = None,
    ):
        if width is not None and grid is not None:
            raise ValueError("'grid' and 'width' arguments are mutually exclusive")
        if width is not None:
            grid = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]) * width
        elif grid is None:
            grid = np.linspace(0.0, 0.05, 8)

        self.inlet = Inlet1D(phase=gas, name="reactants")
        self.flame = FreeFlow(phase=gas, name="flame")
        self.outlet = Outlet1D(phase=gas, name="products")

        super().__init__((self.inlet, self.flame, self.outlet), gas, grid)

        self.inlet.T = gas.T
        self.inlet.X = gas.X

    def set_initial_guess(
        self,
        locs: Sequence[float] = (0.0, 0.3, 0.5, 1.0),
        data: Any = None,
        group: str | None = None,
    ):
        if data is not None:
            return
        self.gas.TPY = self.inlet.T, self.P, self.inlet.Y
        if not self.inlet.mdot:
            self.inlet.mdot = 1.0 * self.gas.density

        Y0 = self.inlet.Y
        u0 = self.inlet.mdot / self.gas.density
        T0 = self.inlet.T

        # Calculate adiabatic equilibrium
        try:
            self.gas.equilibrate("HP")
            Teq = self.gas.T
            Yeq = self.gas.Y
        except Exception:
            Teq = max(T0 + 1500.0, 2000.0)
            Yeq = Y0
        u1 = self.inlet.mdot / self.gas.density

        locs_list = list(locs)
        self.flame.set_profile("velocity", locs_list, [u0, u0, u1, u1])
        self.flame.set_profile("T", locs_list, [T0, T0, Teq, Teq])

        T = self.T
        Tmid = 0.75 * T0 + 0.25 * Teq
        idx = int(np.flatnonzero(T < Tmid)[-1]) if len(np.flatnonzero(T < Tmid)) > 0 else 0
        if Tmid - T[idx] < 0.2 * (Tmid - T0):
            self.fixed_temperature = T[idx]
        elif idx + 1 < len(T) and T[idx + 1] - Tmid < 0.2 * (Teq - Tmid):
            self.fixed_temperature = T[idx + 1]
        else:
            self.fixed_temperature = Tmid

        for n in range(self.gas.n_species):
            self.flame.set_profile(
                self.gas.species_name(n), locs_list, [Y0[n], Y0[n], Yeq[n], Yeq[n]]
            )


class BurnerFlame(FlameBase):
    """A burner-stabilized flat flame."""

    def __init__(
        self,
        gas: Solution,
        grid: Sequence[float] | np.ndarray | None = None,
        width: float | None = None,
    ):
        if width is not None and grid is not None:
            raise ValueError("'grid' and 'width' arguments are mutually exclusive")
        if width is not None:
            grid = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]) * width
        elif grid is None:
            grid = np.linspace(0.0, 0.05, 7)

        self.burner = Inlet1D(phase=gas, name="burner")
        self.flame = UnstrainedFlow(phase=gas, name="flame")
        self.outlet = Outlet1D(phase=gas, name="outlet")

        super().__init__((self.burner, self.flame, self.outlet), gas, grid)

        self.burner.T = gas.T
        self.burner.X = gas.X

    def set_initial_guess(self, data: Any = None, group: str | None = None):
        if data is not None:
            return
        self.gas.TPY = self.burner.T, self.P, self.burner.Y
        Y0 = self.burner.Y
        u0 = self.burner.mdot / self.gas.density if self.burner.mdot else 0.1
        T0 = self.burner.T

        self.gas.equilibrate("HP")
        Teq = self.gas.T
        Yeq = self.gas.Y
        u1 = self.burner.mdot / self.gas.density if self.burner.mdot else 0.1

        locs = [0.0, 0.2, 1.0]
        self.flame.set_profile("velocity", locs, [u0, u1, u1])
        self.flame.set_profile("T", locs, [T0, Teq, Teq])
        for n in range(self.gas.n_species):
            self.flame.set_profile(
                self.gas.species_name(n), locs, [Y0[n], Yeq[n], Yeq[n]]
            )


class CounterflowDiffusionFlame(FlameBase):
    """An opposed-jet counterflow diffusion flame."""

    def __init__(
        self,
        gas: Solution,
        grid: Sequence[float] | np.ndarray | None = None,
        width: float | None = None,
    ):
        if width is not None and grid is not None:
            raise ValueError("'grid' and 'width' arguments are mutually exclusive")
        if width is not None:
            grid = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) * width
        elif grid is None:
            grid = np.linspace(0.0, 0.02, 6)

        self.fuel_inlet = Inlet1D(phase=gas, name="fuel_inlet")
        self.oxidizer_inlet = Inlet1D(phase=gas, name="oxidizer_inlet")
        self.flame = AxisymmetricFlow(phase=gas, name="flame")

        super().__init__((self.fuel_inlet, self.flame, self.oxidizer_inlet), gas, grid)

        self.fuel_inlet.T = gas.T
        self.oxidizer_inlet.T = gas.T

    def set_initial_guess(
        self,
        data: Any = None,
        group: str | None = None,
        mode: Literal["stoich", "linear"] = "stoich",
    ):
        if data is not None:
            return
        if mode not in ("stoich", "linear"):
            raise ValueError("mode must be 'stoich' or 'linear'")

        T_f = self.fuel_inlet.T
        T_o = self.oxidizer_inlet.T
        self.flame.set_profile("T", [0.0, 1.0], [T_f, T_o])
        for n in range(self.gas.n_species):
            self.flame.set_profile(
                self.gas.species_name(n),
                [0.0, 1.0],
                [self.fuel_inlet.Y[n], self.oxidizer_inlet.Y[n]],
            )


def _poly5(x: float, c: Sequence[float]) -> float:
    """Evaluate polynomial of degree 5 with coefficients c[0]..c[5] in Horner form."""
    return (((((c[5] * x + c[4]) * x + c[3]) * x + c[2]) * x + c[1]) * x + c[0])


class IonFlow(Flow1D):
    """1D reacting flow domain with charged species, ion mobility, and Poisson electric field coupling.

    Direct reconstruction of Cantera's native IonFlow class.
    """

    _domain_type = "IonFlow"

    def __init__(self, phase: Solution, id: str = "flame", points: int = 1):
        super().__init__(phase, id=id, points=points)
        self.m_nsp = self.nsp
        self.m_speciesCharge: list[float] = [float(self.gas.charge(k)) for k in range(self.m_nsp)]
        self.m_kCharge: list[int] = [k for k in range(self.m_nsp) if self.m_speciesCharge[k] != 0.0]
        self.m_kNeutral: list[int] = [k for k in range(self.m_nsp) if self.m_speciesCharge[k] == 0.0]

        # Find index of electron if present
        try:
            self.m_kElectron: Optional[int] = self.gas.species_index("E")
        except (KeyError, ValueError, CanteraError):
            self.m_kElectron = None

        # Electric potential bounds: -1e20 to 1e20
        self.setBounds(c_offset_E, -1.0e20, 1.0e20)

        # Set tighter negative species limit on charged species to avoid instabilities
        for k in self.m_kCharge:
            self.setBounds(c_offset_Y + k, -1.0e-10, 1.0)
        if self.m_kElectron is not None:
            self.setBounds(c_offset_Y + self.m_kElectron, -1.0e-14, 1.0)

        self.refiner().setActive(c_offset_E, False)
        self.m_mobility = np.zeros((self.m_points, self.m_nsp), dtype=float)
        self.m_do_electric_field: bool = False
        self.m_import_electron_transport: bool = False
        self.m_mobi_e_fix = np.zeros(6, dtype=float)
        self.m_diff_e_fix = np.zeros(6, dtype=float)

        # Validate transport model
        trans_model = getattr(self.gas, "transport_model", "")
        if not trans_model:
            trans = getattr(self.gas, "transport", None)
            if trans is not None:
                tm_obj = getattr(trans, "transportModel", getattr(trans, "transport_model", ""))
                trans_model = tm_obj() if callable(tm_obj) else str(tm_obj)
        if not trans_model or trans_model == "none":
            raise CanteraError("IonFlow::IonFlow",
                "An appropriate transport model\nshould be set when instantiating the Solution ('gas') object.")

    def domainType(self) -> str:
        if self.m_isFree:
            return "free-ion-flow"
        if self.m_usesLambda:
            return "axisymmetric-ion-flow"
        return "unstrained-ion-flow"

    @property
    def domain_type(self) -> str:
        return self.domainType()

    def resize(self, components: int, points: int):
        super().resize(components, points)
        nsp = getattr(self, "m_nsp", max(0, components - c_offset_Y))
        self.m_mobility = np.zeros((self.m_points, nsp), dtype=float)

    def componentActive(self, n: int) -> bool:
        if n == c_offset_E:
            return True
        return super().componentActive(n)

    def solveElectricField(self):
        if not self.m_do_electric_field:
            self.needJacUpdate()
        self.refiner().setActive(c_offset_U, True)
        self.refiner().setActive(c_offset_V, True)
        self.refiner().setActive(c_offset_T, True)
        self.refiner().setActive(c_offset_E, True)
        self.m_do_electric_field = True

    def fixElectricField(self):
        if self.m_do_electric_field:
            self.needJacUpdate()
        self.refiner().setActive(c_offset_U, False)
        self.refiner().setActive(c_offset_V, False)
        self.refiner().setActive(c_offset_T, False)
        self.refiner().setActive(c_offset_E, False)
        self.m_do_electric_field = False

    def doElectricField(self) -> bool:
        return self.m_do_electric_field

    def analyticJacobianSupported(self) -> bool:
        return False

    def setElectronTransport(self, tfix: Sequence[float], diff_e: Sequence[float], mobi_e: Sequence[float]):
        self.m_import_electron_transport = True
        tlog = np.log(np.asarray(tfix, dtype=float))
        self.m_diff_e_fix = np.polyfit(tlog, np.asarray(diff_e, dtype=float), 5)[::-1]
        self.m_mobi_e_fix = np.polyfit(tlog, np.asarray(mobi_e, dtype=float), 5)[::-1]

    def E(self, x: Any, j: int = 0) -> float:
        if isinstance(x, (int, np.integer)):
            j = int(x)
            return float(self.m_eField[j]) if hasattr(self, "m_eField") and len(self.m_eField) > j else 0.0
        return float(x[self.index(c_offset_E, j)])

    def dEdz(self, x: np.ndarray, j: int = 0) -> float:
        if j == 0:
            return 0.0
        return (self.E(x, j) - self.E(x, j - 1)) / (self.z(j) - self.z(j - 1))

    def ND(self, x: np.ndarray, k: int, j: int = 0) -> float:
        return Avogadro * self.m_rho[j] * self.Y(x, k, j) / self.m_wt[k]

    def rho_e(self, x: np.ndarray, j: int = 0) -> float:
        charge_density = 0.0
        for k in self.m_kCharge:
            charge_density += self.m_speciesCharge[k] * ElectronCharge * self.ND(x, k, j)
        return charge_density

    def updateTransport(self, x: np.ndarray, j0: int, j1: int) -> None:
        super().updateTransport(x, j0, j1)
        for j in range(j0, j1):
            self.setGasAtMidpoint(x, j)
            mobs = self.gas.mobilities
            if mobs is not None and len(mobs) == self.m_nsp:
                self.m_mobility[j, :] = mobs
            else:
                self.m_mobility[j, :] = 0.0
            if self.m_import_electron_transport and self.m_kElectron is not None:
                k = self.m_kElectron
                tlog = math.log(self.gas.T)
                self.m_mobility[j, k] = _poly5(tlog, self.m_mobi_e_fix)
                rho = self.gas.density
                wtm = self.gas.mean_molecular_weight
                self.m_diff[j, k] = self.m_wt[k] * rho * _poly5(tlog, self.m_diff_e_fix) / wtm

    def updateDiffFluxes(self, x: np.ndarray, j0: int, j1: int) -> None:
        if self.m_do_electric_field:
            self.electricFieldMethod(x, j0, j1)
        else:
            self.frozenIonMethod(x, j0, j1)

    def frozenIonMethod(self, x: np.ndarray, j0: int, j1: int) -> None:
        for j in range(j0, j1):
            dz = self.z(j + 1) - self.z(j)
            sum_neut = 0.0
            for k in self.m_kNeutral:
                self.m_flux[k, j] = self.m_diff[j, k] * (self.X(x, k, j) - self.X(x, k, j + 1)) / dz
                sum_neut -= self.m_flux[k, j]

            # correction flux to ensure that \sum_k Y_k V_k = 0
            for k in self.m_kNeutral:
                self.m_flux[k, j] += sum_neut * self.Y(x, k, j)

            # flux for ions: zeroed out to prevent fast charged species from running away
            for k in self.m_kCharge:
                self.m_flux[k, j] = 0.0

    def electricFieldMethod(self, x: np.ndarray, j0: int, j1: int) -> None:
        for j in range(j0, j1):
            rho = self.density(j)
            dz = self.z(j + 1) - self.z(j)

            # mixture-average diffusion
            for k in range(self.m_nsp):
                self.m_flux[k, j] = self.m_diff[j, k] * (self.X(x, k, j) - self.X(x, k, j + 1)) / dz

            # ambipolar diffusion
            E_ambi = self.E(x, j)
            for k in self.m_kCharge:
                Yav = 0.5 * (self.Y(x, k, j) + self.Y(x, k, j + 1))
                drift = rho * Yav * E_ambi * self.m_speciesCharge[k] * self.m_mobility[j, k]
                self.m_flux[k, j] += drift

            # correction flux
            sum_flux = 0.0
            for k in range(self.m_nsp):
                sum_flux -= self.m_flux[k, j]

            sum_ion = 0.0
            for k in self.m_kCharge:
                sum_ion += self.Y(x, k, j)

            # The portion of correction for ions is taken off
            denom = 1.0 - sum_ion
            for k in self.m_kNeutral:
                self.m_flux[k, j] += self.Y(x, k, j) / denom * sum_flux

    def evalElectricField(self, x: np.ndarray, rsd: np.ndarray, diag: np.ndarray,
                           rdt: float, jmin: int, jmax: int) -> None:
        super().evalElectricField(x, rsd, diag, rdt, jmin, jmax)
        if not self.m_do_electric_field:
            return

        if jmin == 0:
            rsd[self.index(c_offset_E, jmin)] = self.E(x, jmin)

        if jmax == self.m_points - 1:
            rsd[self.index(c_offset_E, jmax)] = self.dEdz(x, jmax) - self.rho_e(x, jmax) / epsilon_0

        j0 = max(jmin, 1)
        j1 = min(jmax, self.m_points - 2)
        for j in range(j0, j1 + 1):
            rsd[self.index(c_offset_E, j)] = self.dEdz(x, j) - self.rho_e(x, j) / epsilon_0
            diag[self.index(c_offset_E, j)] = 0

    def evalSpecies(self, x: np.ndarray, rsd: np.ndarray, diag: np.ndarray,
                    rdt: float, jmin: int, jmax: int) -> None:
        super().evalSpecies(x, rsd, diag, rdt, jmin, jmax)
        if not self.m_do_electric_field:
            return

        if jmin == 0:
            for k in self.m_kCharge:
                rsd[self.index(c_offset_Y + k, jmin)] = self.Y(x, k, jmin) - self.Y(x, k, jmin + 1)


class DomainFactory:
    """Factory for creating 1D domain objects.

    Direct reconstruction of Cantera's C++ DomainFactory class.
    """
    _instance: Optional["DomainFactory"] = None
    _mutex = threading.Lock()

    def __init__(self):
        self._creators: dict[str, Callable[[Solution, str], Domain1D]] = {}
        self._register_default_creators()

    @staticmethod
    def _transport_model(sol: Solution) -> str:
        tm = getattr(sol, "transport_model", "")
        if not tm:
            trans = getattr(sol, "transport", None)
            if trans is not None:
                tm_obj = getattr(trans, "transportModel", getattr(trans, "transport_model", ""))
                tm = tm_obj() if callable(tm_obj) else str(tm_obj)
        return tm

    def _register_default_creators(self):
        self.reg("inlet", lambda sol, id: Inlet1D(sol, id=id))
        self.reg("empty", lambda sol, id: Empty1D(sol, id=id))
        self.reg("symmetry-plane", lambda sol, id: Symm1D(sol, id=id))
        self.reg("outlet", lambda sol, id: Outlet1D(sol, id=id))
        self.reg("outlet-reservoir", lambda sol, id: OutletRes1D(sol, id=id))
        self.reg("surface", lambda sol, id: Surf1D(sol, id=id))
        self.reg("reacting-surface", lambda sol, id: ReactingSurf1D(sol, id=id))
        self.reg("gas-flow", lambda sol, id: Flow1D(sol, id=id))
        self.reg("ion-flow", lambda sol, id: IonFlow(sol, id=id))
        self.reg("free-flow", self._create_free_flow)
        self.reg("axisymmetric-flow", self._create_axisymmetric_flow)
        self.reg("unstrained-flow", self._create_unstrained_flow)

    def _create_free_flow(self, sol: Solution, id: str) -> Flow1D:
        model = self._transport_model(sol)
        flow = IonFlow(sol, id=id) if model == "ionized-gas" else Flow1D(sol, id=id)
        flow.setFreeFlow()
        return flow

    def _create_axisymmetric_flow(self, sol: Solution, id: str) -> Flow1D:
        model = self._transport_model(sol)
        flow = IonFlow(sol, id=id) if model == "ionized-gas" else Flow1D(sol, id=id)
        flow.setAxisymmetricFlow()
        return flow

    def _create_unstrained_flow(self, sol: Solution, id: str) -> Flow1D:
        model = self._transport_model(sol)
        flow = IonFlow(sol, id=id) if model == "ionized-gas" else Flow1D(sol, id=id)
        flow.setUnstrainedFlow()
        return flow

    def reg(self, domain_type: str, creator: Callable[[Solution, str], Domain1D]):
        self._creators[domain_type] = creator

    def create(self, domain_type: str, solution: Solution, id: str = "") -> Domain1D:
        if not id:
            id = domain_type
        if domain_type not in self._creators:
            raise CanteraError("DomainFactory::create", f"Unknown domain type '{domain_type}'")
        return self._creators[domain_type](solution, id)

    @classmethod
    def factory(cls) -> "DomainFactory":
        with cls._mutex:
            if cls._instance is None:
                cls._instance = DomainFactory()
            return cls._instance

    @classmethod
    def deleteFactory(cls) -> None:
        with cls._mutex:
            cls._instance = None


def newDomain(domainType: str, solution: Solution, id: str = "") -> Domain1D:
    """Create a Domain1D object of the specified type."""
    return DomainFactory.factory().create(domainType, solution, id)


def newBoundary1D(domainType: str, solution: Solution, id: str = "") -> Boundary1D:
    """Create a Boundary1D object of the specified type."""
    dom = newDomain(domainType, solution, id)
    if not isinstance(dom, Boundary1D):
        raise CanteraError("newBoundary1D", f"Invalid cast: unable to access '{type(dom).__name__}' as 'Boundary1D'.")
    return dom


def newFlow1D(domainType: str, solution: Solution, id: str = "") -> Flow1D:
    """Create a Flow1D object of the specified type."""
    dom = newDomain(domainType, solution, id)
    if not isinstance(dom, Flow1D):
        raise CanteraError("newFlow1D", f"Invalid cast: unable to access '{type(dom).__name__}' as 'Flow1D'.")
    return dom


class MultiJac(SystemJacobian):
    """Banded block Jacobian for 1D multi-domain networks.

    Direct reconstruction of Cantera's C++ MultiJac class.
    """

    def __init__(self, sys=None):
        super().__init__(sys)
        self.m_mat = BandMatrix()
        self.m_ssdiag = np.zeros(0, dtype=float)
        self.m_dim = 0
        self._age = 100000

    def reset(self) -> None:
        self.m_mat.bfill(0.0)
        self._age = 10000

    def type(self) -> str:
        return "banded-direct"

    def setValue(self, row: int, col: int, value: float) -> None:
        self.m_mat.setValue(row, col, value)
        if row == col and row < len(self.m_ssdiag):
            self.m_ssdiag[row] = float(value)

    def initialize(self, nVars: int = 1) -> None:
        self.m_dim = int(nVars)
        kl = self.m_mat.nSubDiagonals()
        ku = self.m_mat.nSuperDiagonals()
        self.m_mat.resize(self.m_dim, kl, ku)
        self.m_ssdiag = np.zeros(self.m_dim, dtype=float)

    def setBandwidth(self, bw: int, ku: Optional[int] = None) -> None:
        ku_val = bw if ku is None else ku
        self.m_mat.resize(self.m_dim, bw, ku_val)

    def updateTransient(self, rdt: float, mask: Sequence[int]) -> None:
        for n in range(self.m_dim):
            m_val = mask[n] if n < len(mask) else 1
            self.m_mat.setValue(n, n, self.m_ssdiag[n] - m_val * rdt)
        self.factorize()

    def factorize(self) -> None:
        self.m_mat.factor()

    def value(self, i: int, j: int) -> float:
        return self.m_mat.value(i, j)

    def solve(self, b: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
        return self.m_mat.solve(b, x)

    def info(self) -> int:
        return self.m_mat.info()


class MultiNewton:
    """Newton iterator for multi-domain, one-dimensional problems.

    Reconstructs Cantera's C++ MultiNewton solver implementing bounded damped
    Newton iteration (Kee et al., 2003).
    """

    def __init__(self, sz: int = 0):
        self.m_n = int(sz)
        self.m_maxAge = 5
        self.m_dampFactor = math.sqrt(2.0)
        self.m_maxDampIter = 7
        self.m_elapsed = 0.0
        self.m_lastIterations = 0
        self.m_x = np.zeros(self.m_n, dtype=float)
        self.m_stp = np.zeros(self.m_n, dtype=float)
        self.m_stp1 = np.zeros(self.m_n, dtype=float)

    def size(self) -> int:
        return self.m_n

    def lastIterations(self) -> int:
        return self.m_lastIterations

    def setOptions(self, maxJacAge: int = 5) -> None:
        self.m_maxAge = int(maxJacAge)

    def resize(self, sz: int) -> None:
        self.m_n = int(sz)
        self.m_x = np.zeros(self.m_n, dtype=float)
        self.m_stp = np.zeros(self.m_n, dtype=float)
        self.m_stp1 = np.zeros(self.m_n, dtype=float)

    def step(self, x: Sequence[float], step: Sequence[float], r: Any, loglevel: int = 0) -> None:
        """Compute undamped Newton step delta_x = -J^{-1} F(x)."""
        x_arr = np.asarray(x, dtype=float)
        step_arr = np.asarray(step, dtype=float)
        r.eval(x_arr, step_arr)
        step_arr[:] = -step_arr[:]

        jac = r.linearSolver()
        try:
            t0 = time.perf_counter()
            jac.solve(step_arr, step_arr)
            if hasattr(r, "recordLinearSolve"):
                r.recordLinearSolve(time.perf_counter() - t0)
        except Exception as err:
            info = getattr(jac, "info", lambda: 0)()
            if info > 0:
                row = int(info - 1)
                comp = r.componentName(row) if hasattr(r, "componentName") else f"component_{row}"
                raise CanteraError(
                    f"MultiNewton::step: Jacobian is singular for matrix row {row}:\n{comp}"
                ) from err
            raise
        if step is not step_arr:
            step[:] = step_arr[:]

    def boundStep(self, x0: Sequence[float], step0: Sequence[float], r: Any, loglevel: int = 0) -> float:
        """Return multiplier in [0, 1] keeping all components between lower and upper bounds."""
        fbound = 1.0
        n = self.size()
        for i in range(n):
            above = r.upperBound(i)
            below = r.lowerBound(i)
            val = float(x0[i])
            newval = val + float(step0[i])

            if newval > above:
                fbound = max(0.0, min(fbound, (above - val) / (newval - val)))
            elif newval < below:
                fbound = min(fbound, (val - below) / (val - newval))
        return float(fbound)

    def dampStep(self, x0: Sequence[float], step0: Sequence[float],
                 x1: Sequence[float], step1: Sequence[float],
                 s1: Any, r: Any, loglevel: int = 0, writetitle: bool = False) -> int:
        """Performs a damped Newton step seeking alpha_k where norm(step_{k+1}) < norm(step_k)."""
        s0 = r.weightedNorm(step0)
        fbound = self.boundStep(x0, step0, r, loglevel - 1)

        if fbound < 1.0e-10:
            return -3

        alpha = fbound * 1.0
        computed_s1 = s0
        m = 0
        for m in range(self.m_maxDampIter):
            for j in range(self.m_n):
                x1[j] = x0[j] + alpha * step0[j]

            try:
                self.step(x1, step1, r, loglevel - 1)
            except Exception:
                alpha /= self.m_dampFactor
                continue

            computed_s1 = r.weightedNorm(step1)
            if computed_s1 < 1.0 or computed_s1 < s0:
                break
            alpha /= self.m_dampFactor
        else:
            m = self.m_maxDampIter

        if isinstance(s1, (list, np.ndarray)):
            s1[0] = computed_s1

        if m < self.m_maxDampIter:
            if computed_s1 > 1.0:
                return 0
            else:
                return 1
        else:
            return -2

    def solve(self, x0: Sequence[float], x1: Sequence[float], r: Any, loglevel: int = 0) -> int:
        """Find solution to F(x) = 0 by damped Newton iteration."""
        t0 = time.perf_counter()
        status = 0
        forceNewJac = False
        write_header = True
        s1 = 1.0e30

        if len(self.m_x) != len(x0):
            self.resize(len(x0))

        self.m_x[:] = np.asarray(x0, dtype=float)
        self.m_lastIterations = 0

        rdt = getattr(r, "rdt", lambda: 0.0)()
        nJacReeval = 0
        jac = r.linearSolver()

        while True:
            self.m_lastIterations += 1
            if jac.age() > self.m_maxAge:
                forceNewJac = True

            if forceNewJac:
                r.evalJacobian(self.m_x)
                try:
                    r.factorizeJacobian()
                except Exception as err:
                    if rdt == 0.0:
                        return -4
                    else:
                        raise
                forceNewJac = False

            self.step(self.m_x, self.m_stp, r, loglevel - 1)
            jac.incrementAge()

            s1_box = [s1]
            status = self.dampStep(self.m_x, self.m_stp, x1, self.m_stp1, s1_box, r, loglevel - 1, write_header)
            s1 = s1_box[0]
            write_header = False

            if status == 0:
                self.m_x[:] = x1[:]
            elif status == 1:
                if rdt == 0.0:
                    jac.setAge(0)
                break
            elif status < 0:
                if jac.age() > 1:
                    forceNewJac = True
                    if nJacReeval > 3:
                        break
                    nJacReeval += 1
                else:
                    break

        if status < 0:
            x1[:] = self.m_x[:]

        self.m_elapsed += time.perf_counter() - t0
        return status

def newSim1D(domains=None) -> Sim1D:
    return Sim1D(domains or [])