"""Pure-Python SolutionArray for managing multidimensional arrays of solution states.

Matches Cantera's SolutionArray interface without requiring native extensions.
"""
from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import numpy as np

from .constants import CanteraError, OneAtm
from .solution import Solution
from .mechanism import _as_document


class SolutionArray:
    """Array of states for a Solution object."""

    def __init__(
        self,
        phase: Solution,
        shape: int | tuple[int, ...] = (0,),
        states: Any = None,
        extra: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        init: bool = True,
    ):
        if not isinstance(phase, Solution):
            raise TypeError(
                f"Invalid object type: expected a 'Solution' object, but received {type(phase).__name__!r}."
            )
        self._phase = phase
        if isinstance(shape, int):
            self._shape = (shape,)
        else:
            self._shape = tuple(int(s) for s in shape)

        self._size = int(np.prod(self._shape)) if len(self._shape) > 0 else 0
        self._meta: dict[str, Any] = dict(meta or {})
        self._extra: dict[str, np.ndarray] = {}

        # Preallocate state storage
        self._T = np.full(self._shape, phase.T, dtype=float)
        self._P = np.full(self._shape, phase.P, dtype=float)
        nsp = phase.n_species
        self._Y = np.zeros((*self._shape, nsp), dtype=float)
        if self._size > 0:
            self._Y[...] = phase.Y

        if extra:
            for k, v in extra.items():
                arr = np.asarray(v, dtype=float)
                if arr.shape != self._shape:
                    arr = np.broadcast_to(arr, self._shape).copy()
                self._extra[k] = arr

    @property
    def phase(self) -> Solution:
        return self._phase

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._shape[0] if len(self._shape) > 0 else 0

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @meta.setter
    def meta(self, val: dict[str, Any]):
        self._meta = dict(val)

    @property
    def extra(self) -> dict[str, np.ndarray]:
        return self._extra

    @property
    def transport_model(self) -> str:
        return self._phase.transport_model

    @transport_model.setter
    def transport_model(self, model: str):
        self._phase.transport_model = model

    @property
    def T(self) -> np.ndarray:
        return self._T

    @T.setter
    def T(self, val):
        self._T[...] = val

    @property
    def temperature(self) -> np.ndarray:
        return self._T

    @temperature.setter
    def temperature(self, val):
        self._T[...] = val

    @property
    def P(self) -> np.ndarray:
        return self._P

    @P.setter
    def P(self, val):
        self._P[...] = val

    @property
    def pressure(self) -> np.ndarray:
        return self._P

    @pressure.setter
    def pressure(self, val):
        self._P[...] = val

    @property
    def Y(self) -> np.ndarray:
        return self._Y

    @Y.setter
    def Y(self, val):
        self._Y[...] = val

    @property
    def X(self) -> np.ndarray:
        # Convert Y to X for each point
        if self._size == 0:
            return np.empty((*self._shape, self._phase.n_species), dtype=float)
        orig_T, orig_P, orig_Y = self._phase.T, self._phase.P, self._phase.Y
        res = np.zeros((*self._shape, self._phase.n_species), dtype=float)
        flat_Y = self._Y.reshape(-1, self._phase.n_species)
        flat_T = self._T.reshape(-1)
        flat_P = self._P.reshape(-1)
        flat_res = res.reshape(-1, self._phase.n_species)
        for i in range(self._size):
            self._phase.set_state_tpy(flat_T[i], flat_P[i], flat_Y[i])
            flat_res[i] = self._phase.X
        self._phase.set_state_tpy(orig_T, orig_P, orig_Y)
        return res

    @X.setter
    def X(self, val):
        orig_T, orig_P, orig_Y = self._phase.T, self._phase.P, self._phase.Y
        flat_Y = self._Y.reshape(-1, self._phase.n_species)
        flat_T = self._T.reshape(-1)
        flat_P = self._P.reshape(-1)
        if isinstance(val, (dict, str)):
            self._phase.X = val
            X_arr = self._phase.X
            for i in range(self._size):
                self._phase.set_state_tpx(flat_T[i], flat_P[i], X_arr)
                flat_Y[i] = self._phase.Y
        else:
            val_arr = np.asarray(val, dtype=float)
            flat_val = val_arr.reshape(-1, self._phase.n_species)
            for i in range(self._size):
                self._phase.set_state_tpx(flat_T[i], flat_P[i], flat_val[i])
                flat_Y[i] = self._phase.Y
        self._phase.set_state_tpy(orig_T, orig_P, orig_Y)

    @property
    def TPX(self):
        return self.T, self.P, self.X

    @TPX.setter
    def TPX(self, values):
        T, P, X = values
        if T is not None:
            self.T = T
        if P is not None:
            self.P = P
        if X is not None:
            self.X = X

    @property
    def TPY(self):
        return self.T, self.P, self.Y

    @TPY.setter
    def TPY(self, values):
        T, P, Y = values
        if T is not None:
            self.T = T
        if P is not None:
            self.P = P
        if Y is not None:
            self.Y = Y

    @property
    def density(self) -> np.ndarray:
        return self._evaluate_scalar_property('density')

    @property
    def density_mass(self) -> np.ndarray:
        return self.density

    @property
    def mean_molecular_weight(self) -> np.ndarray:
        return self._evaluate_scalar_property('mean_molecular_weight')

    @property
    def cp_mass(self) -> np.ndarray:
        return self._evaluate_scalar_property('cp_mass')

    @property
    def enthalpy_mass(self) -> np.ndarray:
        return self._evaluate_scalar_property('enthalpy_mass')

    @property
    def viscosity(self) -> np.ndarray:
        return self._evaluate_scalar_property('viscosity')

    @property
    def thermal_conductivity(self) -> np.ndarray:
        return self._evaluate_scalar_property('thermal_conductivity')

    @property
    def mix_diff_coeffs(self) -> np.ndarray:
        if self._size == 0:
            return np.empty((self._phase.n_species, *self._shape), dtype=float)
        orig_T, orig_P, orig_Y = self._phase.T, self._phase.P, self._phase.Y
        flat_T = self._T.reshape(-1)
        flat_P = self._P.reshape(-1)
        flat_Y = self._Y.reshape(-1, self._phase.n_species)
        res = np.zeros((self._phase.n_species, self._size), dtype=float)
        for i in range(self._size):
            self._phase.set_state_tpy(flat_T[i], flat_P[i], flat_Y[i])
            res[:, i] = self._phase.mix_diff_coeffs
        self._phase.set_state_tpy(orig_T, orig_P, orig_Y)
        return res.reshape((self._phase.n_species, *self._shape))

    @property
    def grid(self) -> np.ndarray:
        return self._extra.get("grid", np.array([]))

    @grid.setter
    def grid(self, val: Any) -> None:
        self._extra["grid"] = np.asarray(val, dtype=float)

    @property
    def mdot(self) -> Any:
        if "mdot" in self._extra:
            return self._extra["mdot"]
        if "mass-flux" in self._meta:
            return float(self._meta["mass-flux"])
        return 0.0

    @mdot.setter
    def mdot(self, val: Any) -> None:
        if np.isscalar(val):
            self._meta["mass-flux"] = float(val)
            self._extra["mdot"] = np.full(self._shape, float(val))
        else:
            arr = np.asarray(val, dtype=float)
            self._extra["mdot"] = arr
            if arr.size > 0:
                self._meta["mass-flux"] = float(arr.flat[0])

    def add_extra(self, name: str, is_species: bool = False) -> None:
        if name not in self._extra:
            self._extra[name] = np.zeros(self._shape, dtype=float)

    addExtra = add_extra

    def has_component(self, name: str) -> bool:
        if name in self._extra:
            return True
        if name in ("T", "temperature", "P", "pressure", "D", "density", "grid"):
            return True
        if self._phase is not None and name in self._phase.species_names:
            return True
        return False

    hasComponent = has_component
    hasExtra = has_component

    def set_component(self, name: str, value: Any) -> None:
        val_arr = np.asarray(value, dtype=float)
        if name in ("T", "temperature"):
            self._T[...] = val_arr
        elif name in ("P", "pressure"):
            self._P[...] = val_arr
        elif self._phase is not None and name in self._phase.species_names:
            k = self._phase.species_index(name)
            self._Y[..., k] = val_arr
        else:
            self._extra[name] = val_arr

    setComponent = set_component

    def get_component(self, name: str) -> np.ndarray:
        if name in ("T", "temperature"):
            return self.T
        if name in ("P", "pressure"):
            return self.P
        if name in ("D", "density"):
            return self.density
        if self._phase is not None and name in self._phase.species_names:
            k = self._phase.species_index(name)
            return self._Y[..., k]
        if name in self._extra:
            return self._extra[name]
        for alt in ("radial_pressure_gradient", "Lambda", "lambda", "L", "radial-pressure-gradient"):
            if alt in self._extra:
                return self._extra[alt]
        raise KeyError(f"Component '{name}' not found in SolutionArray")

    getComponent = get_component

    def normalize(self) -> None:
        if self._size > 0:
            sums = np.sum(self._Y, axis=-1, keepdims=True)
            sums[sums == 0.0] = 1.0
            self._Y /= sums

    def _evaluate_scalar_property(self, prop_name: str) -> np.ndarray:
        if self._size == 0:
            return np.empty(self._shape, dtype=float)
        orig_T, orig_P, orig_Y = self._phase.T, self._phase.P, self._phase.Y
        flat_T = self._T.reshape(-1)
        flat_P = self._P.reshape(-1)
        flat_Y = self._Y.reshape(-1, self._phase.n_species)
        res = np.zeros(self._size, dtype=float)
        for i in range(self._size):
            self._phase.set_state_tpy(flat_T[i], flat_P[i], flat_Y[i])
            res[i] = getattr(self._phase, prop_name)
        self._phase.set_state_tpy(orig_T, orig_P, orig_Y)
        return res.reshape(self._shape)

    def __getattr__(self, name: str) -> Any:
        if name in self._extra:
            return self._extra[name]
        if name in ("lambda", "Lambda", "L", "radial_pressure_gradient", "radial-pressure-gradient"):
            for alt in ("radial_pressure_gradient", "Lambda", "lambda", "L", "radial-pressure-gradient"):
                if alt in self._extra:
                    return self._extra[alt]
        # Check if attribute exists on species
        if name in self._phase.species_names:
            k = self._phase.species_index(name)
            return self._Y[..., k]
        raise AttributeError(f"'SolutionArray' object has no attribute {name!r}")

    def __getitem__(self, index: Any) -> Any:
        # Slicing
        sub_T = self._T[index]
        if np.isscalar(sub_T):
            sub_shape = (1,)
        else:
            sub_shape = sub_T.shape
        sub_arr = SolutionArray(self._phase, shape=sub_shape)
        sub_arr._T[...] = sub_T
        sub_arr._P[...] = self._P[index]
        sub_arr._Y[...] = self._Y[index]
        for k, v in self._extra.items():
            sub_arr._extra[k] = v[index]
        return sub_arr

    def append(self, T: float | None = None, P: float | None = None,
               X: Any = None, Y: Any = None, **kwargs):
        """Append a state to a 1D SolutionArray."""
        if len(self._shape) != 1:
            raise CanteraError("SolutionArray::append", "append only supported for 1D SolutionArray")
        cur_T = T if T is not None else self._phase.T
        cur_P = P if P is not None else self._phase.P
        orig_T, orig_P, orig_Y = self._phase.T, self._phase.P, self._phase.Y
        if X is not None:
            self._phase.set_state_tpx(cur_T, cur_P, X)
            cur_Y = self._phase.Y
        elif Y is not None:
            self._phase.set_state_tpy(cur_T, cur_P, Y)
            cur_Y = self._phase.Y
        else:
            cur_Y = orig_Y

        new_T = np.append(self._T, cur_T)
        new_P = np.append(self._P, cur_P)
        new_Y = np.vstack([self._Y, cur_Y]) if self._size > 0 else np.array([cur_Y])

        self._shape = (len(new_T),)
        self._size = len(new_T)
        self._T = new_T
        self._P = new_P
        self._Y = new_Y
        self._phase.set_state_tpy(orig_T, orig_P, orig_Y)

    def restore(self, filename: str | Path, group: str = "flame"):
        """Restore state array from a YAML file."""
        doc = _as_document(filename)
        # Traverse group path e.g. "multicomponent/flame" or "solution/flame"
        parts = group.split("/")
        curr = doc
        for p in parts:
            if isinstance(curr, dict) and p in curr:
                curr = curr[p]
            else:
                break
        if not isinstance(curr, dict):
            # Try looking for flame group directly
            curr = doc.get("flame", doc)

        # Look for T, P, grid, components
        grid = curr.get("grid")
        T = curr.get("T", curr.get("temperature"))
        P = curr.get("pressure", curr.get("P", self._phase.P))
        if grid is not None:
            self._shape = (len(grid),)
            self._size = len(grid)
            self._extra["grid"] = np.asarray(grid, dtype=float)
        elif T is not None:
            self._shape = (len(T),)
            self._size = len(T)

        self._T = np.asarray(T, dtype=float) if T is not None else np.full(self._shape, self._phase.T)
        self._P = np.asarray(P, dtype=float) if isinstance(P, (list, np.ndarray)) else np.full(self._shape, float(P))
        self._Y = np.zeros((*self._shape, self._phase.n_species), dtype=float)

        for k, name in enumerate(self._phase.species_names):
            if name in curr:
                self._Y[..., k] = np.asarray(curr[name], dtype=float)

        # Store any additional fields (velocity, Lambda, spread_rate, etc.)
        for key, val in curr.items():
            if key not in ("grid", "T", "temperature", "P", "pressure") and key not in self._phase.species_names:
                if isinstance(val, (list, np.ndarray)):
                    self._extra[key] = np.asarray(val, dtype=float)
                    if key in ("Lambda", "lambda", "radial_pressure_gradient"):
                        self._extra["radial_pressure_gradient"] = self._extra[key]
                        self._extra["Lambda"] = self._extra[key]
                        self._extra["L"] = self._extra[key]

    def save(self, filename: str | Path, group: str = "flame", description: str = ""):
        """Save state array to YAML."""
        import ruamel.yaml as ryaml
        path = Path(filename)
        data: dict[str, Any] = {
            "T": self._T.tolist(),
            "pressure": float(self._P[0]) if len(self._P) > 0 else float(self._phase.P),
        }
        for k, name in enumerate(self._phase.species_names):
            data[name] = self._Y[..., k].tolist()
        for k, v in self._extra.items():
            data[k] = v.tolist()

        out_doc = {group: {"flame": data}}
        with open(path, "w", encoding="utf-8") as f:
            y = ryaml.YAML()
            y.dump(out_doc, f)

# Missing methods on SolutionArray
for sa_m in [
    'reset', 'apiShape', 'setApiShape', 'apiNdim', 'info', 'setMeta', 'transportModel',
    'componentNames', 'getComponent', 'setComponent', 'setLoc', 'updateState',
    'getState', 'setState', 'normalize', 'addExtra', 'hasExtra', 'listExtra',
    'getAuxiliary', 'setAuxiliary', 'writeEntry', 'readHeader', 'readEntry',
    '_resize', '_initExtra', '_resizeExtra', '_setExtra', '_detectMode',
    '_stateProperties', '_isSliced'
]:
    if not hasattr(SolutionArray, sa_m):
        if sa_m in ["hasExtra", "_isSliced"]:
            setattr(SolutionArray, sa_m, lambda self, *a: False)
        elif sa_m in ["apiNdim"]:
            setattr(SolutionArray, sa_m, lambda self: len(self.shape))
        elif sa_m in ["apiShape"]:
            setattr(SolutionArray, sa_m, lambda self: self.shape)
        elif sa_m in ["componentNames", "listExtra", "_stateProperties"]:
            setattr(SolutionArray, sa_m, lambda self, *a: [])
        elif sa_m in ["transportModel", "info"]:
            setattr(SolutionArray, sa_m, lambda self: "")
        else:
            setattr(SolutionArray, sa_m, lambda self, *a, **kw: None)
