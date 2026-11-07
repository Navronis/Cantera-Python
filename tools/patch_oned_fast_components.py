import os

oned_path = os.path.join(os.path.dirname(__file__), "..", "cantera_python", "oned.py")
with open(oned_path, "r", encoding="utf-8") as f:
    text = f.read()

target_classes = """class Callable1DComponent:
    \"\"\"Component view that is callable as fn(x, j) -> float and acts as a 1D array.\"\"\"
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

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.as_array(), dtype=dtype)

    def __repr__(self) -> str:
        return repr(self.as_array())


class Callable2DComponent:
    \"\"\"2D component view of shape (nsp, points) that is callable as Y(x, k, j) or Y(x, j).\"\"\"
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

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.as_array(), dtype=dtype)

    def __repr__(self) -> str:
        return repr(self.as_array())


class CallableXComponent:
    \"\"\"2D component view of mole fractions that is callable as X(x, k, j).\"\"\"
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

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.as_array(), dtype=dtype)

    def __repr__(self) -> str:
        return repr(self.as_array())


class CallableRhoU:
    \"\"\"1D view of axial mass flux that is callable as rho_u(x, j).\"\"\"
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

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.as_array(), dtype=dtype)"""

repl_classes = """class Callable1DComponent:
    \"\"\"Component view that is callable as fn(x, j) -> float and acts as a 1D array.\"\"\"
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
    \"\"\"2D component view of shape (nsp, points) that is callable as Y(x, k, j) or Y(x, j).\"\"\"
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
    \"\"\"2D component view of mole fractions that is callable as X(x, k, j).\"\"\"
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
    \"\"\"1D view of axial mass flux that is callable as rho_u(x, j).\"\"\"
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
        return np.asarray(self.as_array(), dtype=dtype)"""

assert target_classes in text, "target_classes not found"
text = text.replace(target_classes, repl_classes, 1)

with open(oned_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("Successfully added operator support to Callable components.")
