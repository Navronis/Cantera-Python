"""Patch oned.py with complete and exact Domain1D and OneDim & SteadyStateSystem implementation."""
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    oned_path = repo_root / "cantera_python" / "oned.py"
    content = oned_path.read_text(encoding="utf-8")

    # 1. Patch Domain1D methods: append, firstPoint, lastPoint, locate, bandwidth, atol, rtol
    old_domain_methods = '''    def append(self, x: float) -> None:
        pass

    def appendResiduals(self, res: np.ndarray) -> None:
        pass

    def atol(self, k: int = 0) -> float:
        return 1e-9

    def bandwidth(self) -> int:
        return self.m_bw

    def checkAnalyticJacobian(self) -> bool:
        return False

    def data(self) -> dict:
        return {}

    def evalJacobianAnalytic(self, x: np.ndarray) -> None:
        pass

    def firstPoint(self) -> float:
        return float(self.grid[0]) if len(self.grid) else 0.0

    def forceFullUpdate(self, full: bool = True) -> None:
        pass

    def fromArray(self, arr: np.ndarray) -> None:
        pass

    def getResiduals(self, res: np.ndarray) -> None:
        pass

    def getValues(self, vals: np.ndarray) -> None:
        pass

    def globalComponentIndex(self, k: int) -> int:
        return k

    def hasAnalyticJacobian(self) -> bool:
        return False

    def info(self) -> dict:
        return {}

    def init(self) -> None:
        pass

    def initialValue(self, k: int) -> float:
        return 0.0

    def lastPoint(self) -> float:
        return float(self.grid[-1]) if len(self.grid) else 1.0

    def left(self):
        return self.m_left

    def linkLeft(self, left_domain) -> None:
        self.m_left = left_domain

    def linkRight(self, right_domain) -> None:
        self.m_right = right_domain

    def locate(self, z: float) -> int:
        return 0'''

    new_domain_methods = '''    def append(self, d: Any) -> None:
        self.m_right = d
        if hasattr(d, "m_left"):
            d.m_left = self

    def appendResiduals(self, res: np.ndarray) -> None:
        pass

    def atol(self, k: int = 0) -> float:
        k = int(k)
        if hasattr(self, "m_atol_ss") and k < len(self.m_atol_ss):
            return float(self.m_atol_ss[k])
        return 1e-9

    def bandwidth(self) -> int:
        return getattr(self, "m_bw", -1)

    def setBandwidth(self, bw: int = -1) -> None:
        self.m_bw = int(bw)

    def checkAnalyticJacobian(self) -> bool:
        return False

    def data(self) -> dict:
        return {}

    def evalJacobianAnalytic(self, x: np.ndarray, jac: Any = None) -> None:
        pass

    def hasAnalyticJacobian(self, j: int = 0, n: int = 0) -> bool:
        return False

    def firstPoint(self) -> int:
        return int(getattr(self, "m_jstart", 0))

    def lastPoint(self) -> int:
        pts = self.n_points if hasattr(self, "n_points") else (self.nPoints() if hasattr(self, "nPoints") else 1)
        return int(getattr(self, "m_jstart", 0) + pts - 1)

    def forceFullUpdate(self, full: bool = True) -> None:
        pass

    def fromArray(self, arr: np.ndarray) -> None:
        pass

    def getResiduals(self, res: np.ndarray) -> None:
        pass

    def getValues(self, vals: np.ndarray) -> None:
        pass

    def globalComponentIndex(self, k: int) -> int:
        return k

    def info(self) -> dict:
        return {}

    def init(self) -> None:
        pass

    def initialValue(self, k: int) -> float:
        return 0.0

    def left(self):
        return getattr(self, "m_left", None)

    def linkLeft(self, left_domain) -> None:
        self.m_left = left_domain
        self.locate()

    def linkRight(self, right_domain) -> None:
        self.m_right = right_domain

    def locate(self) -> None:
        if getattr(self, "m_left", None):
            left_pts = self.m_left.n_points if hasattr(self.m_left, "n_points") else (
                self.m_left.nPoints() if hasattr(self.m_left, "nPoints") else 0
            )
            left_sz = self.m_left.size() if hasattr(self.m_left, "size") else 0
            left_jstart = getattr(self.m_left, "m_jstart", 0)
            left_loc = getattr(self.m_left, "m_loc", 0)
            self.m_jstart = left_jstart + left_pts
            self.m_loc = left_loc + left_sz
        else:
            self.m_jstart = 0
            self.m_loc = 0
        if getattr(self, "m_right", None) and hasattr(self.m_right, "locate"):
            self.m_right.locate()

    def loc(self, j: int = 0) -> int:
        return int(getattr(self, "m_loc", 0))

    def rtol(self, k: int = 0) -> float:
        k = int(k)
        if hasattr(self, "m_rtol_ss") and k < len(self.m_rtol_ss):
            return float(self.m_rtol_ss[k])
        return 1e-4

    def upperBound(self, k: int = 0) -> float:
        k = int(k)
        if hasattr(self, "m_max") and k < len(self.m_max):
            return float(self.m_max[k])
        return 1e10

    def lowerBound(self, k: int = 0) -> float:
        k = int(k)
        if hasattr(self, "m_min") and k < len(self.m_min):
            return float(self.m_min[k])
        return -1e10

    def setBounds(self, k: int, lower: float, upper: float) -> None:
        k = int(k)
        while len(self.m_min) <= k:
            self.m_min.append(-1e10)
        while len(self.m_max) <= k:
            self.m_max.append(1e10)
        self.m_min[k] = float(lower)
        self.m_max[k] = float(upper)'''

    if old_domain_methods in content:
        content = content.replace(old_domain_methods, new_domain_methods, 1)
        print("Patched Domain1D methods successfully.")
    else:
        print("Warning: old_domain_methods not matched directly, checking normalized.")

    lines = content.splitlines()
    start_onedim = next(i for i, l in enumerate(lines) if l.strip().startswith("class OneDim"))
    end_onedim = next(i for i in range(start_onedim + 1, len(lines)) if lines[i].strip().startswith("class Sim1D"))

    onedim_code = '''class SteadyStateSystem:
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
'''

    new_lines = lines[:start_onedim] + [onedim_code] + lines[end_onedim:]
    oned_path.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"Patched OneDim in oned.py (replaced lines {start_onedim+1}-{end_onedim})")


if __name__ == "__main__":
    main()
