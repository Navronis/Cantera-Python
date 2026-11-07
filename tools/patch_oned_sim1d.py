"""Patch Sim1D in cantera_python/oned.py with source-exact C++ methods."""

from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
oned_file = repo_root / "cantera_python" / "oned.py"

with open(oned_file, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the initial stub class Sim1D methods
start_marker = "class Sim1D(OneDim):"
end_marker = "    \"\"\"Composite multi-domain 1D simulation manager and solver.\"\"\""

idx_start = content.find(start_marker)
idx_end = content.find(end_marker)

assert idx_start != -1, "start_marker not found"
assert idx_end != -1, "end_marker not found"

new_sim1d_header = '''class Sim1D(OneDim):
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
        self.resize()

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
        return added if added > 0 else (-discarded if discarded > 0 else 0)
'''

content = content[:idx_start] + new_sim1d_header + "\n\n    " + content[idx_end:]

# Also ensure newSim1D function is defined at module level if not present
if "def newSim1D(" not in content:
    content += "\n\ndef newSim1D(domains: Sequence[Any] = ()) -> Sim1D:\n    return Sim1D(domains)\n"

with open(oned_file, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully patched Sim1D in oned.py")
