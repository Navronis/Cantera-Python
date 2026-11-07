"""Patch oned.py with complete, exact, and optimized Refiner implementation."""
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    oned_path = repo_root / "cantera_python" / "oned.py"
    lines = oned_path.read_text(encoding="utf-8").splitlines()

    start_idx = next(i for i, l in enumerate(lines) if "class Refiner:" in l)
    end_idx = next(i for i in range(start_idx + 1, len(lines)) if lines[i].startswith("class Domain1D"))

    refiner_code = '''class Refiner:
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
'''

    new_lines = lines[:start_idx] + [refiner_code] + lines[end_idx:]
    oned_path.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"Patched Refiner in oned.py (replaced lines {start_idx+1}-{end_idx})")


if __name__ == "__main__":
    main()
