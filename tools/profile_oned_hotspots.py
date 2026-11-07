"""Profile Flow1D.eval hotspots to identify root causes of slowdown."""

import cProfile
import pstats
import io
import time
from pathlib import Path
import numpy as np

import cantera_python as ct
from cantera_python.oned import Flow1D, Inlet1D, Outlet1D, Sim1D, c_offset_U, c_offset_T, c_offset_Y


def main():
    gas = ct.Solution("h2o2.yaml")
    gas.TPX = 1200.0, ct.OneAtm, "H2:2, O2:1, AR:4"
    n_pts = 25
    grid = np.linspace(0.0, 0.05, n_pts)

    inlet = Inlet1D(gas, name="inlet")
    flow = Flow1D(gas, id="flame", points=n_pts)
    outlet = Outlet1D(gas, name="outlet")

    flow.setupGrid(grid)
    flow.setPressure(ct.OneAtm)
    flow.setFreeFlow()
    flow.solveEnergyEqn(True)

    sim = Sim1D((inlet, flow, outlet))
    sim.init()

    x = np.zeros(sim.size(), dtype=float)
    flow_loc = flow.loc()
    nc = flow.nComponents()

    for j in range(n_pts):
        base = flow_loc + j * nc
        x[base + c_offset_U] = 1.0 + 2.0 * (j / n_pts)
        x[base + c_offset_T] = 300.0 + 1500.0 * (j / n_pts)
        for k in range(gas.n_species):
            x[base + c_offset_Y + k] = gas.Y[k]

    rsd = np.zeros(sim.size(), dtype=float)
    diag = np.zeros(sim.size(), dtype=int)

    # Warmup
    for _ in range(1):
        flow.eval(None, x, rsd, diag, 0.0)

    # Profile
    pr = cProfile.Profile()
    n_iters = 50
    t0 = time.perf_counter()
    pr.enable()
    for _ in range(n_iters):
        flow.eval(None, x, rsd, diag, 0.0)
    pr.disable()
    elapsed = time.perf_counter() - t0

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(35)

    print(f"Total time for {n_iters} evals: {elapsed:.3f} s ({elapsed / n_iters * 1000:.2f} ms/eval)")
    print(s.getvalue())


if __name__ == "__main__":
    main()
