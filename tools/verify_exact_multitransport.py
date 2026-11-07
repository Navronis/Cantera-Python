#!/usr/bin/env python3
"""Verification of exact pure-Python MultiTransport against native Cantera."""
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cantera_python as ct
from cantera_python.constants import (
    Boltzmann, GasConstant, Avogadro, Pi, epsilon_0, lightSpeed, OneAtm
)
from cantera_python.transport import (
    _astar_table, _bstar_table, _cstar_table, _omega22_table,
    GasTransportData, Frot
)

# Reference delta and tstar grids from Monchick & Mason
delta_grid = np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5])
tstar_grid = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
    18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0, 500.0
])
tstar22 = tstar_grid[1:38]

def polyfit_cpp(deg, x, y, w=None):
    """Exact reproduction of Cantera's polyfit (weighted least squares)."""
    A = np.vander(x, deg + 1, increasing=True)
    if w is not None and len(w) > 0 and w[0] > 0:
        ww = np.sqrt(w)
        A = A * ww[:, np.newaxis]
        y_w = y * ww
    else:
        y_w = y
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y_w, rcond=None)
    return coeffs

class ExactMMCollisionInt:
    def __init__(self):
        self.m_logTemp = np.log(tstar_grid[1:38])
        self.m_o22poly = []
        self.m_apoly = []
        self.m_bpoly = []
        self.m_cpoly = []
        
        # DeltaDegree = 6
        for i in range(37):
            self.m_o22poly.append(polyfit_cpp(6, delta_grid, _omega22_table[i, :]))
            self.m_apoly.append(polyfit_cpp(6, delta_grid, _astar_table[i + 1, :]))
            self.m_bpoly.append(polyfit_cpp(6, delta_grid, _bstar_table[i + 1, :]))
            self.m_cpoly.append(polyfit_cpp(6, delta_grid, _cstar_table[i + 1, :]))

    def init(self, tsmin, tsmax):
        self.m_nmin = -1
        self.m_nmax = -1
        for n in range(37):
            if tsmin > tstar_grid[n + 1]:
                self.m_nmin = n
            if tsmax > tstar_grid[n + 1]:
                self.m_nmax = n + 1
        if self.m_nmin < 0 or self.m_nmin >= 36 or self.m_nmax < 0 or self.m_nmax > 36:
            self.m_nmin = 0
            self.m_nmax = 36

    def quadInterp(self, x0, x, y):
        dx21 = x[1] - x[0]
        dx32 = x[2] - x[1]
        dx31 = dx21 + dx32
        dy32 = y[2] - y[1]
        dy21 = y[1] - y[0]
        a = (dx21 * dy32 - dy21 * dx32) / (dx21 * dx31 * dx32)
        return a * (x0 - x[0]) * (x0 - x[1]) + (dy21 / dx21) * (x0 - x[1]) + y[1]

    def omega22(self, ts, deltastar):
        i = 0
        while i < 37 and ts >= tstar22[i]:
            i += 1
        i1 = max(i - 1, 0)
        i2 = i1 + 3
        if i2 > 36:
            i2 = 36
            i1 = i2 - 3
        values = np.zeros(3)
        for idx in range(i1, i2):
            if deltastar == 0.0:
                values[idx - i1] = _omega22_table[idx, 0]
            else:
                values[idx - i1] = np.polynomial.polynomial.polyval(deltastar, self.m_o22poly[idx])
        return self.quadInterp(math.log(ts), self.m_logTemp[i1:i2], values)

    def astar(self, ts, deltastar):
        i = 0
        while i < 37 and ts >= tstar22[i]:
            i += 1
        i1 = max(i - 1, 0)
        i2 = i1 + 3
        if i2 > 36:
            i2 = 36
            i1 = i2 - 3
        values = np.zeros(3)
        for idx in range(i1, i2):
            if deltastar == 0.0:
                values[idx - i1] = _astar_table[idx + 1, 0]
            else:
                values[idx - i1] = np.polynomial.polynomial.polyval(deltastar, self.m_apoly[idx])
        return self.quadInterp(math.log(ts), self.m_logTemp[i1:i2], values)

    def bstar(self, ts, deltastar):
        i = 0
        while i < 37 and ts >= tstar22[i]:
            i += 1
        i1 = max(i - 1, 0)
        i2 = i1 + 3
        if i2 > 36:
            i2 = 36
            i1 = i2 - 3
        values = np.zeros(3)
        for idx in range(i1, i2):
            if deltastar == 0.0:
                values[idx - i1] = _bstar_table[idx + 1, 0]
            else:
                values[idx - i1] = np.polynomial.polynomial.polyval(deltastar, self.m_bpoly[idx])
        return self.quadInterp(math.log(ts), self.m_logTemp[i1:i2], values)

    def cstar(self, ts, deltastar):
        i = 0
        while i < 37 and ts >= tstar22[i]:
            i += 1
        i1 = max(i - 1, 0)
        i2 = i1 + 3
        if i2 > 36:
            i2 = 36
            i1 = i2 - 3
        values = np.zeros(3)
        for idx in range(i1, i2):
            if deltastar == 0.0:
                values[idx - i1] = _cstar_table[idx + 1, 0]
            else:
                values[idx - i1] = np.polynomial.polynomial.polyval(deltastar, self.m_cpoly[idx])
        return self.quadInterp(math.log(ts), self.m_logTemp[i1:i2], values)

    def omega11(self, ts, deltastar):
        return self.omega22(ts, deltastar) / self.astar(ts, deltastar)

    def fit(self, degree, deltastar):
        n = self.m_nmax - self.m_nmin + 1
        logT = self.m_logTemp[self.m_nmin : self.m_nmax + 1]
        def fit_one(table, polys):
            vals = np.zeros(n)
            for i in range(n):
                idx = i + self.m_nmin
                if deltastar == 0.0:
                    vals[i] = table[idx + 1, 0]
                else:
                    vals[i] = np.polynomial.polynomial.polyval(deltastar, polys[idx])
            return polyfit_cpp(degree, logT, vals)
        return (fit_one(_astar_table, self.m_apoly),
                fit_one(_bstar_table, self.m_bpoly),
                fit_one(_cstar_table, self.m_cpoly))

    def fit_omega22(self, degree, deltastar):
        n = self.m_nmax - self.m_nmin + 1
        logT = self.m_logTemp[self.m_nmin : self.m_nmax + 1]
        vals = np.zeros(n)
        for i in range(n):
            idx = i + self.m_nmin
            if deltastar == 0.0:
                vals[i] = _omega22_table[idx, 0]
            else:
                vals[i] = np.polynomial.polynomial.polyval(deltastar, self.m_o22poly[idx])
        return polyfit_cpp(degree, logT, vals)


# Let's test loading GRI30 and setting up GasTransport
sol = ct.Solution("gri30.yaml")
thermo = sol.thermo
nsp = thermo.n_species

print(f"Setting up exact MultiTransport for {nsp} species...")
integrals = ExactMMCollisionInt()
tstar_min = 1e8
tstar_max = 0.0

# Extract transport data
tdata_dict = sol.transport._transport_data
crot = np.zeros(nsp)
zrot = np.zeros(nsp)
sigma = np.zeros(nsp)
eps = np.zeros(nsp)
dipole = np.zeros((nsp, nsp))
polar = np.zeros(nsp, dtype=bool)
alpha = np.zeros(nsp)

for k, s in enumerate(thermo.species):
    tdata = tdata_dict.get(s.name, GasTransportData())
    if tdata.geometry == "atom":
        crot[k] = 0.0
    elif tdata.geometry == "linear":
        crot[k] = 1.0
    else:
        crot[k] = 1.5
    sigma[k] = tdata.diameter * 1e-10  # m
    eps[k] = tdata.well_depth * Boltzmann  # J
    dip = tdata.dipole * (1e-21 / lightSpeed)  # C*m
    dipole[k, k] = dip
    polar[k] = (tdata.dipole > 0)
    alpha[k] = tdata.polarizability * 1e-30  # m^3
    zrot[k] = tdata.rotational_relaxation

mw = np.array(thermo.molecular_weights)
reducedMass = np.zeros((nsp, nsp))
diam = np.zeros((nsp, nsp))
epsilon = np.zeros((nsp, nsp))
dipole_mat = np.zeros((nsp, nsp))
delta = np.zeros((nsp, nsp))

for i in range(nsp):
    for j in range(i, nsp):
        reducedMass[i, j] = mw[i] * mw[j] / (Avogadro * (mw[i] + mw[j]))
        diam[i, j] = 0.5 * (sigma[i] + sigma[j])
        epsilon[i, j] = math.sqrt(eps[i] * eps[j])
        dipole_mat[i, j] = math.sqrt(dipole[i, i] * dipole[j, j])
        d = diam[i, j]
        delta[i, j] = 0.5 * (dipole_mat[i, j] ** 2) / (4.0 * Pi * epsilon_0 * epsilon[i, j] * (d ** 3))

        # Polar corrections
        if polar[i] != polar[j]:
            kp = i if polar[i] else j
            knp = j if polar[i] else i
            d3np = sigma[knp] ** 3
            d3p = sigma[kp] ** 3
            alpha_star = alpha[knp] / d3np
            mu_p_star = dipole[kp, kp] / math.sqrt(4.0 * Pi * epsilon_0 * d3p * eps[kp])
            xi = 1.0 + 0.25 * alpha_star * (mu_p_star ** 2) * math.sqrt(eps[kp] / eps[knp])
            f_sigma = xi ** (-1.0 / 6.0)
            f_eps = xi * xi
            diam[i, j] *= f_sigma
            epsilon[i, j] *= f_eps

        reducedMass[j, i] = reducedMass[i, j]
        diam[j, i] = diam[i, j]
        epsilon[j, i] = epsilon[i, j]
        dipole_mat[j, i] = dipole_mat[i, j]
        delta[j, i] = delta[i, j]

        tstar_min = min(tstar_min, Boltzmann * thermo.min_temp / epsilon[i, j])
        tstar_max = max(tstar_max, Boltzmann * thermo.max_temp / epsilon[i, j])

integrals.init(tstar_min, tstar_max)

# Fit collision integrals (degree 8)
omega22_poly = []
astar_poly = []
bstar_poly = []
cstar_poly = []
poly = np.zeros((nsp, nsp), dtype=int)
fitlist = []

for i in range(nsp):
    for j in range(i, nsp):
        dstar = delta[i, j]
        found = False
        for idx, f in enumerate(fitlist):
            if abs(f - dstar) < 1e-12:
                poly[i, j] = poly[j, i] = idx
                found = True
                break
        if not found:
            idx = len(fitlist)
            fitlist.append(dstar)
            ca, cb, cc = integrals.fit(8, dstar)
            co22 = integrals.fit_omega22(8, dstar)
            omega22_poly.append(co22)
            astar_poly.append(ca)
            bstar_poly.append(cb)
            cstar_poly.append(cc)
            poly[i, j] = poly[j, i] = idx

# Fit properties (np=50, degree=4)
np_pts = 50
dt = (thermo.max_temp - thermo.min_temp) / (np_pts - 1)
tlog = np.array([math.log(thermo.min_temp + dt * n) for n in range(np_pts)])

visccoeffs = []
condcoeffs = []
diffcoeffs = []

spvisc = np.zeros(np_pts)
spcond = np.zeros(np_pts)
w = np.zeros(np_pts)
w2 = np.zeros(np_pts)

T_save = thermo.temperature
for k in range(nsp):
    tstar_298 = Boltzmann * 298.0 / eps[k]
    fz_298 = 1.0 + (Pi ** 1.5) / math.sqrt(tstar_298) * (0.5 + 1.0 / tstar_298) + (0.25 * Pi * Pi + 2.0) / tstar_298

    for n in range(np_pts):
        t = thermo.min_temp + dt * n
        thermo.TP = t, thermo.pressure
        cp_R = thermo.partial_molar_cp[k] / GasConstant
        tstar = Boltzmann * t / eps[k]
        sqrt_T = math.sqrt(t)
        om22 = integrals.omega22(tstar, delta[k, k])
        om11 = integrals.omega11(tstar, delta[k, k])

        diffcoeff = (3.0 / 16.0) * math.sqrt(2.0 * Pi / reducedMass[k, k]) * ((Boltzmann * t) ** 1.5) / (Pi * (sigma[k] ** 2) * om11)
        v = (5.0 / 16.0) * math.sqrt(Pi * mw[k] * Boltzmann * t / Avogadro) / (om22 * Pi * (sigma[k] ** 2))

        f_int = mw[k] / (GasConstant * t) * diffcoeff / v
        cv_rot = crot[k]
        A_factor = 2.5 - f_int
        fz_tstar = 1.0 + (Pi ** 1.5) / math.sqrt(tstar) * (0.5 + 1.0 / tstar) + (0.25 * Pi * Pi + 2.0) / tstar
        B_factor = zrot[k] * fz_298 / fz_tstar + (2.0 / Pi) * (5.0 / 3.0 * cv_rot + f_int)
        c1 = (2.0 / Pi) * A_factor / B_factor
        cv_int = cp_R - 2.5 - cv_rot
        f_rot = f_int * (1.0 + c1)
        f_trans = 2.5 * (1.0 - c1 * cv_rot / 1.5)
        cond = (v / mw[k]) * GasConstant * (f_trans * 1.5 + f_rot * cv_rot + f_int * cv_int)

        spvisc[n] = math.sqrt(v / sqrt_T)
        spcond[n] = cond / sqrt_T
        w[n] = 1.0 / (spvisc[n] ** 2)
        w2[n] = 1.0 / (spcond[n] ** 2)

    c_visc = polyfit_cpp(4, tlog, spvisc, w)
    c_cond = polyfit_cpp(4, tlog, spcond, w2)
    visccoeffs.append(c_visc)
    condcoeffs.append(c_cond)

thermo.TP = T_save, thermo.pressure

# Fit binary diffusivities (np=50, degree=4)
diff_vals = np.zeros(np_pts)
w_diff = np.zeros(np_pts)
for k in range(nsp):
    for j in range(k, nsp):
        for n in range(np_pts):
            t = thermo.min_temp + dt * n
            eps_val = epsilon[j, k]
            tstar = Boltzmann * t / eps_val
            sig_val = diam[j, k]
            om11 = integrals.omega11(tstar, delta[j, k])
            diffcoeff = (3.0 / 16.0) * math.sqrt(2.0 * Pi / reducedMass[k, j]) * ((Boltzmann * t) ** 1.5) / (Pi * (sig_val ** 2) * om11)
            diff_vals[n] = diffcoeff / (t ** 1.5)
            w_diff[n] = 1.0 / (diff_vals[n] ** 2)
        c_diff = polyfit_cpp(4, tlog, diff_vals, w_diff)
        diffcoeffs.append(c_diff)

print("Polynomial fitting completed successfully.")

# Now test at T = 1200 K, 1 atm, CH4:0.1, O2:0.2, N2:0.7 (exact WSL test case)
sol.TPX = 1200.0, OneAtm, "CH4:0.1, O2:0.2, N2:0.7"
T = 1200.0
P = OneAtm
logt = math.log(T)
sqrt_t = math.sqrt(T)
t14 = math.sqrt(sqrt_t)
poly_t = np.array([1.0, logt, logt**2, logt**3, logt**4])

# Evaluate pure species viscosities
visc = np.zeros(nsp)
sqvisc = np.zeros(nsp)
for k in range(nsp):
    sqvisc[k] = t14 * np.dot(poly_t, visccoeffs[k])
    visc[k] = sqvisc[k] ** 2

# Evaluate binary diffusion coefficients at unit pressure (1 Pa)
m_bdiff = np.zeros((nsp, nsp))
ic = 0
for i in range(nsp):
    for j in range(i, nsp):
        m_bdiff[i, j] = T * sqrt_t * np.dot(poly_t, diffcoeffs[ic])
        m_bdiff[j, i] = m_bdiff[i, j]
        ic += 1

# Wilke mixture viscosity
wratjk = np.zeros((nsp, nsp))
wratkj1 = np.zeros((nsp, nsp))
for j in range(nsp):
    for k in range(j, nsp):
        wratjk[j, k] = math.sqrt(mw[j] / mw[k])
        wratjk[k, j] = math.sqrt(wratjk[j, k])
        wratkj1[j, k] = math.sqrt(1.0 + mw[k] / mw[j])

phi = np.zeros((nsp, nsp))
for j in range(nsp):
    for k in range(j, nsp):
        factor1 = 1.0 + (sqvisc[k] / sqvisc[j]) * wratjk[k, j]
        phi[k, j] = factor1 * factor1 / (math.sqrt(8.0) * wratkj1[j, k])
        phi[j, k] = phi[k, j] / ((visc[k] / visc[j]) * (mw[j] / mw[k]))

X = np.array(thermo.X)
X_safe = np.array([max(1e-14, x) for x in X])
spwork = np.dot(phi, X_safe)
viscmix = sum(X_safe[k] * visc[k] / spwork[k] for k in range(nsp))

print(f"Calculated Viscosity: {viscmix:.16e}")
print(f"Native Cantera Visc: 4.6879608034318800e-05")
print(f"Relative Viscosity Diff: {abs(viscmix - 4.68796080343188e-05) / 4.68796080343188e-05:.6e}")
