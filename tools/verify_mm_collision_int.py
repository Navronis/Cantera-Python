#!/usr/bin/env python3
"""Verify MMCollisionInt in Python vs C++."""
import math
import numpy as np

# Monchick and Mason tables
delta_grid = np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5])
tstar_grid = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
    18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0, 500.0
])
tstar22_grid = tstar_grid[1:38] # 37 points

# Load tables from transport.py or definition
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cantera_python.transport import (
    _astar_table, _bstar_table, _cstar_table, _omega22_table
)

def polyfit_cpp(deg, x, y, w=None):
    """Exact reproduction of Cantera's polyfit (Householder QR / least squares)."""
    A = np.vander(x, deg + 1, increasing=True)
    if w is not None and len(w) > 0 and w[0] > 0:
        ww = np.sqrt(w)
        A = A * ww[:, np.newaxis]
        y_w = y * ww
    else:
        y_w = y
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y_w, rcond=None)
    return coeffs

class PyMMCollisionInt:
    def __init__(self):
        self.m_logTemp = np.log(tstar_grid[1:38])
        self.m_o22poly = []
        self.m_apoly = []
        self.m_bpoly = []
        self.m_cpoly = []
        
        # DeltaDegree = 6
        for i in range(37):
            c_o22 = polyfit_cpp(6, delta_grid, _omega22_table[i, :])
            c_a = polyfit_cpp(6, delta_grid, _astar_table[i + 1, :])
            c_b = polyfit_cpp(6, delta_grid, _bstar_table[i + 1, :])
            c_c = polyfit_cpp(6, delta_grid, _cstar_table[i + 1, :])
            self.m_o22poly.append(c_o22)
            self.m_apoly.append(c_a)
            self.m_bpoly.append(c_b)
            self.m_cpoly.append(c_c)

    def init(self, tsmin, tsmax):
        self.m_nmin = 0
        self.m_nmax = 36
        for n in range(37):
            if tsmin > tstar_grid[n + 1]:
                self.m_nmin = n
            if tsmax > tstar_grid[n + 1]:
                self.m_nmax = n + 1
        if self.m_nmin < 0 or self.m_nmin >= 36 or self.m_nmax < 0 or self.m_nmax > 36:
            self.m_nmin = 0
            self.m_nmax = 36

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

        a_poly = fit_one(_astar_table, self.m_apoly)
        b_poly = fit_one(_bstar_table, self.m_bpoly)
        c_poly = fit_one(_cstar_table, self.m_cpoly)
        return a_poly, b_poly, c_poly

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

if __name__ == "__main__":
    mm = PyMMCollisionInt()
    mm.init(0.1, 100.0)
    a, b, c = mm.fit(8, 0.0)
    print("A* poly for delta*=0 (len=%d):" % len(a), a[:3])
    om = mm.fit_omega22(8, 0.0)
    print("omega22 poly for delta*=0 (len=%d):" % len(om), om[:3])
