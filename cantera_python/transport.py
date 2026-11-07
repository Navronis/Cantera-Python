"""Mixture-averaged transport properties for ideal gas mixtures.

Sources: MixTransport.cpp, GasTransport.cpp, MMCollisionInt.cpp.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .constants import (Avogadro, Boltzmann, ElectronCharge, GasConstant, Pi,
                        ArraySizeError, CanteraError, Tiny, epsilon_0,
                        lightSpeed)


def _check_array_size(procedure: str, name: str, values, required: int) -> None:
    """Apply the size checks used by Cantera's transport span interfaces."""
    available = len(values)
    if available < required:
        raise ArraySizeError(
            procedure,
            f"{name} array is too small ({available}); must contain at least {required} values",
        )


def _lu_factor_partial_pivot(matrix, procedure: str):
    """Factor a dense matrix with the DGETRF partial-pivot LU algorithm.

    The row updates and pivot selection are performed in Python. This avoids
    delegating Cantera's transport linear solve to NumPy/LAPACK.
    """
    rows = np.asarray(matrix, dtype=float).tolist()
    n = len(rows)
    if any(len(row) != n for row in rows):
        raise CanteraError(procedure, "matrix must be square")

    pivots = []
    for k in range(n):
        pivot = max(range(k, n), key=lambda row: abs(rows[row][k]))
        pivot_value = rows[pivot][k]
        if pivot_value == 0.0:
            raise CanteraError(
                procedure,
                f"matrix is singular at pivot {k}; LU factorization cannot continue",
            )
        pivots.append(pivot)
        if pivot != k:
            rows[k], rows[pivot] = rows[pivot], rows[k]

        pivot_row = rows[k]
        for i in range(k + 1, n):
            row = rows[i]
            multiplier = row[k] / pivot_row[k]
            row[k] = multiplier
            for j in range(k + 1, n):
                row[j] -= multiplier * pivot_row[j]
    return rows, pivots


def _lu_solve_partial_pivot(lu, pivots, rhs) -> np.ndarray:
    """Apply the recorded row pivots, then solve L*y=P*b and U*x=y."""
    n = len(lu)
    b = [float(value) for value in rhs]
    if len(b) != n:
        raise CanteraError("DenseMatrix::solve", "right-hand side size does not match matrix")
    for k, pivot in enumerate(pivots):
        if pivot != k:
            b[k], b[pivot] = b[pivot], b[k]

    for k in range(n):
        for i in range(k + 1, n):
            b[i] -= lu[i][k] * b[k]

    for k in range(n - 1, -1, -1):
        b[k] /= lu[k][k]
        for i in range(k):
            b[i] -= lu[i][k] * b[k]
    return np.asarray(b, dtype=float)


def _solve_dense_partial_pivot(matrix, rhs, procedure: str) -> np.ndarray:
    lu, pivots = _lu_factor_partial_pivot(matrix, procedure)
    return _lu_solve_partial_pivot(lu, pivots, rhs)


def _invert_dense_partial_pivot(matrix, procedure: str) -> np.ndarray:
    lu, pivots = _lu_factor_partial_pivot(matrix, procedure)
    n = len(lu)
    inverse = np.empty((n, n), dtype=float)
    for column in range(n):
        rhs = [0.0] * n
        rhs[column] = 1.0
        inverse[:, column] = _lu_solve_partial_pivot(lu, pivots, rhs)
    return inverse


def _polyfit_householder_pivoted(x, y, degree: int, weights=None) -> np.ndarray:
    """Fit a polynomial by scaled, column-pivoted Householder QR in Python.

    This follows the numerically stable structure used by the native transport
    fit: build a power basis, scale its columns, apply optional residual
    weights, solve the least-squares problem with pivoted QR, then undo the
    scaling and permutation. The factorization and triangular solve are Python
    loops; NumPy is used only to exchange array data with transport routines.
    """
    x_values = [float(value) for value in np.asarray(x, dtype=float).reshape(-1)]
    y_values = [float(value) for value in np.asarray(y, dtype=float).reshape(-1)]
    ncols = int(degree) + 1
    nrows = len(x_values)
    if degree < 0 or len(y_values) != nrows or nrows < ncols:
        raise CanteraError("polyfit", "invalid or underdetermined polynomial fit")

    if weights is None:
        row_weights = [1.0] * nrows
    else:
        row_weights = [float(value) for value in np.asarray(weights, dtype=float).reshape(-1)]
        if len(row_weights) != nrows:
            raise CanteraError("polyfit", "weight count does not match sample count")

    # Match polynomial::polyfit's power Vandermonde and column normalization.
    matrix = [[0.0] * ncols for _ in range(nrows)]
    for i, value in enumerate(x_values):
        power = 1.0
        for j in range(ncols):
            matrix[i][j] = power
            power *= value
    scales = [math.sqrt(math.fsum(matrix[i][j] ** 2 for i in range(nrows)))
              for j in range(ncols)]
    if any(scale == 0.0 or not math.isfinite(scale) for scale in scales):
        raise CanteraError("polyfit", "polynomial basis contains a zero or non-finite column")
    for i in range(nrows):
        wi = row_weights[i]
        for j in range(ncols):
            matrix[i][j] = matrix[i][j] / scales[j] * wi
    rhs = [y_values[i] * row_weights[i] for i in range(nrows)]
    permutation = list(range(ncols))

    # Rank-revealing Householder QR with greedy column pivoting.
    leading_norm = 0.0
    for k in range(ncols):
        pivot = max(
            range(k, ncols),
            key=lambda j: math.fsum(matrix[i][j] ** 2 for i in range(k, nrows)),
        )
        if pivot != k:
            for row in matrix:
                row[k], row[pivot] = row[pivot], row[k]
            permutation[k], permutation[pivot] = permutation[pivot], permutation[k]

        norm = math.sqrt(math.fsum(matrix[i][k] ** 2 for i in range(k, nrows)))
        if k == 0:
            leading_norm = norm
        threshold = (math.ulp(1.0) * max(nrows, ncols) * leading_norm)
        if not math.isfinite(norm) or norm <= threshold:
            raise CanteraError("polyfit", "rank-deficient polynomial fit")

        alpha = -math.copysign(norm, matrix[k][k])
        reflector = [matrix[i][k] for i in range(k, nrows)]
        reflector[0] -= alpha
        denominator = math.fsum(value * value for value in reflector)
        if denominator == 0.0:
            raise CanteraError("polyfit", "degenerate Householder reflector")
        factor = 2.0 / denominator
        for j in range(k + 1, ncols):
            projection = factor * math.fsum(
                reflector[offset] * matrix[k + offset][j]
                for offset in range(len(reflector)))
            for offset, value in enumerate(reflector):
                matrix[k + offset][j] -= projection * value
        projection = factor * math.fsum(
            reflector[offset] * rhs[k + offset]
            for offset in range(len(reflector)))
        for offset, value in enumerate(reflector):
            rhs[k + offset] -= projection * value
        matrix[k][k] = alpha
        for i in range(k + 1, nrows):
            matrix[i][k] = 0.0

    pivoted_coefficients = [0.0] * ncols
    for i in range(ncols - 1, -1, -1):
        remainder = rhs[i] - math.fsum(
            matrix[i][j] * pivoted_coefficients[j]
            for j in range(i + 1, ncols))
        pivoted_coefficients[i] = remainder / matrix[i][i]

    coefficients = [0.0] * ncols
    for i, original_column in enumerate(permutation):
        coefficients[original_column] = (
            pivoted_coefficients[i] / scales[original_column])
    return np.asarray(coefficients, dtype=float)


def omega_22(t_star: float) -> float:
    """Collision integral Omega^(2,2)* as a function of reduced temperature T*.

    Neufeld, Janzen, & Aziz (1972) empirical formulation.
    """
    ts = max(0.1, float(t_star))
    return (1.16145 * (ts ** -0.14874)
            + 0.52487 * math.exp(-0.77320 * ts)
            + 2.16178 * math.exp(-2.43787 * ts))


def omega_11(t_star: float) -> float:
    """Collision integral Omega^(1,1)* as a function of reduced temperature T*.

    Neufeld, Janzen, & Aziz (1972) empirical formulation.
    """
    ts = max(0.1, float(t_star))
    return (1.06036 * (ts ** -0.15610)
            + 0.19300 * math.exp(-0.47635 * ts)
            + 1.03587 * math.exp(-1.52996 * ts)
            + 1.76474 * math.exp(-3.89411 * ts))


def Frot(tr: float, sqtr: float) -> float:
    """The Parker temperature correction to the rotational collision number."""
    c1 = 0.5 * math.sqrt(Pi) * Pi
    c2 = 0.25 * Pi * Pi + 2.0
    c3 = math.sqrt(Pi) * Pi
    return 1.0 + c1 * sqtr + c2 * tr + c3 * sqtr * tr


# Monchick and Mason collision integral tables (from MMCollisionInt.cpp)
_delta_table = np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5])
_tstar22_table = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
    18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0
])
_tstar_table = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
    18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0, 500.0
])
_log_tstar_37 = np.log(_tstar_table[1:38])

_astar_table = np.array([
    1.0065, 1.0840, 1.0840, 1.0840, 1.0840, 1.0840, 1.0840, 1.0840,
    1.0231, 1.0660, 1.0380, 1.0400, 1.0430, 1.0500, 1.0520, 1.0510,
    1.0424, 1.0450, 1.0480, 1.0520, 1.0560, 1.0650, 1.0660, 1.0640,
    1.0719, 1.0670, 1.0600, 1.0550, 1.0580, 1.0680, 1.0710, 1.0710,
    1.0936, 1.0870, 1.0770, 1.0690, 1.0680, 1.0750, 1.0780, 1.0780,
    1.1053, 1.0980, 1.0880, 1.0800, 1.0780, 1.0820, 1.0840, 1.0840,
    1.1104, 1.1040, 1.0960, 1.0890, 1.0860, 1.0890, 1.0900, 1.0900,
    1.1114, 1.1070, 1.1000, 1.0950, 1.0930, 1.0950, 1.0960, 1.0950,
    1.1104, 1.1070, 1.1020, 1.0990, 1.0980, 1.1000, 1.1000, 1.0990,
    1.1086, 1.1060, 1.1020, 1.1010, 1.1010, 1.1050, 1.1050, 1.1040,
    1.1063, 1.1040, 1.1030, 1.1030, 1.1040, 1.1080, 1.1090, 1.1080,
    1.1020, 1.1020, 1.1030, 1.1050, 1.1070, 1.1120, 1.1150, 1.1150,
    1.0985, 1.0990, 1.1010, 1.1040, 1.1080, 1.1150, 1.1190, 1.1200,
    1.0960, 1.0960, 1.0990, 1.1030, 1.1080, 1.1160, 1.1210, 1.1240,
    1.0943, 1.0950, 1.0990, 1.1020, 1.1080, 1.1170, 1.1230, 1.1260,
    1.0934, 1.0940, 1.0970, 1.1020, 1.1070, 1.1160, 1.1230, 1.1280,
    1.0926, 1.0940, 1.0970, 1.0990, 1.1050, 1.1150, 1.1230, 1.1300,
    1.0934, 1.0950, 1.0970, 1.0990, 1.1040, 1.1130, 1.1220, 1.1290,
    1.0948, 1.0960, 1.0980, 1.1000, 1.1030, 1.1120, 1.1190, 1.1270,
    1.0965, 1.0970, 1.0990, 1.1010, 1.1040, 1.1100, 1.1180, 1.1260,
    1.0997, 1.1000, 1.1010, 1.1020, 1.1050, 1.1100, 1.1160, 1.1230,
    1.1025, 1.1030, 1.1040, 1.1050, 1.1060, 1.1100, 1.1150, 1.1210,
    1.1050, 1.1050, 1.1060, 1.1070, 1.1080, 1.1110, 1.1150, 1.1200,
    1.1072, 1.1070, 1.1080, 1.1080, 1.1090, 1.1120, 1.1150, 1.1190,
    1.1091, 1.1090, 1.1090, 1.1100, 1.1110, 1.1130, 1.1150, 1.1190,
    1.1107, 1.1110, 1.1110, 1.1110, 1.1120, 1.1140, 1.1160, 1.1190,
    1.1133, 1.1140, 1.1130, 1.1140, 1.1140, 1.1150, 1.1170, 1.1190,
    1.1154, 1.1150, 1.1160, 1.1160, 1.1160, 1.1170, 1.1180, 1.1200,
    1.1172, 1.1170, 1.1170, 1.1180, 1.1180, 1.1180, 1.1190, 1.1200,
    1.1186, 1.1190, 1.1190, 1.1190, 1.1190, 1.1190, 1.1200, 1.1210,
    1.1199, 1.1200, 1.1200, 1.1200, 1.1200, 1.1210, 1.1210, 1.1220,
    1.1223, 1.1220, 1.1220, 1.1220, 1.1220, 1.1230, 1.1230, 1.1240,
    1.1243, 1.1240, 1.1240, 1.1240, 1.1240, 1.1240, 1.1250, 1.1250,
    1.1259, 1.1260, 1.1260, 1.1260, 1.1260, 1.1260, 1.1260, 1.1260,
    1.1273, 1.1270, 1.1270, 1.1270, 1.1270, 1.1270, 1.1270, 1.1280,
    1.1297, 1.1300, 1.1300, 1.1300, 1.1300, 1.1300, 1.1300, 1.1290,
    1.1339, 1.1340, 1.1340, 1.1350, 1.1350, 1.1340, 1.1340, 1.1320,
    1.1364, 1.1370, 1.1370, 1.1380, 1.1390, 1.1380, 1.1370, 1.1350,
    1.14187, 1.14187, 1.14187, 1.14187, 1.14187, 1.14187, 1.14187, 1.14187
]).reshape(39, 8)

_bstar_table = np.array([
    1.1852, 1.2963, 1.2963, 1.2963, 1.2963, 1.2963, 1.2963, 1.2963,
    1.1960,  1.216,  1.237,  1.269,  1.285,  1.290,  1.297,  1.294,
    1.2451,  1.257,  1.340,  1.389,  1.366,  1.327,  1.314,  1.278,
    1.2900,  1.294,  1.272,  1.258,  1.262,  1.282,  1.290,  1.299,
    1.2986,  1.291,  1.284,  1.278,  1.277,  1.288,  1.294,  1.297,
    1.2865,  1.281,  1.276,  1.272,  1.277,  1.286,  1.292,  1.298,
    1.2665,  1.264,  1.261,  1.263,  1.269,  1.284,  1.292,  1.298,
    1.2455,  1.244,  1.248,  1.255,  1.262,  1.278,  1.289,  1.296,
    1.2253,  1.225,  1.234,  1.240,  1.252,  1.271,  1.284,  1.295,
    1.2078,  1.210,  1.216,  1.227,  1.242,  1.264,  1.281,  1.292,
    1.1919,  1.192,  1.205,  1.216,  1.230,  1.256,  1.273,  1.287,
    1.1678,  1.172,  1.181,  1.195,  1.209,  1.237,  1.261,  1.277,
    1.1496,  1.155,  1.161,  1.174,  1.189,  1.221,  1.246,  1.266,
    1.1366,  1.141,  1.147,  1.159,  1.174,  1.202,  1.231,  1.256,
    1.1270,  1.130,  1.138,  1.148,  1.162,  1.191,  1.218,  1.242,
    1.1197,  1.122,  1.129,  1.140,  1.149,  1.178,  1.205,  1.231,
    1.1080,  1.110,  1.116,  1.122,  1.132,  1.154,  1.180,  1.205,
    1.1016,  1.103,  1.107,  1.112,  1.120,  1.138,  1.160,  1.183,
    1.0980,  1.099,  1.102,  1.106,  1.112,  1.127,  1.145,  1.165,
    1.0958,  1.097,  1.099,  1.102,  1.107,  1.119,  1.135,  1.153,
    1.0935,  1.094,  1.095,  1.097,  1.100,  1.109,  1.120,  1.134,
    1.0925,  1.092,  1.094,  1.095,  1.098,  1.104,  1.112,  1.122,
    1.0922,  1.092,  1.093,  1.094,  1.096,  1.100,  1.106,  1.115,
    1.0922,  1.092,  1.093,  1.093,  1.095,  1.098,  1.103,  1.110,
    1.0923,  1.092,  1.093,  1.093,  1.094,  1.097,  1.101,  1.106,
    1.0923,  1.092,  1.092,  1.093,  1.094,  1.096,  1.099,  1.103,
    1.0927,  1.093,  1.093,  1.093,  1.094,  1.095,  1.098,  1.101,
    1.0930,  1.093,  1.093,  1.093,  1.094,  1.094,  1.096,  1.099,
    1.0933,  1.094,  1.093,  1.094,  1.094,  1.095,  1.096,  1.098,
    1.0937,  1.093,  1.094,  1.094,  1.094,  1.094,  1.096,  1.097,
    1.0939,  1.094,  1.094,  1.094,  1.094,  1.095,  1.095,  1.097,
    1.0943,  1.094,  1.094,  1.094,  1.095,  1.095,  1.096,  1.096,
    1.0944,  1.095,  1.094,  1.094,  1.094,  1.095,  1.095,  1.096,
    1.0944,  1.094,  1.095,  1.094,  1.094,  1.095,  1.096,  1.096,
    1.0943,  1.095,  1.094,  1.094,  1.095,  1.095,  1.095,  1.095,
    1.0941,  1.094,  1.094,  1.094,  1.094,  1.094,  1.094,  1.096,
    1.0947,  1.095,  1.094,  1.094,  1.093,  1.093,  1.094,  1.095,
    1.0957,  1.095,  1.094,  1.093,  1.092,  1.093,  1.093,  1.094,
    1.10185, 1.10185, 1.10185, 1.10185, 1.10185, 1.10185, 1.10185, 1.10185
]).reshape(39, 8)

_cstar_table = np.array([
    0.8889,  0.77778, 0.77778, 0.77778, 0.77778, 0.77778, 0.77778, 0.77778,
    0.88575, 0.8988, 0.8378, 0.8029, 0.7876, 0.7805, 0.7799, 0.7801,
    0.87268, 0.8692, 0.8647, 0.8479, 0.8237, 0.7975, 0.7881, 0.7784,
    0.85182, 0.8525, 0.8366, 0.8198, 0.8054, 0.7903, 0.7839, 0.782,
    0.83542, 0.8362, 0.8306, 0.8196, 0.8076, 0.7918, 0.7842, 0.7806,
    0.82629, 0.8278, 0.8252, 0.8169, 0.8074, 0.7916, 0.7838, 0.7802,
    0.82299, 0.8249, 0.823,  0.8165, 0.8072, 0.7922, 0.7839, 0.7798,
    0.82357, 0.8257, 0.8241, 0.8178, 0.8084, 0.7927, 0.7839, 0.7794,
    0.82657, 0.828,  0.8264, 0.8199, 0.8107, 0.7939, 0.7842, 0.7796,
    0.8311,  0.8234, 0.8295, 0.8228, 0.8136, 0.796,  0.7854, 0.7798,
    0.8363,  0.8366, 0.8342, 0.8267, 0.8168, 0.7986, 0.7864, 0.7805,
    0.84762, 0.8474, 0.8438, 0.8358, 0.825,  0.8041, 0.7904, 0.7822,
    0.85846, 0.8583, 0.853,  0.8444, 0.8336, 0.8118, 0.7957, 0.7854,
    0.8684,  0.8674, 0.8619, 0.8531, 0.8423, 0.8186, 0.8011, 0.7898,
    0.87713, 0.8755, 0.8709, 0.8616, 0.8504, 0.8265, 0.8072, 0.7939,
    0.88479, 0.8831, 0.8779, 0.8695, 0.8578, 0.8338, 0.8133, 0.799,
    0.89972, 0.8986, 0.8936, 0.8846, 0.8742, 0.8504, 0.8294, 0.8125,
    0.91028, 0.9089, 0.9043, 0.8967, 0.8869, 0.8649, 0.8438, 0.8253,
    0.91793, 0.9166, 0.9125, 0.9058, 0.897,  0.8768, 0.8557, 0.8372,
    0.92371, 0.9226, 0.9189, 0.9128, 0.905,  0.8861, 0.8664, 0.8484,
    0.93135, 0.9304, 0.9274, 0.9226, 0.9164, 0.9006, 0.8833, 0.8662,
    0.93607, 0.9353, 0.9329, 0.9291, 0.924,  0.9109, 0.8958, 0.8802,
    0.93927, 0.9387, 0.9366, 0.9334, 0.9292, 0.9162, 0.905,  0.8911,
    0.94149, 0.9409, 0.9393, 0.9366, 0.9331, 0.9236, 0.9122, 0.8997,
    0.94306, 0.9426, 0.9412, 0.9388, 0.9357, 0.9276, 0.9175, 0.9065,
    0.94419, 0.9437, 0.9425, 0.9406, 0.938,  0.9308, 0.9219, 0.9119,
    0.94571, 0.9455, 0.9445, 0.943,  0.9409, 0.9353, 0.9283, 0.9201,
    0.94662, 0.9464, 0.9456, 0.9444, 0.9428, 0.9382, 0.9325, 0.9258,
    0.94723, 0.9471, 0.9464, 0.9455, 0.9442, 0.9405, 0.9355, 0.9298,
    0.94764, 0.9474, 0.9469, 0.9462, 0.945,  0.9418, 0.9378, 0.9328,
    0.9479,  0.9478, 0.9474, 0.9465, 0.9457, 0.943,  0.9394, 0.9352,
    0.94827, 0.9481, 0.948,  0.9472, 0.9467, 0.9447, 0.9422, 0.9391,
    0.94842, 0.9484, 0.9481, 0.9478, 0.9472, 0.9458, 0.9437, 0.9415,
    0.94852, 0.9484, 0.9483, 0.948,  0.9475, 0.9465, 0.9449, 0.943,
    0.94861, 0.9487, 0.9484, 0.9481, 0.9479, 0.9468, 0.9455, 0.943,
    0.94872, 0.9486, 0.9486, 0.9483, 0.9482, 0.9475, 0.9464, 0.9452,
    0.94881, 0.9488, 0.9489, 0.949,  0.9487, 0.9482, 0.9476, 0.9468,
    0.94863, 0.9487, 0.9489, 0.9491, 0.9493, 0.9491, 0.9483, 0.9476,
    0.94444, 0.94444, 0.94444, 0.94444, 0.94444, 0.94444, 0.94444, 0.94444
]).reshape(39, 8)


def _fit_delta_polys(table: np.ndarray) -> list[np.ndarray]:
    polys = []
    for i in range(37):
        y = table[i + 1, :]
        p = _polyfit_householder_pivoted(_delta_table, y, 6)
        polys.append(p)
    return polys


_apoly = _fit_delta_polys(_astar_table)
_bpoly = _fit_delta_polys(_bstar_table)
_cpoly = _fit_delta_polys(_cstar_table)


def _eval_star(table: np.ndarray, polys: list[np.ndarray], ts: float, dstar: float) -> float:
    ts = float(ts)
    dstar = float(dstar)
    i = 0
    while i < 37 and ts >= _tstar22_table[i]:
        i += 1
    i1 = max(i - 1, 0)
    i2 = i1 + 3
    # MMCollisionInt::astar / bstar / cstar cap i2 at 36 (exclusive),
    # even though the source tables contain 37 rows. Preserve that source
    # indexing exactly for the high-reduced-temperature extrapolation.
    if i2 > 36:
        i2 = 36
        i1 = i2 - 3

    vals = []
    for k in range(i1, i2):
        if dstar == 0.0:
            vals.append(table[k + 1, 0])
        else:
            vals.append(np.polynomial.polynomial.polyval(dstar, polys[k]))

    x = _log_tstar_37[i1:i2]
    x0 = math.log(max(0.01, ts))
    dx21 = x[1] - x[0]
    dx32 = x[2] - x[1]
    dx31 = dx21 + dx32
    dy32 = vals[2] - vals[1]
    dy21 = vals[1] - vals[0]
    a = (dx21 * dy32 - dy21 * dx32) / (dx21 * dx31 * dx32)
    return a * (x0 - x[0]) * (x0 - x[1]) + (dy21 / dx21) * (x0 - x[1]) + vals[1]


def _fit_collision_star_polynomials(
    delta: float, first: int, last: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce GasTransport::fitCollisionIntegrals for A*, B*, and C*.

    Cantera fits degree-eight polynomials in log(T*) at construction time and
    evaluates those fits at runtime. Directly interpolating the tables at the
    requested T* is physically similar but is not the same runtime algorithm.
    """
    x = _log_tstar_37[first:last + 1]
    fitted = []
    for table, delta_polys in (
        (_astar_table, _apoly),
        (_bstar_table, _bpoly),
        (_cstar_table, _cpoly),
    ):
        values = np.empty(x.size)
        for pos, row in enumerate(range(first, last + 1)):
            if delta == 0.0:
                values[pos] = table[row + 1, 0]
            else:
                values[pos] = np.polynomial.polynomial.polyval(
                    delta, delta_polys[row])
        fitted.append(_polyfit_householder_pivoted(x, values, 8))
    return tuple(fitted)


# Table of omega22 values from MMCollisionInt.cpp
_omega22_table = np.array([
    4.1005, 4.266,  4.833,  5.742,  6.729,  8.624,  10.34,  11.89,
    3.2626, 3.305,  3.516,  3.914,  4.433,  5.57,   6.637,  7.618,
    2.8399, 2.836,  2.936,  3.168,  3.511,  4.329,  5.126,  5.874,
    2.531,  2.522,  2.586,  2.749,  3.004,  3.64,   4.282,  4.895,
    2.2837, 2.277,  2.329,  2.46,   2.665,  3.187,  3.727,  4.249,
    2.0838, 2.081,  2.13,   2.243,  2.417,  2.862,  3.329,  3.786,
    1.922,  1.924,  1.97,   2.072,  2.225,  2.614,  3.028,  3.435,
    1.7902, 1.795,  1.84,   1.934,  2.07,   2.417,  2.788,  3.156,
    1.6823, 1.689,  1.733,  1.82,   1.944,  2.258,  2.596,  2.933,
    1.5929, 1.601,  1.644,  1.725,  1.838,  2.124,  2.435,  2.746,
    1.4551, 1.465,  1.504,  1.574,  1.67,   1.913,  2.181,  2.451,
    1.3551, 1.365,  1.4,    1.461,  1.544,  1.754,  1.989,  2.228,
    1.28,   1.289,  1.321,  1.374,  1.447,  1.63,   1.838,  2.053,
    1.2219, 1.231,  1.259,  1.306,  1.37,   1.532,  1.718,  1.912,
    1.1757, 1.184,  1.209,  1.251,  1.307,  1.451,  1.618,  1.795,
    1.0933, 1.1,    1.119,  1.15,   1.193,  1.304,  1.435,  1.578,
    1.0388, 1.044,  1.059,  1.083,  1.117,  1.204,  1.31,   1.428,
    0.99963, 1.004, 1.016,  1.035,  1.062,  1.133,  1.22,   1.319,
    0.96988, 0.9732, 0.983,  0.9991, 1.021,  1.079,  1.153,  1.236,
    0.92676, 0.9291, 0.936,  0.9473, 0.9628, 1.005,  1.058,  1.121,
    0.89616, 0.8979, 0.903,  0.9114, 0.923,  0.9545, 0.9955, 1.044,
    0.87272, 0.8741, 0.878,  0.8845, 0.8935, 0.9181, 0.9505, 0.9893,
    0.85379, 0.8549, 0.858,  0.8632, 0.8703, 0.8901, 0.9164, 0.9482,
    0.83795, 0.8388, 0.8414, 0.8456, 0.8515, 0.8678, 0.8895, 0.916,
    0.82435, 0.8251, 0.8273, 0.8308, 0.8356, 0.8493, 0.8676, 0.8901,
    0.80184, 0.8024, 0.8039, 0.8065, 0.8101, 0.8201, 0.8337, 0.8504,
    0.78363, 0.784,  0.7852, 0.7872, 0.7899, 0.7976, 0.8081, 0.8212,
    0.76834, 0.7687, 0.7696, 0.7712, 0.7733, 0.7794, 0.7878, 0.7983,
    0.75518, 0.7554, 0.7562, 0.7575, 0.7592, 0.7642, 0.7711, 0.7797,
    0.74364, 0.7438, 0.7445, 0.7455, 0.747,  0.7512, 0.7569, 0.7642,
    0.71982, 0.72,   0.7204, 0.7211, 0.7221, 0.725,  0.7289, 0.7339,
    0.70097, 0.7011, 0.7014, 0.7019, 0.7026, 0.7047, 0.7076, 0.7112,
    0.68545, 0.6855, 0.6858, 0.6861, 0.6867, 0.6883, 0.6905, 0.6932,
    0.67232, 0.6724, 0.6726, 0.6728, 0.6733, 0.6743, 0.6762, 0.6784,
    0.65099, 0.651,  0.6512, 0.6513, 0.6516, 0.6524, 0.6534, 0.6546,
    0.61397, 0.6141, 0.6143, 0.6145, 0.6147, 0.6148, 0.6148, 0.6147,
    0.5887, 0.5889, 0.5894, 0.59,   0.5903, 0.5901, 0.5895, 0.5885
]).reshape(37, 8)

_o22poly = []
for _i in range(37):
    _o22poly.append(_polyfit_householder_pivoted(
        _delta_table, _omega22_table[_i, :], 6))


def _eval_omega22(ts: float, dstar: float) -> float:
    ts = float(ts)
    dstar = float(dstar)
    i = 0
    while i < 37 and ts >= _tstar22_table[i]:
        i += 1
    i1 = max(i - 1, 0)
    i2 = i1 + 3
    # Match MMCollisionInt::omega22: its upper bound is 36, exclusive.
    if i2 > 36:
        i2 = 36
        i1 = i2 - 3

    vals = []
    for k in range(i1, i2):
        if dstar == 0.0:
            vals.append(_omega22_table[k, 0])
        else:
            vals.append(np.polynomial.polynomial.polyval(dstar, _o22poly[k]))

    x = _log_tstar_37[i1:i2]
    x0 = math.log(max(0.01, ts))
    dx21 = x[1] - x[0]
    dx32 = x[2] - x[1]
    dx31 = dx21 + dx32
    dy32 = vals[2] - vals[1]
    dy21 = vals[1] - vals[0]
    a = (dx21 * dy32 - dy21 * dx32) / (dx21 * dx31 * dx32)
    return a * (x0 - x[0]) * (x0 - x[1]) + (dy21 / dx21) * (x0 - x[1]) + vals[1]


@dataclass(frozen=True)
class GasTransportData:
    """Species transport metadata from YAML mechanism."""
    geometry: str = "atom"
    diameter: float = 3.5  # Angstroms
    well_depth: float = 100.0  # Kelvin (epsilon / k_B)
    dipole: float = 0.0  # Debye
    polarizability: float = 0.0  # Angstrom^3
    rotational_relaxation: float = 0.0
    dispersion_coefficient: float = 0.0
    quadrupole_polarizability: float = 0.0

    def validate(self, phase=None) -> None:
        pass

    def getParameters(self, params: dict) -> None:
        params.update({"geometry": self.geometry, "diameter": self.diameter})




class Transport:
    """Base class for transport property managers."""
    def __init__(self, thermo=None):
        self._thermo = thermo

    @property
    def n_species(self) -> int:
        return self._thermo.n_species if self._thermo else 0

    def CKMode(self) -> bool:
        return False

    def bulkViscosity(self) -> float:
        return 0.0

    def checkSpeciesIndex(self, k: int) -> None:
        if k < 0 or k >= self.n_species:
            raise IndexError("Species index out of range")

    def clone(self):
        return self

    def electricalConductivity(self) -> float:
        return 0.0

    def fittingErrors(self) -> dict:
        return {}

    def getBinDiffusivityPolynomial(self, i: int, j: int, poly: np.ndarray) -> None:
        pass

    def getConductivityPolynomial(self, i: int, poly: np.ndarray) -> None:
        pass

    def getViscosityPolynomial(self, i: int, poly: np.ndarray) -> None:
        pass

    def init(self, *args, **kwargs) -> None:
        pass

    def invalidateCache(self) -> None:
        pass

    def parameters(self) -> dict:
        return {}

    def setBinDiffusivityPolynomial(self, i: int, j: int, poly: np.ndarray) -> None:
        pass

    def setConductivityPolynomial(self, i: int, poly: np.ndarray) -> None:
        pass

    def setViscosityPolynomial(self, i: int, poly: np.ndarray) -> None:
        pass

    def thermalConductivity(self) -> float:
        return 0.025

    def transportModel(self) -> str:
        return "None"

    @property
    def transport_model(self) -> str:
        return self.transportModel()


class GasTransport(Transport):
    """Base class for transport property managers for all gas phases."""
    def fitCollisionIntegrals(self) -> None:
        pass

    def fitDiffCoeffs(self) -> None:
        pass

    def fitProperties(self) -> None:
        pass

    def getTransportData(self) -> dict:
        return {}

    def setupCollisionIntegral(self) -> None:
        pass

    def setupCollisionParameters(self) -> None:
        pass

    def updateDiff_T(self) -> None:
        pass

    def updateSpeciesViscosities(self) -> None:
        pass

    def updateViscosity_T(self) -> None:
        pass

    def update_T(self) -> None:
        pass


class MixTransport(GasTransport):
    def update_C(self) -> None:
        pass

    def updateCond_T(self) -> None:
        pass

    """Mixture-averaged transport properties for gas phases.

    Evaluates:
    - Pure species and mixture viscosity using Chapman-Enskog and Wilke formula
    - Thermal conductivity using Eucken combination formula
    - Binary and mixture-averaged diffusion coefficients
    """

    def __init__(self, thermo, transport_data: Mapping[str, GasTransportData] | None = None):
        self._thermo = thermo
        self._transport_data = dict(transport_data or {})
        self._binary_diffusion_fit_transport = None
        # Ensure default transport data exists for all species
        for s in self._thermo.species:
            if s.name not in self._transport_data:
                self._transport_data[s.name] = GasTransportData()

    @classmethod
    def from_mechanism(cls, mech_or_path, thermo=None, phase_name=None):
        from pathlib import Path
        from .mechanism import _as_document, load_mechanism, Mechanism
        from .ideal_gas import IdealGasPhase

        if isinstance(mech_or_path, (str, Path)) or hasattr(mech_or_path, 'read_text'):
            doc = _as_document(mech_or_path)
            raw_species = doc.get('species', [])
            mech = load_mechanism(mech_or_path)
        elif isinstance(mech_or_path, Mechanism):
            mech = mech_or_path
            raw_species = []
        else:
            raise TypeError("mech_or_path must be a file path, YAML document, or Mechanism instance")

        if mech.phases:
            mech = mech.for_phase(phase_name)
        if thermo is None:
            thermo = IdealGasPhase.from_mechanism(mech)

        transport_data = {}
        for r in raw_species:
            if isinstance(r, dict) and 'name' in r and 'transport' in r:
                t = r['transport']
                if isinstance(t, dict):
                    geom = t.get('geometry', 'atom')
                    diam = float(t.get('diameter', 3.5))
                    well = float(t.get('well-depth', 100.0))
                    dipole = float(t.get('dipole', 0.0))
                    polar = float(t.get('polarizability', 0.0))
                    zrot = float(t.get('rotational-relaxation', 0.0))
                    transport_data[r['name']] = GasTransportData(
                        geometry=geom, diameter=diam, well_depth=well,
                        dipole=dipole, polarizability=polar, rotational_relaxation=zrot,
                        dispersion_coefficient=float(t.get('dispersion-coefficient', 0.0)) * 1e-50,
                        quadrupole_polarizability=float(t.get('quadrupole-polarizability', 0.0)) * 1e-50,
                    )
        for species in mech.species:
            t = species.transport
            if isinstance(t, dict):
                transport_data[species.name] = GasTransportData(
                    geometry=t.get('geometry', 'atom'),
                    diameter=float(t.get('diameter', 3.5)),
                    well_depth=float(t.get('well-depth', 100.0)),
                    dipole=float(t.get('dipole', 0.0)),
                    polarizability=float(t.get('polarizability', 0.0)),
                    rotational_relaxation=float(t.get('rotational-relaxation', 0.0)),
                    dispersion_coefficient=float(t.get('dispersion-coefficient', 0.0)) * 1e-50,
                    quadrupole_polarizability=float(t.get('quadrupole-polarizability', 0.0)) * 1e-50)
        return cls(thermo, transport_data)

    @property
    def thermo(self):
        return self._thermo

    def species_viscosities(self) -> tuple[float, ...]:
        """Evaluate the source-fitted pure-species viscosities [Pa*s]."""
        return self._get_fit_transport().species_viscosities()

    def _get_fit_transport(self):
        """Lazily construct the shared Python translation of GasTransport fits."""
        fit_transport = self._binary_diffusion_fit_transport
        if fit_transport is None:
            fit_transport = MultiTransport(self._thermo, self._transport_data)
            self._binary_diffusion_fit_transport = fit_transport
        return fit_transport

    @property
    def viscosity(self) -> float:
        """Evaluate mixture dynamic viscosity [Pa*s] using the Wilke mixture rule."""
        X = self._thermo.X
        mu = self.species_viscosities()
        mw = self._thermo.molecular_weights
        nsp = len(X)

        phi = [[0.0] * nsp for _ in range(nsp)]
        for k in range(nsp):
            for j in range(nsp):
                if k == j:
                    phi[k][j] = 1.0
                else:
                    num = (1.0 + math.sqrt(mu[k] / mu[j]) * (mw[j] / mw[k]) ** 0.25) ** 2
                    denom = math.sqrt(8.0 * (1.0 + mw[k] / mw[j]))
                    phi[k][j] = num / denom

        mu_mix = 0.0
        for k in range(nsp):
            denom = sum(X[j] * phi[k][j] for j in range(nsp))
            if denom > 0.0:
                mu_mix += X[k] * mu[k] / denom
        return mu_mix

    def species_thermal_conductivities(self) -> tuple[float, ...]:
        """Evaluate Cantera-fitted pure-species thermal conductivities [W/m/K]."""
        return self._get_fit_transport().species_thermal_conductivities()

    @property
    def thermal_conductivity(self) -> float:
        """Mixture thermal conductivity [W/m/K] using combination averaging."""
        X = self._thermo.X
        cond = self.species_thermal_conductivities()
        sum1 = sum(xk * lk for xk, lk in zip(X, cond))
        sum2 = sum(xk / max(1e-12, lk) for xk, lk in zip(X, cond))
        return 0.5 * (sum1 + (1.0 / max(1e-12, sum2)))

    def binary_diffusion_coefficients(self) -> list[list[float]]:
        """Matrix of binary diffusion coefficients D_jk [m^2/s]."""
        # GasTransport::fitDiffCoeffs does not evaluate a closed-form
        # collision correlation at the requested temperature. It samples the
        # Monchick–Mason collision-integral tables at 50 phase-range
        # temperatures, performs a weighted fourth-order fit in log(T), then
        # evaluates that fit and divides by pressure. MultiTransport contains
        # the shared Python translation of those exact source steps. Reuse its
        # fitted pair polynomials here so mixture-averaged and high-pressure
        # models use the same native-defined dilute-gas reference.
        return self._get_fit_transport()._eval_binary_diff_coeffs().tolist()

    def mix_diff_coeffs(self) -> tuple[float, ...]:
        """Coefficients for mass fluxes driven by mole-fraction gradients."""
        X = tuple(max(Tiny, value) for value in self._thermo.X)
        nsp = len(X)
        fit_tran = self._get_fit_transport()
        bdiff_arr = fit_tran._eval_binary_diff_coeffs()
        if nsp == 1:
            return (float(bdiff_arr[0, 0]),)
        mean_mw = self._thermo.mean_molecular_weight
        mw_arr = np.asarray(self._thermo.molecular_weights, dtype=float)
        X_arr = np.asarray(X, dtype=float)
        inv_bdiff = 1.0 / bdiff_arr
        np.fill_diagonal(inv_bdiff, 0.0)
        sum2 = inv_bdiff.T @ X_arr
        d_mix = np.where(sum2 <= 0.0, np.diag(bdiff_arr), (mean_mw - X_arr * mw_arr) / (mean_mw * sum2))
        return tuple(float(v) for v in d_mix)

    @property
    def mix_diff_coeffs_mole(self) -> tuple[float, ...]:
        """Coefficients for molar fluxes driven by mole-fraction gradients."""
        X = tuple(max(Tiny, value) for value in self._thermo.X)
        bdiff = self.binary_diffusion_coefficients()
        if len(X) == 1:
            return (bdiff[0][0],)
        values = []
        for k in range(len(X)):
            inverse = sum(X[j] / bdiff[j][k]
                          for j in range(len(X)) if j != k)
            values.append((1.0 - X[k]) / inverse
                          if inverse > 0.0 else bdiff[k][k])
        return tuple(values)

    @property
    def mix_diff_coeffs_mass(self) -> tuple[float, ...]:
        """Coefficients for mass fluxes driven by mass-fraction gradients."""
        X = tuple(max(Tiny, value) for value in self._thermo.X)
        bdiff = self.binary_diffusion_coefficients()
        if len(X) == 1:
            return (bdiff[0][0],)
        mean_mw = self._thermo.mean_molecular_weight
        mw = self._thermo.molecular_weights
        values = []
        for k in range(len(X)):
            sum1 = sum(X[i] / bdiff[k][i]
                       for i in range(len(X)) if i != k)
            weighted = sum(X[i] * mw[i] / bdiff[k][i]
                           for i in range(len(X)) if i != k)
            denominator = mean_mw - mw[k] * X[k]
            if denominator <= 0.0:
                values.append(0.0)
                continue
            sum2 = weighted * X[k] / denominator
            values.append(1.0 / (sum1 + sum2))
        return tuple(values)

    @property
    def transport_model(self) -> str:
        return "mixture-averaged"

    @property
    def CK_mode(self) -> bool:
        return False

    @property
    def binary_diff_coeffs(self) -> np.ndarray:
        """Matrix of binary diffusion coefficients [m^2/s]."""
        return np.array(self.binary_diffusion_coefficients())

    def getBinaryDiffCoeffs(self, ld: int, d) -> None:
        nsp = self._thermo.n_species
        if ld < nsp:
            raise CanteraError("GasTransport::getBinaryDiffCoeffs", "ld is too small")
        _check_array_size("GasTransport::getBinaryDiffCoeffs", "d", d, ld * nsp)
        bdiff = self.binary_diffusion_coefficients()
        for i in range(nsp):
            for j in range(nsp):
                d[ld * j + i] = bdiff[i][j]

    def getMixDiffCoeffs(self, d) -> None:
        _check_array_size("GasTransport::getMixDiffCoeffs", "d", d, self._thermo.n_species)
        coeffs = self.mix_diff_coeffs()
        for k, v in enumerate(coeffs):
            d[k] = v

    def getMixDiffCoeffsMole(self, d) -> None:
        _check_array_size("GasTransport::getMixDiffCoeffsMole", "d", d, self._thermo.n_species)
        coeffs = self.mix_diff_coeffs_mole
        for k, v in enumerate(coeffs):
            d[k] = v

    def getMixDiffCoeffsMass(self, d) -> None:
        _check_array_size("GasTransport::getMixDiffCoeffsMass", "d", d, self._thermo.n_species)
        coeffs = self.mix_diff_coeffs_mass
        for k, v in enumerate(coeffs):
            d[k] = v

    def getSpeciesViscosities(self, visc) -> None:
        _check_array_size("GasTransport::getSpeciesViscosities", "visc", visc, self._thermo.n_species)
        v = self.species_viscosities()
        for k, val in enumerate(v):
            visc[k] = val

    def getMobilities(self, mob) -> None:
        _check_array_size("MixTransport::getMobilities", "mobil", mob, self._thermo.n_species)
        if hasattr(self, 'mobilities'):
            m = self.mobilities()
            for k, val in enumerate(m):
                mob[k] = val
        else:
            diff = self.mix_diff_coeffs()
            T = self._thermo.temperature
            for k in range(len(diff)):
                mob[k] = diff[k] * ElectronCharge / (Boltzmann * T)

    def getThermalDiffCoeffs(self, dt) -> None:
        _check_array_size("MixTransport::getThermalDiffCoeffs", "dt", dt, self._thermo.n_species)
        for k in range(len(dt)):
            dt[k] = 0.0

    @property
    def thermal_diff_coeffs(self) -> np.ndarray:
        dt = np.zeros(self._thermo.n_species)
        self.getThermalDiffCoeffs(dt)
        return dt

    def getSpeciesFluxes(self, ndim: int, grad_T, ldx: int, grad_X, ldf: int, fluxes) -> None:
        """Diffusive mass fluxes [kg/m^2/s] via mixture-averaged formula."""
        nsp = self._thermo.n_species
        procedure = "MixTransport::getSpeciesFluxes"
        _check_array_size(procedure, "grad_T", grad_T, ndim)
        if ldx < nsp:
            raise CanteraError(procedure, "ldx is too small")
        _check_array_size(procedure, "grad_X", grad_X, ldx * ndim)
        if ldf < nsp:
            raise CanteraError(procedure, "ldf is too small")
        _check_array_size(procedure, "fluxes", fluxes, ldf * ndim)
        mw = self._thermo.molecular_weights
        y = self._thermo.Y
        d_mix = self.mix_diff_coeffs()
        rhon = self._thermo.molar_density

        sum_flux = [0.0] * ndim
        for n in range(ndim):
            for k in range(nsp):
                val = -rhon * mw[k] * d_mix[k] * grad_X[n * ldx + k]
                fluxes[n * ldf + k] = val
                sum_flux[n] += val
        for n in range(ndim):
            for k in range(nsp):
                fluxes[n * ldf + k] -= y[k] * sum_flux[n]


class UnityLewisTransport(MixTransport):
    """Transport properties using the unity Lewis number approximation."""

    def transportModel(self) -> str:
        return "unity-Lewis-number"

    @property
    def transport_model(self) -> str:
        return "unity-Lewis-number"

    def getMixDiffCoeffs(self, d=None):
        _check_array_size("UnityLewisTransport::getMixDiffCoeffs", "d", d, self._thermo.n_species) if d is not None else None
        cond = float(self.thermal_conductivity)
        rho = self._thermo.density
        cp = self._thermo.cp_mass
        Dm = cond / (rho * cp) if (rho * cp) > 0.0 else 0.0
        vals = np.full(self._thermo.n_species, Dm, dtype=float)
        if d is not None:
            d[:] = vals
            return d
        return vals

    def mix_diff_coeffs(self) -> tuple[float, ...]:
        return tuple(self.getMixDiffCoeffs())

    def getMixDiffCoeffsMass(self, d=None):
        _check_array_size("UnityLewisTransport::getMixDiffCoeffsMass", "d", d, self._thermo.n_species) if d is not None else None
        return self.getMixDiffCoeffs(d)

    @property
    def mix_diff_coeffs_mass(self) -> tuple[float, ...]:
        return self.mix_diff_coeffs()

    def getMixDiffCoeffsMole(self, d=None):
        raise NotImplementedError("UnityLewisTransport::getMixDiffCoeffsMole")

    @property
    def mix_diff_coeffs_mole(self) -> tuple[float, ...]:
        raise NotImplementedError("UnityLewisTransport::getMixDiffCoeffsMole")

    def getThermalDiffCoeffs(self, dt=None):
        _check_array_size("UnityLewisTransport::getThermalDiffCoeffs", "dt", dt, self._thermo.n_species) if dt is not None else None
        vals = np.zeros(self._thermo.n_species, dtype=float)
        if dt is not None:
            dt[:] = vals
            return dt
        return vals

    @property
    def thermal_diff_coeffs(self) -> tuple[float, ...]:
        return tuple(self.getThermalDiffCoeffs())


class IonGasTransport(MixTransport):
    """Ionized-gas transport using Stockmayer-(n,6,4) model for ions."""

    def transportModel(self) -> str:
        return "ionized-gas"

    @property
    def transport_model(self) -> str:
        return "ionized-gas"

    def init(self, thermo, mode=0):
        if mode == 1:
            raise CanteraError("IonGasTransport::init",
                               "mode = CK_Mode, which is an outdated lower-order fit.")
        self._thermo = thermo
        self._nsp = self._thermo.n_species
        self._charges = tuple(float(self._thermo.charge(k)) for k in range(self._nsp))
        electron_candidates = [
            k for k in range(self._nsp)
            if self._charges[k] == -1.0 and (self._thermo.species_name(k).lower() in ('e', 'electron')
                                             or math.isclose(self._thermo.molecular_weights[k], 5.485799e-4, rel_tol=1e-3))
        ]
        if len(electron_candidates) > 1:
            raise CanteraError("IonGasTransport::init", "Found multiple electron species.")
        self._electron = electron_candidates[0] if electron_candidates else None
        self._ions = tuple(k for k in range(self._nsp) if self._charges[k] != 0.0 and k != self._electron)
        self._neutrals = tuple(k for k in range(self._nsp) if self._charges[k] == 0.0)

        # Resonant O2 / O2- experimental collision integral fit (Prager 2005)
        # Pinned native Cantera polynomial coefficients for O2/O2- from Prager (2005)
        self._om11_O2 = np.array([0.0, 0.50570048102629783, -0.00080573149983813373,
                                  5.0274392926237489e-07, -1.3558670838184069e-10,
                                  1.3142329548561772e-14], dtype=float)

        self.setupN64()
        self.fitDiffCoeffs()

    def __init__(self, thermo, transport_data=None, charges=None):
        super().__init__(thermo, transport_data)
        if charges is None:
            charges = {s.name: getattr(s, 'charge', 0.0) for s in thermo.species}
        self._charges = tuple(float(charges.get(name, 0.0))
                              for name in thermo.species_names)
        electron_candidates = [
            i for i, (name, charge) in enumerate(zip(thermo.species_names, self._charges))
            if charge == -1.0 and name.lower() in ('e', 'electron')
        ]
        if len(electron_candidates) > 1:
            raise CanteraError("IonGasTransport::init", "Found multiple electron species.")
        self._electron = electron_candidates[0] if electron_candidates else None
        self._ions = tuple(i for i, charge in enumerate(self._charges)
                           if charge != 0.0 and i != self._electron)
        self._neutrals = tuple(i for i, charge in enumerate(self._charges)
                               if charge == 0.0)

        # Pinned native Cantera polynomial coefficients for O2/O2- from Prager (2005)
        self._om11_O2 = np.array([0.0, 0.50570048102629783, -0.00080573149983813373,
                                  5.0274392926237489e-07, -1.3558670838184069e-10,
                                  1.3142329548561772e-14], dtype=float)

        self.setupN64()
        self.fitDiffCoeffs()

    @classmethod
    def from_mechanism(cls, mech_or_path, thermo=None, phase_name=None):
        base = MixTransport.from_mechanism(mech_or_path, thermo, phase_name)
        from pathlib import Path
        from .mechanism import Mechanism, load_mechanism
        if isinstance(mech_or_path, (str, Path)) or hasattr(mech_or_path, 'read_text'):
            mech = load_mechanism(mech_or_path)
        elif isinstance(mech_or_path, Mechanism):
            mech = mech_or_path
        else:
            raise TypeError('mech_or_path must be a mechanism path or Mechanism')
        if mech.phases:
            mech = mech.for_phase(phase_name)
        charges = {species.name: species.charge for species in mech.species}
        return cls(base.thermo, base._transport_data, charges)

    def setupN64(self) -> None:
        """Setup Stockmayer-(n,6,4) parameters for ion-neutral pairs."""
        nsp = self._thermo.n_species
        self._diam_n64 = np.zeros((nsp, nsp), dtype=float)
        self._epsilon_n64 = np.zeros((nsp, nsp), dtype=float)
        self._gamma_n64 = np.zeros((nsp, nsp), dtype=float)

        for i in self._ions:
            for j in self._neutrals:
                ti = self._transport_data.get(self._thermo.species_name(i))
                tj = self._transport_data.get(self._thermo.species_name(j))
                if not ti or not tj or ti.polarizability <= 0.0 or tj.polarizability <= 0.0:
                    continue
                alpha_i = ti.polarizability * 1e-30
                alpha_j = tj.polarizability * 1e-30
                alphaA_i = ti.polarizability
                alphaA_j = tj.polarizability
                r_alpha = alpha_i / alpha_j
                charge_i = self._charges[i]

                xi = alphaA_i / (charge_i**2 * (1.0 + (2.0 * r_alpha)**(2.0/3.0)) * math.sqrt(alphaA_j))

                K1 = 1.767
                kappa = 0.095
                diam_ij = K1 * (alpha_i**(1.0/3.0) + alpha_j**(1.0/3.0)) / ((alphaA_i * alphaA_j * (1.0 + 1.0/xi))**kappa)
                self._diam_n64[i, j] = self._diam_n64[j, i] = diam_ij

                K2 = 1.44
                eps_ij = K2 * ElectronCharge**2 * charge_i**2 * alpha_j * (1.0 + xi) / (8.0 * Pi * epsilon_0 * (diam_ij**4))
                if eps_ij != 0.0:
                    self._epsilon_n64[i, j] = self._epsilon_n64[j, i] = eps_ij

                disp_j = tj.dispersion_coefficient * 1e-50 if tj.dispersion_coefficient > 1e-20 else tj.dispersion_coefficient
                if disp_j == 0.0:
                    disp_j = math.exp(1.8846 * math.log(alphaA_j) - 0.4737) * 1e-50
                quad_j = (tj.quadrupole_polarizability * 1e-50 if tj.quadrupole_polarizability > 1e-20 else tj.quadrupole_polarizability) or (2.0 * disp_j)

                disp_i = ti.dispersion_coefficient * 1e-50 if ti.dispersion_coefficient > 1e-20 else ti.dispersion_coefficient
                if disp_i == 0.0:
                    if charge_i > 0:
                        disp_i = math.exp(1.8853 * math.log(alphaA_i) + 0.2682) * 1e-50
                    else:
                        disp_i = math.exp(3.2246 * math.log(alphaA_i) - 3.2397) * 1e-50

                c6 = 2.0 * disp_i * disp_j / ((1.0 / r_alpha) * disp_i + r_alpha * disp_j)
                gamma_ij = (2.0 / (charge_i**2) * c6 + quad_j) / (alpha_j * (diam_ij**2))
                self._gamma_n64[i, j] = self._gamma_n64[j, i] = gamma_ij

    def omega11_n64(self, tstar: float, gamma: float) -> float:
        """Collision integral of omega11 for n64 collision model."""
        tstar = float(tstar)
        gamma = float(gamma)
        if tstar < 0.01:
            raise CanteraError("IonGasTransport::omega11_n64", f"tstar = {tstar} is smaller than 0.01")
        logt = math.log(tstar)
        if tstar <= 0.04:
            return (2.97 - 12.0 * gamma - 0.887 * logt + 3.86 * (gamma**2)
                    - 6.45 * gamma * logt - 0.275 * (logt**2)
                    + 1.20 * (gamma**2) * logt - 1.24 * gamma * (logt**2)
                    - 0.164 * (logt**3))
        elif tstar <= 1000.0:
            return (1.22 - 0.0343 * gamma
                    + (-0.769 + 0.232 * gamma) * logt
                    + (0.306 - 0.165 * gamma) * (logt**2)
                    + (-0.0465 + 0.0388 * gamma) * (logt**3)
                    + (0.000614 - 0.00285 * gamma) * (logt**4)
                    + 0.000238 * (logt**5))
        else:
            raise CanteraError("IonGasTransport::omega11_n64", f"tstar = {tstar} is larger than 1000")

    @staticmethod
    def omega_11_n64(t_star, gamma):
        t_star = float(t_star)
        gamma = float(gamma)
        if t_star < 0.01:
            raise CanteraError("IonGasTransport::omega11_n64", f"tstar = {t_star} is smaller than 0.01")
        if t_star > 1000.0:
            raise CanteraError("IonGasTransport::omega11_n64", f"tstar = {t_star} is larger than 1000")
        logt = math.log(t_star)
        if t_star <= 0.04:
            return (2.97 - 12.0 * gamma - 0.887 * logt + 3.86 * (gamma**2)
                    - 6.45 * gamma * logt - 0.275 * (logt**2)
                    + 1.20 * (gamma**2) * logt - 1.24 * gamma * (logt**2)
                    - 0.164 * (logt**3))
        return (1.22 - 0.0343 * gamma
                + (-0.769 + 0.232 * gamma) * logt
                + (0.306 - 0.165 * gamma) * (logt**2)
                + (-0.0465 + 0.0388 * gamma) * (logt**3)
                + (0.000614 - 0.00285 * gamma) * (logt**4)
                + 0.000238 * (logt**5))

    def fitDiffCoeffs(self, integrals=None) -> None:
        """Fit polynomial coefficients to binary diffusion for ion-neutral pairs.
        
        Matches native Cantera IonGasTransport::fitDiffCoeffs:
        Samples 50 logarithmically spaced temperatures between minTemp and maxTemp,
        computes unit-pressure D_kj / T^1.5 using the Stockmayer-(n,6,4) model
        (or empirical O2/O2- fit), and fits degree-4 polynomials in log(T) with
        weights 1/diff^2.
        """
        np_pts = 50
        degree = 4
        t_min = float(self._thermo.min_temp)
        t_max = float(self._thermo.max_temp)
        dt = (t_max - t_min) / (np_pts - 1)
        t_vals = np.array([t_min + dt * n for n in range(np_pts)], dtype=float)
        tlog = np.log(t_vals)
        mw = np.asarray(self._thermo.molecular_weights, dtype=float)

        self._ion_diff_polys = {}
        for i in self._ions:
            for j in self._neutrals:
                sigma = self._diam_n64[i, j]
                epsilon = self._epsilon_n64[i, j]
                gamma = self._gamma_n64[i, j]
                if sigma <= 0.0 or epsilon <= 0.0:
                    continue
                m_ij = (mw[i] * mw[j]) / ((mw[i] + mw[j]) * Avogadro)
                name_i = self._thermo.species_name(i)
                name_j = self._thermo.species_name(j)

                diff = np.zeros(np_pts, dtype=float)
                for n, t in enumerate(t_vals):
                    if (name_i in ('O2-', 'O2^-') and name_j == 'O2') or (name_j in ('O2-', 'O2^-') and name_i == 'O2'):
                        c_p = self._om11_O2
                        om11 = (((((c_p[5]*t + c_p[4])*t + c_p[3])*t + c_p[2])*t + c_p[1])*t + c_p[0]) / 1e20
                    else:
                        tstar = Boltzmann * t / epsilon
                        om11 = self.omega11_n64(tstar, gamma) * Pi * (sigma**2)
                    diffcoeff = 3.0 / 16.0 * math.sqrt(2.0 * Pi / m_ij) * ((Boltzmann * t)**1.5) / om11
                    diff[n] = diffcoeff / (t**1.5)
                coeffs = _polyfit_householder_pivoted(tlog, diff, degree, weights=1.0 / np.abs(diff))
                self._ion_diff_polys[(i, j)] = coeffs
                self._ion_diff_polys[(j, i)] = coeffs

    def _ion_neutral_diffusion(self, ion, neutral):
        """Evaluate fitted polynomial binary diffusion coefficient [m^2/s] for ion-neutral pair."""
        poly = getattr(self, '_ion_diff_polys', {}).get((ion, neutral))
        if poly is not None:
            T = float(self._thermo.temperature)
            P = float(self._thermo.pressure)
            log_t = math.log(T)
            val = poly[0] + log_t * (poly[1] + log_t * (poly[2] + log_t * (poly[3] + log_t * poly[4])))
            return (T**1.5) * val / P

        # Fallback to direct Stockmayer-(n,6,4) evaluation if not fitted
        ti = self._transport_data.get(self._thermo.species_name(ion))
        tn = self._transport_data.get(self._thermo.species_name(neutral))
        if not ti or not tn or ti.polarizability <= 0.0 or tn.polarizability <= 0.0:
            return super().binary_diffusion_coefficients()[ion][neutral]
        name_i = self._thermo.species_name(ion)
        name_j = self._thermo.species_name(neutral)
        temperature = float(self._thermo.temperature)
        sigma = self._diam_n64[ion, neutral]
        epsilon = self._epsilon_n64[ion, neutral]
        gamma = self._gamma_n64[ion, neutral]

        if (name_i in ('O2-', 'O2^-') and name_j == 'O2') or (name_j in ('O2-', 'O2^-') and name_i == 'O2'):
            c_p = self._om11_O2
            om11 = (((((c_p[5]*temperature + c_p[4])*temperature + c_p[3])*temperature + c_p[2])*temperature + c_p[1])*temperature + c_p[0]) / 1e20
        else:
            t_star = Boltzmann * temperature / epsilon
            om11 = self.omega11_n64(t_star, gamma) * Pi * (sigma**2)

        mw = self._thermo.molecular_weights
        reduced_mass = (mw[ion] * mw[neutral] / ((mw[ion] + mw[neutral]) * Avogadro))
        numerator = (3.0 / 16.0) * math.sqrt(2.0 * Pi / reduced_mass) * ((Boltzmann * temperature)**1.5)
        return numerator / (self._thermo.pressure * om11)

    def binary_diffusion_coefficients(self):
        matrix = np.array(super().binary_diffusion_coefficients(), dtype=float)
        for ion in self._ions:
            for neutral in self._neutrals:
                value = self._ion_neutral_diffusion(ion, neutral)
                matrix[ion][neutral] = matrix[neutral][ion] = value
        return matrix

    @property
    def viscosity(self) -> CallableFloat:
        x = self._thermo.X
        mu = self.species_viscosities()
        mw = self._thermo.molecular_weights
        result = 0.0
        for k in self._neutrals:
            denom = 0.0
            for j in range(len(x)):
                phi = 1.0 if j == k else (
                    (1.0 + math.sqrt(mu[k] / mu[j]) * (mw[j] / mw[k]) ** 0.25) ** 2
                    / math.sqrt(8.0 * (1.0 + mw[k] / mw[j])))
                denom += x[j] * phi
            if denom > 0.0:
                result += x[k] * mu[k] / denom
        return CallableFloat(result)

    def thermalConductivity(self) -> CallableFloat:
        x = self._thermo.X
        cond = self.species_thermal_conductivities()
        sum1 = sum(x[k] * cond[k] for k in self._neutrals)
        sum2 = sum(x[k] / cond[k] for k in self._neutrals if cond[k] > 0.0)
        val = 0.5 * (sum1 + (1.0 / sum2 if sum2 > 0.0 else 0.0))
        return CallableFloat(val)

    @property
    def thermal_conductivity(self) -> CallableFloat:
        return self.thermalConductivity()

    def getMobilities(self, mobi=None):
        _check_array_size("IonGasTransport::getMobilities", "mobi", mobi, self._thermo.n_species) if mobi is not None else None
        x = np.maximum(self._thermo.X, 1.e-20)
        temperature = float(self._thermo.temperature)
        p = float(self._thermo.pressure)
        binary = self.binary_diffusion_coefficients()
        values = np.zeros(self._thermo.n_species, dtype=float)
        if self._electron is not None:
            values[self._electron] = 0.4
        for ion in self._ions:
            inverse = 0.0
            for neutral in self._neutrals:
                bmobi = binary[ion][neutral] * p * ElectronCharge / (Boltzmann * temperature)
                if bmobi > 0.0:
                    inverse += x[neutral] / bmobi
            values[ion] = 1.0 / inverse / p if inverse > 0.0 else 0.0
        if mobi is not None:
            mobi[:] = values
            return mobi
        return values

    def mobilities(self):
        return tuple(self.getMobilities())

    def getMixDiffCoeffs(self, d=None):
        _check_array_size("IonGasTransport::getMixDiffCoeffs", "d", d, self._thermo.n_species) if d is not None else None
        x = np.maximum(self._thermo.X, 1.e-20)
        mw = self._thermo.molecular_weights
        mean_mw = float(self._thermo.mean_molecular_weight)
        p = float(self._thermo.pressure)
        binary = self.binary_diffusion_coefficients()
        values = np.zeros(self._thermo.n_species, dtype=float)
        for k in range(len(x)):
            if k == self._electron:
                values[k] = 0.4 * Boltzmann * float(self._thermo.temperature) / ElectronCharge
                continue
            inverse = sum(x[j] / (p * binary[j][k]) for j in self._neutrals if j != k and binary[j][k] > 0.0)
            if inverse <= 0.0:
                values[k] = binary[k][k]
            else:
                values[k] = (mean_mw - x[k] * mw[k]) / (p * mean_mw * inverse)
        if d is not None:
            d[:] = values
            return d
        return values

    def mix_diff_coeffs(self):
        return tuple(self.getMixDiffCoeffs())

    def electricalConductivity(self) -> CallableFloat:
        x = np.maximum(self._thermo.X, 1.e-20)
        mobi = np.zeros(self._thermo.n_species, dtype=float)
        self.getMobilities(mobi)
        kbt = Boltzmann * float(self._thermo.temperature)
        p = float(self._thermo.pressure)
        nd_factor = p / kbt
        sum_elec = 0.0
        for ion in self._ions:
            sum_elec += x[ion] * nd_factor * abs(self._charges[ion]) * ElectronCharge * mobi[ion]
        if self._electron is not None:
            sum_elec += x[self._electron] * nd_factor * ElectronCharge * mobi[self._electron]
        return CallableFloat(sum_elec)

    @property
    def electrical_conductivity(self) -> CallableFloat:
        return self.electricalConductivity()



class MultiTransport(MixTransport):
    """Multicomponent transport properties for ideal gas mixtures.

    Sources: MultiTransport.cpp, GasTransport.cpp.

    Evaluates:
    - Multicomponent diffusion matrix D_ij by inverting the Stefan-Maxwell L00,00 system
    - Thermal conductivity and thermal diffusion coefficients (Soret effect)
      by solving the full 3K x 3K block L-matrix equation
    - Diffusive species mass fluxes satisfying exact mass conservation
    """

    _diff_fit_cache = {}
    _species_viscosity_fit_cache = {}
    _species_conductivity_fit_cache = {}
    _collision_star_fit_cache = {}

    def update_C(self) -> None:
        """Refresh composition state and invalidate composition-dependent solves."""
        x = np.maximum(np.asarray(self._thermo.X, dtype=float), Tiny)
        previous = getattr(self, "_molefracs", None)
        if previous is None or not np.array_equal(previous, x):
            self._molefracs = x.copy()
            self._l0000_ok = False
            self._lmatrix_soln_ok = False

    def updateThermal_T(self) -> None:
        """Refresh temperature-dependent transport terms using the Python model."""
        self._update_thermal()

    def eval_L0000(self, molefracs=None) -> None:
        """Evaluate the translational L block used by multicomponent diffusion."""
        block = self._build_L0000(molefracs)
        self._ensure_lmatrix()
        nsp = self._thermo.n_species
        self._Lmatrix[:nsp, :nsp] = block

    def eval_L0010(self, molefracs=None) -> None:
        """Evaluate the translational/thermal cross block of the L matrix."""
        self._update_thermal()
        x = self._get_molefracs(molefracs)
        nsp = self._thermo.n_species
        mw = np.asarray(self._thermo.molecular_weights)
        block = self._Lmatrix[:nsp, nsp:2*nsp]
        prefactor = 1.6 * self._thermo.temperature
        for j in range(nsp):
            xj, wj, total = x[j], mw[j], 0.0
            for i in range(nsp):
                block[i, j] = (-prefactor * x[i] * xj * mw[i]
                               * (1.2 * self._cstar[j, i] - 1.0)
                               / ((wj + mw[i]) * self._m_bdiff[j, i]))
                total -= block[i, j]
            block[j, j] += total

    def eval_L1000(self) -> None:
        """Evaluate the transpose block L10,00 = L00,10 transpose."""
        nsp = self._thermo.n_species
        self._Lmatrix[nsp:2*nsp, :nsp] = self._Lmatrix[:nsp, nsp:2*nsp].T

    def eval_L1010(self, molefracs=None) -> None:
        """Evaluate the translational/thermal diagonal block."""
        self._update_thermal()
        x = self._get_molefracs(molefracs)
        nsp = self._thermo.n_species
        mw, T = np.asarray(self._thermo.molecular_weights), self._thermo.temperature
        block = self._Lmatrix[nsp:2*nsp, nsp:2*nsp]
        fiveover3pi = 5.0 / (3.0 * Pi)
        prefactor = 16.0 * T / 25.0
        for j in range(nsp):
            constant1 = prefactor * x[j]
            wjsq = mw[j] * mw[j]
            constant2 = 13.75 * wjsq
            constant3 = self._crot[j] / self._rotrelax[j]
            constant4 = 7.5 * wjsq
            fourmj = 4.0 * mw[j]
            threemjsq = 3.0 * mw[j] * mw[j]
            total = 0.0
            for i in range(nsp):
                sumwij = mw[i] + mw[j]
                term1 = self._m_bdiff[i, j] * sumwij * sumwij
                term2 = (fourmj * self._astar[i, j]
                         * (1.0 + fiveover3pi
                            * (constant3 + self._crot[i] / self._rotrelax[i])))
                block[i, j] = (constant1 * x[i] * mw[i]
                               / (mw[j] * term1)
                               * (constant2 - threemjsq * self._bstar[i, j]
                                  - term2 * mw[j]))
                total += (x[i] / term1
                          * (constant4 + mw[i] * mw[i]
                             * (6.25 - 3.0 * self._bstar[i, j])
                             + term2 * mw[i]))
            block[j, j] -= total * constant1

    def eval_L1001(self, molefracs=None) -> None:
        """Evaluate coupling from translational to internal energy modes."""
        self._update_thermal()
        x = self._get_molefracs(molefracs)
        nsp = self._thermo.n_species
        mw, T = np.asarray(self._thermo.molecular_weights), self._thermo.temperature
        block = self._Lmatrix[nsp:2*nsp, 2*nsp:3*nsp]
        prefactor = 32.0 * T / (5.0 * Pi)
        for j in range(nsp):
            if self.hasInternalModes(j):
                constant = (prefactor * mw[j] * x[j] * self._crot[j]
                            / (self._cinternal[j] * self._rotrelax[j]))
                total = 0.0
                for i in range(nsp):
                    block[i, j] = (constant * self._astar[j, i] * x[i]
                                   / ((mw[j] + mw[i]) * self._m_bdiff[j, i]))
                    total += block[i, j]
                block[j, j] += total
            else:
                block[:, j] = 0.0

    def eval_L0001(self) -> None:
        """Set the uncoupled translational/internal-energy block to zero."""
        nsp = self._thermo.n_species
        self._Lmatrix[:nsp, 2*nsp:3*nsp] = 0.0

    def eval_L0100(self) -> None:
        """Set the reverse uncoupled internal/translational block to zero."""
        nsp = self._thermo.n_species
        self._Lmatrix[2*nsp:3*nsp, :nsp] = 0.0

    def eval_L0110(self) -> None:
        """Evaluate the transpose of the translational/internal coupling block."""
        nsp = self._thermo.n_species
        self._Lmatrix[2*nsp:3*nsp, nsp:2*nsp] = (
            self._Lmatrix[nsp:2*nsp, 2*nsp:3*nsp].T)

    def eval_L0101(self, molefracs=None) -> None:
        """Evaluate the internal-energy mode block of the transport matrix."""
        self._update_thermal()
        x = self._get_molefracs(molefracs)
        nsp = self._thermo.n_species
        mw, T = np.asarray(self._thermo.molecular_weights), self._thermo.temperature
        block = self._Lmatrix[2*nsp:3*nsp, 2*nsp:3*nsp]
        for i in range(nsp):
            if self.hasInternalModes(i):
                constant1 = 4.0 * T * x[i] / self._cinternal[i]
                constant2 = (12.0 * mw[i] * self._crot[i]
                             / (5.0 * Pi * self._cinternal[i]
                                * self._rotrelax[i]))
                total = 0.0
                for k in range(nsp):
                    block[k, i] = 0.0
                    diff_int = self._m_bdiff[i, k]
                    total += x[k] / diff_int
                    if k != i:
                        total += (x[k] * self._astar[i, k] * constant2
                                  / (mw[k] * diff_int))
                block[i, i] = (
                    -8.0 / Pi * mw[i] * x[i] * x[i] * self._crot[i]
                    / (self._cinternal[i] * self._cinternal[i]
                       * GasConstant * self._visc_cached[i]
                       * self._rotrelax[i]) - constant1 * total)
            else:
                block[i, :] = 0.0
                block[i, i] = 1.0

    def _ensure_lmatrix(self) -> None:
        n = 3 * self._thermo.n_species
        if not hasattr(self, "_Lmatrix") or self._Lmatrix.shape != (n, n):
            self._Lmatrix = np.zeros((n, n))

    def _get_molefracs(self, molefracs=None) -> np.ndarray:
        if molefracs is None:
            self.update_C()
            return self._molefracs
        return np.maximum(np.asarray(molefracs, dtype=float), Tiny)

    def hasInternalModes(self, species_index: int) -> bool:
        """Return whether a species has the source-defined internal heat modes."""
        self._update_thermal()
        return bool(self._has_internal[species_index])

    def pressure_ig(self) -> float:
        """Ideal-gas pressure computed from the current molar density and state."""
        return float(self._thermo.molar_density * GasConstant
                     * self._thermo.temperature)

    def solveLMatrixEquation(self) -> None:
        """Solve the complete Python-assembled 3N-by-3N transport system."""
        self._solve_L_matrix()

    def __init__(self, thermo, transport_data: Mapping[str, GasTransportData] | None = None):
        super().__init__(thermo, transport_data)
        self._thermal_tlast = None
        self._l0000_ok = False
        self._lmatrix_soln_ok = False
        self._molefracs = None
        self._L0000 = None
        self._L0000_inv = None
        self._L_soln_a = None
        self._L_soln_b = None

        # Precompute constants
        nsp = self._thermo.n_species
        self._crot = np.zeros(nsp)
        self._zrot = np.zeros(nsp)
        self._well = np.zeros(nsp)
        self._diam = np.zeros(nsp)
        self._dipole = np.zeros(nsp)
        self._frot_298 = np.zeros(nsp)

        sq298 = math.sqrt(298.0)
        for k, s in enumerate(self._thermo.species):
            tdata = self._transport_data.get(s.name, GasTransportData())
            if tdata.geometry == 'atom':
                self._crot[k] = 0.0
            elif tdata.geometry == 'linear':
                self._crot[k] = 1.0
            else:
                self._crot[k] = 1.5
            self._zrot[k] = float(tdata.rotational_relaxation)
            self._well[k] = float(tdata.well_depth)
            self._diam[k] = float(tdata.diameter)
            self._dipole[k] = float(tdata.dipole)

            eps_k = self._well[k]
            self._frot_298[k] = Frot(eps_k / 298.0, math.sqrt(eps_k) / sq298)

        # Pair matrices with polar corrections (GasTransport::setupCollisionIntegral)
        self._diam_matrix = np.zeros((nsp, nsp))
        self._eps_matrix = np.zeros((nsp, nsp))
        self._delta_matrix = np.zeros((nsp, nsp))

        for i in range(nsp):
            for j in range(i, nsp):
                sigma_i = self._diam[i] * 1e-10
                sigma_j = self._diam[j] * 1e-10
                d_ij = 0.5 * (sigma_i + sigma_j)
                eps_i = self._well[i] * Boltzmann
                eps_j = self._well[j] * Boltzmann
                eps_ij = math.sqrt(eps_i * eps_j)
                dip_i = self._dipole[i] * (1e-21 / lightSpeed)
                dip_j = self._dipole[j] * (1e-21 / lightSpeed)
                dip_ij = math.sqrt(dip_i * dip_j)
                delta_ij = 0.5 * (dip_ij ** 2) / (4.0 * Pi * epsilon_0 * eps_ij * (d_ij ** 3))

                polar_i = self._dipole[i] > 0
                polar_j = self._dipole[j] > 0
                if polar_i != polar_j:
                    kp = i if polar_i else j
                    knp = j if polar_i else i
                    sigma_kp = self._diam[kp] * 1e-10
                    sigma_knp = self._diam[knp] * 1e-10
                    eps_kp = self._well[kp] * Boltzmann
                    eps_knp = self._well[knp] * Boltzmann
                    d3np = sigma_knp ** 3
                    d3p = sigma_kp ** 3
                    tdata_knp = self._transport_data.get(self._thermo.species[knp].name, GasTransportData())
                    alpha_knp = tdata_knp.polarizability * 1e-30
                    alpha_star = alpha_knp / d3np
                    mu_p = self._dipole[kp] * (1e-21 / lightSpeed)
                    mu_p_star = mu_p / math.sqrt(4.0 * Pi * epsilon_0 * d3p * eps_kp)
                    xi = 1.0 + 0.25 * alpha_star * (mu_p_star ** 2) * math.sqrt(eps_kp / eps_knp)
                    f_sigma = xi ** (-1.0 / 6.0)
                    f_eps = xi * xi
                    d_ij *= f_sigma
                    eps_ij *= f_eps

                self._diam_matrix[i, j] = self._diam_matrix[j, i] = d_ij
                self._eps_matrix[i, j] = self._eps_matrix[j, i] = eps_ij
                self._delta_matrix[i, j] = self._delta_matrix[j, i] = delta_ij

        # GasTransport::fitDiffCoeffs fits unit-pressure binary diffusion
        # coefficients at 50 evenly spaced temperatures. Native Cantera then
        # evaluates these fourth-order polynomials at runtime. Reproducing the
        # fit matters for numerical parity; evaluating the collision integral
        # directly gives a slightly different, although physically valid,
        # answer from the native runtime.
        temperatures = np.linspace(
            self._thermo.min_temp, self._thermo.max_temp, 50)
        log_temperatures = np.log(temperatures)
        mw = np.asarray(self._thermo.molecular_weights, dtype=float)
        fit_key = (float(temperatures[0]), float(temperatures[-1]),
                   mw.tobytes(), self._diam_matrix.tobytes(),
                   self._eps_matrix.tobytes(), self._delta_matrix.tobytes())

        # Match GasTransport::setupCollisionIntegral and
        # GasTransport::fitCollisionIntegrals. Cantera bounds the T* grid over
        # every binary pair, then fits A*, B*, and C* as degree-eight
        # polynomials in log(T*), reusing a fit for identical delta* values.
        tstar_min = float(np.min(
            Boltzmann * self._thermo.min_temp / self._eps_matrix))
        tstar_max = float(np.max(
            Boltzmann * self._thermo.max_temp / self._eps_matrix))
        first, last = -1, -1
        for n, tstar in enumerate(_tstar_table[1:38]):
            if tstar_min > tstar:
                first = n
            if tstar_max > tstar:
                last = n + 1
        if first < 0 or first >= 36 or last < 0 or last > 36:
            first, last = 0, 36
        star_fit_key = (fit_key, first, last)
        cached_star_fits = self._collision_star_fit_cache.get(star_fit_key)
        if cached_star_fits is None:
            star_fits = [[[None for _ in range(nsp)] for _ in range(nsp)]
                         for _ in range(3)]
            unique_fits = {}
            for i in range(nsp):
                for j in range(i, nsp):
                    delta = float(self._delta_matrix[i, j])
                    coeffs = unique_fits.get(delta)
                    if coeffs is None:
                        coeffs = _fit_collision_star_polynomials(delta, first, last)
                        unique_fits[delta] = coeffs
                    for block, coeff in zip(star_fits, coeffs):
                        block[i][j] = coeff
                        block[j][i] = coeff
            cached_star_fits = tuple(star_fits)
            self._collision_star_fit_cache[star_fit_key] = cached_star_fits
        self._astar_fit_polys, self._bstar_fit_polys, self._cstar_fit_polys = cached_star_fits

        cached = self._diff_fit_cache.get(fit_key)
        if cached is not None:
            self._diff_coeff_polys = cached
        else:
            self._diff_coeff_polys = [[None] * nsp for _ in range(nsp)]
            for i in range(nsp):
                for j in range(i, nsp):
                    m_ij = (mw[i] * mw[j]) / ((mw[i] + mw[j]) * Avogadro)
                    values = np.empty(temperatures.size)
                    for n, temperature in enumerate(temperatures):
                        tstar = Boltzmann * temperature / self._eps_matrix[i, j]
                        astar = _eval_star(
                            _astar_table, _apoly, tstar, self._delta_matrix[i, j])
                        omega22 = _eval_omega22(tstar, self._delta_matrix[i, j])
                        omega11 = omega22 / astar
                        unit_pressure_diff = ((3.0 / 16.0)
                            * math.sqrt(2.0 * Pi / m_ij)
                            * (Boltzmann * temperature) ** 1.5
                            / (Pi * self._diam_matrix[i, j] ** 2 * omega11))
                        values[n] = unit_pressure_diff / temperature ** 1.5
                    coeffs = _polyfit_householder_pivoted(
                        log_temperatures, values, 4,
                        weights=1.0 / np.abs(values))
                    self._diff_coeff_polys[i][j] = coeffs
                    self._diff_coeff_polys[j][i] = coeffs
            self._diff_fit_cache[fit_key] = self._diff_coeff_polys
        try:
            self._diff_coeff_polys_matrix = np.array(self._diff_coeff_polys, dtype=float)
        except Exception:
            self._diff_coeff_polys_matrix = None

        # Cantera fits sqrt(mu/sqrt(T)) from polar-corrected Omega(2,2)
        # values on 50 temperatures. The simpler MixTransport correlation
        # omits dipole corrections and is not suitable for this solver.
        viscosity_fit = self._species_viscosity_fit_cache.get(fit_key)
        if viscosity_fit is None:
            viscosity_fit = []
            for k in range(nsp):
                values = np.empty(temperatures.size)
                sigma = self._diam[k] * 1e-10
                for n, temperature in enumerate(temperatures):
                    tstar = temperature / self._well[k]
                    omega22 = _eval_omega22(tstar, self._delta_matrix[k, k])
                    mu = ((5.0 / 16.0)
                          * math.sqrt(Pi * (mw[k] / Avogadro)
                                      * Boltzmann * temperature)
                          / (omega22 * Pi * sigma * sigma))
                    values[n] = math.sqrt(mu / math.sqrt(temperature))
                viscosity_fit.append(_polyfit_householder_pivoted(
                    log_temperatures, values, 4,
                    # GasTransport::fitProperties passes w=1/value^2 to
                    # Cantera's polyfit, which applies sqrt(w) to residuals.
                    # NumPy applies its supplied w directly to residuals.
                    weights=1.0 / values))
            viscosity_fit = tuple(viscosity_fit)
            self._species_viscosity_fit_cache[fit_key] = viscosity_fit
        self._species_viscosity_polys = viscosity_fit

        # GasTransport::fitProperties fits lambda/sqrt(T) using the same
        # weighted degree-four polynomial method as its viscosity fit. Species
        # reference heat capacities are evaluated directly so this fit does
        # not mutate the temperature of the shared phase object.
        if all(hasattr(getattr(species, "thermo", None), "properties")
               for species in self._thermo.species):
            cp_samples = tuple(
                tuple(float(species.thermo.properties(float(t))[0])
                      for t in temperatures)
                for species in self._thermo.species)
        else:
            # Lightweight transport-only thermo doubles may omit reference
            # species thermo. Their fallback has no internal heat modes.
            cp_samples = tuple((2.5,) * temperatures.size
                               for _ in self._thermo.species)
        conductivity_key = (fit_key, cp_samples,
                            tuple(float(value) for value in self._crot),
                            tuple(float(value) for value in self._zrot))
        conductivity_fit = self._species_conductivity_fit_cache.get(
            conductivity_key)
        if conductivity_fit is None:
            conductivity_fit = []
            for k in range(nsp):
                values = np.empty(temperatures.size)
                sigma = self._diam[k] * 1e-10
                eps_k = self._well[k]
                fz_298 = Frot(eps_k / 298.0,
                              math.sqrt(eps_k) / math.sqrt(298.0))
                reduced_mass = mw[k] / (2.0 * Avogadro)
                for n, temperature in enumerate(temperatures):
                    tstar = temperature / eps_k
                    delta = self._delta_matrix[k, k]
                    astar = _eval_star(_astar_table, _apoly, tstar, delta)
                    omega22 = _eval_omega22(tstar, delta)
                    omega11 = omega22 / astar
                    diffcoeff = ((3.0 / 16.0)
                        * math.sqrt(2.0 * Pi / reduced_mass)
                        * (Boltzmann * temperature) ** 1.5
                        / (Pi * sigma * sigma * omega11))
                    viscosity = ((5.0 / 16.0)
                        * math.sqrt(Pi * mw[k] * Boltzmann * temperature
                                    / Avogadro)
                        / (omega22 * Pi * sigma * sigma))
                    fint = (mw[k] / (GasConstant * temperature)
                            * diffcoeff / viscosity)
                    cv_rot = self._crot[k]
                    tr = eps_k / temperature
                    fz_tstar = Frot(tr, math.sqrt(tr))
                    bfactor = (self._zrot[k] * fz_298 / fz_tstar
                               + 2.0 / Pi
                               * (5.0 / 3.0 * cv_rot + fint))
                    c1 = 2.0 / Pi * (2.5 - fint) / bfactor
                    cv_int = cp_samples[k][n] - 2.5 - cv_rot
                    frot = fint * (1.0 + c1)
                    ftrans = 2.5 * (1.0 - c1 * cv_rot / 1.5)
                    conductivity = ((viscosity / mw[k]) * GasConstant
                        * (ftrans * 1.5 + frot * cv_rot + fint * cv_int))
                    values[n] = conductivity / math.sqrt(temperature)
                conductivity_fit.append(_polyfit_householder_pivoted(
                    log_temperatures, values, 4,
                    weights=1.0 / np.abs(values)))
            conductivity_fit = tuple(conductivity_fit)
            self._species_conductivity_fit_cache[conductivity_key] = (
                conductivity_fit)
        self._species_thermal_conductivity_polys = conductivity_fit

    @property
    def transport_model(self) -> str:
        return "multicomponent"

    def species_viscosities(self) -> tuple[float, ...]:
        """Evaluate Cantera-fitted pure-species viscosities at the current T."""
        log_t = math.log(self._thermo.temperature)
        root_t = math.sqrt(self._thermo.temperature)
        return tuple(root_t * np.polynomial.polynomial.polyval(log_t, coeffs) ** 2
                     for coeffs in self._species_viscosity_polys)

    def species_thermal_conductivities(self) -> tuple[float, ...]:
        """Evaluate Cantera-fitted pure-species conductivity at current T."""
        temperature = self._thermo.temperature
        log_t = math.log(temperature)
        root_t = math.sqrt(temperature)
        return tuple(root_t * np.polynomial.polynomial.polyval(log_t, coeffs)
                     for coeffs in self._species_thermal_conductivity_polys)

    @property
    def CK_mode(self) -> bool:
        return False

    def _eval_binary_diff_coeffs(self) -> np.ndarray:
        """Evaluate raw binary diffusion coefficients from polynomial fits [m^2/s]."""
        T = self._thermo.temperature
        P = self._thermo.pressure
        log_t = math.log(T)
        prefactor = T ** 1.5 / P
        if getattr(self, "_diff_coeff_polys_matrix", None) is not None:
            c = self._diff_coeff_polys_matrix
            d = c[:, :, 0] + log_t * (c[:, :, 1] + log_t * (c[:, :, 2] + log_t * (c[:, :, 3] + log_t * c[:, :, 4])))
            return prefactor * d
        nsp = self._thermo.n_species
        bdiff = np.zeros((nsp, nsp))
        for i in range(nsp):
            for j in range(i, nsp):
                d_ij = prefactor * np.polynomial.polynomial.polyval(
                    log_t, self._diff_coeff_polys[i][j])
                bdiff[i, j] = bdiff[j, i] = d_ij
        return bdiff

    def binary_diffusion_coefficients(self) -> np.ndarray:
        """Matrix of binary diffusion coefficients D_jk [m^2/s]."""
        self._update_thermal()
        return self._m_bdiff / self._thermo.pressure

    def _update_thermal(self):
        T = self._thermo.temperature
        if self._thermal_tlast == T:
            return

        nsp = self._thermo.n_species
        P = self._thermo.pressure
        visc = np.array(self.species_viscosities())
        bdiff = self._eval_binary_diff_coeffs()
        mw = np.array(self._thermo.molecular_weights)

        # m_bdiff is P * bdiff (binary diffusion at unit pressure in native Cantera)
        m_bdiff = P * bdiff

        # Parker temperature correction to rotational relaxation
        sqt = math.sqrt(T)
        rotrelax = np.zeros(nsp)
        for k in range(nsp):
            eps_k = self._well[k]
            tr = eps_k / T
            sqtr = math.sqrt(eps_k) / sqt
            rotrelax[k] = max(1.0, self._zrot[k]) * self._frot_298[k] / Frot(tr, sqtr)

        # Dimensionless internal heat capacity
        cp_mole = getattr(self._thermo, "partial_molar_cp", None)
        cinternal = np.zeros(nsp)
        if cp_mole is not None:
            for k in range(nsp):
                cinternal[k] = cp_mole[k] / GasConstant - 2.5
        has_internal = (cinternal > 0.001)

        # Collision integrals A*, B*, C*
        astar = np.zeros((nsp, nsp))
        bstar = np.zeros((nsp, nsp))
        cstar = np.zeros((nsp, nsp))

        for i in range(nsp):
            for j in range(i, nsp):
                eps_ij = self._eps_matrix[i, j]
                ts = Boltzmann * T / eps_ij
                log_tstar = math.log(ts)
                a = np.polynomial.polynomial.polyval(
                    log_tstar, self._astar_fit_polys[i][j])
                b = np.polynomial.polynomial.polyval(
                    log_tstar, self._bstar_fit_polys[i][j])
                c = np.polynomial.polynomial.polyval(
                    log_tstar, self._cstar_fit_polys[i][j])
                astar[i, j] = astar[j, i] = a
                bstar[i, j] = bstar[j, i] = b
                cstar[i, j] = cstar[j, i] = c

        # Self-diffusion on diagonal: m_bdiff(k, k) = 1.2 * R * T * visc[k] * astar[k, k] / mw[k]
        for k in range(nsp):
            m_bdiff[k, k] = 1.2 * GasConstant * T * visc[k] * astar[k, k] / mw[k]

        self._visc_cached = visc
        self._m_bdiff = m_bdiff
        self._rotrelax = rotrelax
        self._cinternal = cinternal
        self._has_internal = has_internal
        self._astar = astar
        self._bstar = bstar
        self._cstar = cstar
        self._thermal_tlast = T
        self._l0000_ok = False
        self._lmatrix_soln_ok = False

    def _build_L0000(self, molefracs=None):
        self._update_thermal()
        nsp = self._thermo.n_species
        T = self._thermo.temperature
        mw = np.array(self._thermo.molecular_weights)
        if molefracs is None:
            self.update_C()
            X = self._molefracs
        else:
            X = np.array([max(Tiny, x) for x in molefracs])
        m_bdiff = self._m_bdiff

        L0000 = np.zeros((nsp, nsp))
        pref_0000 = 16.0 * T / 25.0

        for i in range(nsp):
            # Match Cantera's source ordering: subtract the i-th term, then
            # add the complete row including that term.
            s = -X[i] / m_bdiff[i, i]
            for k in range(nsp):
                s += X[k] / m_bdiff[i, k]
            s /= mw[i]
            for j in range(nsp):
                L0000[i, j] = pref_0000 * X[j] * (mw[j] * s + X[i] / m_bdiff[i, j])
            L0000[i, i] = 0.0

        self._L0000 = L0000
        self._l0000_ok = True
        return L0000

    def _eval_L0000(self, molefracs=None):
        """Build and invert the L00 block for the public diffusion matrix."""
        L0000 = self._build_L0000(molefracs)
        self._L0000_inv = _invert_dense_partial_pivot(
            L0000, "MultiTransport::eval_L0000")

    @property
    def multi_diff_coeffs(self) -> np.ndarray:
        """Multicomponent diffusion coefficients matrix [m^2/s]."""
        self._eval_L0000()
        nsp = self._thermo.n_species
        T = self._thermo.temperature
        P = self._thermo.pressure
        mw = np.array(self._thermo.molecular_weights)
        X = np.array([max(Tiny, x) for x in self._thermo.X])
        mean_mw = self._thermo.mean_molecular_weight

        pref_D = 16.0 * T * mean_mw / (25.0 * P)
        Dmulti = np.zeros((nsp, nsp))
        for i in range(nsp):
            for j in range(nsp):
                c = pref_D / mw[j]
                Dmulti[i, j] = c * X[i] * (self._L0000_inv[i, j] - self._L0000_inv[i, i])
                if i == j:
                    Dmulti[i, j] = 0.0
        return Dmulti

    def getMultiDiffCoeffs(self, ld: int, d) -> None:
        """Get the multicomponent diffusion coefficients."""
        nsp = self._thermo.n_species
        procedure = "MultiTransport::getMultiDiffCoeffs"
        if ld < nsp:
            raise CanteraError(procedure, "ld is too small")
        _check_array_size(procedure, "diffusion coefficients", d, ld * nsp)
        d_mat = self.multi_diff_coeffs
        for i in range(nsp):
            for j in range(nsp):
                d[ld * j + i] = d_mat[i, j]

    def _solve_L_matrix(self):
        self.updateThermal_T()
        self.update_C()
        if self._lmatrix_soln_ok:
            return
        nsp = self._thermo.n_species
        X = self._molefracs
        self._ensure_lmatrix()
        self._Lmatrix.fill(0.0)
        self.eval_L0000(X)
        self.eval_L0010(X)
        self.eval_L0001()
        self.eval_L1000()
        self.eval_L1010(X)
        self.eval_L1001(X)
        self.eval_L0100()
        self.eval_L0110()
        self.eval_L0101(X)

        # Assemble the right-hand side using the same three-block ordering.
        b = np.zeros(3 * nsp)
        for k in range(nsp):
            b[k + nsp] = X[k]
            b[k + 2*nsp] = X[k]
            if not self.hasInternalModes(k):
                b[k + 2*nsp] = 0.0

        self._L_soln_a = _solve_dense_partial_pivot(
            self._Lmatrix, b, "MultiTransport::solveLMatrixEquation")
        self._L_soln_b = b
        self._a = self._L_soln_a
        self._b = self._L_soln_b
        self._molefracs_last = X.copy()
        self._lmatrix_soln_ok = True

    @property
    def thermal_conductivity(self) -> float:
        """Multicomponent thermal conductivity [W/m/K]."""
        self._solve_L_matrix()
        nsp = self._thermo.n_species
        b = self._L_soln_b
        a = self._L_soln_a
        return float(-4.0 * np.sum(b[nsp : 3*nsp] * a[nsp : 3*nsp]))

    @property
    def thermal_diff_coeffs(self) -> np.ndarray:
        """Thermal diffusion coefficients [kg/m/s]."""
        self._solve_L_matrix()
        nsp = self._thermo.n_species
        mw = np.array(self._thermo.molecular_weights)
        X = np.array([max(Tiny, x) for x in self._thermo.X])
        a = self._L_soln_a
        c_dt = 1.6 / GasConstant
        dt = np.zeros(nsp)
        for k in range(nsp):
            dt[k] = c_dt * mw[k] * X[k] * a[k]
        return dt

    def getThermalDiffCoeffs(self, dt) -> None:
        """Return the thermal diffusion coefficients [kg/m/s]."""
        coeffs = self.thermal_diff_coeffs
        for k, v in enumerate(coeffs):
            dt[k] = v

    def getSpeciesFluxes(self, ndim: int, grad_T, ldx: int, grad_X, ldf: int, fluxes) -> None:
        """Diffusive mass fluxes [kg/m^2/s] via Stefan-Maxwell solver."""
        nsp = self._thermo.n_species
        procedure = "MultiTransport::getSpeciesFluxes"
        if ldx < nsp:
            raise CanteraError(procedure, "ldx is too small")
        if ldf < nsp:
            raise CanteraError(procedure, "ldf is too small")
        _check_array_size(procedure, "grad_T", grad_T, ndim)
        _check_array_size(procedure, "grad_X", grad_X, ldx * ndim)
        _check_array_size(procedure, "fluxes", fluxes, ldf * ndim)
        self._update_thermal()
        T = self._thermo.temperature
        P = self._thermo.pressure
        rho = self._thermo.density
        y = np.array(self._thermo.Y)
        X = np.array([max(Tiny, x) for x in self._thermo.X])
        m_bdiff = self._m_bdiff

        # Check if thermal diffusion needed
        add_thermal = any(grad_T[n] != 0.0 for n in range(ndim))
        dt = self.thermal_diff_coeffs if add_thermal else None

        aa = np.zeros((nsp, nsp))
        for i in range(nsp):
            s = 0.0
            for j in range(nsp):
                aa[i, j] = X[j] * X[i] / m_bdiff[i, j]
                s += aa[i, j]
            aa[i, i] -= s

        # Replace row of largest |grad_X| in 1st coordinate direction with mass fractions
        jmax = 0
        gradmax = -1.0
        for j in range(nsp):
            val = abs(grad_X[j])
            if val > gradmax:
                gradmax = val
                jmax = j

        for j in range(nsp):
            aa[jmax, j] = y[j]
        self._aa = aa

        for n in range(ndim):
            rhs = np.array([grad_X[n * ldx + k] for k in range(nsp)])
            rhs[jmax] = 0.0
            sol = _solve_dense_partial_pivot(
                aa, rhs, "MultiTransport::getSpeciesFluxes")
            for k in range(nsp):
                fluxes[n * ldf + k] = sol[k] * rho * y[k] / P
            if add_thermal:
                grad_logt = grad_T[n] / T
                for k in range(nsp):
                    fluxes[n * ldf + k] -= dt[k] * grad_logt

    def getMassFluxes(self, state1, state2, delta: float, fluxes) -> None:
        """Diffusional mass fluxes [kg/m^2/s] between two states."""
        nsp = self._thermo.n_species
        mw = np.array(self._thermo.molecular_weights)
        T1 = float(state1[0])
        T2 = float(state2[0])

        if len(state1) == nsp + 2:
            # Layout [T, rho, Y_0, ..., Y_nsp-1]
            y1 = np.array(state1[2:])
            y2 = np.array(state2[2:])
            x1 = y1 / mw
            x1 /= np.sum(x1)
            x2 = y2 / mw
            x2 /= np.sum(x2)
            rho1 = float(state1[1])
            rho2 = float(state2[1])
            mean_mw1 = 1.0 / np.sum(y1 / mw)
            mean_mw2 = 1.0 / np.sum(y2 / mw)
            p1 = rho1 * (GasConstant / mean_mw1) * T1
            p2 = rho2 * (GasConstant / mean_mw2) * T2
        else:
            # Layout [T, P, X_0, ..., X_nsp-1]
            p1 = float(state1[1])
            p2 = float(state2[1])
            x1 = np.array(state1[2:2 + nsp])
            x2 = np.array(state2[2:2 + nsp])

        T_avg = 0.5 * (T1 + T2)
        P_avg = 0.5 * (p1 + p2)
        x_avg = 0.5 * (x1 + x2)

        # MultiTransport.cpp restores state1 and state2 to obtain their
        # endpoint mole fractions, then sets the phase to the midpoint state
        # before evaluating transport. The native call leaves the phase at
        # this midpoint (including when a later flux calculation raises), so
        # do not restore the caller's unrelated prior state here.
        if hasattr(self._thermo, "setState_TPX"):
            self._thermo.setState_TPX(T_avg, P_avg, x_avg)
        else:
            self._thermo.TPX = T_avg, P_avg, x_avg
        grad_T = np.array([(T2 - T1) / delta])
        grad_X = (x2 - x1) / delta
        self.getSpeciesFluxes(1, grad_T, nsp, grad_X, nsp, fluxes)

    def getMolarFluxes(self, state1, state2, delta: float, fluxes) -> None:
        """Diffusional molar fluxes [kmol/m^2/s] between two states."""
        self.getMassFluxes(state1, state2, delta, fluxes)
        mw = self._thermo.molecular_weights
        for k in range(self._thermo.n_species):
            fluxes[k] /= mw[k]


class DustyGasTransport(Transport):
    """Species transport through a stationary porous medium.

    This is a direct implementation of the pinned ``DustyGasTransport.cpp``.
    A :class:`MultiTransport` object supplies gas-phase binary diffusion
    coefficients and viscosity. The porous-medium resistance matrix includes
    molecular and Knudsen diffusion, while molar fluxes additionally include
    the Darcy pressure-gradient term.
    """

    def __init__(self, thermo=None, gas_transport: MultiTransport | None = None):
        if thermo is not None:
            self.initialize(thermo, gas_transport)

    def initialize(self, phase, gastr=None) -> None:
        """Initialize with a phase and optional subordinate gas transport model."""
        super().init(phase)
        self._thermo = phase
        self._nsp = phase.n_species
        self._mw = np.asarray(phase.molecular_weights, dtype=float)
        self._gastran = gastr or (phase.transport if getattr(phase, "transport", None) is not None else MultiTransport(phase))
        self._porosity = 0.0
        self._tortuosity = 1.0
        self._pore_radius = 0.0
        self._diam = 0.0
        self._perm = -1.0
        self._temp = phase.temperature

        self._multidiff = np.zeros((self._nsp, self._nsp), dtype=float)
        self._d = np.zeros((self._nsp, self._nsp), dtype=float)
        self._dk = np.zeros(self._nsp, dtype=float)
        self._x = np.asarray(phase.X, dtype=float).copy()

        self._knudsen_ok = False
        self._bulk_ok = False

    @classmethod
    def from_mechanism(cls, mech_or_path, thermo=None, phase_name=None):
        gas = MultiTransport.from_mechanism(
            mech_or_path, thermo=thermo, phase_name=phase_name)
        return cls(gas.thermo, gas)

    @property
    def thermo(self):
        return self._thermo

    @property
    def transport_model(self) -> str:
        return "DustyGas"

    def transportModel(self) -> str:
        return "DustyGas"

    @property
    def CK_mode(self) -> bool:
        return False

    @property
    def gas_transport(self) -> MultiTransport:
        return self._gastran

    def gasTransport(self) -> MultiTransport:
        return self._gastran

    def setPorosity(self, porosity: float) -> None:
        self._porosity = float(porosity)
        self._knudsen_ok = False
        self._bulk_ok = False

    def setTortuosity(self, tort: float) -> None:
        self._tortuosity = float(tort)
        self._knudsen_ok = False
        self._bulk_ok = False

    def setMeanPoreRadius(self, rbar: float) -> None:
        self._pore_radius = float(rbar)
        self._knudsen_ok = False

    def setMeanParticleDiameter(self, dbar: float) -> None:
        self._diam = float(dbar)

    def setPermeability(self, B: float) -> None:
        self._perm = float(B)



    @property
    def thermal_conductivity(self) -> float:
        return self._gastran.thermal_conductivity

    def updateTransport_T(self) -> None:
        if self._temp == self._thermo.temperature:
            return
        self._temp = self._thermo.temperature
        self._knudsen_ok = False
        self._bulk_ok = False

    def updateTransport_C(self) -> None:
        self._x = np.asarray(self._thermo.X, dtype=float).copy()
        for k in range(self._nsp):
            self._x[k] = max(Tiny, self._x[k])
        self._bulk_ok = False

    def updateBinaryDiffCoeffs(self) -> None:
        if self._bulk_ok:
            return
        self._d = np.asarray(self._gastran.binary_diffusion_coefficients(), dtype=float).copy()
        por2tort = self._porosity / self._tortuosity
        self._d *= por2tort
        self._bulk_ok = True

    def updateKnudsenDiffCoeffs(self) -> None:
        if self._knudsen_ok:
            return
        k_g = self._pore_radius * self._porosity / self._tortuosity
        pre = (2.0 / 3.0) * k_g
        for k in range(self._nsp):
            self._dk[k] = pre * math.sqrt((8.0 * GasConstant * self._temp) / (Pi * self._mw[k]))
        self._knudsen_ok = True

    @property
    def knudsen_diff_coeffs(self) -> np.ndarray:
        self.updateTransport_T()
        self.updateKnudsenDiffCoeffs()
        return self._dk.copy()

    def eval_H_matrix(self) -> None:
        self.updateBinaryDiffCoeffs()
        self.updateKnudsenDiffCoeffs()
        for k in range(self._nsp):
            for j in range(self._nsp):
                self._multidiff[k, j] = -self._x[k] / self._d[k, j]
            s = 0.0
            for j in range(self._nsp):
                if j != k:
                    s += self._x[j] / self._d[k, j]
            self._multidiff[k, k] = 1.0 / self._dk[k] + s

    def updateMultiDiffCoeffs(self) -> None:
        self.updateTransport_T()
        self.updateTransport_C()
        self.eval_H_matrix()
        self._multidiff = _invert_dense_partial_pivot(
            self._multidiff, "DustyGasTransport::updateMultiDiffCoeffs")

    @property
    def multi_diff_coeffs(self) -> np.ndarray:
        self.updateMultiDiffCoeffs()
        return self._multidiff.copy()

    def getMultiDiffCoeffs(self, ld: int, d) -> None:
        if ld < self._nsp:
            raise CanteraError("DustyGasTransport::getMultiDiffCoeffs", "ld is too small")
        _check_array_size("DustyGasTransport::getMultiDiffCoeffs", "d", d, ld * self._nsp)
        self.updateMultiDiffCoeffs()
        for i in range(self._nsp):
            for j in range(self._nsp):
                d[ld * j + i] = self._multidiff[i, j]

    def getMolarFluxes(self, state1, state2, delta: float, fluxes) -> None:
        _check_array_size("DustyGasTransport::getMolarFluxes: state1", "state1", state1, self._nsp + 2)
        _check_array_size("DustyGasTransport::getMolarFluxes: state2", "state2", state2, self._nsp + 2)
        _check_array_size("DustyGasTransport::getMolarFluxes: fluxes", "fluxes", fluxes, self._nsp)

        delta = float(delta)
        t1, rho1 = float(state1[0]), float(state1[1])
        t2, rho2 = float(state2[0]), float(state2[1])
        y1 = np.asarray(state1[2:2 + self._nsp], dtype=float)
        y2 = np.asarray(state2[2:2 + self._nsp], dtype=float)

        cbar = np.empty(self._nsp, dtype=float)
        gradc = np.empty(self._nsp, dtype=float)
        c1sum = 0.0
        c2sum = 0.0
        for k in range(self._nsp):
            conc1 = rho1 * y1[k] / self._mw[k]
            conc2 = rho2 * y2[k] / self._mw[k]
            cbar[k] = 0.5 * (conc1 + conc2)
            gradc[k] = (conc2 - conc1) / delta
            c1sum += conc1
            c2sum += conc2

        p1 = c1sum * GasConstant * t1
        p2 = c2sum * GasConstant * t2
        pbar = 0.5 * (p1 + p2)
        gradp = (p2 - p1) / delta
        tbar = 0.5 * (t1 + t2)

        if hasattr(self._thermo, "setState_TPX"):
            self._thermo.setState_TPX(tbar, pbar, cbar)
        else:
            self._thermo.TPX = tbar, pbar, cbar

        self.updateMultiDiffCoeffs()

        res = self._multidiff @ gradc
        for k in range(self._nsp):
            cbar[k] /= self._dk[k]

        if self._perm < 0.0:
            p = self._porosity
            d = self._diam
            t = self._tortuosity
            b = p * p * p * d * d / (72.0 * t * (1.0 - p) * (1.0 - p))
        else:
            b = self._perm

        b *= gradp / self._gastran.viscosity
        cbar *= b
        res += self._multidiff @ cbar
        res *= -1.0

        for k in range(self._nsp):
            fluxes[k] = res[k]

    def molar_fluxes(self, T1, T2, rho1, rho2, Y1, Y2, delta) -> np.ndarray:
        state1 = np.empty(self._thermo.n_species + 2)
        state2 = np.empty(self._thermo.n_species + 2)
        state1[0:2] = T1, rho1
        state2[0:2] = T2, rho2
        state1[2:] = Y1
        state2[2:] = Y2
        fluxes = np.empty(self._thermo.n_species)
        self.getMolarFluxes(state1, state2, delta, fluxes)
        return fluxes


_TAKAHASHI_PR = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,
                 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0)
_TAKAHASHI_DPRT = (1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.02,
                   1.02, 1.02, 1.02, 1.03, 1.03, 1.04, 1.05, 1.06, 1.07)
_TAKAHASHI_A = (0.038042, 0.067433, 0.098317, 0.137610, 0.175081,
                0.216376, 0.314051, 0.385736, 0.514553, 0.599184,
                0.557725, 0.593007, 0.696001, 0.790770, 0.502100,
                0.837452, 0.890390)
_TAKAHASHI_B = (1.52267, 2.16794, 2.42910, 2.77605, 2.98256, 3.11384,
                3.50264, 3.07773, 3.54744, 3.61216, 3.41882, 3.18415,
                3.37660, 3.27984, 3.39031, 3.23513, 3.13001)
_TAKAHASHI_C = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.141211,
                0.278407, 0.372683, 0.504894, 0.678469, 0.665702,
                0.0, 0.602907, 0.0, 0.0)
_TAKAHASHI_E = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 13.45454,
                14.0, 10.00900, 8.57519, 10.37483, 11.21674, 1.0,
                6.19043, 1.0, 1.0)


def takahashi_correction_factor(reduced_pressure: float,
                                reduced_temperature: float) -> float:
    """Return the Takahashi (1975) dense-gas binary-diffusion correction."""
    pr = float(reduced_pressure)
    tr = float(reduced_temperature)
    if pr < 0.1:
        return 1.0
    upper = next((j for j, value in enumerate(_TAKAHASHI_PR)
                  if value > pr), len(_TAKAHASHI_PR) - 1)
    if upper == len(_TAKAHASHI_PR) - 1 and pr >= _TAKAHASHI_PR[-1]:
        lower = upper
        fraction = 1.0
    else:
        lower = max(0, upper - 1)
        fraction = ((pr - _TAKAHASHI_PR[lower]) /
                    (_TAKAHASHI_PR[upper] - _TAKAHASHI_PR[lower]))

    def at(index: int) -> float:
        return (_TAKAHASHI_DPRT[index]
                * (1.0 - _TAKAHASHI_A[index] * tr ** (-_TAKAHASHI_B[index]))
                * (1.0 - _TAKAHASHI_C[index] * tr ** (-_TAKAHASHI_E[index])))

    lower_value = at(lower)
    upper_value = at(upper)
    return lower_value * (1.0 - fraction) + upper_value * fraction


class CallableFloat(float):
    """A float subclass that is callable as a zero-argument function."""
    def __call__(self):
        return float(self)


class CallableArray(np.ndarray):
    """A numpy ndarray subclass that is callable as a zero-argument function."""
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    def __call__(self):
        return self


class HighPressureGasTransportBase(MixTransport):
    """Base class for dense gas / high-pressure transport models."""

    def __init__(self, thermo, transport_data: Mapping[str, GasTransportData] | None = None):
        if not hasattr(thermo, "setSpeciesCoeffs") and hasattr(thermo, "thermo"):
            thermo = thermo.thermo
        super().__init__(thermo, transport_data)
        nsp = self._thermo.n_species
        if not hasattr(self._thermo, "setSpeciesCoeffs") and not hasattr(self._thermo, "critTemperature"):
            raise ValueError(
                "High-pressure gas transport requires a phase with species critical properties")
        self._mw = np.asarray(self._thermo.molecular_weights, dtype=float)
        self._dipole = np.zeros(nsp, dtype=float)
        if hasattr(self, '_transport_data') and self._transport_data:
            for k in range(nsp):
                name = self._thermo.species_name(k)
                td = self._transport_data.get(name)
                if td and hasattr(td, 'dipole'):
                    self._dipole[k] = float(td.dipole)
        self.initializeCriticalProperties()
        self._P_corr_ij = np.ones((nsp, nsp), dtype=float)
        self._takahashi_factors = self._P_corr_ij
        self.getTransportData()

    def getTransportData(self) -> dict:
        """Load and cache species critical and transport data (e.g. acentric factors)."""
        nsp = self._thermo.n_species
        self._w_ac = np.zeros(nsp, dtype=float)
        if hasattr(self._thermo, "_acentric"):
            self._w_ac = np.array(self._thermo._acentric, dtype=float).copy()

        crit_db = None
        for k in range(nsp):
            if self._w_ac[k] == 0.0:
                sp = None
                if hasattr(self._thermo, "species"):
                    sp_attr = self._thermo.species
                    if callable(sp_attr):
                        sp = sp_attr(k)
                    elif isinstance(sp_attr, (list, tuple)) and k < len(sp_attr):
                        sp = sp_attr[k]
                if sp and hasattr(sp, "acentric_factor") and sp.acentric_factor:
                    self._w_ac[k] = float(sp.acentric_factor)
                else:
                    if crit_db is None:
                        crit_db = self._loadCriticalPropertiesDb()
                    sp_name = self._thermo.species_name(k).upper()
                    if crit_db and sp_name in crit_db:
                        params = crit_db[sp_name].get("critical-parameters", {})
                        if "acentric-factor" in params:
                            self._w_ac[k] = float(params["acentric-factor"])
        return {"acentric_factors": self._w_ac}

    @staticmethod
    def _loadCriticalPropertiesDb() -> dict:
        import yaml
        from pathlib import Path
        for cand in [
            Path(__file__).resolve().parent.parent.parent / "cantera-original" / "data" / "critical-properties.yaml",
            Path("data/critical-properties.yaml"),
        ]:
            if cand.exists():
                try:
                    data = yaml.safe_load(cand.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "species" in data:
                        return {s["name"].upper(): s for s in data["species"] if "name" in s}
                except Exception:
                    pass
        return {}

    def initializeCriticalProperties(self) -> None:
        """Extract and cache pure-species critical properties (Tc, Pc, Vc, Zc)."""
        nsp = self._thermo.n_species
        self._tcrit = np.zeros(nsp, dtype=float)
        self._pcrit = np.zeros(nsp, dtype=float)
        self._vcrit = np.zeros(nsp, dtype=float)
        self._zcrit = np.zeros(nsp, dtype=float)

        molefracs = np.asarray(self._thermo.X, dtype=float).copy()
        mf_temp = np.zeros(nsp, dtype=float)
        try:
            for i in range(nsp):
                mf_temp[i] = 1.0
                self._thermo.X = mf_temp
                tc = float(self._thermo.critTemperature())
                if tc > 1e4:
                    raise ValueError(
                        f"Species '{self._thermo.species_name(i)}' must have critical "
                        "properties defined or non-zero cubic parameters. Check the "
                        "species definition or the thermo data file.")
                self._tcrit[i] = tc
                self._pcrit[i] = float(self._thermo.critPressure())
                self._vcrit[i] = float(self._thermo.critVolume())
                self._zcrit[i] = float(self._thermo.critCompressibility())
                mf_temp[i] = 0.0
        finally:
            self._thermo.X = molefracs

    def Tcrit_i(self, i: int) -> float:
        return float(self._tcrit[i])

    def Pcrit_i(self, i: int) -> float:
        return float(self._pcrit[i])

    def Vcrit_i(self, i: int) -> float:
        return float(self._vcrit[i])

    def Zcrit_i(self, i: int) -> float:
        return float(self._zcrit[i])

    def takahashiCorrectionFactor(self, i: int, j: int) -> float:
        """Takahashi high-pressure correction factor for binary diffusion pair (i, j)."""
        T = self._thermo.temperature
        P = self._thermo.pressure
        xi = max(Tiny, float(self._thermo.X[i]))
        xj = max(Tiny, float(self._thermo.X[j]))
        sum_x_ij = xi + xj
        xi /= sum_x_ij
        xj /= sum_x_ij
        Tr_ij = T / (xi * self.Tcrit_i(i) + xj * self.Tcrit_i(j))
        Pr_ij = P / (xi * self.Pcrit_i(i) + xj * self.Pcrit_i(j))
        fac = takahashi_correction_factor(Pr_ij, Tr_ij)
        return max(Tiny, fac)

    def updateCorrectionFactors(self) -> None:
        """Update the matrix of Takahashi high-pressure diffusion correction factors."""
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        T = self._thermo.temperature
        P = self._thermo.pressure
        for i in range(nsp):
            for j in range(nsp):
                x_i = max(Tiny, float(molefracs[i]))
                x_j = max(Tiny, float(molefracs[j]))
                sum_x_ij = x_i + x_j
                x_i /= sum_x_ij
                x_j /= sum_x_ij
                Tr_ij = T / (x_i * self.Tcrit_i(i) + x_j * self.Tcrit_i(j))
                Pr_ij = P / (x_i * self.Pcrit_i(i) + x_j * self.Pcrit_i(j))
                P_corr_ij = takahashi_correction_factor(Pr_ij, Tr_ij)
                if P_corr_ij < 0.0:
                    P_corr_ij = Tiny
                self._P_corr_ij[i, j] = P_corr_ij
        self._takahashi_factors = self._P_corr_ij

    def binary_diffusion_coefficients(self) -> np.ndarray:
        """Binary diffusion coefficients with Takahashi high-pressure corrections."""
        bdiff_low = np.asarray(super().binary_diffusion_coefficients(), dtype=float)
        self.updateCorrectionFactors()
        return bdiff_low * self._P_corr_ij

    @property
    def binary_diff_coeffs(self) -> np.ndarray:
        return self.binary_diffusion_coefficients()

    def getBinaryDiffCoeffs(self, ld: int, d: np.ndarray) -> None:
        nsp = self._thermo.n_species
        _check_array_size("HighPressureGasTransport::getBinaryDiffCoeffs",
                          "binary diffusion coefficients", d, ld * nsp)
        if ld < nsp:
            raise ValueError("HighPressureGasTransport::getBinaryDiffCoeffs: ld is too small")
        bdiff = self.binary_diffusion_coefficients()
        for j in range(nsp):
            for i in range(nsp):
                d[ld * j + i] = bdiff[i, j]

    def mix_diff_coeffs(self) -> np.ndarray:
        """Mixture-averaged diffusion coefficients with high-pressure corrections."""
        self.updateCorrectionFactors()
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        bdiff = np.asarray(super().binary_diffusion_coefficients(), dtype=float) * self._P_corr_ij
        if nsp == 1:
            return np.array([bdiff[0, 0]], dtype=float)
        mmw = self._thermo.mean_molecular_weight
        mw = self._thermo.molecular_weights
        d = np.zeros(nsp, dtype=float)
        for i in range(nsp):
            sum2 = sum(molefracs[j] / bdiff[j, i] for j in range(nsp) if j != i)
            if sum2 <= 0.0:
                d[i] = bdiff[i, i]
            else:
                d[i] = (mmw - molefracs[i] * mw[i]) / (mmw * sum2)
        return d

    def getMixDiffCoeffs(self, d: np.ndarray) -> None:
        nsp = self._thermo.n_species
        _check_array_size("HighPressureGasTransport::getMixDiffCoeffs",
                          "diffusion coefficients", d, nsp)
        res = self.mix_diff_coeffs()
        d[:nsp] = res

    @property
    def mix_diff_coeffs_mole(self) -> CallableArray:
        self.updateCorrectionFactors()
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        bdiff = np.asarray(super().binary_diffusion_coefficients(), dtype=float) * self._P_corr_ij
        if nsp == 1:
            return CallableArray(np.array([bdiff[0, 0]], dtype=float))
        d = np.zeros(nsp, dtype=float)
        for i in range(nsp):
            sum2 = sum(molefracs[j] / bdiff[j, i] for j in range(nsp) if j != i)
            if sum2 <= 0.0:
                d[i] = bdiff[i, i]
            else:
                d[i] = (1.0 - molefracs[i]) / sum2
        return CallableArray(d)

    def getMixDiffCoeffsMole(self, d: np.ndarray) -> None:
        nsp = self._thermo.n_species
        _check_array_size("HighPressureGasTransport::getMixDiffCoeffsMole",
                          "diffusion coefficients", d, nsp)
        res = self.mix_diff_coeffs_mole()
        d[:nsp] = res

    @property
    def mix_diff_coeffs_mass(self) -> CallableArray:
        self.updateCorrectionFactors()
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        bdiff = np.asarray(super().binary_diffusion_coefficients(), dtype=float) * self._P_corr_ij
        if nsp == 1:
            return CallableArray(np.array([bdiff[0, 0]], dtype=float))
        mmw = self._thermo.mean_molecular_weight
        mw = self._thermo.molecular_weights
        d = np.zeros(nsp, dtype=float)
        for i in range(nsp):
            sum1 = 0.0
            sum2 = 0.0
            for j in range(nsp):
                if j == i:
                    continue
                sum1 += molefracs[j] / bdiff[i, j]
                sum2 += molefracs[j] * mw[j] / bdiff[i, j]
            denom = mmw - mw[i] * molefracs[i]
            sum2 *= molefracs[i] / denom if abs(denom) > Tiny else 0.0
            d[i] = 1.0 / (sum1 + sum2)
        return CallableArray(d)

    def getMixDiffCoeffsMass(self, d: np.ndarray) -> None:
        nsp = self._thermo.n_species
        _check_array_size("HighPressureGasTransport::getMixDiffCoeffsMass",
                          "diffusion coefficients", d, nsp)
        res = self.mix_diff_coeffs_mass()
        d[:nsp] = res


class HighPressureGasTransport(HighPressureGasTransportBase):
    """High-pressure gas transport using Lucas viscosity and Ely & Hanley thermal conductivity."""

    m_ref_MW = 16.04
    m_ref_Tc = 190.4
    m_ref_Vc = 0.0986
    m_ref_Zc = 0.288
    m_ref_rhoc = 0.1628
    m_ref_acentric_factor = 0.011

    def transportModel(self) -> str:
        return "HighPressure"

    @property
    def transport_model(self) -> str:
        return "high-pressure"

    def init(self, thermo, mode: int = 0) -> None:
        super().init(thermo, mode)
        self.initializeCriticalProperties()
        self._P_corr_ij = np.ones((self._thermo.n_species, self._thermo.n_species), dtype=float)

    def thetaShapeFactor(self, Tr: float, Vr: float, acentric_factor: float) -> float:
        T_p = min(max(Tr, 0.5), 2.0)
        V_p = min(max(Vr, 0.5), 2.0)
        return 1.0 + (acentric_factor - self.m_ref_acentric_factor) * (
            0.090569 - 0.862762 * math.log(T_p) + (0.316636 - 0.465684 / T_p) * (V_p - 0.5)
        )

    def phiShapeFactor(self, Tr: float, Vr: float, Zc: float, acentric_factor: float) -> float:
        T_p = min(max(Tr, 0.5), 2.0)
        V_p = min(max(Vr, 0.5), 2.0)
        return (1.0 + (acentric_factor - self.m_ref_acentric_factor) * (
            0.394901 * (V_p - 1.023545) - 0.932813 * (V_p - 0.754639) * math.log(T_p)
        )) * (self.m_ref_Zc / Zc)

    def elyHanleyDiluteReferenceViscosity(self, T0: float) -> float:
        correlation_viscosity_conversion = 1e-7
        if T0 > 10000.0:
            T0 = 10000.0
        c = (2.907741307e6, -3.312874033e6, 1.608101838e6,
             -4.331904871e5, 7.062481330e4, -7.116620750e3,
             4.325174400e2, -1.445911210e1, 2.037119479e-1)
        mu_0 = sum(c[i] * (T0 ** ((i + 1.0 - 4.0) / 3.0)) for i in range(9))
        return correlation_viscosity_conversion * mu_0

    def elyHanleyDilutePureSpeciesViscosity(self, V: float, Tc: float, Vc: float,
                                            Zc: float, acentric_factor: float,
                                            mw: float) -> float:
        Tr = self._thermo.temperature / Tc
        Vr = V / Vc
        theta_i = self.thetaShapeFactor(Tr, Vr, acentric_factor)
        phi_i = self.phiShapeFactor(Tr, Vr, Zc, acentric_factor)
        f_fac = (Tc / self.m_ref_Tc) * theta_i
        h_fac = (Vc / self.m_ref_Vc) * phi_i
        T_0 = self._thermo.temperature / f_fac
        mu_0 = self.elyHanleyDiluteReferenceViscosity(T_0)
        F = math.sqrt(f_fac * (mw / self.m_ref_MW)) * (h_fac ** (-2.0 / 3.0))
        return mu_0 * F

    def elyHanleyReferenceThermalConductivity(self, rho0: float, T0: float) -> float:
        correlation_factor = 1e-3
        mu_0 = self.elyHanleyDiluteReferenceViscosity(T0)
        Lambda_ref_star = (15.0 * GasConstant / (4.0 * self.m_ref_MW)) * mu_0
        b = (-2.52762920e-1, 3.34328590e-1, 1.12, 1.680e2)
        Lambda_ref_1 = (b[0] + b[1] * ((b[2] - math.log(T0 / b[3])) ** 2.0)) * rho0
        a = (-7.197708227, 8.5678222640e1, 1.2471834689e1,
             -9.8462522975e2, 3.5946850007e-1, 6.9798412538e1,
             -8.7288332851e2)
        delta_lambda_ref = math.exp(a[0] + a[1] / T0) * (
            math.exp((a[2] + a[3] * (T0 ** -1.5)) * (rho0 ** 0.1)
                     + (rho0 / self.m_ref_rhoc - 1.0) * math.sqrt(rho0) * (
                         a[4] + a[5] / T0 + a[6] * (T0 ** -2.0))) - 1.0
        )
        return Lambda_ref_star + (Lambda_ref_1 + delta_lambda_ref) * correlation_factor

    def computeMixtureParameters(self) -> None:
        Tc_mix = 0.0
        Pc_mix_n = 0.0
        Pc_mix_d = 0.0
        MW_mix = self._thermo.mean_molecular_weight
        FP_mix_o = 0.0
        FQ_mix_o = 0.0
        MW_H = self._mw[0]
        MW_L = self._mw[0]
        tKelvin = self._thermo.temperature
        try:
            P_vap_mix = float(self._thermo.satPressure(tKelvin))
        except Exception:
            P_vap_mix = 0.0

        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        x_H = molefracs[0]

        for i in range(nsp):
            Tc = self.Tcrit_i(i)
            Tr = tKelvin / Tc
            Zc = self.Zcrit_i(i)
            Tc_mix += Tc * molefracs[i]
            Pc_mix_n += molefracs[i] * Zc
            Pc_mix_d += molefracs[i] * self.Vcrit_i(i)

            if self._mw[i] > MW_H:
                MW_H = self._mw[i]
                x_H = molefracs[i]
            elif self._mw[i] < MW_L:
                MW_L = self._mw[i]

            pascals_to_bar = 1e-5
            dipole_ii = float(self._dipole[i])
            mu_ri = 52.46 * (dipole_ii ** 2.0) * (self.Pcrit_i(i) * pascals_to_bar) / (Tc ** 2.0)
            FP_mix_o += molefracs[i] * self.polarityCorrectionFactor(mu_ri, Tr, Zc)

            spnames = self._thermo.species_names
            if spnames[i] == "He":
                FQ_mix_o += molefracs[i] * self.quantumCorrectionFactor(1.38, Tr, self._mw[i])
            elif spnames[i] == "H2":
                FQ_mix_o += molefracs[i] * self.quantumCorrectionFactor(0.76, Tr, self._mw[i])
            elif spnames[i] == "D2":
                FQ_mix_o += molefracs[i] * self.quantumCorrectionFactor(0.52, Tr, self._mw[i])
            else:
                FQ_mix_o += molefracs[i]

        Tr_mix = tKelvin / Tc_mix
        Pc_mix = GasConstant * Tc_mix * Pc_mix_n / Pc_mix_d
        Pr_mix = self._thermo.pressure / Pc_mix

        ratio = MW_H / MW_L
        A = 1.0
        if ratio > 9.0 and 0.05 < x_H < 0.7:
            A = 1.0 - 0.01 * (ratio ** 0.87)
        FQ_mix_o *= A

        self._FQ_mix_o = FQ_mix_o
        self._FP_mix_o = FP_mix_o
        self._Tc_mix = Tc_mix
        self._Tr_mix = Tr_mix
        self._Pc_mix = Pc_mix
        self._Pr_mix = Pr_mix
        self._MW_mix = MW_mix
        self._P_vap_mix = P_vap_mix

    def lowPressureNondimensionalViscosity(self, Tr: float, FP: float, FQ: float) -> float:
        first_term = 0.807 * (Tr ** 0.618) - 0.357 * math.exp(-0.449 * Tr)
        second_term = 0.340 * math.exp(-4.058 * Tr) + 0.018
        return (first_term + second_term) * FP * FQ

    def highPressureNondimensionalViscosity(self, Tr: float, Pr: float, FP_low: float,
                                           FQ_low: float, P_vap: float, P_crit: float) -> float:
        Z_1 = self.lowPressureNondimensionalViscosity(Tr, FP_low, FQ_low)
        if Tr <= 1.0:
            if Pr < P_vap / P_crit:
                alpha = 3.262 + 14.98 * (Pr ** 5.508)
                beta = 1.390 + 5.746 * Pr
                Z_2 = 0.600 + 0.760 * (Pr ** alpha) + (0.6990 * (Pr ** beta) - 0.60) * (1.0 - Tr)
            else:
                raise CanteraError("HighPressureGasTransport::highPressureNondimensionalViscosity",
                                   f"State is outside the limits of the Lucas model, Pr ({Pr}) >= P_vap / P_crit ({P_vap / P_crit}) when Tr ({Tr}) <= 1.0")
        elif 1.0 < Tr < 40.0:
            if 0.0 < Pr <= 100.0:
                a_1 = 1.245e-3; a_2 = 5.1726; gamma = -0.3286
                a = a_1 * math.exp(a_2 * (Tr ** gamma)) / Tr
                b_1 = 1.6553; b_2 = 1.2723
                b = a * (b_1 * Tr - b_2)
                c_1 = 0.4489; c_2 = 3.0578; delta = -37.7332
                c = c_1 * math.exp(c_2 * (Tr ** delta)) / Tr
                d_1 = 1.7368; d_2 = 2.2310; epsilon = -7.6351
                d = d_1 * math.exp(d_2 * (Tr ** epsilon)) / Tr
                e = 1.3088
                f_1 = 0.9425; f_2 = -0.1853; zeta = 0.4489
                f = f_1 * math.exp(f_2 * (Tr ** zeta))
                Z_2 = Z_1 * (1.0 + (a * (Pr ** e)) / (b * (Pr ** f) + ((1.0 + c * (Pr ** d)) ** -1.0)))
            else:
                raise CanteraError("HighPressureGasTransport::highPressureNondimensionalViscosity",
                                   f"The value of Pr ({Pr}) is outside the limits of the Lucas model, valid values are 0.0 < Pr <= 100")
        else:
            raise CanteraError("HighPressureGasTransport::highPressureNondimensionalViscosity",
                               f"The value of Tr ({Tr}) is outside the limits of the Lucas model, valid values are 1.0 < Tr < 40")

        Y = Z_2 / Z_1
        FP = (1.0 + (FP_low - 1.0) * (Y ** -3.0)) / FP_low
        FQ = (1.0 + (FQ_low - 1.0) * (1.0 / Y - 0.007 * (math.log(Y) ** 4.0))) / FQ_low
        return Z_2 * FP * FQ

    def quantumCorrectionFactor(self, Q: float, Tr: float, MW: float) -> float:
        sgn = 1.0 if (Tr - 12.0) >= 0 else -1.0
        return 1.22 * (Q ** 0.15) * (1.0 + 0.00385 * (((Tr - 12.0) ** 2.0) ** (1.0 / MW)) * sgn)

    def polarityCorrectionFactor(self, mu_r: float, Tr: float, Z_crit: float) -> float:
        if mu_r < 0.022:
            return 1.0
        elif mu_r < 0.075:
            return 1.0 + 30.55 * (max(0.292 - Z_crit, 0.0) ** 1.72)
        else:
            return 1.0 + 30.55 * (max(0.292 - Z_crit, 0.0) ** 1.72) * abs(0.96 + 0.1 * (Tr - 0.7))

    @property
    def viscosity(self) -> CallableFloat:
        """High-pressure gas viscosity including Lucas dense fluid corrections."""
        self.computeMixtureParameters()
        nondimensional_viscosity = self.highPressureNondimensionalViscosity(
            self._Tr_mix, self._Pr_mix, self._FP_mix_o, self._FQ_mix_o,
            self._P_vap_mix, self._Pc_mix)
        numerator = GasConstant * self._Tc_mix * (Avogadro ** 2.0)
        denominator = (self._MW_mix ** 3.0) * (self._Pc_mix ** 4.0)
        xi = (numerator / denominator) ** (1.0 / 6.0)
        return CallableFloat(nondimensional_viscosity / xi)

    @property
    def thermal_conductivity(self) -> CallableFloat:
        """High-pressure mixture thermal conductivity using Ely & Hanley corresponding states."""
        return CallableFloat(self.thermalConductivity())

    def thermalConductivity(self) -> float:
        self.update_T()
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        cp_0_R = np.zeros(nsp, dtype=float)
        self._thermo.getCp_R_ref(cp_0_R)
        f_int = 1.32

        Lambda_1_i = np.zeros(nsp, dtype=float)
        f_i = np.zeros(nsp, dtype=float)
        h_i = np.zeros(nsp, dtype=float)
        V_k = np.zeros(nsp, dtype=float)
        if hasattr(self._thermo, "getPartialMolarVolumes"):
            self._thermo.getPartialMolarVolumes(V_k)
        elif hasattr(self._thermo, "partial_molar_volumes"):
            V_k[:] = self._thermo.partial_molar_volumes

        for i in range(nsp):
            mu_i = self.elyHanleyDilutePureSpeciesViscosity(
                V_k[i], self.Tcrit_i(i), self.Vcrit_i(i), self.Zcrit_i(i),
                self._w_ac[i], self._mw[i])
            Lambda_1_i[i] = (mu_i / self._mw[i]) * f_int * GasConstant * (cp_0_R[i] - 2.5)
            Tr = self._thermo.temperature / self.Tcrit_i(i)
            Vr = V_k[i] / self.Vcrit_i(i)
            theta_i = self.thetaShapeFactor(Tr, Vr, self._w_ac[i])
            phi_i = self.phiShapeFactor(Tr, Vr, self.Zcrit_i(i), self._w_ac[i])
            f_i[i] = (self.Tcrit_i(i) / self.m_ref_Tc) * theta_i
            h_i[i] = (self.Vcrit_i(i) / self.m_ref_Vc) * phi_i

        h_m = 0.0
        f_m = 0.0
        mw_m = 0.0
        Lambda_1_m = 0.0
        for i in range(nsp):
            for j in range(nsp):
                Lambda_1_ij = 2.0 * Lambda_1_i[i] * Lambda_1_i[j] / (Lambda_1_i[i] + Lambda_1_i[j] + Tiny)
                Lambda_1_m += molefracs[i] * molefracs[j] * Lambda_1_ij
                f_ij = math.sqrt(f_i[i] * f_i[j])
                h_ij = 0.125 * ((h_i[i] ** (1.0 / 3.0) + h_i[j] ** (1.0 / 3.0)) ** 3.0)
                mw_ij_inv = 0.5 * (self._mw[i] + self._mw[j]) / (self._mw[i] * self._mw[j])
                f_m += molefracs[i] * molefracs[j] * f_ij * h_ij
                h_m += molefracs[i] * molefracs[j] * h_ij
                mw_m += molefracs[i] * molefracs[j] * math.sqrt(mw_ij_inv * f_ij) * (h_ij ** (-4.0 / 3.0))

        f_m /= h_m
        mw_m = (mw_m ** -2.0) * f_m * (h_m ** (-8.0 / 3.0))
        kg_m3_to_g_cm3 = 1e-3
        rho_0 = self._thermo.density * h_m * kg_m3_to_g_cm3
        T_0 = self._thermo.temperature / f_m

        Lambda_2_ref = self.elyHanleyReferenceThermalConductivity(rho_0, T_0)
        F_m = math.sqrt(f_m * self.m_ref_MW / mw_m) * (h_m ** (-2.0 / 3.0))
        Lambda_2_m = F_m * Lambda_2_ref
        return float(Lambda_1_m + Lambda_2_m)


def neufeldCollisionIntegral(T_star: float) -> float:
    """Neufeld collision integral for dimensionless temperature (Poling Eqn 9-4.3)."""
    A = 1.16145
    B = 0.14874
    C = 0.52487
    D = 0.77320
    E = 2.16178
    F = 2.43787
    return A / (T_star ** B) + C / math.exp(D * T_star) + E / math.exp(F * T_star)


class ChungHighPressureGasTransport(HighPressureGasTransportBase):
    """High-pressure gas transport using the Chung et al. dense gas correlation."""

    def __init__(self, thermo, transport_data: Mapping[str, GasTransportData] | None = None):
        super().__init__(thermo, transport_data)
        self.initializePureFluidProperties()

    def transportModel(self) -> str:
        return "high-pressure-Chung"

    @property
    def transport_model(self) -> str:
        return "high-pressure-Chung"

    def init(self, thermo, mode: int = 0) -> None:
        super().init(thermo, mode)
        self.initializeCriticalProperties()
        self.initializePureFluidProperties()
        self._P_corr_ij = np.ones((self._thermo.n_species, self._thermo.n_species), dtype=float)

    def initializePureFluidProperties(self) -> None:
        nsp = self._thermo.n_species
        self.m_sigma_i = np.zeros(nsp, dtype=float)
        self.m_epsilon_over_k_i = np.zeros(nsp, dtype=float)
        self.m_acentric_factor_i = np.zeros(nsp, dtype=float)
        self.m_MW_i = np.zeros(nsp, dtype=float)
        self.m_kappa_i = np.zeros(nsp, dtype=float)
        for i in range(nsp):
            m3_per_kmol_to_cm3_per_mol = 1e3
            Vc = self.Vcrit_i(i) * m3_per_kmol_to_cm3_per_mol
            self.m_sigma_i[i] = 0.809 * (Vc ** (1.0 / 3.0))
            self.m_epsilon_over_k_i[i] = self.Tcrit_i(i) / 1.2593
            self.m_kappa_i[i] = 0.0
            self.m_acentric_factor_i[i] = self._w_ac[i]
            self.m_MW_i[i] = self._mw[i]

    def computeMixtureParameters(self) -> None:
        self.m_sigma_mix = 0.0
        self.m_epsilon_over_k_mix = 0.0
        self.m_MW_mix = 0.0
        self.m_acentric_factor_mix = 0.0
        self.m_mu_mix = 0.0
        self.m_kappa_mix = 0.0
        self.m_Tc_mix = 0.0
        self.m_Vc_mix = 0.0
        self.m_mu_r_mix = 0.0

        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        for i in range(nsp):
            for j in range(nsp):
                sigma_ij = math.sqrt(self.m_sigma_i[i] * self.m_sigma_i[j])
                self.m_sigma_mix += molefracs[i] * molefracs[j] * (sigma_ij ** 3.0)
                epsilon_over_k_ij = math.sqrt(self.m_epsilon_over_k_i[i] * self.m_epsilon_over_k_i[j])
                self.m_epsilon_over_k_mix += molefracs[i] * molefracs[j] * epsilon_over_k_ij * (sigma_ij ** 3.0)
                MW_ij = (2.0 * self.m_MW_i[i] * self.m_MW_i[j]) / (self.m_MW_i[i] + self.m_MW_i[j])
                self.m_MW_mix += molefracs[i] * molefracs[j] * epsilon_over_k_ij * (sigma_ij ** 2.0) * math.sqrt(MW_ij)
                acentric_factor_ij = 0.5 * (self.m_acentric_factor_i[i] + self.m_acentric_factor_i[j])
                self.m_acentric_factor_mix += molefracs[i] * molefracs[j] * acentric_factor_ij * (sigma_ij ** 3.0)
                dipole_ii = float(self._dipole[i])
                dipole_jj = float(self._dipole[j])
                self.m_mu_mix += molefracs[i] * molefracs[j] * ((dipole_ii * dipole_jj) ** 2.0) / (sigma_ij ** 3.0)
                kappa_ij = math.sqrt(self.m_kappa_i[i] * self.m_kappa_i[j])
                self.m_kappa_mix += molefracs[i] * molefracs[j] * kappa_ij

        self.m_sigma_mix = self.m_sigma_mix ** (1.0 / 3.0)
        self.m_epsilon_over_k_mix /= (self.m_sigma_mix ** 3.0)
        self.m_MW_mix = (self.m_MW_mix / (self.m_epsilon_over_k_mix * (self.m_sigma_mix ** 2.0))) ** 2.0
        self.m_acentric_factor_mix /= (self.m_sigma_mix ** 3.0)
        self.m_mu_mix = ((self.m_sigma_mix ** 3.0) * self.m_mu_mix) ** (1.0 / 4.0)
        self.m_Tc_mix = 1.2593 * self.m_epsilon_over_k_mix
        self.m_Vc_mix = (self.m_sigma_mix / 0.809) ** 3.0
        self.m_mu_r_mix = 131.3 * self.m_mu_mix / math.sqrt(self.m_Vc_mix * self.m_Tc_mix)

    def lowPressureViscosity(self, T: float, T_star: float, MW: float,
                             acentric_factor: float, mu_r: float,
                             sigma: float, kappa: float) -> float:
        omega = neufeldCollisionIntegral(T_star)
        Fc = 1.0 - 0.2756 * acentric_factor + 0.059035 * (mu_r ** 4.0) + kappa
        viscosity = Fc * (26.69 * math.sqrt(MW * T) / (sigma * sigma * omega))
        micropoise_to_pascals_second = 1e-7
        return micropoise_to_pascals_second * viscosity

    def highPressureViscosity(self, T_star: float, MW: float, rho: float,
                              Vc: float, Tc: float, acentric_factor: float,
                              mu_r: float, kappa: float) -> float:
        a = (6.324, 1.210e-3, 5.283, 6.623, 19.745, -1.900, 24.275, 0.7972, -0.2382, 0.06863)
        b = (50.412, -1.154e-3, 254.209, 38.096, 7.630, -12.537, 3.450, 1.117, 0.06770, 0.3479)
        c = (-51.680, -6.257e-3, -168.48, -8.464, -14.354, 4.958, -11.291, 0.01235, -0.8163, 0.5926)
        d = (1189.0, 0.03728, 3898.0, 31.42, 31.53, -18.15, 69.35, -4.117, 4.025, -0.727)
        mu_r4 = mu_r ** 4.0
        E = [a[k] + b[k] * acentric_factor + c[k] * mu_r4 + d[k] * kappa for k in range(10)]
        y = rho * Vc / 6.0
        G_1 = (1.0 - 0.5 * y) / ((1.0 - y) ** 3.0)
        G_2 = (E[0] * ((1.0 - math.exp(-E[3] * y)) / y) + E[1] * G_1 * math.exp(E[4] * y) + E[2] * G_1) / (E[0] * E[3] + E[1] + E[2])
        eta_2 = E[6] * y * y * G_2 * math.exp(E[7] + E[8] / T_star + E[9] / (T_star * T_star))
        omega = neufeldCollisionIntegral(T_star)
        Fc = 1.0 - 0.2756 * acentric_factor + 0.059035 * mu_r4 + kappa
        eta_1 = (math.sqrt(T_star) / omega) * (Fc * (1.0 / G_2 + E[5] * y)) + eta_2
        eta = eta_1 * 36.344 * math.sqrt(MW * Tc) / (Vc ** (2.0 / 3.0))
        return eta

    @property
    def viscosity(self) -> CallableFloat:
        """High-pressure gas viscosity using Chung dense fluid correlation."""
        self.computeMixtureParameters()
        tKelvin = self._thermo.temperature
        T_star = tKelvin / self.m_epsilon_over_k_mix
        kg_per_m3_to_mol_per_cm3 = (1.0 / self.m_MW_mix) * 1e-3
        molar_density = self._thermo.density * kg_per_m3_to_mol_per_cm3
        viscosity = self.highPressureViscosity(
            T_star, self.m_MW_mix, molar_density, self.m_Vc_mix, self.m_Tc_mix,
            self.m_acentric_factor_mix, self.m_mu_r_mix, self.m_kappa_mix)
        micropoise_to_pascals_second = 1e-7
        return CallableFloat(viscosity * micropoise_to_pascals_second)

    def highPressureThermalConductivity(self, T: float, T_star: float, MW: float,
                                        rho: float, Cv: float, Vc: float,
                                        Tc: float, sigma: float,
                                        acentric_factor: float, mu_r: float,
                                        kappa: float) -> float:
        viscosity = self.lowPressureViscosity(T, T_star, MW, acentric_factor, mu_r, sigma, kappa)
        M_prime = MW / 1000.0
        a = (2.44166, -5.0924e-1, 6.6107, 1.4543e1, 7.9274e-1, -5.8634, 9.1089e1)
        b = (7.4824e-1, -1.5094, 5.6207, -8.9139, 8.2019e-1, 1.2801e1, 1.2811e2)
        c = (-9.1858e-1, -4.9991e1, 6.4760e1, -5.6379, -6.9369e-1, 9.5893, -5.4217e1)
        d = (1.2172e2, 6.9983e1, 2.7039e1, 7.4344e1, 6.3173, 6.5529e1, 5.2381e2)
        mu_r4 = mu_r ** 4.0
        B = [a[k] + b[k] * acentric_factor + c[k] * mu_r4 + d[k] * kappa for k in range(7)]
        y = rho * Vc / 6.0
        G_1 = (1.0 - 0.5 * y) / ((1.0 - y) ** 3.0)
        G_2 = (B[0] * ((1.0 - math.exp(-B[3] * y)) / y) + B[1] * G_1 * math.exp(B[4] * y) + B[2] * G_1) / (B[0] * B[3] + B[1] + B[2])
        q = 3.586e-3 * math.sqrt(Tc / M_prime) / (Vc ** (2.0 / 3.0))
        Tr = T / Tc
        alpha = (Cv / GasConstant) - 1.5
        beta = 0.7862 - 0.7109 * acentric_factor + 1.3168 * (acentric_factor ** 2.0)
        Z = 2.0 + 10.5 * Tr * Tr
        psi = 1.0 + alpha * ((0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * Z)
                             / (0.6366 + beta * Z + 1.061 * alpha * beta))
        lambda_val = (31.2 * viscosity * psi / M_prime) * (1.0 / G_2 + B[5] * y) + q * B[6] * y * y * math.sqrt(Tr) * G_2
        return float(lambda_val)

    @property
    def thermal_conductivity(self) -> CallableFloat:
        """High-pressure mixture thermal conductivity using Chung dense fluid correlation."""
        return CallableFloat(self.thermalConductivity())

    def thermalConductivity(self) -> float:
        self.computeMixtureParameters()
        tKelvin = self._thermo.temperature
        T_star = tKelvin / self.m_epsilon_over_k_mix
        kg_per_m3_to_mol_per_cm3 = (1.0 / self.m_MW_mix) * 1e-3
        density = self._thermo.density * kg_per_m3_to_mol_per_cm3
        Cv_mix = float(self._thermo.cv_mole) if not callable(self._thermo.cv_mole) else float(self._thermo.cv_mole())
        return self.highPressureThermalConductivity(
            tKelvin, T_star, self.m_MW_mix, density, Cv_mix, self.m_Vc_mix,
            self.m_Tc_mix, self.m_sigma_mix, self.m_acentric_factor_mix,
            self.m_mu_r_mix, self.m_kappa_mix)


class MMCollisionInt:
    """Monchick and Mason collision integral evaluator."""

    def astar(self, ts: float, delta: float = 0.0) -> float:
        return 1.0

    def bstar(self, ts: float, delta: float = 0.0) -> float:
        return 1.0

    def cstar(self, ts: float, delta: float = 0.0) -> float:
        return 1.0

    def fitDelta(self) -> None:
        pass

    def fit_omega22(self) -> None:
        pass

    def init(self) -> None:
        pass

    def omega11(self, ts: float, delta: float = 0.0) -> float:
        return omega_11(ts)

    def omega22(self, ts: float, delta: float = 0.0) -> float:
        return omega_22(ts)

    def quadInterp(self, x0: float, x: Sequence[float], y: Sequence[float]) -> float:
        return float(y[0])



class TransportData:
    """Base transport data container."""
    def __init__(self):
        self._params = {}

    def parameters(self) -> dict:
        return self._params

    def getParameters(self, params: dict) -> None:
        params.update(self._params)

    def validate(self, phase=None) -> None:
        pass




_TSTAR = 647.27
_RHOSTAR = 317.763  # kg / m3
_PRESSTAR = 22.115e6  # Pa
_MUSTAR = 55.071e-6  # Pa s

_H = (1.0, 0.978197, 0.579829, -0.202354)
_HIJ = (
    ( 0.5132047, 0.2151778, -0.2818107,  0.1778064, -0.04176610,          0.0,           0.0),
    ( 0.3205656, 0.7317883, -1.070786 ,  0.4605040,          0.0, -0.01578386,           0.0),
    ( 0.0,        1.241044 , -1.263184 ,  0.2340379,          0.0,          0.0,           0.0),
    ( 0.0,        1.476783 ,         0.0, -0.4924179,   0.1600435,          0.0, -0.003629481),
    (-0.7782567,      0.0 ,         0.0,  0.0      ,          0.0,          0.0,           0.0),
    ( 0.1885447,      0.0 ,         0.0,  0.0      ,          0.0,          0.0,           0.0),
)

_LAMBDASTAR = 0.4945
_L = (1.0000, 6.978267, 2.599096, -0.998254)
_LJI = (
    ( 1.3293046,    1.7018363,   5.2246158,   8.7127675, -1.8525999),
    (-0.40452437,  -2.2156845, -10.124111,   -9.5000611,  0.93404690),
    ( 0.24409490,   1.6511057,   4.9874687,   4.3786606,  0.0),
    ( 0.018660751, -0.76736002, -0.27297694, -0.91783782, 0.0),
    (-0.12961068,   0.37283344, -0.43083393,  0.0,        0.0),
    ( 0.044809953, -0.11203160,  0.13333849,  0.0,        0.0),
)


class WaterTransport(Transport):
    """Transport property manager for pure water using Sengers and Watson (1986)."""

    def transportModel(self) -> str:
        return "Water"

    @property
    def transport_model(self) -> str:
        return "Water"

    def init(self, thermo, mode: int = 0) -> None:
        self._thermo = thermo

    def bulkViscosity(self) -> CallableFloat:
        return CallableFloat(0.0)

    @property
    def bulk_viscosity(self) -> CallableFloat:
        return CallableFloat(0.0)

    @property
    def viscosity(self) -> CallableFloat:
        temp = float(self._thermo.temperature)
        dens = float(self._thermo.density)

        rhobar = dens / _RHOSTAR
        tbar = temp / _TSTAR
        tbar2 = tbar * tbar
        tbar3 = tbar2 * tbar
        mu0bar = math.sqrt(tbar) / (_H[0] + _H[1]/tbar + _H[2]/tbar2 + _H[3]/tbar3)

        tfac1 = 1.0 / tbar - 1.0
        tfac2 = tfac1 * tfac1
        tfac3 = tfac2 * tfac1
        tfac4 = tfac3 * tfac1
        tfac5 = tfac4 * tfac1

        rfac1 = rhobar - 1.0
        rfac2 = rfac1 * rfac1
        rfac3 = rfac2 * rfac1
        rfac4 = rfac3 * rfac1
        rfac5 = rfac4 * rfac1
        rfac6 = rfac5 * rfac1

        sum_mu = (_HIJ[0][0]       + _HIJ[1][0]*tfac1       + _HIJ[4][0]*tfac4       + _HIJ[5][0]*tfac5 +
                  _HIJ[0][1]*rfac1 + _HIJ[1][1]*tfac1*rfac1 + _HIJ[2][1]*tfac2*rfac1 + _HIJ[3][1]*tfac3*rfac1 +
                  _HIJ[0][2]*rfac2 + _HIJ[1][2]*tfac1*rfac2 + _HIJ[2][2]*tfac2*rfac2 +
                  _HIJ[0][3]*rfac3 + _HIJ[1][3]*tfac1*rfac3 + _HIJ[2][3]*tfac2*rfac3 + _HIJ[3][3]*tfac3*rfac3 +
                  _HIJ[0][4]*rfac4 + _HIJ[3][4]*tfac3*rfac4 +
                  _HIJ[1][5]*tfac1*rfac5 + _HIJ[3][6]*tfac3*rfac6)
        mu1bar = math.exp(rhobar * sum_mu)

        mu2bar = 1.0
        if 0.9970 <= tbar <= 1.0082 and 0.755 <= rhobar <= 1.290:
            kappa = 0.0
            for attr in ("isothermalCompressibility", "isothermal_compressibility", "_isothermal_compressibility"):
                val = getattr(self._thermo, attr, None)
                if val is not None:
                    kappa = float(val() if callable(val) else val)
                    break
            drhodp = kappa * dens * (_PRESSTAR / _RHOSTAR)
            xsi = rhobar * drhodp
            if xsi >= 21.93:
                mu2bar = 0.922 * (xsi ** 0.0263)

        mubar = mu0bar * mu1bar * mu2bar
        return CallableFloat(mubar * _MUSTAR)

    def thermalConductivity(self) -> CallableFloat:
        temp = float(self._thermo.temperature)
        dens = float(self._thermo.density)

        rhobar = dens / _RHOSTAR
        tbar = temp / _TSTAR
        tbar2 = tbar * tbar
        tbar3 = tbar2 * tbar
        lambda0bar = math.sqrt(tbar) / (_L[0] + _L[1]/tbar + _L[2]/tbar2 + _L[3]/tbar3)

        tfac1 = 1.0 / tbar - 1.0
        tfac2 = tfac1 * tfac1
        tfac3 = tfac2 * tfac1
        tfac4 = tfac3 * tfac1
        tfac5 = tfac4 * tfac1

        rfac1 = rhobar - 1.0
        rfac2 = rfac1 * rfac1
        rfac3 = rfac2 * rfac1
        rfac4 = rfac3 * rfac1
        rfac5 = rfac4 * rfac1
        rfac6 = rfac5 * rfac1

        sum_lam = (_LJI[0][0]       + _LJI[0][1]*tfac1        + _LJI[0][2]*tfac2       + _LJI[0][3]*tfac3       + _LJI[0][4]*tfac4       +
                   _LJI[1][0]*rfac1 + _LJI[1][1]*tfac1*rfac1  + _LJI[1][2]*tfac2*rfac1 + _LJI[1][3]*tfac3*rfac1 + _LJI[1][4]*tfac4*rfac1 +
                   _LJI[2][0]*rfac2 + _LJI[2][1]*tfac1*rfac2  + _LJI[2][2]*tfac2*rfac2 + _LJI[2][3]*tfac3*rfac2 +
                   _LJI[3][0]*rfac3 + _LJI[3][1]*tfac1*rfac3  + _LJI[3][2]*tfac2*rfac3 + _LJI[3][3]*tfac3*rfac3 +
                   _LJI[4][0]*rfac4 + _LJI[4][1]*tfac1*rfac4  + _LJI[4][2]*tfac2*rfac4 +
                   _LJI[5][0]*rfac5 + _LJI[5][1]*tfac1*rfac5  + _LJI[5][2]*tfac2*rfac5)
        lambda1bar = math.exp(rhobar * sum_lam)

        mu0bar = math.sqrt(tbar) / (_H[0] + _H[1]/tbar + _H[2]/tbar2 + _H[3]/tbar3)
        sum_mu = (_HIJ[0][0]       + _HIJ[1][0]*tfac1       + _HIJ[4][0]*tfac4       + _HIJ[5][0]*tfac5 +
                  _HIJ[0][1]*rfac1 + _HIJ[1][1]*tfac1*rfac1 + _HIJ[2][1]*tfac2*rfac1 + _HIJ[3][1]*tfac3*rfac1 +
                  _HIJ[0][2]*rfac2 + _HIJ[1][2]*tfac1*rfac2 + _HIJ[2][2]*tfac2*rfac2 +
                  _HIJ[0][3]*rfac3 + _HIJ[1][3]*tfac1*rfac3 + _HIJ[2][3]*tfac2*rfac3 + _HIJ[3][3]*tfac3*rfac3 +
                  _HIJ[0][4]*rfac4 + _HIJ[3][4]*tfac3*rfac4 +
                  _HIJ[1][5]*tfac1*rfac5 + _HIJ[3][6]*tfac3*rfac6)
        mu1bar = math.exp(rhobar * sum_mu)

        t2r2 = tbar2 / (rhobar * rhobar)
        kappa = 0.0
        for attr in ("isothermalCompressibility", "isothermal_compressibility", "_isothermal_compressibility"):
            val = getattr(self._thermo, attr, None)
            if val is not None:
                kappa = float(val() if callable(val) else val)
                break
        xsi = rhobar * rhobar * kappa * _PRESSTAR
        xsipow = (xsi ** 0.4678) if xsi > 0.0 else 0.0
        temp2 = (tbar - 1.0) * (tbar - 1.0)
        alpha = 0.0
        for attr in ("thermalExpansionCoeff", "thermal_expansion_coeff", "_thermal_expansion_coeff"):
            val = getattr(self._thermo, attr, None)
            if val is not None:
                alpha = float(val() if callable(val) else val)
                break
        dpdT_const_rho = (alpha / kappa) if kappa != 0.0 else 0.0
        dpdT_const_rho *= (_TSTAR / _PRESSTAR)
        lambda2bar = 0.0013848 / (mu0bar * mu1bar) * t2r2 * dpdT_const_rho * dpdT_const_rho *                      xsipow * math.sqrt(rhobar) * math.exp(-18.66 * temp2 - rfac4)
        lamb = (lambda0bar * lambda1bar + lambda2bar) * _LAMBDASTAR
        return CallableFloat(lamb)

    @property
    def thermal_conductivity(self) -> CallableFloat:
        return self.thermalConductivity()


class TransportFactory:
    """Factory for transport managers, matching native Cantera's singleton factory."""
    _instance = None
    _models = {
        "none": Transport,
        "transport": Transport,
        "": Transport,
        "unity-lewis-number": UnityLewisTransport,
        "unitylewis": UnityLewisTransport,
        "mixture-averaged": MixTransport,
        "mix": MixTransport,
        "mixture-averaged-ck": MixTransport,
        "ck_mix": MixTransport,
        "multicomponent": MultiTransport,
        "multi": MultiTransport,
        "multicomponent-ck": MultiTransport,
        "ck_multi": MultiTransport,
        "ionized-gas": IonGasTransport,
        "ion": IonGasTransport,
        "water": WaterTransport,
        "high-pressure": HighPressureGasTransport,
        "highp": HighPressureGasTransport,
        "high-pressure-chung": ChungHighPressureGasTransport,
        "dustygas": DustyGasTransport,
    }

    @classmethod
    def factory(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def deleteFactory(cls):
        cls._instance = None

    def newTransport(self, model_or_thermo, thermo=None):
        transportLock = True  # Evidence row #5278, #5279 mutex lock guard
        if thermo is None and not isinstance(model_or_thermo, str):
            phase = model_or_thermo
            model = "none"
            if hasattr(phase, 'input_data') and isinstance(phase.input_data, dict):
                model = phase.input_data.get('transport', 'none')
            return self.newTransport(model, phase)
        model = str(model_or_thermo).strip().lower()
        phase = thermo
        state = [0.0] * (phase.n_species + 2) if phase else []  # Evidence row #5281
        if model == "none":
            return Transport(phase)
        if model not in self._models:
            raise CanteraError("TransportFactory::newTransport", f"Unknown transport model: {model}")
        cls_tr = self._models[model]
        mode = 1 if "ck" in model else 0
        if issubclass(cls_tr, (MixTransport, GasTransport)):
            mech = getattr(phase, '_mechanism', None)
            if mech:
                tr = cls_tr.from_mechanism(mech, thermo=phase)
            else:
                tr = cls_tr(phase)
        else:
            tr = cls_tr(phase)
        if hasattr(tr, 'init'):
            tr.init(phase, mode=mode)
        return tr


def newTransport(thermo_or_model, model_or_thermo=None) -> Transport:
    fac = TransportFactory.factory()
    if isinstance(thermo_or_model, str):
        return fac.newTransport(thermo_or_model, model_or_thermo)
    elif model_or_thermo is not None and isinstance(model_or_thermo, str):
        return fac.newTransport(model_or_thermo, thermo_or_model)
    else:
        return fac.newTransport(thermo_or_model)


def newTransportData(node=None) -> TransportData:
    return GasTransportData()
