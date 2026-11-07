"""Independent reference evaluator for pinned Cantera cubic EOS root solver.

Directly ports C++ MixtureFugacityTP::solveCubic from pinned Cantera
commit 726522be4e2a13454d8415b7ef799d621f665cf3 (MixtureFugacityTP.cpp lines 774-1014).
Does NOT import or rely on cantera_python.cubic_eos.
"""
from __future__ import annotations
import math

GasConstant = 8314.46261815324  # J / kmol / K matching Cantera

def solve_cubic_source_oracle(T: float, pres: float, a: float, b: float,
                              aAlpha: float, an: float, bn: float, cn: float, dn: float,
                              tc: float, vc: float) -> tuple[int, list[float]]:
    """Direct reference translation of MixtureFugacityTP::solveCubic."""
    if T <= 0.0:
        raise ValueError(f"negative temperature T = {T}")

    Vroot = [0.0, 0.0, 0.0]
    xN = -bn / (3.0 * an)
    deltaNumerator = bn * bn - 3.0 * an * cn
    delta2 = deltaNumerator / (9.0 * an * an)
    delta = math.sqrt(delta2) if delta2 > 0.0 else 0.0

    ratio1 = 3.0 * an * cn / (bn * bn) if bn != 0.0 else 1.0
    ratio2 = pres * b / (GasConstant * T)
    if abs(ratio1) < 1e-7:
        ratio3 = (aAlpha / (GasConstant * T)) * (pres / (GasConstant * T))
        if abs(ratio2) < 1e-5 and abs(ratio3) < 1e-5:
            zz = 1.0
            for _ in range(10):
                znew = zz / (zz - ratio2) - ratio3 / (zz + ratio1)
                deltaz = znew - zz
                zz = znew
                if abs(deltaz) < 1e-14:
                    break
            v = zz * GasConstant * T / pres
            Vroot[0] = v
            return 1, Vroot

    h2 = 4.0 * an * an * (delta2 ** 3)
    h = 2.0 * an * delta * delta2
    yTerm1 = 2.0 * (bn ** 3) / (27.0 * an * an)
    yTerm2 = -bn * cn / (3.0 * an)
    yN = yTerm1 + yTerm2 + dn
    disc = yN * yN - h2

    cancellationTol = 64.0 * 2.220446049250313e-16
    deltaScale = max(abs(bn * bn), abs(3.0 * an * cn))
    yScale = max(abs(yTerm1), abs(yTerm2), abs(dn))
    tripleRoot = (abs(deltaNumerator) <= cancellationTol * deltaScale
                  and abs(yN) <= cancellationTol * yScale)
    if tripleRoot:
        delta = 0.0
        disc = 0.0

    if not tripleRoot and abs(abs(h) - abs(yN)) < 1e-10:
        if disc > 1e-10:
            raise ValueError("value of yN and h are too high, unrealistic roots may be obtained")
        disc = 0.0

    if disc < -1e-14:
        nSolnValues = 3
    elif abs(disc) <= 1e-14:
        nSolnValues = 1 if tripleRoot else 2
    else:
        nSolnValues = 1

    def physicalRoot(v: float) -> bool:
        vmin = max(0.0, b * (1.0 + 1e-12))
        return math.isfinite(v) and v > vmin

    if disc > 1e-14:
        tmpD = math.sqrt(disc)
        tmp1 = (-yN + tmpD) / (2.0 * an)
        sgn1 = 1.0 if tmp1 >= 0.0 else -1.0
        p1 = (abs(tmp1) ** (1.0 / 3.0)) * sgn1
        tmp2 = (-yN - tmpD) / (2.0 * an)
        sgn2 = 1.0 if tmp2 >= 0.0 else -1.0
        p2 = (abs(tmp2) ** (1.0 / 3.0)) * sgn2
        alpha = xN + p1 + p2
        Vroot[0] = alpha
        Vroot[1] = 0.0
        Vroot[2] = 0.0
    elif disc < -1e-14:
        val = max(-1.0, min(1.0, -yN / h))
        theta = math.acos(val) / 3.0
        twoThirdPi = 2.0 * math.pi / 3.0
        alpha = xN + 2.0 * delta * math.cos(theta)
        beta = xN + 2.0 * delta * math.cos(theta + twoThirdPi)
        gamma = xN + 2.0 * delta * math.cos(theta + 2.0 * twoThirdPi)
        Vroot[0] = beta   # liquid phase root
        Vroot[1] = gamma  # intermediate / unstable root
        Vroot[2] = alpha  # gas phase root
    elif abs(disc) <= 1e-14:
        if tripleRoot:
            Vroot[0] = xN
            Vroot[1] = xN
            Vroot[2] = xN
        else:
            if yN > 0.0:
                Vroot[0] = xN - 2.0 * delta  # liquid
                Vroot[1] = xN + delta
                Vroot[2] = xN + delta
            else:
                Vroot[0] = xN + delta        # gas
                Vroot[1] = xN - 2.0 * delta
                Vroot[2] = xN - 2.0 * delta

    # Polishing matching lines 953-984
    for i in range(nSolnValues):
        if not physicalRoot(Vroot[i]):
            continue
        for n in range(20):
            res = an * Vroot[i]**3 + bn * Vroot[i]**2 + cn * Vroot[i] + dn
            if abs(res) < 1e-14:
                break
            dresdV = 3.0 * an * Vroot[i]**2 + 2.0 * bn * Vroot[i] + cn
            if dresdV == 0.0:
                break
            del_v = -res / dresdV
            Vroot[i] += del_v
            if abs(del_v) / (abs(Vroot[i]) + abs(del_v)) < 1e-14:
                break
            res2 = an * Vroot[i]**3 + bn * Vroot[i]**2 + cn * Vroot[i] + dn
            if abs(res2) < abs(res):
                continue
            else:
                Vroot[i] -= del_v
                Vroot[i] += 0.1 * del_v

        res = an * Vroot[i]**3 + bn * Vroot[i]**2 + cn * Vroot[i] + dn
        dresdV = 3.0 * an * Vroot[i]**2 + 2.0 * bn * Vroot[i] + cn
        if abs(res) > 1e-14 and abs(res) > 1e-14 * abs(dresdV) * abs(Vroot[i]):
            raise ValueError(f"root failed to converge for T = {T}, p = {pres} with V = {Vroot[i]}")

    if nSolnValues == 1 and not physicalRoot(Vroot[0]):
        raise ValueError(f"single real root is non-physical for T = {T}, p = {pres} (V = {Vroot[0]}, b = {b})")

    # Branch sign convention
    if nSolnValues == 1:
        if T > tc:
            if Vroot[0] < vc:
                nSolnValues = -1
        else:
            if Vroot[0] < xN:
                nSolnValues = -1
    else:
        if nSolnValues == 2 and delta > 1e-14:
            nSolnValues = -2

    return nSolnValues, Vroot
