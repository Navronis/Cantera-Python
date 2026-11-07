"""TPX pure substance thermodynamic models.

Pure-Python port of Cantera's tpx library (Sub.cpp, Water.cpp, CarbonDioxide.cpp, etc.)
implements exact multiparameter Helmholtz and analytical equations of state,
saturation properties, and property pairs.
"""
from __future__ import annotations

import math
from math import exp, log, sqrt
from typing import Optional
import numpy as np

from .constants import GasConstant, OneAtm

class TPXError(Exception):
    """Exception for TPX thermodynamic failures."""
    pass


def _ieee_divide(numerator: float, denominator: float) -> float:
    """Match IEEE-754 C++ floating-point division for a zero denominator.

    Python raises ``ZeroDivisionError`` for float division by signed zero,
    whereas the pinned C++ TPX solver performs the division directly. Keep the
    native NaN/infinity result so the subsequent source steps determine the
    solver outcome; do not replace the Newton direction with a regularized
    gradient step.
    """
    # Inputs can be NumPy scalar values from the analytic TPX equations.
    # Convert to Python float so division by zero follows the explicit
    # IEEE-754 handling below instead of NumPy's warning-based behavior.
    numerator = float(numerator)
    denominator = float(denominator)
    try:
        return numerator / denominator
    except ZeroDivisionError:
        if numerator == 0.0 or math.isnan(numerator):
            return math.nan
        sign = math.copysign(1.0, numerator) * math.copysign(1.0, denominator)
        return math.copysign(math.inf, sign)

class PropertyPair:
    TV = 12
    HP = 34
    SP = 54
    PV = 42
    TP = 14
    UV = 62
    ST = 51
    SV = 52
    UP = 64
    VH = 23
    TH = 13
    SH = 53
    PX = 47
    TX = 17
    VT = -12
    PH = -34
    PS = -54
    VP = -42
    PT = -14
    VU = -62
    TS = -51
    VS = -52
    PU = -64
    HV = -23
    HT = -13
    HS = -53
    XP = -47
    XT = -17

class PropertyFlag:
    H = 0
    S = 1
    U = 2
    V = 3
    P = 4
    T = 5

Undef = 999.1234
Tgiven = 0
Pgiven = 1


class Substance:
    def BracketSlope(self, Pressure: float) -> None:
        """Volume step and bracket slope update from pinned Sub.cpp lines 768-785."""
        if getattr(self, 'kbr', 0) == 0:
            if getattr(self, 'v_here', 1.0) < self.Vcrit():
                self.dv = -0.05 * getattr(self, 'v_here', 1.0)
            else:
                self.dv = 0.2 * getattr(self, 'v_here', 1.0)
            if getattr(self, 'Vmin', 0.0) > 0.0:
                self.dv = 0.2 * getattr(self, 'v_here', 1.0)
            if getattr(self, 'Vmax', 1e30) < 1e30:
                self.dv = -0.05 * getattr(self, 'v_here', 1.0)
        else:
            dpdv = (getattr(self, 'Pmax', 1.0) - getattr(self, 'Pmin', 0.0)) / max(1e-300, getattr(self, 'Vmax', 1.0) - getattr(self, 'Vmin', 0.0))
            self.v_here = getattr(self, 'Vmax', 1.0)
            self.P_here = getattr(self, 'Pmax', 1.0)
            self.dv = getattr(self, 'dvbf', 1.0) * (Pressure - self.P_here) / dpdv
            self.dvbf = 0.5 * getattr(self, 'dvbf', 1.0)

    def Lever(self, itp: int, sat: float, val: float, ifunc: int) -> int:
        """Port of Substance::Lever from pinned Sub.cpp lines 596-637."""
        Tsave = self.T
        Rhosave = self.Rho
        if itp == Tgiven:
            if sat >= self.Tcrit():
                return 0
            self.T = sat
            psat = self.Ps()
        elif itp == Pgiven:
            if sat >= self.Pcrit():
                return 0
            psat = sat
            try:
                self.T = self.Tsat(psat)
            except (TPXError, ValueError):
                self.T = Tsave
                self.Rho = Rhosave
                return 0
        else:
            raise TPXError("Substance::Lever: General error")
        self.Set(PropertyPair.TX, self.T, 1.0)
        Valg = self.vprop(ifunc)
        self.Set(PropertyPair.TX, self.T, 0.0)
        Valf = self.vprop(ifunc)
        if Valf <= val <= Valg:
            xx = (val - Valf) / max(1e-300, Valg - Valf)
            vv = (1.0 - xx) / self.Rhf + xx / self.Rhv
            self.set_v(vv)
            return 1
        else:
            self.T = Tsave
            self.Rho = Rhosave
            return 0

    def dPsdT(self, T: Optional[float] = None) -> float:
        """Finite-difference dPsat/dT from pinned Sub.cpp lines 254-269."""
        DeltaT = 1e-6
        tsave = self.T
        if T is not None:
            self.T = float(T)
        ps1 = self.Ps()
        if self.T + DeltaT < self.Tcrit():
            self.T += DeltaT
            dpdt = (self.Ps() - ps1) / DeltaT
        else:
            self.T -= DeltaT
            dpdt = (ps1 - self.Ps()) / DeltaT
        self.T = tsave
        return float(dpdt)

    def dPsatdT(self, T: Optional[float] = None) -> float:
        """Return dPsdT evaluated at T or current temperature."""
        t_eval = self.T if T is None else float(T)
        return float(self.dPsdT(t_eval))

    def ideal(self) -> bool:
        """Return True if substance is modeled as ideal gas."""
        flag = getattr(self, '_is_ideal', False)
        return bool(flag)

    def setStdState(self, h0: float = 0.0, s0: float = 0.0,
                    t0: float = 298.15, p0: float = 1.01325e5) -> None:
        """Set the source reference-state offsets after evaluating T,P."""
        self.Set(PropertyPair.TP, t0, p0)
        self.m_energy_offset = 0.0
        self.m_entropy_offset = 0.0
        self.m_energy_offset = h0 - self.hp()
        self.m_entropy_offset = s0 - self.sp()

    def set_Rho(self, rho: float) -> None:
        r = float(rho)
        self.Rho = r

    def vp(self) -> float:
        """Specific volume v = 1/rho from pinned Sub.h line 208."""
        r = float(self.Rho)
        return 1.0 / r

    def C(self, *args) -> float:
        """Virial coefficient multiplier, overridden in subclasses."""
        idx = int(args[0]) if args else 0
        return float(idx * 0.0)

    def Cprime(self, *args) -> float:
        """Temperature derivative of C, overridden in subclasses."""
        idx = int(args[0]) if args else 0
        return float(idx * 0.0)

    def H(self, *args) -> float:
        """Enthalpy or density multiplier, overridden in subclasses."""
        if not args:
            return float(self.h())
        idx = int(args[0])
        return float(idx * 0.0)

    def I(self, *args) -> float:
        """Density integral term, overridden in subclasses."""
        idx = int(args[0]) if args else 0
        return float(idx * 0.0)

    def Pp(self) -> float:
        """Pressure computed directly from underlying EOS."""
        raise NotImplementedError("Substance::Pp must be implemented by subclass")

    def Psat(self) -> float:
        """Saturation pressure correlation [Pa]."""
        raise NotImplementedError("Substance::Psat must be implemented by subclass")

    def W(self, *args) -> float:
        """Molecular weight or W term."""
        if not args:
            return float(self.MolWt())
        idx = int(args[0])
        return float(idx * 0.0)

    def icv(self, *args) -> float:
        """Specific heat integral term."""
        if not args:
            return float(self.cv())
        idx = int(args[0])
        return float(idx * 0.0)

    def ldens(self) -> float:
        """Liquid density correlation [kg/m^3]."""
        raise NotImplementedError("Substance::ldens must be implemented by subclass")

    def sp(self) -> float:
        """Entropy of single-phase state [J/kg/K]."""
        raise NotImplementedError("Substance::sp must be implemented by subclass")

    def up(self) -> float:
        """Internal energy of single-phase state [J/kg]."""
        raise NotImplementedError("Substance::up must be implemented by subclass")

    """Base class from which all pure substances are derived."""
    def __init__(self):
        self.T = Undef
        self.Rho = Undef
        self.Tslast = Undef
        self.Rhf = Undef
        self.Rhv = Undef
        self.Pst = Undef
        self.m_energy_offset = 0.0
        self.m_entropy_offset = 0.0
        self.m_name = ""
        self.m_formula = ""

    def MolWt(self) -> float:
        raise NotImplementedError

    def Tcrit(self) -> float:
        raise NotImplementedError

    def Pcrit(self) -> float:
        raise NotImplementedError

    def Vcrit(self) -> float:
        raise NotImplementedError

    def Tmin(self) -> float:
        raise NotImplementedError

    def Tmax(self) -> float:
        raise NotImplementedError

    def name(self) -> str:
        return self.m_name

    def formula(self) -> str:
        return self.m_formula

    def P(self) -> float:
        return self.Ps() if self.TwoPhase() else self.Pp()

    def Temp(self) -> float:
        return self.T

    def v(self) -> float:
        return self.prop(PropertyFlag.V)

    def u(self) -> float:
        return self.prop(PropertyFlag.U)

    def h(self) -> float:
        return self.prop(PropertyFlag.H)

    def s(self) -> float:
        return self.prop(PropertyFlag.S)

    def f(self) -> float:
        return self.u() - self.T * self.s()

    def g(self) -> float:
        return self.h() - self.T * self.s()

    def cv(self) -> float:
        if self.TwoPhase(strict=True):
            return float('nan')
        Tsave = self.T
        dt = 1e-4 * self.T
        x0 = self.x()
        T1 = max(self.Tmin(), Tsave - dt)
        T2 = min(self.Tmax(), Tsave + dt)
        self.set_T(T1)
        x1 = self.x()
        if (x0 == 1.0 or x0 == 0.0) and x1 != x0:
            T1 = Tsave
            self.set_T(T1)
        s1 = self.s()

        self.set_T(T2)
        x2 = self.x()
        if (x0 == 1.0 or x0 == 0.0) and x2 != x0:
            T2 = Tsave
            self.set_T(T2)
        s2 = self.s()

        self.set_T(Tsave)
        return self.T * (s2 - s1) / (T2 - T1)

    def cp(self) -> float:
        """Cp from pinned Sub.cpp lines 82-132 with phase-aware saturation boundary."""
        if self.TwoPhase(strict=True):
            return float('inf')
        Tsave = self.T
        dt = 1e-4 * self.T
        RhoSave = self.Rho
        p0 = self.P()
        rhf_val = getattr(self, 'Rhf', float('inf'))
        if self.Rho >= rhf_val:
            # Initial state is pure liquid
            T1 = max(self.Tmin(), Tsave - dt)
            self.Set(PropertyPair.TP, T1, p0)
            s1 = self.s()
            try:
                T2 = min(self.Tsat(p0), Tsave + dt)
            except Exception:
                T2 = Tsave + dt
            if T2 < Tsave + dt:
                self.Set(PropertyPair.TX, T2, 0.0)
            else:
                self.Set(PropertyPair.TP, T2, p0)
            s2 = self.s()
        else:
            # Initial state is pure vapor
            try:
                T1 = max(self.Tsat(p0), Tsave - dt)
            except Exception:
                T1 = Tsave - dt
            if T1 > Tsave - dt:
                self.Set(PropertyPair.TX, T1, 1.0)
            else:
                self.Set(PropertyPair.TP, T1, p0)
            s1 = self.s()
            T2 = min(self.Tmax(), Tsave + dt)
            self.Set(PropertyPair.TP, T2, p0)
            s2 = self.s()
        self.Set(PropertyPair.TV, Tsave, 1.0 / RhoSave)
        return self.T * (s2 - s1) / (T2 - T1)

    def thermalExpansionCoeff(self) -> float:
        """Isobaric thermal expansion coefficient beta from pinned Sub.cpp lines 134-185."""
        if self.TwoPhase(strict=True):
            return float('inf')
        Tsave = self.T
        dt = 1e-4 * self.T
        RhoSave = self.Rho
        p0 = self.P()
        rhf_val = getattr(self, 'Rhf', float('inf'))
        if self.Rho >= rhf_val:
            T1 = max(self.Tmin(), Tsave - dt)
            self.Set(PropertyPair.TP, T1, p0)
            v1 = self.v()
            try:
                T2 = min(self.Tsat(p0), Tsave + dt)
            except Exception:
                T2 = Tsave + dt
            if T2 < Tsave + dt:
                self.Set(PropertyPair.TX, T2, 0.0)
            else:
                self.Set(PropertyPair.TP, T2, p0)
            v2 = self.v()
        else:
            try:
                T1 = max(self.Tsat(p0), Tsave - dt)
            except Exception:
                T1 = Tsave - dt
            if T1 > Tsave - dt:
                self.Set(PropertyPair.TX, T1, 1.0)
            else:
                self.Set(PropertyPair.TP, T1, p0)
            v1 = self.v()
            T2 = min(self.Tmax(), Tsave + dt)
            self.Set(PropertyPair.TP, T2, p0)
            v2 = self.v()
        self.Set(PropertyPair.TV, Tsave, 1.0 / RhoSave)
        return 2.0 * (v2 - v1) / ((v2 + v1) * (T2 - T1))

    def isothermalCompressibility(self) -> float:
        """Isothermal compressibility kappa from pinned Sub.cpp lines 187-233."""
        if self.TwoPhase(strict=True):
            return float('inf')
        Psave = self.P()
        dp = 1e-4 * Psave
        RhoSave = self.Rho
        v0 = self.v()
        rhf_val = getattr(self, 'Rhf', float('inf'))
        if self.Rho >= rhf_val:
            P2 = Psave + dp
            self.Set(PropertyPair.TP, self.T, P2)
            v2 = self.v()
            try:
                P1 = max(self.Ps(), Psave - dp)
            except Exception:
                P1 = max(1e-12, Psave - dp)
            if P1 > Psave - dp:
                self.Set(PropertyPair.TX, self.T, 0.0)
            else:
                self.Set(PropertyPair.TP, self.T, P1)
            v1 = self.v()
        else:
            P1 = max(0.0, Psave - dp)
            self.Set(PropertyPair.TP, self.T, P1)
            v1 = self.v()
            try:
                P2 = min(self.Ps(), Psave + dp)
            except Exception:
                P2 = Psave + dp
            if P2 < Psave + dp:
                self.Set(PropertyPair.TX, self.T, 1.0)
            else:
                self.Set(PropertyPair.TP, self.T, P2)
            v2 = self.v()
        self.Set(PropertyPair.TV, self.T, 1.0 / RhoSave)
        return -(v2 - v1) / (v0 * (P2 - P1))

    def internalPressure(self) -> float:
        """Internal pressure pi_T = (dU/dV)|T from pinned Sub.cpp lines 235-252."""
        if self.TwoPhase(strict=True):
            return float('nan')
        Tsave = self.T
        vsave = self.v()
        dv = max(1e-8, 1e-6 * abs(vsave))
        v1 = max(1e-300, vsave - dv)
        v2 = vsave + dv
        self.Set(PropertyPair.TV, Tsave, v1)
        u1 = self.u()
        self.Set(PropertyPair.TV, Tsave, v2)
        u2 = self.u()
        self.Set(PropertyPair.TV, Tsave, vsave)
        return (u2 - u1) / (v2 - v1)

    def Ps(self) -> float:
        if self.T < self.Tmin() or self.T > self.Tcrit():
            raise TPXError(f"Illegal temperature value: {self.T}")
        self.update_sat()
        return self.Pst

    def Tsat(self, p: float) -> float:
        """Newton iteration for Tsat from pinned Sub.cpp lines 301-341."""
        Tsave = self.T
        self.T = self.Tmin()
        try:
            Pmin = self.Ps()
        except Exception:
            Pmin = 0.0
        self.T = Tsave
        if p <= 0.0 or p > self.Pcrit() or p < Pmin:
            raise TPXError(f"Illegal pressure value for Tsat: {p}")
        if self.T < self.Tmin() or self.T > self.Tcrit():
            self.T = 0.5 * (self.Tcrit() - self.Tmin())
        tol = 1e-6 * p
        for _ in range(100):
            self.T = min(self.T, self.Tcrit())
            self.T = max(self.T, self.Tmin())
            dp = p - self.Ps()
            if abs(dp) <= tol:
                tsat = self.T
                self.T = Tsave
                return tsat
            dpdt = self.dPsdT()
            if not math.isfinite(dpdt) or abs(dpdt) < 1e-300:
                break
            dt = dp / dpdt
            dta = abs(dt)
            dtm = 0.1 * self.T
            if dta > dtm:
                dt = dt * dtm / dta
            self.T += dt
        self.T = Tsave
        raise TPXError(f"Tsat did not converge at pressure {p}")


    def x(self) -> float:
        if self.T >= self.Tcrit():
            return 0.0 if (1.0 / self.Rho) < self.Vcrit() else 1.0
        self.update_sat()
        # Preserve pinned Sub.cpp branch precedence at coincident saturation
        # densities: the vapor-boundary condition is checked first.
        if self.Rho <= self.Rhv:
            return 1.0
        elif self.Rho >= self.Rhf:
            return 0.0
        else:
            return (1.0 / self.Rho - 1.0 / self.Rhf) / (1.0 / self.Rhv - 1.0 / self.Rhf)

    def TwoPhase(self, strict: bool = False) -> int:
        if self.T >= self.Tcrit():
            return 0
        self.update_sat()
        if strict:
            return 1 if (self.Rho < self.Rhf and self.Rho > self.Rhv) else 0
        return 1 if (self.Rho <= self.Rhf and self.Rho >= self.Rhv) else 0

    def hp(self) -> float:
        return self.up() + self.Pp() / self.Rho

    def gp(self) -> float:
        return self.hp() - self.T * self.sp()

    def prop(self, ijob: int) -> float:
        """Property evaluation from pinned Sub.cpp lines 742-763."""
        if ijob == PropertyFlag.P:
            return float(self.P())
        if ijob == PropertyFlag.T:
            return float(self.T)
        xx = self.x()
        if 0.0 < xx < 1.0:
            Rho_save = self.Rho
            self.Rho = self.Rhv
            vp = self.vprop(ijob)
            self.Rho = self.Rhf
            lp = self.vprop(ijob)
            pp = (1.0 - xx) * lp + xx * vp
            self.Rho = Rho_save
            return float(pp)
        else:
            return float(self.vprop(ijob))


    def vprop(self, ijob: int) -> float:
        if ijob == PropertyFlag.H:
            return self.hp()
        elif ijob == PropertyFlag.S:
            return self.sp()
        elif ijob == PropertyFlag.U:
            return self.up()
        elif ijob == PropertyFlag.V:
            return 1.0 / self.Rho
        elif ijob == PropertyFlag.P:
            return self.Pp()
        elif ijob == PropertyFlag.T:
            return self.T
        else:
            raise TPXError("Invalid property job flag")

    def set_T(self, val: float):
        if val < self.Tmin() or val > self.Tmax():
            raise TPXError(f"Temperature out of range: {val}")
        self.T = val

    def set_v(self, val: float):
        # Match the native positive-value predicate: NaN is rejected because
        # ``NaN > 0`` is false, while a ``<= 0`` test would let it through.
        if not val > 0.0:
            raise TPXError(f"Negative specific volume: {val}")
        self.Rho = 1.0 / val

    def set_TPp(self, t0: float, p0: float):
        # Direct translation of pinned Substance::set_TPp. The independent
        # variable is specific volume and the bracket is retained across
        # pressure-slope changes; this handles steep liquid and non-monotonic
        # subcritical branches where an unconstrained density Newton step fails.
        self.set_T(t0)
        v_save = _ieee_divide(1.0, self.Rho)
        v_here = v_save
        self.kbr = 0; dvbf = 1.0; vmin = 0.0; vmax = 1e30; pmin = 1e30; pmax = 0.0
        dvs1 = 2.0 * self.Vcrit(); dvs2 = 0.7 * self.Vcrit()
        v_prev = v_here; p_prev = 0.0; res_prev = 0.0; have_prev = False
        for loop in range(201):
            self.set_v(v_here)
            p_here = self.Pp()
            if loop != 0 and abs(p0-p_here) < 1e-7*p0:
                self.set_v(v_here); return
            res = p_here-p0
            if have_prev and self.kbr == 0 and res*res_prev < 0.0:
                if v_here < v_prev:
                    vmin,pmin,vmax,pmax = v_here,p_here,v_prev,p_prev
                else:
                    vmin,pmin,vmax,pmax = v_prev,p_prev,v_here,p_here
                self.kbr=1
            v_prev,p_prev,res_prev,have_prev = v_here,p_here,res,True
            if p_here < 0.0:
                v_here,dv,dvbf = self._bracket_slope(
                    p0,v_here,vmin,vmax,pmin,pmax,dvbf)
            else:
                dv = .001*v_here * (-1.0 if v_here <= self.Vcrit() else 1.0)
                self.set_v(v_here+dv)
                dpdv = (self.Pp()-p_here)/dv
                self.set_v(v_here)
                if dpdv > 0.0:
                    v_here,dv,dvbf = self._bracket_slope(
                        p0,v_here,vmin,vmax,pmin,pmax,dvbf)
                else:
                    if p_here > p0 and v_here > vmin: vmin=v_here
                    elif p_here < p0 and v_here < vmax: vmax=v_here
                    if v_here == vmin: pmin=p_here
                    if v_here == vmax: pmax=p_here
                    if vmin >= vmax: raise TPXError('TP density inversion: Vmin >= Vmax')
                    if vmin > 0.0 and vmax < 1e30: self.kbr=1
                    dvbf=1.0
                    if dpdv == 0.0:
                        dvbf=.5
                        v_here,dv,dvbf = self._bracket_slope(
                            p0,v_here,vmin,vmax,pmin,pmax,dvbf)
                    else: dv=(p0-p_here)/dpdv
            dvm=.2*v_here
            if v_here < dvs1: dvm*=.5
            if v_here < dvs2: dvm*=.5
            if self.kbr:
                vt=v_here+dv
                if vt < vmin or vt > vmax:
                    den=pmax-pmin
                    dv=vmin+_ieee_divide((p0-pmin)*(vmax-vmin), den)-v_here
            if abs(dv)>dvm: dv*=dvm/abs(dv)
            v_here += dv
            if abs(dv) <= 4.0*math.ulp(1.0)*abs(v_here):
                self.set_v(v_here)
                return
        self.set_v(v_save)
        raise TPXError(f'TP density inversion did not converge at T={t0}, P={p0}')

    def _bracket_slope(self, pressure, v_here, vmin, vmax, pmin, pmax, dvbf):
        if self.kbr == 0:
            dv = -0.05*v_here if v_here < self.Vcrit() else .2*v_here
            if vmin > 0.0: dv=.2*v_here
            if vmax < 1e30: dv=-.05*v_here
            return v_here, dv, dvbf
        dpdv=_ieee_divide(pmax-pmin,vmax-vmin)
        dv=_ieee_divide(dvbf*(pressure-pmax),dpdv)
        # Native BracketSlope mutates v_here to Vmax and halves dvbf when a
        # bracket is active; returning all state changes keeps that behavior
        # explicit in the Python translation.
        return vmax, dv, 0.5*dvbf

    def Set(self, XY: int, x0: float, y0: float):
        reverse = {
            PropertyPair.VT: PropertyPair.TV, PropertyPair.PH: PropertyPair.HP,
            PropertyPair.PS: PropertyPair.SP, PropertyPair.VP: PropertyPair.PV,
            PropertyPair.PT: PropertyPair.TP, PropertyPair.VU: PropertyPair.UV,
            PropertyPair.TS: PropertyPair.ST, PropertyPair.VS: PropertyPair.SV,
            PropertyPair.PU: PropertyPair.UP, PropertyPair.HV: PropertyPair.VH,
            PropertyPair.HT: PropertyPair.TH, PropertyPair.HS: PropertyPair.SH,
            PropertyPair.XP: PropertyPair.PX, PropertyPair.XT: PropertyPair.TX,
        }
        if XY in reverse:
            XY, x0, y0 = reverse[XY], y0, x0
        if XY == PropertyPair.TV:
            self.set_T(x0)
            self.set_v(y0)
        elif XY == PropertyPair.TP:
            self.set_T(x0)
            if self.T < self.Tcrit():
                ps = self.Ps()
                if abs(y0 - ps) / max(abs(y0), 1e-300) < 1e-8:
                    raise TPXError("Saturated mixture detected: use vapor fraction")
                # This is the pinned phase-selection branch. The subsequent
                # set_xy route performs the source Newton inversion.
                self.Set(PropertyPair.TX, x0, 1.0 if y0 < ps else 0.0)
            self._set_xy_source(PropertyFlag.T, x0, PropertyFlag.P, y0,
                                 1e-3, 0.0, 1e-8, 1e-8)
        elif XY == PropertyPair.TX:
            # Preserve the pinned Sub.cpp validation order and admissible
            # two-phase domain. set_T alone permits supercritical states,
            # while TX is defined only from Tmin through Tcrit.
            if y0 > 1.0 or y0 < 0.0:
                raise TPXError(f"Invalid vapor fraction, {y0}")
            if x0 < self.Tmin() or x0 > self.Tcrit():
                raise TPXError(
                    f"Illegal temperature value: {x0} "
                    "(supercritical or below triple point)"
                )
            self.set_T(x0)
            self.update_sat()
            vf = 1.0 / self.Rhf
            vg = 1.0 / self.Rhv
            v = (1.0 - y0) * vf + y0 * vg
            self.set_v(v)
        elif XY == PropertyPair.HP:
            if not self._lever(Pgiven, y0, x0, PropertyFlag.H):
                self._set_xy_source(PropertyFlag.H, x0, PropertyFlag.P, y0,
                                     1e-4, 0.0, 1e-8, 1e-8)
        elif XY == PropertyPair.SP:
            if not self._lever(Pgiven, y0, x0, PropertyFlag.S):
                self._set_xy_source(PropertyFlag.S, x0, PropertyFlag.P, y0,
                                     1e-7, 0.0, 1e-8, 1e-8)
        elif XY == PropertyPair.UV:
            self._set_xy_source(PropertyFlag.U, x0, PropertyFlag.V, y0,
                                 1e-4, 1e-8, 1e-8, 1e-8)
        elif XY == PropertyPair.PV:
            if not self._lever(Pgiven, x0, y0, PropertyFlag.V):
                self._set_xy_source(PropertyFlag.P, x0, PropertyFlag.V, y0,
                                     0.0, 1e-8, 1e-8, 1e-8)
        elif XY == PropertyPair.VH:
            self._set_xy_source(PropertyFlag.V, x0, PropertyFlag.H, y0,
                                 1e-8, 1e-4, 1e-8, 1e-8)
        elif XY == PropertyPair.TH:
            self._set_xy_source(PropertyFlag.T, x0, PropertyFlag.H, y0,
                                 1e-3, 1e-4, 1e-8, 1e-8)
        elif XY == PropertyPair.ST:
            if not self._lever(Tgiven, y0, x0, PropertyFlag.S):
                self._set_xy_source(PropertyFlag.S, x0, PropertyFlag.T, y0,
                                     1e-7, 1e-3, 1e-8, 1e-8)
        elif XY == PropertyPair.SV:
            self._set_xy_source(PropertyFlag.S, x0, PropertyFlag.V, y0,
                                 1e-7, 1e-8, 1e-8, 1e-8)
        elif XY == PropertyPair.UP:
            if not self._lever(Pgiven, y0, x0, PropertyFlag.U):
                self._set_xy_source(PropertyFlag.U, x0, PropertyFlag.P, y0,
                                     1e-4, 0.0, 1e-8, 1e-8)
        elif XY == PropertyPair.SH:
            self._set_xy_source(PropertyFlag.S, x0, PropertyFlag.H, y0,
                                 1e-7, 1e-4, 1e-8, 1e-8)
        elif XY == PropertyPair.PX:
            # Native PX first moves the phase to Tmin, then validates quality
            # and pressure. Keep this mutation order on exceptional inputs.
            self.set_T(self.Tmin())
            if y0 > 1.0 or y0 < 0.0:
                raise TPXError(f"Invalid vapor fraction, {y0}")
            if x0 < self.Ps() or x0 > self.Pcrit():
                raise TPXError(
                    f"Illegal pressure value: {x0} "
                    "(supercritical or below triple point)"
                )
            self.T = self.Tsat(x0); self.update_sat(); self.set_v((1-y0)/self.Rhf+y0/self.Rhv)
        else:
            raise TPXError(f"Unsupported property pair: {XY}")

    def _lever(self, independent, sat_value, target, flag):
        """Port of Substance::Lever for two-phase property inversion."""
        tsave, rsave = self.T, self.Rho
        try:
            if independent == Tgiven:
                if sat_value >= self.Tcrit():
                    return False
                self.T = sat_value
                psat = self.Ps()
            else:
                if sat_value >= self.Pcrit():
                    return False
                psat = sat_value
                try:
                    self.T = self.Tsat(psat)
                except Exception:
                    self.T, self.Rho = tsave, rsave
                    return False
            self.Set(PropertyPair.TX, self.T, 1.0)
            valg = self.vprop(flag)
            self.Set(PropertyPair.TX, self.T, 0.0)
            valf = self.vprop(flag)
            if target < valf or target > valg or valg == valf:
                self.T, self.Rho = tsave, rsave
                return False
            xx = (target-valf)/(valg-valf)
            self.set_v((1.0-xx)/self.Rhf + xx/self.Rhv)
            return True
        except (TPXError, ValueError, ZeroDivisionError, OverflowError):
            self.T, self.Rho = tsave, rsave
            return False

    def _set_xy_source(self, ifx, X, ify, Y, atx, aty, rtx, rty):
        """Direct Python translation of pinned ``Substance::set_xy``."""
        record_iterations = bool(getattr(self, "_record_solver_iterations", False))
        if record_iterations:
            self._last_set_xy_iterations = 0
        v_save = 1.0 / self.Rho if self.Rho != Undef else None
        t_save = self.T

        if self.T == Undef and self.Rho == Undef:
            self.Set(PropertyPair.TV, self.Tcrit() * 1.1, self.Vcrit() * 1.1)
            t_here = self.T
            v_here = 1.0 / self.Rho
        elif self.Rho == Undef:
            self.Set(PropertyPair.TV, self.T, self.Vcrit() * 1.1)
            t_here = self.T
            v_here = 1.0 / self.Rho
        else:
            v_here = v_save
            t_here = t_save

        dvs1, dvs2 = 2.0*self.Vcrit(), 0.7*self.Vcrit()
        for iteration in range(201):
            if record_iterations:
                self._last_set_xy_iterations = iteration + 1
            x_here, y_here = self.prop(ifx), self.prop(ify)
            if (abs(X-x_here) < atx + rtx*abs(X)
                    and abs(Y-y_here) < aty + rty*abs(Y)):
                return
            dt = 0.001*t_here
            if t_here + dt > self.Tmax(): dt = -dt
            dv = 0.001*v_here
            if v_here <= self.Vcrit(): dv = -dv
            self.Set(PropertyPair.TV, t_here+dt, v_here)
            dxdt, dydt = (self.prop(ifx)-x_here)/dt, (self.prop(ify)-y_here)/dt
            self.Set(PropertyPair.TV, t_here, v_here+dv)
            dxdv, dydv = (self.prop(ifx)-x_here)/dv, (self.prop(ify)-y_here)/dv
            det = dxdt*dydv - dydt*dxdv
            # Substance::set_xy in the pinned source divides both Newton
            # numerators directly by det, without a singular-Jacobian
            # regularization branch.
            dtn = _ieee_divide((X-x_here)*dydv - (Y-y_here)*dxdv, det)
            dvn = _ieee_divide((Y-y_here)*dxdt - (X-x_here)*dydt, det)
            dvm = 0.2*v_here
            if v_here < dvs1: dvm *= 0.5
            if v_here < dvs2: dvm *= 0.5
            dtm = 0.1*t_here
            if abs(dvn) > dvm: dvn *= dvm/abs(dvn)
            if abs(dtn) > dtm: dtn *= dtm/abs(dtn)
            v_here += dvn
            t_here += dtn
            t_here = max(self.Tmin(), min(self.Tmax(), t_here))
            if v_here <= 0.0:
                v_here = 0.0001
            self.Set(PropertyPair.TV, t_here, v_here)
        raise TPXError(f"No convergence. properties {ifx}={X}, {ify}={Y}")


    def _solve_scalar(self, flag, target, guess, temperature=False):
        """Pinned-style safeguarded scalar inversion with bracket expansion."""
        lo, hi = ((self.Tmin(), self.Tmax()) if temperature else (1e-9, max(1e5, 10*self.Roc if hasattr(self,'Roc') else 1e5)))
        def f(z):
            if temperature: self.set_T(z)
            else: self.Rho = z
            return self.vprop(flag)-target
        if temperature:
            grid=[lo+(hi-lo)*i/96 for i in range(97)]
        else:
            llo=math.log10(lo); lhi=math.log10(hi)
            grid=[10**(llo+(lhi-llo)*i/128) for i in range(129)]
        vals=[]
        for z in grid:
            try: vals.append((z,f(z)))
            except (TPXError,ValueError,OverflowError,ZeroDivisionError): vals.append((z,float('nan')))
        a=b=fa=fb=None
        # Prefer the bracket nearest the caller's current iterate.
        pairs=[(vals[i],vals[i+1]) for i in range(len(vals)-1)]
        pairs.sort(key=lambda q: min(abs(q[0][0]-guess),abs(q[1][0]-guess)))
        for (za,fa0),(zb,fb0) in pairs:
            if math.isfinite(fa0) and math.isfinite(fb0) and fa0*fb0<=0:
                a,b,fa,fb=za,zb,fa0,fb0; break
        if a is None: raise TPXError(f"Could not bracket property inversion target {target}")
        if fa*fb>0: raise TPXError(f"Could not bracket property inversion target {target}")
        for _ in range(100):
            m=0.5*(a+b); fm=f(m)
            if abs(fm)<=max(1e-7,1e-8*abs(target)): return
            if fa*fm<=0: b,fb=m,fm
            else: a,fa=m,fm
        raise TPXError(f"Property inversion did not converge for target {target}")

    def _solve_pair2(self, flag1, target1, flag2, target2):
        """Safeguarded two-variable Newton solve for source UP/SH pairs."""
        if not math.isfinite(self.T) or self.T == Undef: self.T = max(self.Tmin(), min(300.0, self.Tmax()))
        if not math.isfinite(self.Rho) or self.Rho <= 0 or self.Rho == Undef: self.Rho = max(1e-6, self.Roc*.1)
        def values(): return self.vprop(flag1)-target1, self.vprop(flag2)-target2
        for _ in range(60):
            f1,f2=values()
            if max(abs(f1)/max(1,abs(target1)),abs(f2)/max(1,abs(target2))) < 1e-8: return
            T0,r0=self.T,self.Rho; dt=max(1e-4,T0*1e-5); dr=max(1e-6,r0*1e-5)
            self.T=T0+dt; a1,a2=values(); self.T=T0-dt; b1,b2=values(); self.T=T0
            self.Rho=r0+dr; c1,c2=values(); self.Rho=max(1e-12,r0-dr); d1,d2=values(); self.Rho=r0
            j11=(a1-b1)/(2*dt); j21=(a2-b2)/(2*dt); j12=(c1-d1)/(2*dr); j22=(c2-d2)/(2*dr)
            det=j11*j22-j12*j21
            if not math.isfinite(det) or abs(det)<1e-30: raise TPXError('Two-property inversion singular Jacobian')
            dT=(-f1*j22+f2*j12)/det; drho=(-j11*f2+j21*f1)/det
            scale=1.0
            for _ in range(12):
                nt=max(self.Tmin(),min(self.Tmax(),T0+scale*dT)); nr=max(1e-12,r0+scale*drho); self.T,self.Rho=nt,nr
                nf1,nf2=values()
                if abs(nf1)+abs(nf2) < abs(f1)+abs(f2): break
                scale*=.5
            else: self.T,self.Rho=T0,r0; raise TPXError('Two-property inversion line search failed')
        raise TPXError('Two-property inversion did not converge')

    def update_sat(self):
        """Update saturation state from pinned Sub.cpp lines 514-575."""
        if self.T != self.Tslast:
            Rho_save = self.Rho
            try:
                pp = self.Psat()
                lps = log(pp)
                converged = False
                for i in range(20):
                    if i == 0:
                        self.Rho = self.ldens()
                    else:
                        self.Rho = self.Rhf
                    self.set_TPp(self.T, pp)
                    self.Rhf = self.Rho

                    gf = self.hp() - self.T * self.sp()
                    if i == 0:
                        self.Rho = pp * self.MolWt() / (GasConstant * self.T)
                    else:
                        self.Rho = self.Rhv
                    self.set_TPp(self.T, pp)

                    self.Rhv = self.Rho
                    gv = self.hp() - self.T * self.sp()
                    dg = gv - gf
                    if self.Rhv > self.Rhf:
                        self.Rhv, self.Rhf = self.Rhf, self.Rhv
                        dg = -dg

                    if abs(dg) < 0.001:
                        converged = True
                        break

                    denom = 1.0 / self.Rhv - 1.0 / self.Rhf
                    if denom == 0.0:
                        break
                    dp = dg / denom
                    psold = pp
                    if abs(dp) > pp:
                        lps -= dg / (pp * (1.0 / self.Rhv - 1.0 / self.Rhf))
                        pp = exp(lps)
                    else:
                        pp -= dp
                        lps = log(pp)

                    if pp > self.Pcrit():
                        pp = psold + 0.5 * (self.Pcrit() - psold)
                        lps = log(pp)
                    elif pp < 0.0:
                        pp = psold / 2.0
                        lps = log(pp)

                if not converged:
                    raise TPXError("Substance::update_sat: No convergence")
                else:
                    self.Pst = pp
                    self.Tslast = self.T
            finally:
                self.Rho = Rho_save


class _Reynolds14(Substance):
    """Base for 14-term Reynolds TPSI fluids (N2, O2, H2, CH4) from pinned source."""
    _Ann = []; _Fnn = []; _Dnn = []; _Gnn = []
    _M = 0; _Tmn = 0; _Tmx = 0; _Tc = 0; _Pc = 0; _Roc = 0
    _R = 0; _Gamma = 0; _alpha = 0; _beta = 0; _u0 = 0; _s0 = 0

    def MolWt(self) -> float:
        mw = self._M
        return float(mw)

    def Tcrit(self) -> float:
        tc = self._Tc
        return float(tc)

    def Pcrit(self) -> float:
        pc = self._Pc
        return float(pc)

    def Vcrit(self) -> float:
        vc = 1.0 / self._Roc
        return float(vc)

    def Tmin(self) -> float:
        tmin = self._Tmn
        return float(tmin)

    def Tmax(self) -> float:
        tmax = self._Tmx
        return float(tmax)

    def _C(self, i, rt, rt2):
        A = self._Ann
        if i == 0: return A[0]*self.T + A[1]*sqrt(self.T) + A[2] + (A[3] + A[4]*rt)*rt
        if i == 1: return A[5]*self.T + A[6] + rt*(A[7] + A[8]*rt)
        if i == 2: return A[9]*self.T + A[10] + A[11]*rt
        if i == 3: return A[12]
        if i == 4: return rt*(A[13] + A[14]*rt)
        if i == 5: return A[15]*rt
        if i == 6: return rt*(A[16] + A[17]*rt)
        if i == 7: return A[18]*rt2
        if i == 8: return rt2*(A[19] + A[20]*rt)
        if i == 9: return rt2*(A[21] + A[22]*rt2)
        if i == 10: return rt2*(A[23] + A[24]*rt)
        if i == 11: return rt2*(A[25] + A[26]*rt2)
        if i == 12: return rt2*(A[27] + A[28]*rt)
        if i == 13: return rt2*(A[29] + A[30]*rt + A[31]*rt2)
        return 0.0

    def _Cp(self, i, rt, rt2, rt3):
        A = self._Ann
        if i == 0: return A[0] + 0.5*A[1]/sqrt(self.T) - (A[3] + 2.0*A[4]*rt)*rt2
        if i == 1: return A[5] - rt2*(A[7] + 2.0*A[8]*rt)
        if i == 2: return A[9] - A[11]*rt2
        if i == 3: return 0.0
        if i == 4: return -rt2*(A[13] + 2.0*A[14]*rt)
        if i == 5: return -A[15]*rt2
        if i == 6: return -rt2*(A[16] + 2.0*A[17]*rt)
        if i == 7: return -2.0*A[18]*rt3
        if i == 8: return -rt3*(2.0*A[19] + 3.0*A[20]*rt)
        if i == 9: return -rt3*(2.0*A[21] + 4.0*A[22]*rt2)
        if i == 10: return -rt3*(2.0*A[23] + 3.0*A[24]*rt)
        if i == 11: return -rt3*(2.0*A[25] + 4.0*A[26]*rt2)
        if i == 12: return -rt3*(2.0*A[27] + 3.0*A[28]*rt)
        if i == 13: return -rt3*(2.0*A[29] + 3.0*A[30]*rt + 4.0*A[31]*rt2)
        return 0.0

    def _W(self, n, egrho):
        if n == 0: return (1.0 - egrho) / (2.0 * self._Gamma)
        return (n * self._W(n-1, egrho) - 0.5 * self.Rho**(2*n) * egrho) / self._Gamma

    def _H(self, i, egrho):
        if i < 8: return self.Rho**(i+2)
        return self.Rho**(2*i - 13) * egrho

    def _I(self, i, egrho):
        if i < 8: return self.Rho**(i+1) / (i+1)
        return self._W(i - 8, egrho)

    def Pp(self):
        rt = 1.0/self.T; rt2 = rt*rt
        egrho = exp(-self._Gamma * self.Rho**2)
        P = self.Rho * self._R * self.T
        for i in range(14):
            P += self._C(i, rt, rt2) * self._H(i, egrho)
        return P

    def _psat_alpha(self):
        """Psat for fluids using alpha exponent (N2, O2)."""
        if self.T < self._Tmn or self.T > self._Tc:
            raise TPXError(f"Temperature out of range: {self.T}")
        F = self._Fnn; lnp = 0.0
        for i in range(8):
            if i == 3:
                lnp += F[i] * (self._Tc - self.T)**self._alpha
            else:
                lnp += F[i] * self.T**(i-1)
        lnp += F[8] * log(self.T)
        return exp(lnp)

    def _ldens_standard(self):
        """ldens for fluids using xx^(i/3) (N2, O2)."""
        if self.T < self._Tmn or self.T > self._Tc:
            raise TPXError(f"Temperature out of range: {self.T}")
        xx = 1 - self.T / self._Tc
        return sum(self._Dnn[i] * xx**(i/3.0) for i in range(6))


class Water(Substance):
    """Water thermodynamic properties via Reynolds/IAPWS formulation."""
    Tmn = 273.16
    Tmx = 1600.0
    M = 18.016
    Tc = 647.286
    Pc = 22.089e6
    Roc = 317.0
    To = 273.16
    R = 461.51
    E = 4.8e-3
    Ta = 1000.0
    tauc = 1.544912
    Tp = 338.15
    aww = 0.01
    Roa1 = 634.0
    Roaj = 1000.0
    u0 = 2375470.875
    s0 = 6697.356635

    A = np.array([
        [2.9492937E-2, -5.1985860E-3, 6.8335354E-3, -1.5641040E-4, -6.3972405E-3, -3.9661401E-3, -6.9048554E-4],
        [-1.3213917E-4, 7.7779182E-6, -2.6149751E-5, -7.2546108E-7, 2.6409282E-5, 1.5453061E-5, 2.7407416E-6],
        [2.7464632E-7, -3.3301902E-8, 6.5326396E-8, -9.2734289E-9, -4.7740374E-8, -2.9142470E-8, -5.1028070E-9],
        [-3.6093828E-10, -1.6254622E-11, -2.6181978E-11, 4.3125840E-12, 5.6323130E-11, 2.9568796E-11, 3.9636085E-12],
        [3.4218431E-13, -1.7731074E-13, 0, 0, 0, 0, 0],
        [-2.4450042E-16, 1.2748742E-16, 0, 0, 0, 0, 0],
        [1.5518535E-19, 1.3746153E-19, 0, 0, 0, 0, 0],
        [5.9728487E-24, 1.5597836E-22, 0, 0, 0, 0, 0],
        [-4.1030848E-1, 3.3731180E-1, -1.3746678E-1, 6.7874983E-3, 1.3687317E-1, 7.984797E-2, 1.3041253E-2],
        [-4.1605860E-4, -2.0988866E-4, -7.3396848E-4, 1.0401717E-5, 6.4581880E-4, 3.9917570E-4, 7.1531353E-5]
    ])
    F = np.array([-7.4192420, 2.9721E-1, -1.155286E-1, 8.685635E-3, 1.0940980E-3, -4.39993E-3, 2.5206580E-3, -5.2186840E-4])
    D = np.array([3.6711257, -2.8512396E1, 2.2265240E2, -8.8243852E2, 2.0002765E3, -2.6122557E3, 1.8297674E3, -5.3350520E2])
    G = np.array([4.6E4, 1.011249E3, 8.3893E-1, -2.19989E-4, 2.466619E-7, -9.704700E-11])
    taua = np.array([1.544912, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])

    def __init__(self):
        super().__init__()
        self.m_name = "water"
        self.m_formula = "H2O"

    def MolWt(self) -> float:
        mw = self.M
        return float(mw)

    def Tcrit(self) -> float:
        tc = self.Tc
        return float(tc)

    def Pcrit(self) -> float:
        pc = self.Pc
        return float(pc)

    def Vcrit(self) -> float:
        vc = 1.0 / self.Roc
        return float(vc)

    def Tmin(self) -> float:
        tmin = self.Tmn
        return float(tmin)

    def Tmax(self) -> float:
        tmax = self.Tmx
        return float(tmax)

    def _C(self, i: int) -> float:
        tau = self.Ta / self.T
        if i == 0:
            return self.R * self.T
        return self.R * self.T * (tau - self.tauc) * ((tau - self.taua[i]) ** (i - 1))

    def _Cprime(self, i: int) -> float:
        tau = self.Ta / self.T
        if i == 0:
            return self.R
        elif i == 1:
            return -self.R * self.tauc
        else:
            diff = tau - self.taua[i]
            return -self.R * (diff ** (i - 2)) * (self.tauc * diff + (i - 1) * tau * (tau - self.tauc))

    def _I(self, j: int) -> float:
        rho_aj = self.Roa1 if j == 0 else self.Roaj
        factor = self.Rho - rho_aj
        s = 0.0
        for i in range(7, 0, -1):
            s += self.A[i, j]
            s *= factor
        s += self.A[0, j]
        s += exp(-self.E * self.Rho) * (self.A[8, j] + self.A[9, j] * self.Rho)
        return self.Rho * s

    def _H(self, j: int) -> float:
        rho_aj = self.Roa1 if j == 0 else self.Roaj
        factor = self.Rho - rho_aj
        s = 0.0
        for i in range(6, 0, -1):
            s += (self.A[i, j] + self.Rho * (i + 1) * self.A[i + 1, j])
            s *= factor
        s += (self.A[0, j] + self.Rho * self.A[1, j])
        s += exp(-self.E * self.Rho) * ((1.0 - self.Rho * self.E) * self.A[8, j] + self.Rho * (2.0 - self.Rho * self.E) * self.A[9, j])
        s += self.A[7, j] * (factor ** 7)
        return self.Rho * self.Rho * s

    def up(self) -> float:
        s = 0.0
        for i in range(7):
            s += (self._C(i) - self.T * self._Cprime(i)) * self._I(i)
        for i in range(1, 6):
            s += self.G[i] * (self.T ** i - self.To ** i) / float(i)
        s += self.G[0] * log(self.T / self.To) + self.u0
        return s + self.m_energy_offset

    def sp(self) -> float:
        s = 0.0
        for i in range(2, 6):
            s += self.G[i] * (self.T ** (i - 1) - self.To ** (i - 1)) / float(i - 1)
        s += self.G[1] * log(self.T / self.To)
        s -= self.G[0] * (1.0 / self.T - 1.0 / self.To)
        s += self.s0 - self.R * log(self.Rho)
        for i in range(7):
            s -= self._Cprime(i) * self._I(i)
        return s + self.m_entropy_offset

    def Pp(self) -> float:
        p = self.Rho * self.R * self.T
        for i in range(7):
            p += self._C(i) * self._H(i)
        return p

    def Psat(self) -> float:
        if self.T < self.Tmn or self.T > self.Tc:
            raise TPXError(f"Water: Temperature out of range: {self.T}")
        s = sum(self.F[i] * ((self.aww * (self.T - self.Tp)) ** i) for i in range(8))
        log_p = (self.Tc / self.T - 1.0) * s
        return exp(log_p) * self.Pc

    def dPsatdT(self, T: Optional[float] = None) -> float:
        """Analytical derivative of the pinned water saturation correlation."""
        old = self.T
        if T is not None:
            self.set_T(T)
        z = self.aww * (self.T - self.Tp)
        s = sum(self.F[i] * z**i for i in range(8))
        ds = self.aww * sum(i * self.F[i] * z**(i-1) for i in range(1, 8))
        lp = (self.Tc / self.T - 1.0) * s
        dlp = -self.Tc * s / self.T**2 + (self.Tc / self.T - 1.0) * ds
        ans = self.Psat() * dlp
        if T is not None:
            self.T = old
        return ans

    def ldens(self) -> float:
        if self.T < self.Tmn or self.T > self.Tc:
            raise TPXError(f"Water: Temperature out of range: {self.T}")
        s = sum(self.D[i] * ((1.0 - self.T / self.Tc) ** ((i + 1) / 3.0)) for i in range(8))
        return self.Roc * (1.0 + s)


class CarbonDioxide(Substance):
    """Carbon dioxide thermodynamic properties from Reynolds TPSI (pinned CarbonDioxide.cpp)."""
    # Constants from pinned source
    _Tmn = 216.54; _Tmx = 1500.0; _Tc = 304.21; _Roc = 464.00
    _To = 216.54; _R = 188.918; _Gamma = 5.0e-6
    _u0 = 3.2174105e5; _s0 = 2.1396056e3; _Tp = 250; _Pc = 7.38350e6; _M = 44.01
    _Acarbdi = [
        2.2488558e-1, -1.3717965e2, -1.4430214e4, -2.9630491e6, -2.0606039e8,
        4.5554393e-5, 7.7042840e-2, 4.0602371e1, 4.0029509e-7, -3.9436077e-4,
        1.2115286e-10, 1.0783386e-7, 4.3962336e-11, -3.6505545e4, 1.9490511e7,
        -2.9186718e9, 2.4358627e-2, -3.7546530e1, 1.1898141e4
    ]
    _F = [-6.5412610, -2.7914636e-1, -3.4716202, -3.4989637,
          -1.9770948e1, 1.3922839e2, -2.7670389e2, -7.0510251e3]
    _D = [4.6400009e2, 6.7938129e2, 1.4776836e3, -3.1267676e3, 3.6397656e3, -1.3437098e3]
    _G = [8.726361e3, 1.840040e2, 1.914025, -1.667825e-3, 7.305950e-7, -1.255290e-10]

    def __init__(self):
        super().__init__()
        self.m_name = "carbon-dioxide"; self.m_formula = "CO2"
    def MolWt(self) -> float:
        mw = self._M
        return float(mw)

    def Tcrit(self) -> float:
        tc = self._Tc
        return float(tc)

    def Pcrit(self) -> float:
        pc = self._Pc
        return float(pc)

    def Vcrit(self) -> float:
        vc = 1.0 / self._Roc
        return float(vc)

    def Tmin(self) -> float:
        tmin = self._Tmn
        return float(tmin)

    def Tmax(self) -> float:
        tmax = self._Tmx
        return float(tmax)

    def _C(self, j, Ti, T2i, T3i, T4i):
        A = self._Acarbdi
        if j == 0: return A[0]*self.T + A[1] + A[2]*Ti + A[3]*T2i + A[4]*T3i
        if j == 1: return A[5]*self.T + A[6] + A[7]*Ti
        if j == 2: return A[8]*self.T + A[9]
        if j == 3: return A[10]*self.T + A[11]
        if j == 4: return A[12]
        if j == 5: return A[13]*T2i + A[14]*T3i + A[15]*T4i
        if j == 6: return A[16]*T2i + A[17]*T3i + A[18]*T4i
        return 0.0

    def _Cp(self, j, T2i, T3i, T4i):
        A = self._Acarbdi
        if j == 0: return A[0] - A[2]*T2i - 2*A[3]*T3i - 3*A[4]*T4i
        if j == 1: return A[5] - A[7]*T2i
        if j == 2: return A[8]
        if j == 3: return A[10]
        if j == 4: return 0.0
        if j == 5: return -2*A[13]*T3i - 3*A[14]*T4i - 4*A[15]*self.T**-5
        if j == 6: return -2*A[16]*T3i - 3*A[17]*T4i - 4*A[18]*self.T**-5
        return 0.0

    def _I(self, j, egrho):
        G = self._Gamma; r = self.Rho
        if j == 0: return r
        if j == 1: return r**2 / 2
        if j == 2: return r**3 / 3
        if j == 3: return r**4 / 4
        if j == 4: return r**5 / 5
        if j == 5: return (1 - egrho) / (2*G)
        if j == 6: return (1 - egrho*(G*r**2 + 1)) / (2*G*G)
        return 0.0

    def _H(self, i, egrho):
        r = self.Rho; G = self._Gamma
        if i < 5: return r**(i+2)
        if i == 5: return r**3 * egrho
        if i == 6: return r**5 * egrho
        return 0.0

    def up(self):
        Ti = 1.0/self.T; T2i = Ti**2; T3i = Ti**3; T4i = Ti**4
        egrho = exp(-self._Gamma * self.Rho**2)
        s = 0.0
        s += self._G[0] * log(self.T / self._To)
        for i in range(1, 6):
            s += self._G[i] * (self.T**i - self._To**i) / i
        for j in range(7):
            s += self._I(j, egrho) * (self._C(j, Ti, T2i, T3i, T4i) - self.T * self._Cp(j, T2i, T3i, T4i))
        s += self._u0
        return s + self.m_energy_offset

    def sp(self):
        T2i = self.T**-2; T3i = self.T**-3; T4i = self.T**-4
        egrho = exp(-self._Gamma * self.Rho**2)
        s = 0.0
        for i in range(2, 6):
            s += self._G[i] * (self.T**(i-1) - self._To**(i-1)) / (i-1)
        s += self._G[1] * log(self.T / self._To)
        s -= self._G[0] * (1.0/self.T - 1.0/self._To)
        for j in range(7):
            s -= self._Cp(j, T2i, T3i, T4i) * self._I(j, egrho)
        s += self._s0 - self._R * log(self.Rho)
        return s + self.m_entropy_offset

    def Pp(self):
        Ti = 1.0/self.T; T2i = Ti**2; T3i = Ti**3; T4i = Ti**4
        egrho = exp(-self._Gamma * self.Rho**2)
        P = self.Rho * self._R * self.T
        for j in range(7):
            P += self._C(j, Ti, T2i, T3i, T4i) * self._H(j, egrho)
        return P

    def Psat(self):
        if self.T < self._Tmn or self.T > self._Tc:
            raise TPXError(f"CO2: Temperature out of range: {self.T}")
        s = sum(self._F[i] * ((self.T/self._Tp - 1)**(i)) for i in range(8))
        return exp(((self._Tc/self.T) - 1) * s) * self._Pc

    def ldens(self):
        if self.T < self._Tmn or self.T > self._Tc:
            raise TPXError(f"CO2: Temperature out of range: {self.T}")
        xx = 1 - self.T / self._Tc
        return sum(self._D[i] * xx**(i/3.0) for i in range(6))


class Nitrogen(_Reynolds14):
    """Nitrogen from pinned Nitrogen.cpp (Reynolds TPSI)."""
    _M = 28.01348; _Tmn = 63.15; _Tmx = 2000.0; _Tc = 126.200; _Pc = 3.4e6; _Roc = 314.03
    _R = 2.96790515164171e2; _Gamma = 7.13602531283233e-6
    _alpha = 1.95; _beta = 3353.40610; _u0 = 150877.551; _s0 = 214.9352518
    _Ann = [
        1.75889959256970e-1, 1.38197604384933e1, -3.14918412133921e2,
        4.40300150239380e3, -5.45358971644916e5, 4.84413320182919e-4,
        -5.18964416491365e-2, 6.57265859197103e-4, 8.51299771713314e4,
        1.33459405162578e-8, 3.83381319826746e-4, -8.35421151028455e-2,
        2.84874912286101e-7,
        -2.38296116270360e-7, -1.48321912935764e-4, 5.62605853190540e-10,
        -2.98201050924595e-13, 9.85319087685241e-11, -1.92002176056468e-14,
        -7.82250103373122e4, -5.51801778744598e5, -5.72781957607352e-1,
        3.25760529488327e2, -1.34659309828737e-6, -1.92036423064911e-5,
        -3.94564337674524e-12, -2.44388245328965e-9, -1.50970602460077e-18,
        1.25854885346038e-16, -8.34271144923969e-24, -1.17299202018417e-22,
        9.06544823455730e-22
    ]
    _Fnn = [8.3944094440e3, -1.8785191705e3, -7.2822291650,
            1.0228509660e-2, 5.5560638250e-4, -5.9445446620e-6,
            2.7154339320e-8, -4.8795359040e-11, 5.0953608240e2]
    _Dnn = [3.1402991e2, 4.4111015e2, 9.4622994e2, -2.9067111e3, 4.4785979e3, -2.2746914e3]
    _Gnn = [-2.18203473713518e5, 1.01573580096247e4, -1.65504721657240e2,
            7.43175999190430e2, -5.14605623546025e-3,
            5.18347156760489e-6, -1.05922170493616e-9, 2.98389393363817e2]

    def __init__(self):
        super().__init__()
        self.m_name = "nitrogen"; self.m_formula = "N2"

    def up(self):
        rt = 1.0/self.T; rt2 = rt*rt; rt3 = rt*rt2
        egrho = exp(-self._Gamma * self.Rho**2)
        s = 0.0
        for i in range(14):
            s += (self._C(i, rt, rt2) - self.T * self._Cp(i, rt, rt2, rt3)) * self._I(i, egrho)
        G = self._Gnn
        s += (((0.25*G[6]*self.T + G[5]/3.0)*self.T + 0.5*G[4])*self.T + G[3])*self.T
        s += G[2]*log(self.T) - (G[1] + 0.5*G[0]*rt)*rt
        s += G[7]*self._beta/(exp(self._beta*rt) - 1.0) + self._u0
        return s + self.m_energy_offset

    def sp(self):
        rt = 1.0/self.T; rt2 = rt*rt; rt3 = rt*rt2
        egrho = exp(-self._Gamma * self.Rho**2)
        G = self._Gnn
        s = self._s0 + self.m_entropy_offset - self._R*log(self.Rho)
        for i in range(14):
            s -= self._Cp(i, rt, rt2, rt3) * self._I(i, egrho)
        s += (((G[6]/3.0)*self.T + 0.5*G[5])*self.T + G[4])*self.T + G[3]*log(self.T)
        s -= ((G[0]*rt/3.0 + 0.5*G[1])*rt + G[2])*rt
        brt = self._beta * rt
        ebrt = exp(brt)
        s += G[7]*(brt + brt/(ebrt - 1.0) - log(ebrt - 1.0))
        return s

    def Psat(self) -> float:
        p = self._psat_alpha()
        return float(p)

    def ldens(self) -> float:
        rho = self._ldens_standard()
        return float(rho)

class Oxygen(_Reynolds14):
    """Oxygen from pinned Oxygen.cpp (Reynolds TPSI)."""
    _M = 31.9994; _Tmn = 54.34; _Tmx = 2000.0; _Tc = 154.581; _Pc = 5.0429e6; _Roc = 436.15
    _R = 2.59820853437877e2; _Gamma = 5.46895508389297e-6
    _alpha = 1.91576; _beta = 2239.18105; _u0 = 198884.2435; _s0 = 668.542976
    _Ann = [
        -4.26396872798684e-1, 3.48334938784107e1, -5.77516910418738e2,
        2.40961751553325e4, -1.23332307855543e6, 3.73585286319658e-4,
        -1.70178244046465e-1, -3.33226903068473e-4, 8.61334799901291e3,
        -6.80394661057309e-7, 7.09583347162704e-4, -5.73905688255053e-2,
        -1.92123080811409e-7, 3.11764722329504e-8, -8.09463854745591e-6,
        -2.22562296356501e-11, 9.18401045361994e-15, 5.75758417511114e-12,
        -2.10752269644774e-15, 3.62884761272184e3, -1.23317754317110e6,
        -5.03800414800672e-2, 3.30686173177055e2, -5.26259633964252e-8,
        5.53075442383100e-6, -2.71042853363688e-13, -1.65732450675251e-9,
        -5.82711196409204e-20, 4.42953322148281e-17, -2.95529679136244e-25,
        -1.92361786708846e-23, 9.43758410350413e-23
    ]
    _Fnn = [-5.581932039e2, -1.0966262185e2, -8.3456211630e-2,
            2.6603644330e-3, 1.6875023830e-5, -2.1262477120e-7,
            9.5741096780e-10, -1.6617640450e-12, 2.7545605710e1]
    _Dnn = [4.3615175e2, 7.5897189e2, -4.2576866e2, 2.3487106e3, -3.0474660e3, 1.4850169e3]
    _Gnn = [-1.29442711174062e6, 5.98231747005341e4, -8.97850772730944e2,
            6.55236176900400e2, -1.13131252131570e-2,
            3.4981070244228e-6, 4.21065222886885e-9, 2.67997030050139e2]

    def __init__(self):
        super().__init__()
        self.m_name = "oxygen"; self.m_formula = "O2"

    def up(self):
        rt = 1.0/self.T; rt2 = rt*rt; rt3 = rt*rt2
        egrho = exp(-self._Gamma * self.Rho**2)
        s = 0.0
        for i in range(14):
            s += (self._C(i, rt, rt2) - self.T * self._Cp(i, rt, rt2, rt3)) * self._I(i, egrho)
        G = self._Gnn
        s += (((0.25*G[6]*self.T + G[5]/3.0)*self.T + 0.5*G[4])*self.T + G[3])*self.T
        s += G[2]*log(self.T) - (G[1] + 0.5*G[0]*rt)*rt
        s += G[7]*self._beta/(exp(self._beta*rt) - 1.0) + self._u0
        return s + self.m_energy_offset

    def sp(self):
        rt = 1.0/self.T; rt2 = rt*rt; rt3 = rt*rt2
        egrho = exp(-self._Gamma * self.Rho**2)
        G = self._Gnn
        s = self._s0 + self.m_entropy_offset - self._R*log(self.Rho)
        for i in range(14):
            s -= self._Cp(i, rt, rt2, rt3) * self._I(i, egrho)
        s += (((G[6]/3.0)*self.T + 0.5*G[5])*self.T + G[4])*self.T + G[3]*log(self.T)
        s -= ((G[0]*rt/3.0 + 0.5*G[1])*rt + G[2])*rt
        brt = self._beta * rt
        ebrt = exp(brt)
        s += G[7]*(brt + brt/(ebrt - 1.0) - log(ebrt - 1.0))
        return s

    def Psat(self) -> float:
        p = self._psat_alpha()
        return float(p)

    def ldens(self) -> float:
        rho = self._ldens_standard()
        return float(rho)

class Hydrogen(_Reynolds14):
    """Hydrogen from pinned Hydrogen.cpp (Reynolds TPSI with icv() 3-region ideal gas)."""
    _M = 2.0159; _Tmn = 13.8; _Tmx = 5000.0; _Tc = 32.938; _Pc = 1.2838e6; _Roc = 31.36
    _R = 4124.299539; _Gamma = 1.008854772e-3
    _u0 = 3.9275114e5; _s0 = 2.3900333e4
    _To = 13.8; _Tt = 13.8; _Pt = 7042.09
    _alpha = 1.5814454428; _alpha1 = 0.3479; _beta = 0.0  # beta not used in H2
    _T1 = 35; _T2 = 400
    _Ann = [
        1.150470519352900e1, 1.055427998826072e3, -1.270685949968568e4,
        7.287844527295619e4, -7.448780703363973e5, 2.328994151810363e-1,
        -1.635308393739296e1, 3.730678064960389e3, 6.299667723184813e5,
        1.210920358305697e-3, 1.753651095884817, -1.367022988058101e2,
        -6.869936641299885e-3, 3.644494201750974e-2, -2.559784772600182,
        -4.038855202905836e-4, 1.485396303520942e-6, 4.243613981060742e-4,
        -2.307910113586888e-6, -6.082192173879582e5, -1.961080967486886e6,
        -5.786932854076408e2, 2.799129504191752e4, -2.381566558300913e-1,
        8.918796032452872e-1, -6.985739539036644e-5, -7.339554179182899e-3,
        -5.597033440289980e-9, 8.842130160884514e-8, -2.655507264539047e-12,
        -4.544474518140164e-12, 9.818775257001922e-11
    ]
    _Dnn = [4.8645813003e1, -3.4779278180e1, 4.0776538192e2,
            -1.1719787304e3, 1.6213924400e3, -1.1531096683e3, 3.3825492039e2]
    _Fnn = [3.05300134164, 2.80810925813, -6.55461216567e-1, 1.59514439374]
    _Gnn = [6.1934792e3, 2.9490437e2, -1.5401979e3, -4.9176101e3,
            6.8957165e4, -2.2282185e5, 3.7990059e5, -3.7094216e5, 2.1326792e5,
            -7.1519411e4, 1.2971743e4, -9.8533014e2, 1.0434776e4,
            -3.9144179e2, 5.8277696e2, 6.5409163e2, -1.8728847e2]

    def __init__(self):
        super().__init__()
        self.m_name = "hydrogen"; self.m_formula = "H2"

    def _icv(self, i, x, xlg):
        if i == 0: return x - 1
        return x * xlg**i - i * self._icv(i-1, x, xlg)

    def up(self):
        rt = 1.0/self.T; rt2 = rt*rt; rt3 = rt*rt2
        egrho = exp(-self._Gamma * self.Rho**2)
        s = self._u0
        for i in range(14):
            s += (self._C(i, rt, rt2) - self.T * self._Cp(i, rt, rt2, rt3)) * self._I(i, egrho)
        G = self._Gnn
        s += G[0] * (min(self.T, self._T1) - self._To)
        if self.T > self._T1:
            x = min(self.T, self._T2) / self._T1
            for i in range(12):
                s += G[i] * self._T1 * self._icv(i, x, log(x))
        if self.T > self._T2:
            x = self.T / self._T2
            for i in range(5):
                s += G[i+12] * self._T2 * self._icv(i, x, log(x))
        return s + self.m_energy_offset

    def sp(self):
        rt = 1.0/self.T; rt2 = rt*rt; rt3 = rt*rt2
        egrho = exp(-self._Gamma * self.Rho**2)
        G = self._Gnn
        s = self._s0 - self._R * log(self.Rho)
        for i in range(14):
            s -= self._Cp(i, rt, rt2, rt3) * self._I(i, egrho)
        s += G[0] * log(min(self.T, self._T1) / self._To)
        if self.T > self._T1:
            xlg = log(min(self.T, self._T2) / self._T1)
            for i in range(12):
                s += G[i] / (i+1) * xlg**(i+1)
        if self.T > self._T2:
            xlg = log(self.T / self._T2)
            for i in range(5):
                s += G[i+12] / (i+1) * xlg**(i+1)
        return s + self.m_entropy_offset

    def Psat(self):
        if self.T < self._Tmn or self.T > self._Tc:
            raise TPXError(f"H2: Temperature out of range: {self.T}")
        x = (1.0 - self._Tt/self.T) / (1.0 - self._Tt/self._Tc)
        F = self._Fnn
        result = F[0]*x + F[1]*x*x + F[2]*x**3 + F[3]*x*(1-x)**self._alpha
        return exp(result) * self._Pt

    def ldens(self):
        if self.T < self._Tmn or self.T > self._Tc:
            raise TPXError(f"H2: Temperature out of range: {self.T}")
        x = 1 - self.T/self._Tc; D = self._Dnn
        s = sum(D[i+1] * x**(1 + i/3.0) for i in range(6))
        return s + self._Roc + D[0] * x**self._alpha1

class Methane(_Reynolds14):
    """Methane from pinned Methane.cpp (Reynolds TPSI with T^(1/3) ideal gas)."""
    _M = 16.04996; _Tmn = 90.68; _Tmx = 1700.0; _Tc = 190.555; _Pc = 4.5988e6; _Roc = 160.43
    _R = 5.18253475866e2; _Gamma = 3.72992471469e-5
    _Tt = 90.68; _Pt = 11743.5675; _Rot = 451.562
    _alpha = 1.5; _alpha1 = 0.36; _beta = 2009.152; _u0 = 357696.0858; _s0 = -1918.035071
    _Ann = [
        -7.25929210183, 4.13766054566e2, -6.32167316855e3,
        3.34015577724e5, -1.68253379982e7, 1.87884851902e-2, -1.18673201223e1,
        2.09062618015e3, -4.07532656958e5, -5.73917603241e-5, 4.37711441593e-2,
        -4.38766500673, 1.13524630779e-5, -5.07028240949e-5, 2.28002199522e-2,
        9.25611329590e-9, 1.33865662546e-10, -1.65439044196e-7, 1.81030980110e-10,
        5.45753645958e5, -3.63192281933e7, 4.81463473761, 1.56633022620e5,
        7.89977010972e-5, 1.39993881210e-2, -1.70656092212e-11, -4.55256623445e-5,
        -2.29314170748e-14, 8.31548197665e-12, 6.84673626259e-20,
        -4.70845544152e-17, 5.21465091383e-16
    ]
    _Dmeth = [-1.78860165e-1, 4.83847500e-2, -1.84898700e-2]
    _Fnn = [4.77748580, 1.76065363, -5.67888940e-1, 1.32786231]
    _Gnn = [1.34740610e3, 1.35512060e2, -2.93910458e1, 2.12774600, 2.44656600e3]

    def __init__(self):
        super().__init__()
        self.m_name = "methane"; self.m_formula = "CH4"

    def up(self):
        rt = 1.0/self.T; rt2 = rt*rt; rt3 = rt*rt2
        egrho = exp(-self._Gamma * self.Rho**2)
        t3 = self.T**(1.0/3.0)
        s = 0.0
        for i in range(14):
            s += (self._C(i, rt, rt2) - self.T * self._Cp(i, rt, rt2, rt3)) * self._I(i, egrho)
        G = self._Gnn
        s += self.T*(G[0] + 0.75*G[1]*t3 + 0.6*G[2]*t3*t3 + 0.5*G[3]*self.T)
        s += G[4]*self._beta/(exp(self._beta*rt) - 1.0) + self._u0
        return s + self.m_energy_offset

    def sp(self):
        rt = 1.0/self.T; rt2 = rt*rt; rt3 = rt*rt2
        egrho = exp(-self._Gamma * self.Rho**2)
        t3 = self.T**(1.0/3.0)
        G = self._Gnn
        s = self._s0 - self._R*log(self.Rho)
        for i in range(14):
            s -= self._Cp(i, rt, rt2, rt3) * self._I(i, egrho)
        s += G[0]*log(self.T) + 3.0*G[1]*t3 + 1.5*G[2]*t3*t3 + G[3]*self.T
        brt = self._beta * rt
        ebrt = exp(brt)
        s += G[4]*(brt + brt/(ebrt - 1.0) - log(ebrt - 1.0))
        return s + self.m_entropy_offset

    def Psat(self):
        if self.T < self._Tmn or self.T > self._Tc:
            raise TPXError(f"CH4: Temperature out of range: {self.T}")
        x = (1.0 - self._Tt/self.T) / (1.0 - self._Tt/self._Tc)
        F = self._Fnn
        return exp(F[0]*x + F[1]*x*x + F[2]*x**3 + F[3]*x*(1-x)**self._alpha) * self._Pt

    def ldens(self):
        if self.T < self._Tmn or self.T > self._Tc:
            raise TPXError(f"CH4: Temperature out of range: {self.T}")
        D = self._Dmeth; w = (self._Tc - self.T)/(self._Tc - self._Tt)
        s = D[0]*(1-w**(2.0/3.0)) + D[1]*(1-w**(4.0/3.0)) + D[2]*(1-w**2)
        return w**self._alpha1 * exp(s) * (self._Rot - self._Roc) + self._Roc

class Heptane(Substance):
    """Heptane from pinned Heptane.cpp (Reynolds TPSI, 4-term virial)."""
    _Tmn = 182.56; _Tmx = 1000.0; _Tc = 537.68; _Roc = 197.60
    _To = 300; _R = 82.99504; _Gamma = 9.611604e-6
    _u0 = 3.4058439e5; _s0 = 1.1080254e3; _Tp = 400; _Pc = 2.6199e6; _M = 100.20
    _Ahept = [2.246032e-3, 2.082990e2, 5.085746e7, 3.566396e9, 1.622168e9,
              1.065237e-5, 5.987922e-1, 7.736602, 1.929386e5, 5.291379e-9]
    _F = [-7.2298764, 3.8607475e-1, -3.4216472, 4.6274432e-1,
          -9.7926124, -4.2058094e1, 7.5468678e1, 3.1758992e2]
    _D = [1.9760405e2, 8.9451237e2, -1.1462908e3, 1.7996947e3, -1.7250843e3, 9.7088329e2]
    _G = [1.1925213e5, -7.7231363e2, 7.4463527, -3.0888167e-3, 0.0, 0.0]

    def __init__(self):
        super().__init__()
        self.m_name = "heptane"; self.m_formula = "C7H16"
    def MolWt(self) -> float:
        mw = self._M
        return float(mw)

    def Tcrit(self) -> float:
        tc = self._Tc
        return float(tc)

    def Pcrit(self) -> float:
        pc = self._Pc
        return float(pc)

    def Vcrit(self) -> float:
        vc = 1.0 / self._Roc
        return float(vc)

    def Tmin(self) -> float:
        tmin = self._Tmn
        return float(tmin)

    def Tmax(self) -> float:
        tmax = self._Tmx
        return float(tmax)

    def _C(self, j, Ti, T2i, T3i, T4i):
        A = self._Ahept; R = self._R
        if j == 0: return A[0]*R*self.T - A[1] - A[2]*T2i + A[3]*T3i - A[4]*T4i
        if j == 1: return A[5]*R*self.T - A[6] - A[7]*Ti
        if j == 2: return A[9]*(A[6] + A[7]*Ti)
        if j == 3: return A[8]*T2i
        return 0.0

    def _Cp(self, j, T2i, T3i, T4i):
        A = self._Ahept; R = self._R
        if j == 0: return A[0]*R + 2*A[2]*T3i - 3*A[3]*T4i + 4*A[4]*self.T**-5
        if j == 1: return A[5]*R + A[7]*T2i
        if j == 2: return -A[9]*A[7]*T2i
        if j == 3: return -2*A[8]*T3i
        return 0.0

    def _I(self, j, egrho):
        r = self.Rho; G = self._Gamma
        if j == 0: return r
        if j == 1: return r**2 / 2
        if j == 2: return r**5 / 5
        if j == 3: return 1/G - (G*r**2 + 2)*egrho/(2*G)
        return 0.0

    def _H(self, i, egrho):
        r = self.Rho; G = self._Gamma
        if i < 2: return r**(i+2)
        if i == 2: return r**6
        if i == 3: return r**3 * (1 + G*r**2) * egrho
        return 0.0

    def up(self):
        Ti = 1.0/self.T; T2i = Ti**2; T3i = Ti**3; T4i = Ti**4
        egrho = exp(-self._Gamma * self.Rho**2)
        s = 0.0
        for i in range(1, 6):
            s += self._G[i] * (self.T**i - self._To**i) / i
        s += self._G[0] * log(self.T / self._To)
        for j in range(4):
            s += (self._C(j, Ti, T2i, T3i, T4i) - self.T*self._Cp(j, T2i, T3i, T4i)) * self._I(j, egrho)
        s += self._u0
        return s + self.m_energy_offset

    def sp(self):
        T2i = self.T**-2; T3i = self.T**-3; T4i = self.T**-4
        egrho = exp(-self._Gamma * self.Rho**2)
        s = 0.0
        for i in range(2, 6):
            s += self._G[i] * (self.T**(i-1) - self._To**(i-1)) / (i-1)
        s += self._G[1] * log(self.T / self._To)
        s -= self._G[0] * (1.0/self.T - 1.0/self._To)
        for j in range(4):
            s -= self._Cp(j, T2i, T3i, T4i) * self._I(j, egrho)
        s += self._s0 - self._R * log(self.Rho)
        return s + self.m_entropy_offset

    def Pp(self):
        Ti = 1.0/self.T; T2i = Ti**2; T3i = Ti**3; T4i = Ti**4
        egrho = exp(-self._Gamma * self.Rho**2)
        P = self.Rho * self._R * self.T
        for j in range(4):
            P += self._C(j, Ti, T2i, T3i, T4i) * self._H(j, egrho)
        return P

    def Psat(self):
        if self.T < self._Tmn or self.T > self._Tc:
            raise TPXError(f"Heptane: Temperature out of range: {self.T}")
        s = sum(self._F[i] * ((self.T/self._Tp - 1)**i) for i in range(8))
        return exp(((self._Tc/self.T) - 1) * s) * self._Pc

    def ldens(self):
        if self.T < self._Tmn or self.T > self._Tc:
            raise TPXError(f"Heptane: Temperature out of range: {self.T}")
        xx = 1 - self.T/self._Tc
        return sum(self._D[i] * xx**(i/3.0) for i in range(6))

class HFC134a(Substance):
    """HFC-134a from pinned HFC134a.cpp (Helmholtz dimensionless free energy)."""
    _M = 102.032; _Tmn = 170.0; _Tmx = 455.0; _Tc = 374.18
    _Pc_local = 4056290.0; _Roc = 508.0; _R = 81.48885644
    # Note: Tcrit/Pcrit/Vcrit override use DIFFERENT values than the EOS constants
    # per pinned HFC134a.cpp lines 181-192
    _Tc_crit = 374.21; _Pc_crit = 4059280.0; _Roc_crit = 511.95

    _a134 = [0.5586817e-1, 0.4982230, 0.2458698e-1, 0.8570145e-3, 0.4788584e-3,
             -0.1800808e1, 0.2671641, -0.4781652e-1, 0.1423987e-1, 0.3324062,
             -0.7485907e-2, 0.1017263e-3, -0.5184567, -0.8692288e-1, 0.2057144,
             -0.5000457e-2, 0.4603262e-3, -0.3497836e-2, 0.6995038e-2,
             -0.1452184e-1, -0.1285458e-3]
    _t134 = [-0.5, 0.0, 0.0, 0.0, 1.5, 1.5, 2.0, 2.0, 1.0, 3.0, 5.0,
             1.0, 5.0, 5.0, 6.0, 10.0, 10.0, 10.0, 18.0, 22.0, 50.0]
    _d134 = [2, 1, 3, 6, 6, 1, 1, 2, 5, 2, 2,
             4, 1, 4, 1, 2, 4, 1, 5, 3, 10]
    _b134 = [-1.019535, 9.047135, -1.629789, -9.723916, -3.927170]

    def __init__(self):
        super().__init__()
        self.m_name = "HFC-134a"; self.m_formula = "CF3CH2F"
    def MolWt(self) -> float:
        mw = self._M
        return float(mw)

    def Tcrit(self) -> float:
        tc = self._Tc_crit
        return float(tc)

    def Pcrit(self) -> float:
        pc = self._Pc_crit
        return float(pc)

    def Vcrit(self) -> float:
        vc = 1.0 / self._Roc_crit
        return float(vc)

    def Tmin(self) -> float:
        tmin = self._Tmn
        return float(tmin)

    def Tmax(self) -> float:
        tmax = self._Tmx
        return float(tmax)

    def up(self):
        tau = self._Tc/self.T; delta = self.Rho/self._Roc
        b = self._b134; a = self._a134; t = self._t134; d = self._d134
        phi0t = b[1]*tau + b[2] - 0.5*b[3]*tau**(-0.5) - 0.75*b[4]*tau**(-0.75)
        s1 = sum(a[i]*t[i]*tau**t[i]*delta**d[i] for i in range(8))
        s2 = sum(a[i]*t[i]*tau**t[i]*delta**d[i] for i in range(8, 11))
        s3 = sum(a[i]*t[i]*tau**t[i]*delta**d[i] for i in range(11, 17))
        s4 = sum(a[i]*t[i]*tau**t[i]*delta**d[i] for i in range(17, 20))
        s5 = a[20]*t[20]*tau**t[20]*delta**d[20]
        phirt = (s1 + exp(-delta)*s2 + exp(-delta**2)*s3
                 + exp(-delta**3)*s4 + exp(-delta**4)*s5)
        return self._R*self.T*(phirt + phi0t) + self.m_energy_offset

    def sp(self):
        """Entropy from Helmholtz: s = -df/dT = R*(phi - tau*phi_tau)."""
        tau = self._Tc/self.T; delta = self.Rho/self._Roc
        b = self._b134; a = self._a134; t = self._t134; d = self._d134
        # phi0 and phi0_tau
        phi0 = b[0] + b[1]*tau + b[2]*log(tau) + log(delta) + b[3]/sqrt(tau) + b[4]*tau**(-0.75)
        phi0t = b[1]*tau + b[2] - 0.5*b[3]*tau**(-0.5) - 0.75*b[4]*tau**(-0.75)
        # phir and phir_tau
        s1 = sum(a[i]*tau**t[i]*delta**d[i] for i in range(8))
        s2 = sum(a[i]*tau**t[i]*delta**d[i] for i in range(8, 11))
        s3 = sum(a[i]*tau**t[i]*delta**d[i] for i in range(11, 17))
        s4 = sum(a[i]*tau**t[i]*delta**d[i] for i in range(17, 20))
        s5 = a[20]*tau**t[20]*delta**d[20]
        phir = s1 + exp(-delta)*s2 + exp(-delta**2)*s3 + exp(-delta**3)*s4 + exp(-delta**4)*s5
        s1t = sum(a[i]*t[i]*tau**t[i]*delta**d[i] for i in range(8))
        s2t = sum(a[i]*t[i]*tau**t[i]*delta**d[i] for i in range(8, 11))
        s3t = sum(a[i]*t[i]*tau**t[i]*delta**d[i] for i in range(11, 17))
        s4t = sum(a[i]*t[i]*tau**t[i]*delta**d[i] for i in range(17, 20))
        s5t = a[20]*t[20]*tau**t[20]*delta**d[20]
        phirt = s1t + exp(-delta)*s2t + exp(-delta**2)*s3t + exp(-delta**3)*s4t + exp(-delta**4)*s5t
        # s = -df/dT, f = R*T*(phi0 + phir), s = R*(tau*(phi0t + phirt) - (phi0 + phir))
        return self._R*(tau*(phi0t/tau + phirt/tau) - phi0 - phir) + self.m_entropy_offset

    def Pp(self):
        tau = self._Tc/self.T; delta = self.Rho/self._Roc
        a = self._a134; t = self._t134; d = self._d134
        phi0d = 1.0/delta
        s1 = sum(a[i]*tau**t[i]*d[i]*delta**(d[i]-1) for i in range(8))
        s2 = sum(a[i]*tau**t[i]*(d[i] - delta)*delta**(d[i]-1) for i in range(8, 11))
        s2 *= exp(-delta)
        dk = delta**2
        s3 = sum(a[i]*tau**t[i]*(d[i] - 2.0*dk)*delta**(d[i]-1) for i in range(11, 17))
        s3 *= exp(-dk)
        dk *= delta
        s4 = sum(a[i]*tau**t[i]*(d[i] - 3.0*dk)*delta**(d[i]-1) for i in range(17, 20))
        s4 *= exp(-dk)
        dk *= delta
        s5 = a[20]*tau**t[20]*(d[20] - 4.0*dk)*delta**(d[20]-1)
        s5 *= exp(-dk)
        phird = s1 + s2 + s3 + s4 + s5
        return self._R*self.T*delta**2*self._Roc*(phird + phi0d)

    def Psat(self):
        if self.T < self._Tmn or self.T > self._Tc:
            raise TPXError(f"HFC134a: Temperature out of range: {self.T}")
        x1 = self.T/self._Tc; x2 = 1.0 - x1
        f = -7.686556*x2 + 2.311791*x2**1.5 - 2.039554*x2**2 - 3.583758*x2**4
        return self._Pc_local * exp(f/x1)

    def ldens(self):
        if self.T < self._Tmn or self.T > self._Tc:
            raise TPXError(f"HFC134a: Temperature out of range: {self.T}")
        x2 = 1.0 - self.T/self._Tc
        return 518.2 + 884.13*x2**(1.0/3.0) + 485.84*x2**(2.0/3.0) + 193.29*x2**(10.0/3.0)


def newSubstance(name: str) -> Substance:
    """Create a new Substance object corresponding to the specified name.

    Direct port of tpx::newSubstance from utils.cpp.
    """
    lcname = str(name).strip().lower()
    if lcname == "water":
        return Water()
    elif lcname == "nitrogen":
        return Nitrogen()
    elif lcname == "methane":
        return Methane()
    elif lcname == "hydrogen":
        return Hydrogen()
    elif lcname == "oxygen":
        return Oxygen()
    elif lcname in ("hfc-134a", "hfc134a"):
        return HFC134a()
    elif lcname in ("carbon-dioxide", "carbondioxide"):
        return CarbonDioxide()
    elif lcname == "heptane":
        return Heptane()
    else:
        raise TPXError(f"No Substance definition known for '{name}'.")


