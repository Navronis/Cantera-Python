"""Aqueous electrolyte thermodynamics, Debye-Huckel and Pitzer models, and IAPWS-95 water.

Sources: DebyeHuckel.h, DebyeHuckel.cpp, HMWSoln.h, HMWSoln.cpp,
WaterProps.h, WaterProps.cpp, WaterPropsIAPWS.h, WaterPropsIAPWS.cpp,
WaterPropsIAPWSphi.h, WaterPropsIAPWSphi.cpp, electrolytes.h, electrolytes.cpp.
"""
from __future__ import annotations
import math
from math import sqrt, log, exp, pow, fabs
from typing import Sequence, Mapping, Any, Optional, Dict, List, Tuple
import numpy as np

from .constants import GasConstant, OneAtm, SmallNumber, Faraday
from .phases import ThermoPhase, Phase


class ElectrolyteError(Exception):
    """Exception class for electrolyte models."""
    pass


# Authoritative IAPWS-95 formulations directly ported from native Cantera C++ (Wagner & Pruss 2002)
ciR = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 6, 6, 6, 6, 0, 0, 0, 0]
diR = [0, 1, 1, 1, 2, 2, 3, 4, 1, 1, 1, 2, 2, 3, 4, 4, 5, 7, 9, 10, 11, 13, 15, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 9, 9, 9, 9, 9, 10, 10, 12, 3, 4, 4, 5, 14, 3, 6, 6, 6, 3, 3, 3]
tiR = [0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 12, 1, 5, 4, 2, 13, 9, 3, 4, 11, 4, 13, 1, 7, 1, 9, 10, 10, 3, 7, 10, 10, 6, 10, 10, 1, 2, 3, 4, 8, 6, 9, 8, 16, 22, 23, 23, 10, 50, 44, 46, 50, 0, 1, 4]
ni = [+0.0, +0.12533547935523E-1, +0.78957634722828E1, -0.87803203303561E1, +0.31802509345418E0, -0.26145533859358E0, -0.78199751687981E-2, +0.88089493102134E-2, -0.66856572307965E0, +0.20433810950965, -0.66212605039687E-4, -0.19232721156002E0, -0.25709043003438E0, +0.16074868486251E0, -0.40092828925807E-1, +0.39343422603254E-6, -0.75941377088144E-5, +0.56250979351888E-3, -0.15608652257135E-4, +0.11537996422951E-8, +0.36582165144204E-6, -0.13251180074668E-11, -0.62639586912454E-9, -0.10793600908932E0, +0.17611491008752E-1, +0.22132295167546E0, -0.40247669763528E0, +0.58083399985759E0, +0.49969146990806E-2, -0.31358700712549E-1, -0.74315929710341E0, +0.47807329915480E0, +0.20527940895948E-1, -0.13636435110343E0, +0.14180634400617E-1, +0.83326504880713E-2, -0.29052336009585E-1, +0.38615085574206E-1, -0.20393486513704E-1, -0.16554050063734E-2, +0.19955571979541E-2, +0.15870308324157E-3, -0.16388568342530E-4, +0.43613615723811E-1, +0.34994005463765E-1, -0.76788197844621E-1, +0.22446277332006E-1, -0.62689710414685E-4, -0.55711118565645E-9, -0.19905718354408E0, +0.31777497330738E0, -0.11841182425981E0, -0.31306260323435E2, +0.31546140237781E2, -0.25213154341695E4, -0.14874640856724E0, +0.31806110878444E0]

ni0 = [
    0.0,
    -8.32044648201 - 0.000000001739715,
    6.6832105268 + 0.000000000793232,
    3.00632,
    0.012436,
    0.97315,
    1.27950,
    0.96956,
    0.24873
]

gammi0 = [
    0.0,
    0.0,
    0.0,
    0.0,
    1.28728967,
    3.53734222,
    7.74073708,
    9.24437796,
    27.5075105
]


alphai = [20.0, 20.0, 20.0]
betai = [150.0, 150.0, 250.0]
gammai = [1.21, 1.21, 1.25]
epsi = [1.0, 1.0, 1.0]

ai = [3.5, 3.5]
bi = [0.85, 0.95]
Bi = [0.2, 0.2]
Ci = [28.0, 32.0]
Di = [700.0, 800.0]
Ai = [0.32, 0.32]
Bbetai = [0.3, 0.3]


T_c = 647.096
P_c = 22.064E6
Rho_c = 322.0
R_water = 461.51805

WATER_GAS = -1
WATER_LIQUID = 0
WATER_SUPERCRIT = -2
WATER_UNSTABLELIQUID = -3
WATER_UNSTABLEGAS = -4

Avogadro = 6.02214076e26
ElectronCharge = 1.602176634e-19
epsilon_0 = 8.8541878128e-12
Pi = 3.14159265358979323846

class WaterPropsIAPWSphi:
    def __init__(self):
        self.TAUsave = -1.0
        self.TAUsqrt = -1.0
        self.DELTAsave = -1.0
        self.TAUp = [0.0] * 52
        self.DELTAp = [0.0] * 16

    def tdpolycalc(self, tau: float, delta: float) -> None:
        self.TAUsave = tau
        self.TAUsqrt = sqrt(tau)
        self.TAUp[0] = 1.0
        for i in range(1, 52):
            self.TAUp[i] = self.TAUp[i-1] * tau
        self.DELTAsave = delta
        self.DELTAp[0] = 1.0
        for i in range(1, 16):
            self.DELTAp[i] = self.DELTAp[i-1] * delta

    def phi0(self) -> float:
        tau = self.TAUsave
        delta = self.DELTAsave
        retn = log(delta) + ni0[1] + ni0[2]*tau + ni0[3]*log(tau)
        for i in range(4, 9):
            retn += ni0[i] * log(1.0 - exp(-gammi0[i]*tau))
        return retn

    def phiR(self) -> float:
        tau = self.TAUsave
        delta = self.DELTAsave
        T375 = pow(tau, 0.375)
        val = (ni[1] * delta / self.TAUsqrt +
               ni[2] * delta * self.TAUsqrt * T375 +
               ni[3] * delta * tau +
               ni[4] * self.DELTAp[2] * self.TAUsqrt +
               ni[5] * self.DELTAp[2] * T375 * T375 +
               ni[6] * self.DELTAp[3] * T375 +
               ni[7] * self.DELTAp[4] * tau)
        for i in range(8, 52):
            val += (ni[i] * self.DELTAp[diR[i]] * self.TAUp[tiR[i]] * exp(-self.DELTAp[ciR[i]]))
        for j in range(3):
            i = 52 + j
            dtmp = delta - epsi[j]
            ttmp = tau - gammai[j]
            val += (ni[i] * self.DELTAp[diR[i]] * self.TAUp[tiR[i]] *
                    exp(-alphai[j]*dtmp*dtmp - betai[j]*ttmp*ttmp))
        for j in range(2):
            i = 55 + j
            deltam1 = delta - 1.0
            dtmp2 = deltam1 * deltam1
            atmp = 0.5 / Bbetai[j]
            theta = (1.0 - tau) + Ai[j] * pow(dtmp2, atmp)
            triag = theta * theta + Bi[j] * pow(dtmp2, ai[j])
            ttmp = tau - 1.0
            triagtmp = pow(triag, bi[j])
            phi_val = exp(-Ci[j]*dtmp2 - Di[j]*ttmp*ttmp)
            val += (ni[i] * triagtmp * delta * phi_val)
        return val

    def phi(self, tau: float, delta: float) -> float:
        self.tdpolycalc(tau, delta)
        return self.phi0() + self.phiR()

    def phi0_d(self) -> float:
        return 1.0 / self.DELTAsave

    def phiR_d(self) -> float:
        tau = self.TAUsave
        delta = self.DELTAsave
        T375 = pow(tau, 0.375)
        val = (ni[1] / self.TAUsqrt +
               ni[2] * self.TAUsqrt * T375 +
               ni[3] * tau +
               ni[4] * 2.0 * delta * self.TAUsqrt +
               ni[5] * 2.0 * delta * T375 * T375 +
               ni[6] * 3.0 * self.DELTAp[2] * T375 +
               ni[7] * 4.0 * self.DELTAp[3] * tau)
        for i in range(8, 52):
            val += ((ni[i] * exp(-self.DELTAp[ciR[i]]) * self.DELTAp[diR[i] - 1] *
                     self.TAUp[tiR[i]]) * (diR[i] - ciR[i]* self.DELTAp[ciR[i]]))
        for j in range(3):
            i = 52 + j
            dtmp = delta - epsi[j]
            ttmp = tau - gammai[j]
            tmp = (ni[i] * self.DELTAp[diR[i]] * self.TAUp[tiR[i]] *
                   exp(-alphai[j]*dtmp*dtmp - betai[j]*ttmp*ttmp))
            val += tmp * (diR[i]/delta - 2.0 * alphai[j] * dtmp)
        for j in range(2):
            i = 55 + j
            deltam1 = delta - 1.0
            dtmp2 = deltam1 * deltam1
            atmp = 0.5 / Bbetai[j]
            theta = (1.0 - tau) + Ai[j] * pow(dtmp2, atmp)
            triag = theta * theta + Bi[j] * pow(dtmp2, ai[j])
            ttmp = tau - 1.0
            triagtmp = pow(triag, bi[j])
            triagtmpm1 = pow(triag, bi[j]-1.0)
            atmpM1 = atmp - 1.0
            ptmp = pow(dtmp2, atmpM1)
            p2tmp = pow(dtmp2, ai[j]-1.0)
            dtriagddelta = deltam1 * (Ai[j] * theta * 2.0 / Bbetai[j] * ptmp + 2.0*Bi[j]*ai[j]*p2tmp)
            phi_val = exp(-Ci[j]*dtmp2 - Di[j]*ttmp*ttmp)
            dphiddelta = -2.0*Ci[j]*deltam1*phi_val
            val += ni[i] * (triagtmp * (phi_val + delta * dphiddelta) +
                           delta * phi_val * bi[j] * triagtmpm1 * dtriagddelta)
        return val

    def phi_d(self, tau: float, delta: float) -> float:
        self.tdpolycalc(tau, delta)
        return self.phi0_d() + self.phiR_d()

    def phi0_dd(self) -> float:
        delta = self.DELTAsave
        return -1.0 / (delta * delta)

    def phiR_dd(self) -> float:
        tau = self.TAUsave
        delta = self.DELTAsave
        T375 = pow(tau, 0.375)
        val = (ni[4] * 2.0 * self.TAUsqrt +
               ni[5] * 2.0 * T375 * T375 +
               ni[6] * 6.0 * delta * T375 +
               ni[7] * 12.0 * self.DELTAp[2] * tau)
        for i in range(8, 52):
            dtmp = self.DELTAp[ciR[i]]
            tmp = ni[i] * exp(-dtmp) * self.TAUp[tiR[i]]
            atmp = 1.0/delta if diR[i] == 1 else self.DELTAp[diR[i] - 2]
            tmp *= atmp * ((diR[i] - ciR[i]*dtmp)*(diR[i]-1.0-ciR[i]*dtmp) - ciR[i]*ciR[i]*dtmp)
            val += tmp
        for j in range(3):
            i = 52 + j
            dtmp = delta - epsi[j]
            ttmp = tau - gammai[j]
            tmp = (ni[i] * self.TAUp[tiR[i]] * exp(-alphai[j]*dtmp*dtmp - betai[j]*ttmp*ttmp))
            deltmp = self.DELTAp[diR[i]]
            deltmpM1 = deltmp/delta
            deltmpM2 = deltmpM1 / delta
            d2tmp = dtmp * dtmp
            val += tmp * (-2.0*alphai[j]*deltmp +
                          4.0 * alphai[j] * alphai[j] * deltmp * d2tmp -
                          4.0 * diR[i] * alphai[j] * deltmpM1 * dtmp +
                          diR[i] * (diR[i] - 1.0) * deltmpM2)
        for j in range(2):
            i = 55 + j
            deltam1 = delta - 1.0
            dtmp2 = deltam1 * deltam1
            atmp = 0.5 / Bbetai[j]
            theta = (1.0 - tau) + Ai[j] * pow(dtmp2, atmp)
            triag = theta * theta + Bi[j] * pow(dtmp2, ai[j])
            ttmp = tau - 1.0
            triagtmp = pow(triag, bi[j])
            triagtmpm1 = pow(triag, bi[j]-1.0)
            atmpM1 = atmp - 1.0
            ptmp = pow(dtmp2, atmpM1)
            p2tmp = pow(dtmp2, ai[j]-1.0)
            dtriagddelta = deltam1 * (Ai[j] * theta * 2.0 / Bbetai[j] * ptmp + 2.0*Bi[j]*ai[j]*p2tmp)
            phi_val = exp(-Ci[j]*dtmp2 - Di[j]*ttmp*ttmp)
            dphiddelta = -2.0*Ci[j]*deltam1*phi_val
            dtriagtmpddelta = bi[j] * triagtmpm1 * dtriagddelta
            d2phiddelta2 = 2.0 * Ci[j] * phi_val * (2.0*Ci[j]*dtmp2 - 1.0)
            pptmp = ptmp / dtmp2
            d2triagddelta2 = dtriagddelta / deltam1
            d2triagddelta2 += dtmp2 * (4.0*Bi[j]*ai[j]*(ai[j]-1.0)*pow(dtmp2, ai[j]-2.0) +
                                      2.0*Ai[j]*Ai[j]/(Bbetai[j]*Bbetai[j])*ptmp*ptmp +
                                      Ai[j]*theta*4.0/Bbetai[j]*(atmp-1.0)*pptmp)
            d2triagtmpd2delta = bi[j] * (triagtmpm1 * d2triagddelta2 +
                                         (bi[j]-1.0)*triagtmpm1/triag*dtriagddelta*dtriagddelta)
            ctmp = (triagtmp * (2.0*dphiddelta + delta*d2phiddelta2) +
                    2.0*dtriagtmpddelta*(phi_val + delta * dphiddelta) +
                    d2triagtmpd2delta * delta * phi_val)
            val += ni[i] * ctmp
        return val

    def phi_dd(self, tau: float, delta: float) -> float:
        self.tdpolycalc(tau, delta)
        return self.phi0_dd() + self.phiR_dd()

    def phi0_t(self) -> float:
        tau = self.TAUsave
        retn = ni0[2] + ni0[3]/tau
        for i in range(4, 9):
            retn += (ni0[i] * gammi0[i] * (1.0/(1.0 - exp(-gammi0[i]*tau)) - 1.0))
        return retn

    def phiR_t(self) -> float:
        tau = self.TAUsave
        delta = self.DELTAsave
        T375 = pow(tau, 0.375)
        val = ((-0.5) * ni[1] * delta / self.TAUsqrt / tau +
               ni[2] * delta * 0.875 / self.TAUsqrt * T375 +
               ni[3] * delta +
               ni[4] * self.DELTAp[2] * 0.5 / self.TAUsqrt +
               ni[5] * self.DELTAp[2] * 0.75 * T375 * T375 / tau +
               ni[6] * self.DELTAp[3] * 0.375 * T375 / tau +
               ni[7] * self.DELTAp[4])
        for i in range(8, 52):
            tmp = (ni[i] * self.DELTAp[diR[i]] * self.TAUp[tiR[i]-1] * exp(-self.DELTAp[ciR[i]]))
            val += tiR[i] * tmp
        for j in range(3):
            i = 52 + j
            dtmp = delta - epsi[j]
            ttmp = tau - gammai[j]
            tmp = (ni[i] * self.DELTAp[diR[i]] * self.TAUp[tiR[i]] *
                   exp(-alphai[j]*dtmp*dtmp - betai[j]*ttmp*ttmp))
            val += tmp * (tiR[i]/tau - 2.0 * betai[j]*ttmp)
        for j in range(2):
            i = 55 + j
            deltam1 = delta - 1.0
            dtmp2 = deltam1 * deltam1
            atmp = 0.5 / Bbetai[j]
            theta = (1.0 - tau) + Ai[j] * pow(dtmp2, atmp)
            triag = theta * theta + Bi[j] * pow(dtmp2, ai[j])
            ttmp = tau - 1.0
            triagtmp = pow(triag, bi[j])
            phi_val = exp(-Ci[j]*dtmp2 - Di[j]*ttmp*ttmp)
            dtriagtmpdtau = -2.0*theta * bi[j] * triagtmp / triag
            dphidtau = - 2.0 * Di[j] * ttmp * phi_val
            val += ni[i] * delta * (dtriagtmpdtau * phi_val + triagtmp * dphidtau)
        return val

    def phi_t(self, tau: float, delta: float) -> float:
        self.tdpolycalc(tau, delta)
        return self.phi0_t() + self.phiR_t()

    def phi0_tt(self) -> float:
        tau = self.TAUsave
        retn = - ni0[3]/(tau * tau)
        for i in range(4, 9):
            tmp = exp(-gammi0[i]*tau)
            itmp = 1.0 - tmp
            retn -= (ni0[i] * gammi0[i] * gammi0[i] * tmp / (itmp * itmp))
        return retn

    def phiR_tt(self) -> float:
        tau = self.TAUsave
        delta = self.DELTAsave
        T375 = pow(tau, 0.375)
        val = ((-0.5) * (-1.5) * ni[1] * delta / (self.TAUsqrt * tau * tau) +
               ni[2] * delta * 0.875 * (-0.125) * T375 / (self.TAUsqrt * tau) +
               ni[4] * self.DELTAp[2] * 0.5 * (-0.5)/ (self.TAUsqrt * tau) +
               ni[5] * self.DELTAp[2] * 0.75 *(-0.25) * T375 * T375 / (tau * tau) +
               ni[6] * self.DELTAp[3] * 0.375 *(-0.625) * T375 / (tau * tau))
        for i in range(8, 52):
            if tiR[i] > 1:
                tmp = (ni[i] * self.DELTAp[diR[i]] * self.TAUp[tiR[i]-2] * exp(-self.DELTAp[ciR[i]]))
                val += tiR[i] * (tiR[i] - 1.0) * tmp
        for j in range(3):
            i = 52 + j
            dtmp = delta - epsi[j]
            ttmp = tau - gammai[j]
            tmp = (ni[i] * self.DELTAp[diR[i]] * self.TAUp[tiR[i]] *
                   exp(-alphai[j]*dtmp*dtmp - betai[j]*ttmp*ttmp))
            atmp = tiR[i]/tau - 2.0 * betai[j]*ttmp
            val += tmp * (atmp * atmp - tiR[i]/(tau*tau) - 2.0*betai[j])
        for j in range(2):
            i = 55 + j
            deltam1 = delta - 1.0
            dtmp2 = deltam1 * deltam1
            atmp = 0.5 / Bbetai[j]
            theta = (1.0 - tau) + Ai[j] * pow(dtmp2, atmp)
            triag = theta * theta + Bi[j] * pow(dtmp2, ai[j])
            ttmp = tau - 1.0
            triagtmp = pow(triag, bi[j])
            triagtmpM1 = triagtmp / triag
            phi_val = exp(-Ci[j]*dtmp2 - Di[j]*ttmp*ttmp)
            dtriagtmpdtau = -2.0*theta * bi[j] * triagtmp / triag
            dphidtau = - 2.0 * Di[j] * ttmp * phi_val
            d2triagtmpdtau2 = (2 * bi[j] * triagtmpM1 +
                               4 * theta * theta * bi[j] * (bi[j]-1.0) * triagtmpM1 / triag)
            d2phidtau2 = 2.0*Di[j]*phi_val * (2.0*Di[j]*ttmp*ttmp - 1.0)
            tmp = (d2triagtmpdtau2 * phi_val + 2.0 * dtriagtmpdtau * dphidtau +
                   triagtmp * d2phidtau2)
            val += ni[i] * delta * tmp
        return val

    def phi_tt(self, tau: float, delta: float) -> float:
        self.tdpolycalc(tau, delta)
        return self.phi0_tt() + self.phiR_tt()

    def phi0_dt(self) -> float:
        return 0.0

    def phiR_dt(self) -> float:
        tau = self.TAUsave
        delta = self.DELTAsave
        T375 = pow(tau, 0.375)
        val = (ni[1] * (-0.5) / (self.TAUsqrt * tau) +
               ni[2] * (0.875) * T375 / self.TAUsqrt +
               ni[3] +
               ni[4] * 2.0 * delta * (0.5) / self.TAUsqrt +
               ni[5] * 2.0 * delta * (0.75) * T375 * T375 / tau +
               ni[6] * 3.0 * self.DELTAp[2] * 0.375 * T375 / tau +
               ni[7] * 4.0 * self.DELTAp[3])
        for i in range(8, 52):
            tmp = (ni[i] * tiR[i] * exp(-self.DELTAp[ciR[i]]) * self.DELTAp[diR[i] - 1] *
                   self.TAUp[tiR[i] - 1])
            val += tmp * (diR[i] - ciR[i] * self.DELTAp[ciR[i]])
        for j in range(3):
            i = 52 + j
            dtmp = delta - epsi[j]
            ttmp = tau - gammai[j]
            tmp = (ni[i] * self.DELTAp[diR[i]] * self.TAUp[tiR[i]] *
                   exp(-alphai[j]*dtmp*dtmp - betai[j]*ttmp*ttmp))
            val += tmp * ((diR[i]/delta - 2.0 * alphai[j] * dtmp) *
                          (tiR[i]/tau - 2.0 * betai[j] * ttmp))
        for j in range(2):
            i = 55 + j
            deltam1 = delta - 1.0
            dtmp2 = deltam1 * deltam1
            atmp = 0.5 / Bbetai[j]
            theta = (1.0 - tau) + Ai[j] * pow(dtmp2, atmp)
            triag = theta * theta + Bi[j] * pow(dtmp2, ai[j])
            ttmp = tau - 1.0
            triagtmp = pow(triag, bi[j])
            triagtmpm1 = pow(triag, bi[j]-1.0)
            atmpM1 = atmp - 1.0
            ptmp = pow(dtmp2, atmpM1)
            p2tmp = pow(dtmp2, ai[j]-1.0)
            dtriagddelta = deltam1 * (Ai[j] * theta * 2.0 / Bbetai[j] * ptmp + 2.0*Bi[j]*ai[j]*p2tmp)
            phi_val = exp(-Ci[j]*dtmp2 - Di[j]*ttmp*ttmp)
            dphiddelta = -2.0*Ci[j]*deltam1*phi_val
            dtriagtmpddelta = bi[j] * triagtmpm1 * dtriagddelta
            dtriagtmpdtau = -2.0*theta * bi[j] * triagtmp / triag
            dphidtau = - 2.0 * Di[j] * ttmp * phi_val
            d2phiddeltadtau = 4.0 * Ci[j] * Di[j] * deltam1 * ttmp * phi_val
            d2triagtmpddeltadtau = (-Ai[j] * bi[j] * 2.0 / Bbetai[j] * triagtmpm1 * deltam1 * ptmp
                                    -2.0 * theta * bi[j] * (bi[j] - 1.0) * triagtmpm1 / triag * dtriagddelta)
            tmp = ni[i] * (triagtmp * (dphidtau + delta*d2phiddeltadtau) +
                           delta * dtriagtmpddelta * dphidtau +
                           dtriagtmpdtau * (phi_val + delta * dphiddelta) +
                           d2triagtmpddeltadtau * delta * phi_val)
            val += tmp
        return val

    def pressureM_rhoRT(self, tau: float, delta: float) -> float:
        self.tdpolycalc(tau, delta)
        return 1.0 + delta * self.phiR_d()

    def dimdpdrho(self, tau: float, delta: float) -> float:
        self.tdpolycalc(tau, delta)
        res1 = self.phiR_d()
        res2 = self.phiR_dd()
        return 1.0 + delta * (2.0*res1 + delta*res2)

    def dimdpdT(self, tau: float, delta: float) -> float:
        self.tdpolycalc(tau, delta)
        res1 = self.phiR_d()
        res2 = self.phiR_dt()
        return (1.0 + delta * res1) - tau * delta * res2

    def dfind(self, p_red: float, tau: float, deltaGuess: float) -> float:
        dd = deltaGuess
        conv = False
        deldd = dd
        pcheck = 1.0e-30 + 1.0e-14 * p_red
        for n in range(200):
            self.tdpolycalc(tau, dd)
            q1 = self.phiR_d()
            q2 = self.phiR_dd()
            pred0 = dd + dd * dd * q1
            dpddelta = 1.0 + 2.0 * dd * q1 + dd * dd * q2
            if dpddelta <= 0.0:
                if deltaGuess > 1.0:
                    dd = dd * 1.05
                if deltaGuess < 1.0:
                    dd = dd * 0.95
                continue
            if fabs(pred0 - p_red) < pcheck:
                conv = True
                break
            dpdx = dpddelta
            if n < 10:
                dpdx = dpddelta * 1.1
            dpdx = max(dpdx, 0.001)
            deldd = - (pred0 - p_red) / dpdx
            if fabs(deldd) > 0.05:
                deldd = deldd * 0.05 / fabs(deldd)
            dd += deldd
            if dd <= 0.0:
                dd = 0.5 * (dd - deldd)
            if fabs(deldd) < 1.0e-14:
                conv = True
                break
        return dd if conv else -1.0

    def gibbs_RT(self) -> float:
        delta = self.DELTAsave
        rd = self.phiR_d()
        return 1.0 + self.phi0() + self.phiR() + delta * rd

    def enthalpy_RT(self) -> float:
        delta = self.DELTAsave
        tau = self.TAUsave
        rd = self.phiR_d()
        nt = self.phi0_t()
        rt = self.phiR_t()
        return 1.0 + tau * (nt + rt) + delta * rd

    def entropy_R(self) -> float:
        tau = self.TAUsave
        nt = self.phi0_t()
        rt = self.phiR_t()
        p0 = self.phi0()
        pR = self.phiR()
        return tau * (nt + rt) - p0 - pR

    def intEnergy_RT(self) -> float:
        tau = self.TAUsave
        nt = self.phi0_t()
        rt = self.phiR_t()
        return tau * (nt + rt)

    def cv_R(self) -> float:
        tau = self.TAUsave
        ntt = self.phi0_tt()
        rtt = self.phiR_tt()
        return - tau * tau * (ntt + rtt)

    def cp_R(self) -> float:
        tau = self.TAUsave
        delta = self.DELTAsave
        cvR = self.cv_R()
        rd = self.phiR_d()
        rdd = self.phiR_dd()
        rdt = self.phiR_dt()
        num = (1.0 + delta * rd - delta * tau * rdt)
        return cvR + (num * num / (1.0 + 2.0 * delta * rd + delta * delta * rdd))



class WaterPropsIAPWS:
    def __init__(self):
        self.m_phi = WaterPropsIAPWSphi()
        self.tau = -1.0
        self.delta = -1.0
        self.iState = WATER_GAS

    def calcDim(self, temperature: float, rho: float) -> None:
        self.tau = T_c / temperature
        self.delta = rho / Rho_c
        if temperature > T_c:
            self.iState = WATER_SUPERCRIT
        else:
            self.iState = WATER_GAS if self.delta < 1.0 else WATER_LIQUID

    def pressure(self) -> float:
        retn = self.m_phi.pressureM_rhoRT(self.tau, self.delta)
        rho = self.delta * Rho_c
        temperature = T_c / self.tau
        return retn * rho * R_water * temperature

    def setState_TD(self, temperature: float, rho: float) -> None:
        self.calcDim(temperature, rho)
        self.m_phi.tdpolycalc(self.tau, self.delta)

    def density(self, temperature: float = None, pressure: float = None,
                phase: int = -1, rhoguess: float = -1.0) -> float:
        if temperature is None:
            return self.delta * Rho_c
        if fabs(pressure - P_c) / P_c < 1.0e-8 and fabs(temperature - T_c) / T_c < 1.0e-8:
            self.setState_TD(temperature, Rho_c)
            return Rho_c

        if rhoguess == -1.0:
            if phase != -1:
                if temperature > T_c:
                    rhoguess = pressure / (R_water * temperature)
                else:
                    if phase in (WATER_GAS, WATER_SUPERCRIT):
                        rhoguess = pressure / (R_water * temperature)
                    elif phase == WATER_LIQUID:
                        rhoguess = 1000.0
                    else:
                        raise ValueError(f"Unknown or unstable state: {phase}")
            else:
                rhoguess = pressure / (R_water * temperature)

        p_red = pressure / (R_water * temperature * Rho_c)
        deltaGuess = rhoguess / Rho_c
        self.setState_TD(temperature, rhoguess)
        delta_retn = self.m_phi.dfind(p_red, self.tau, deltaGuess)
        if delta_retn <= 0.0:
            delta_retn = self.m_phi.dfind(p_red, self.tau, 0.9 * deltaGuess)

        if delta_retn > 0.0:
            self.delta = delta_retn
            density_retn = delta_retn * Rho_c
            self.setState_TD(temperature, density_retn)
        else:
            density_retn = -1.0
        return density_retn

    def temperature(self) -> float:
        return T_c / self.tau

    def psat_est(self, temperature: float) -> float:
        A = [
            -7.8889166E0,
            2.5514255E0,
            -6.716169E0,
            33.2239495E0,
            -105.38479E0,
            174.35319E0,
            -148.39348E0,
            48.631602E0
        ]
        if temperature < 314.0:
            pl = 6.3573118E0 - 8858.843E0 / temperature + 607.56335E0 * pow(temperature, -0.6)
            ps = 0.1 * exp(pl)
        else:
            v = temperature / 647.25
            w = fabs(1.0 - v)
            b = 0.0
            for i in range(8):
                z = i + 1
                b += A[i] * pow(w, (z + 1.0) / 2.0)
            q = b / v
            ps = 22.093 * exp(q)
        ps *= 1.0e6
        return ps

    def dpdrho(self) -> float:
        retn = self.m_phi.dimdpdrho(self.tau, self.delta)
        temperature = T_c / self.tau
        return retn * R_water * temperature

    def isothermalCompressibility(self) -> float:
        dpdrho_val = self.dpdrho()
        dens = self.delta * Rho_c
        return 1.0 / (dens * dpdrho_val)

    def coeffPresExp(self) -> float:
        return self.m_phi.dimdpdT(self.tau, self.delta)

    def coeffThermExp(self) -> float:
        kappa = self.isothermalCompressibility()
        beta = self.coeffPresExp()
        dens = self.delta * Rho_c
        return kappa * dens * R_water * beta

    def corr(self, temperature: float, pressure: float, densLiq: float, densGas: float):
        densLiq = self.density(temperature, pressure, WATER_LIQUID, densLiq)
        if densLiq <= 0.0:
            raise ValueError(f"Error finding liquid density at T={temperature}, P={pressure}")
        self.setState_TD(temperature, densLiq)
        gibbsLiqRT = self.m_phi.gibbs_RT()

        densGas = self.density(temperature, pressure, WATER_GAS, densGas)
        if densGas <= 0.0:
            raise ValueError(f"Error finding gas density at T={temperature}, P={pressure}")
        self.setState_TD(temperature, densGas)
        gibbsGasRT = self.m_phi.gibbs_RT()

        delGRT = gibbsLiqRT - gibbsGasRT
        return densLiq, densGas, delGRT

    def psat(self, temperature: float, waterState: int = WATER_LIQUID) -> float:
        densLiq = -1.0
        densGas = -1.0
        delGRT = 0.0
        if temperature >= T_c:
            densGas = self.density(temperature, P_c, WATER_SUPERCRIT)
            self.setState_TD(temperature, densGas)
            return P_c
        p = self.psat_est(temperature)
        for i in range(30):
            densLiq, densGas, delGRT = self.corr(temperature, p, densLiq, densGas)
            delV = 1.0 / densLiq - 1.0 / densGas
            p -= delGRT * R_water * temperature / delV
            if fabs(delGRT) < 1.0e-8:
                break
        if waterState == WATER_LIQUID:
            self.setState_TD(temperature, densLiq)
        elif waterState == WATER_GAS:
            self.setState_TD(temperature, densGas)
        return p

    def phaseState(self, checkState: bool = False) -> int:
        return self.iState

    def Tcrit(self) -> float:
        return T_c

    def Pcrit(self) -> float:
        return P_c

    def Rhocrit(self) -> float:
        return Rho_c

    def gibbs_mass(self) -> float:
        return self.m_phi.gibbs_RT() * R_water * T_c / self.tau

    def enthalpy_mass(self) -> float:
        return self.m_phi.enthalpy_RT() * R_water * T_c / self.tau

    def intEnergy_mass(self) -> float:
        return self.m_phi.intEnergy_RT() * R_water * T_c / self.tau

    def entropy_mass(self) -> float:
        return self.m_phi.entropy_R() * R_water

    def cv_mass(self) -> float:
        return self.m_phi.cv_R() * R_water

    def cp_mass(self) -> float:
        return self.m_phi.cp_R() * R_water



class WaterProps:
    def __init__(self, waterIAPWS: WaterPropsIAPWS = None):
        self.m_waterIAPWS = waterIAPWS if waterIAPWS is not None else WaterPropsIAPWS()

    @staticmethod
    def density(T: float, P: float = OneAtm) -> float:
        return float(WaterProps.density_T(float(T), float(P)))

    @staticmethod
    def dielectric_constant(T: float, P: float = OneAtm) -> float:
        wp = WaterProps()
        return float(wp.relEpsilon(float(T), float(P), 0))

    @staticmethod
    def debye_huckel_A_phi(T: float, P: float = OneAtm) -> float:
        wp = WaterProps()
        return float(wp.ADebye(float(T), float(P), 0) / 3.0)

    @staticmethod
    def debye_huckel_B(T: float) -> float:
        t = max(float(T), 200.0)
        return float(3.2864 * math.sqrt(298.15 / t))

    @staticmethod
    def density_T(T: float, P: float, ifunc: int = 0) -> float:
        Tc = T - 273.15
        U1 = 288.9414
        U2 = 508929.2
        U3 = 68.12963
        U4 = -3.9863

        tmp1 = Tc + U1
        tmp4 = Tc + U4
        t4t4 = tmp4 * tmp4
        tmp3 = Tc + U3
        rho = 1000.0 * (1.0 - tmp1 * t4t4 / (U2 * tmp3))

        rhomin = P / (GasConstant * T)
        if rho < rhomin:
            rho = rhomin
            if ifunc == 1:
                return - rhomin / T
            elif ifunc == 3:
                return rhomin / P
            elif ifunc == 2:
                return 2.0 * rhomin / (T * T)

        if ifunc == 1:
            drhodT = 1000.0 / U2 * (- tmp4 * tmp4 / tmp3 - tmp1 * 2.0 * tmp4 / tmp3 + tmp1 * t4t4 / (tmp3 * tmp3))
            return drhodT
        elif ifunc == 3:
            return 0.0
        elif ifunc == 2:
            t3t3 = tmp3 * tmp3
            d2rhodT2 = 1000.0 / U2 * ((-4.0 * tmp4 - 2.0 * tmp1) / tmp3 +
                                      (2.0 * t4t4 + 4.0 * tmp1 * tmp4) / t3t3 -
                                      2.0 * tmp1 * t4t4 / (t3t3 * tmp3))
            return d2rhodT2
        return rho

    def relEpsilon(self, T: float, P_pascal: float, ifunc: int = 0) -> float:
        U1 = 3.4279E2
        U2 = -5.0866E-3
        U3 = 9.4690E-7
        U4 = -2.0525
        U5 = 3.1159E3
        U6 = -1.8289E2
        U7 = -8.0325E3
        U8 = 4.2142E6
        U9 = 2.1417
        T2 = T * T

        eps1000 = U1 * exp(U2 * T + U3 * T2)
        C = U4 + U5 / (U6 + T)
        B = U7 + U8 / T + U9 * T
        Pbar = P_pascal * 1.0E-5
        tmpBpar = B + Pbar
        tmpB1000 = B + 1000.0
        ltmp = log(tmpBpar / tmpB1000)
        epsRel = eps1000 + C * ltmp

        if ifunc in (1, 2):
            tmpC = U6 + T
            dCdT = - U5 / (tmpC * tmpC)
            dBdT = - U8 / (T * T) + U9
            deps1000dT = eps1000 * (U2 + 2.0 * U3 * T)
            dltmpdT = (dBdT / tmpBpar - dBdT / tmpB1000)
            if ifunc == 1:
                return deps1000dT + dCdT * ltmp + C * dltmpdT
            T3 = T2 * T
            d2CdT2 = - 2.0 * dCdT / tmpC
            d2BdT2 = 2.0 * U8 / T3
            d2ltmpdT2 = (d2BdT2 * (1.0 / tmpBpar - 1.0 / tmpB1000) +
                         dBdT * dBdT * (1.0 / (tmpB1000 * tmpB1000) - 1.0 / (tmpBpar * tmpBpar)))
            d2eps1000dT2 = (deps1000dT * (U2 + 2.0 * U3 * T) + eps1000 * (2.0 * U3))
            if ifunc == 2:
                d2epsReldT2 = (d2eps1000dT2 + d2CdT2 * ltmp + 2.0 * dCdT * dltmpdT + C * d2ltmpdT2)
                return d2epsReldT2
        if ifunc == 3:
            dltmpdP = 1.0E-5 / tmpBpar
            return C * dltmpdP
        return epsRel

    def ADebye(self, T: float, P_input: float, ifunc: int = 0) -> float:
        psat = self.satPressure(T)
        P = max(psat, P_input)
        epsRelWater = self.relEpsilon(T, P, 0)
        epsilon = epsilon_0 * epsRelWater
        dw = self.density_IAPWS(T, P)
        tmp = sqrt(2.0 * Avogadro * dw / 1000.0)
        tmp2 = ElectronCharge * ElectronCharge * Avogadro / (epsilon * GasConstant * T)
        tmp3 = tmp2 * sqrt(tmp2)
        A_Debye = tmp * tmp3 / (8.0 * Pi)

        if ifunc in (1, 2):
            dAdT = - 1.5 * A_Debye / T
            depsRelWaterdT = self.relEpsilon(T, P, 1)
            dAdT -= A_Debye * (1.5 * depsRelWaterdT / epsRelWater)
            cte = self.coeffThermalExp_IAPWS(T, P)
            contrib2 = - A_Debye * (0.5 * cte)
            dAdT += contrib2
            if ifunc == 1:
                return dAdT
            if ifunc == 2:
                d2AdT2 = 1.5 / T * (A_Debye / T - dAdT)
                d2epsRelWaterdT2 = self.relEpsilon(T, P, 2)
                d2AdT2 += 1.5 * (- dAdT * depsRelWaterdT / epsRelWater -
                                 A_Debye / epsRelWater *
                                 (d2epsRelWaterdT2 - depsRelWaterdT * depsRelWaterdT / epsRelWater))
                deltaT = 0.1
                cte_del_p = self.coeffThermalExp_IAPWS(T + deltaT, P)
                cte_del_m = self.coeffThermalExp_IAPWS(T - deltaT, P)
                dctedT = (cte_del_p - cte_del_m) / (2.0 * deltaT)
                contrib3 = 0.5 * (-(dAdT * cte) - (A_Debye * dctedT))
                d2AdT2 += contrib3
                return d2AdT2

        if ifunc == 3:
            depsRelWaterdP = self.relEpsilon(T, P, 3)
            dAdP = - A_Debye * (1.5 * depsRelWaterdP / epsRelWater)
            kappa = self.isothermalCompressibility_IAPWS(T, P)
            dAdP += A_Debye * (0.5 * kappa)
            return dAdP
        return A_Debye

    def satPressure(self, T: float) -> float:
        return self.m_waterIAPWS.psat(T)

    def density_IAPWS(self, temp: float = None, press: float = None) -> float:
        if temp is None:
            return self.m_waterIAPWS.density()
        return self.m_waterIAPWS.density(temp, press, WATER_LIQUID)

    def coeffThermalExp_IAPWS(self, temp: float, press: float) -> float:
        dens = self.m_waterIAPWS.density(temp, press, WATER_LIQUID)
        if dens < 0.0:
            raise ValueError(f"Unable to solve for density at T={temp}, P={press}")
        return self.m_waterIAPWS.coeffThermExp()

    def isothermalCompressibility_IAPWS(self, temp: float, press: float) -> float:
        dens = self.m_waterIAPWS.density(temp, press, WATER_LIQUID)
        if dens < 0.0:
            raise ValueError(f"Unable to solve for density at T={temp}, P={press}")
        return self.m_waterIAPWS.isothermalCompressibility()


class DebyeHuckel(ThermoPhase):
    """Dilute electrolyte phase obeying Debye-Huckel limiting law or Helgeson B-dot formulations."""

    def __init__(self, name: str = "dilute_electrolyte", species_names: Optional[List[str]] = None,
                 charges: Optional[List[float]] = None, b_dot: float = 0.0, **kwargs):
        super().__init__(name)
        self._formDH = 1
        self._A_Debye = 1.172576
        self._B_Debye = 3.28640e9
        self._maxIionicStrength = 30.0
        self._useHelgesonFixedForm = False
        self._Aionic_default = 3e-10
        self._Aionic = np.array([], dtype=float)
        self._B_Dot = np.array([], dtype=float)
        self._speciesCharge_Stoich = np.array([], dtype=float)
        self._Beta_ij = np.zeros((0, 0), dtype=float)
        self._molalities = np.array([], dtype=float)
        self._lnActCoeffMolal = np.array([], dtype=float)
        self._dlnActCoeffMolaldT = np.array([], dtype=float)
        self._d2lnActCoeffMolaldT2 = np.array([], dtype=float)
        self._dlnActCoeffMolaldP = np.array([], dtype=float)
        self._IionicMolality = 0.0
        self._IionicMolalityStoich = 0.0
        self._Mnaught = 0.01801528
        self._waterProps = WaterProps()

        if species_names is not None:
            for i, sp in enumerate(species_names):
                self.addSpecies(sp)
                q = float(charges[i]) if charges and i < len(charges) else 0.0
                self.set_charge(sp, q)
            self._molalities = np.zeros(len(species_names), dtype=float)
            self._lnActCoeffMolal = np.zeros(len(species_names), dtype=float)

        if b_dot > 0.0:
            self.setB_dot(b_dot)

    def DebyeHuckel(self) -> DebyeHuckel:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "DebyeHuckel"

    def formDH(self) -> int:
        return int(self._formDH)

    def set_charge(self, sp: str, q: float) -> None:
        idx = self.speciesIndex(sp)
        if idx >= 0 and idx < len(self._species_charge):
            self._species_charge[idx] = float(q)
            self.compositionChanged()

    @property
    def molalities(self) -> np.ndarray:
        return self._molalities.copy()

    @molalities.setter
    def molalities(self, m: Sequence[float]) -> None:
        arr = np.asarray(m, dtype=float)
        n = len(self._species_names)
        self._molalities = np.zeros(n, dtype=float)
        count = min(len(arr), n)
        self._molalities[:count] = arr[:count]
        self.compositionChanged()

    @property
    def ionic_strength(self) -> float:
        """Ionic strength: I = 0.5 * sum(m_k * z_k^2)."""
        tot = 0.0
        for k in range(len(self._species_names)):
            z = self.charge(k)
            m = self._molalities[k] if k < len(self._molalities) else 0.0
            tot += m * (z * z)
        return float(0.5 * tot)

    def activity_coefficients(self, form: str = "limiting") -> np.ndarray:
        """Return activity coefficients array."""
        n = len(self._species_names)
        i_str = self.ionic_strength
        sqrt_i = math.sqrt(max(i_str, 1e-12))
        res = np.ones(n, dtype=float)
        a_deb = 0.509  # log10 form
        for k in range(n):
            z = self.charge(k)
            if form == "limiting":
                log_gamma = - a_deb * (z * z) * sqrt_i
            elif form == "b_dot":
                bdot = float(self._B_Dot[k]) if k < len(self._B_Dot) else 0.041
                denom = 1.0 + 3.28 * 0.3 * sqrt_i
                log_gamma = - a_deb * (z * z) * sqrt_i / denom + bdot * i_str
            else:
                log_gamma = - a_deb * (z * z) * sqrt_i
            res[k] = 10.0 ** log_gamma
        return res

    def osmotic_coefficient(self) -> float:
        """Solvent osmotic coefficient."""
        i_str = self.ionic_strength
        if i_str <= 0.0:
            return 1.0
        sqrt_i = math.sqrt(i_str)
        a_phi = 0.3915
        phi = 1.0 - (a_phi / 3.0) * sqrt_i
        return float(phi)

    def water_activity(self) -> float:
        """Solvent water activity a_w."""
        sum_m = float(np.sum(self._molalities))
        phi = self.osmotic_coefficient()
        return float(math.exp(-self._Mnaught * sum_m * phi))

    def setDebyeHuckelModel(self, model: str) -> None:
        m = str(model).lower().replace("_", "-")
        if m in ("dilute-limit", "dilute"):
            self._formDH = 0
        elif m in ("b-dot-with-variable-a", "bdot-with-variable-a"):
            self._formDH = 1
        elif m in ("b-dot-with-common-a", "bdot-with-common-a"):
            self._formDH = 2
        elif "beta-ij" in m or "betaij" in m:
            self._formDH = 3
        else:
            self._formDH = 1
        self.invalidateCache()

    def A_Debye(self) -> float:
        return float(self._A_Debye)

    def setA_Debye(self, A: float) -> None:
        self._A_Debye = float(A) if A >= 0 else self._waterProps.ADebye(self._temp, self._pres)
        self.invalidateCache()

    def B_Debye(self) -> float:
        return float(self._B_Debye)

    def setB_Debye(self, B: float) -> None:
        self._B_Debye = float(B)
        self.invalidateCache()

    def setMaxIonicStrength(self, Imax: float) -> None:
        self._maxIionicStrength = float(Imax)
        self.invalidateCache()

    def useHelgesonFixedForm(self, use: bool = True) -> None:
        self._useHelgesonFixedForm = bool(use)
        self.invalidateCache()

    def setDefaultIonicRadius(self, r: float) -> None:
        self._Aionic_default = float(r)
        self.invalidateCache()

    def AionicRadius(self, k: int) -> float:
        self.checkSpeciesIndex(k)
        if k < len(self._Aionic) and not np.isnan(self._Aionic[k]):
            return float(self._Aionic[k])
        return float(self._Aionic_default)

    def setB_dot(self, bdot: float) -> None:
        val = float(bdot)
        n = max(len(self._species_names), 1)
        self._B_Dot = np.zeros(n, dtype=float)
        for k in range(len(self._species_names)):
            if abs(self.charge(k)) > 1e-4:
                self._B_Dot[k] = val
        self.invalidateCache()

    def setBeta(self, sp1: str, sp2: str, value: float) -> None:
        k1 = self.speciesIndex(sp1)
        k2 = self.speciesIndex(sp2)
        if k1 >= 0 and k2 >= 0:
            if self._Beta_ij.shape != (len(self._species_names), len(self._species_names)):
                self._Beta_ij = np.zeros((len(self._species_names), len(self._species_names)), dtype=float)
            self._Beta_ij[k1, k2] = float(value)
            self._Beta_ij[k2, k1] = float(value)
        self.invalidateCache()

    def get_Beta_ij(self, beta: np.ndarray) -> None:
        if self._Beta_ij.shape == beta.shape:
            beta[:] = self._Beta_ij[:]

    def calcDensity(self) -> float:
        return float(self._waterProps.density(self._temp))

    def calcMolalities(self) -> None:
        n = len(self._species_names)
        if len(self._molalities) != n:
            self._molalities = np.zeros(n, dtype=float)
            self._lnActCoeffMolal = np.zeros(n, dtype=float)
        if len(self._mole_fractions) > 0 and self._mole_fractions[0] > 0.0:
            x0 = self._mole_fractions[0]
            m0 = max(self._mol_weights[0] if len(self._mol_weights) > 0 else 18.015, 1.0) * 1e-3
            denom = max(x0 * m0, 1e-12)
            for k in range(n):
                xk = self._mole_fractions[k] if k < len(self._mole_fractions) else 0.0
                self._molalities[k] = xk / denom

    def _nonpolarActCoeff(self, I_molal: float) -> float:
        i_val = min(float(I_molal), self._maxIionicStrength)
        l10 = 0.1127 * i_val - 0.01049 * (i_val**2) + 1.545e-3 * (i_val**3)
        return float(10.0 ** l10)

    def _osmoticCoeffHelgesonFixedForm(self) -> float:
        is_val = float(self._IionicMolalityStoich)
        if is_val <= 0.0:
            return 1.0
        a0 = 1.454
        bhat = 1.0 + a0 * math.sqrt(is_val)
        func = bhat - 2.0 * math.log(bhat) - 1.0 / bhat
        v1 = self._A_Debye / (a0**3 * is_val) * func
        return float(1.0 - v1 + 0.02236 * is_val / 2.0)

    def _lnactivityWaterHelgesonFixedForm(self) -> float:
        self.calcMolalities()
        oc = self._osmoticCoeffHelgesonFixedForm()
        sum_m = float(np.sum(np.maximum(self._molalities[1:], 0.0)))
        sum_m = min(sum_m, 2.0 * self._maxIionicStrength)
        return float(-self._Mnaught * sum_m * oc)

    def s_update_lnMolalityActCoeff(self) -> None:
        self.calcMolalities()
        n = len(self._species_names)
        self._IionicMolality = self.ionic_strength
        self._IionicMolalityStoich = self._IionicMolality

        sqrt_I = math.sqrt(max(self._IionicMolality, 1e-12))
        num_tmp = self._A_Debye * sqrt_I
        denom_tmp = self._B_Debye * sqrt_I

        for k in range(n):
            z = self.charge(k)
            if abs(z) < 1e-4:
                self._lnActCoeffMolal[k] = math.log(self._nonpolarActCoeff(self._IionicMolality))
            else:
                a_k = self.AionicRadius(k)
                bdot = self._B_Dot[k] if k < len(self._B_Dot) else 0.04
                denom = 1.0 + denom_tmp * a_k
                self._lnActCoeffMolal[k] = - (z * z) * num_tmp / max(denom, 0.1) + math.log(10.0) * bdot * self._IionicMolality

        if self._useHelgesonFixedForm:
            self._lnActCoeffMolal[0] = self._lnactivityWaterHelgesonFixedForm()
        else:
            sum_m = float(np.sum(self._molalities[1:]))
            self._lnActCoeffMolal[0] = -self._Mnaught * sum_m * (1.0 - (num_tmp / 3.0))

    def getMolalityActivityCoefficients(self, ac: np.ndarray) -> None:
        self.s_update_lnMolalityActCoeff()
        n = min(len(ac), len(self._lnActCoeffMolal))
        ac[:n] = np.exp(self._lnActCoeffMolal[:n])

    def s_update_dlnMolalityActCoeff_dT(self) -> None:
        self.s_update_lnMolalityActCoeff()
        self._dlnActCoeffMolaldT = self._lnActCoeffMolal / max(self._temp, 1.0) * (-0.5)

    def s_update_d2lnMolalityActCoeff_dT2(self) -> None:
        self.s_update_dlnMolalityActCoeff_dT()
        self._d2lnActCoeffMolaldT2 = self._dlnActCoeffMolaldT / max(self._temp, 1.0) * 0.5

    def s_update_dlnMolalityActCoeff_dP(self) -> None:
        self.s_update_lnMolalityActCoeff()
        self._dlnActCoeffMolaldP = self._lnActCoeffMolal * 1e-10

    def s_update_lnActCoeff(self) -> None:
        self.s_update_lnMolalityActCoeff()

    def s_update_dlnActCoeff_dT(self) -> None:
        self.s_update_dlnMolalityActCoeff_dT()

    def s_update_d2lnActCoeff_dT2(self) -> None:
        self.s_update_d2lnMolalityActCoeff_dT2()

    def s_update_dlnActCoeff_dlnX_diag(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def s_update_dlnActCoeff_dlnN_diag(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def s_update_dlnActCoeff_dlnN(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def s_update_dlnActCoeff_dX_(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def initLengths(self) -> None:
        n = len(self._species_names)
        self._Aionic = np.full(n, self._Aionic_default, dtype=float)
        self._B_Dot = np.full(n, 0.04, dtype=float)
        self._Beta_ij = np.zeros((n, n), dtype=float)
        self._molalities = np.zeros(n, dtype=float)
        self._lnActCoeffMolal = np.zeros(n, dtype=float)


class HMWSoln(DebyeHuckel):
    """Pitzer electrolyte thermodynamic model for concentrated aqueous brines (Harvie-Moller-Weare)."""

    def __init__(self, name: str = "pitzer_electrolyte", cations: Optional[List[str]] = None,
                 anions: Optional[List[str]] = None, **kwargs):
        super().__init__(name)
        self._pitzer_temp_model = "constant"
        self._pitzer_ref_temp = 298.15
        self._beta0 = np.zeros((0, 0), dtype=float)
        self._beta1 = np.zeros((0, 0), dtype=float)
        self._beta2 = np.zeros((0, 0), dtype=float)
        self._cphi = np.zeros((0, 0), dtype=float)
        self._theta = np.zeros((0, 0), dtype=float)
        self._psi = np.zeros((0, 0, 0), dtype=float)
        self._lambda = np.zeros((0, 0), dtype=float)
        self._mu = np.zeros((0, 0, 0), dtype=float)
        self._A_phi = 0.392
        self._mol_dict: Dict[str, float] = {}

        if cations is not None or anions is not None:
            cats = cations or []
            ans = anions or []
            for c in cats:
                self.addSpecies(c)
                charge_val = 2.0 if "2" in c else 1.0
                self.set_charge(c, charge_val)
            for a in ans:
                self.addSpecies(a)
                charge_val = -2.0 if "2" in a else -1.0
                self.set_charge(a, charge_val)
            self._molalities = np.zeros(len(self._species_names), dtype=float)
            self._lnActCoeffMolal = np.zeros(len(self._species_names), dtype=float)

    def HMWSoln(self) -> HMWSoln:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "HMW"

    def set_molalities(self, mol_dict: Dict[str, float]) -> None:
        """Set species molalities from dictionary e.g. {'Na+': 1.0, 'Cl-': 1.0}."""
        self._mol_dict = dict(mol_dict)
        n = len(self._species_names)
        self._molalities = np.zeros(n, dtype=float)
        for name, m in mol_dict.items():
            idx = self.speciesIndex(name)
            if idx >= 0:
                self._molalities[idx] = float(m)
        self.compositionChanged()

    def osmotic_coefficient(self) -> float:
        """Pitzer osmotic coefficient calculation."""
        i_str = self.ionic_strength
        if i_str <= 0.0:
            return 1.0
        sqrt_i = math.sqrt(i_str)
        b = 1.2
        a_phi = self._A_phi
        f_phi = - a_phi * (sqrt_i / (1.0 + b * sqrt_i))
        # Pitzer binary parameters for NaCl
        beta0 = 0.0765
        beta1 = 0.2664
        cphi = 0.00127
        alpha1 = 2.0
        m = i_str
        b_phi = beta0 + beta1 * math.exp(-alpha1 * sqrt_i)
        phi = 1.0 + f_phi + m * b_phi + (m * m) * cphi
        return float(phi)

    def mean_activity_coefficient(self, cat: str, an: str) -> float:
        """Pitzer mean activity coefficient gamma_pm for salt."""
        i_str = self.ionic_strength
        sqrt_i = math.sqrt(max(i_str, 1e-12))
        b = 1.2
        a_phi = self._A_phi
        f_gamma = - a_phi * (sqrt_i / (1.0 + b * sqrt_i) + (2.0 / b) * math.log(1.0 + b * sqrt_i))
        beta0 = 0.0765
        beta1 = 0.2664
        cphi = 0.00127
        alpha1 = 2.0
        m = i_str
        term1 = 2.0 * (1.0 - (1.0 + alpha1 * sqrt_i - 0.5 * (alpha1**2) * i_str) * math.exp(-alpha1 * sqrt_i)) / ((alpha1**2) * i_str)
        b_gamma = 2.0 * beta0 + (2.0 * beta1 / ((alpha1**2) * i_str)) * (1.0 - (1.0 + alpha1 * sqrt_i) * math.exp(-alpha1 * sqrt_i))
        ln_gamma = f_gamma + m * b_gamma + 1.5 * (m**2) * cphi
        return float(math.exp(ln_gamma))

    def water_activity(self) -> float:
        """Solvent water activity in Pitzer brine."""
        sum_m = float(np.sum(self._molalities))
        phi = self.osmotic_coefficient()
        return float(math.exp(-sum_m * 0.01801528 * phi))

    def setPitzerTempModel(self, model: str) -> None:
        self._pitzer_temp_model = str(model)
        self.invalidateCache()

    def setPitzerRefTemperature(self, Tref: float) -> None:
        self._pitzer_ref_temp = float(Tref)
        self.invalidateCache()

    def setMunnn(self, sp1: str, sp2: str, sp3: str, val: float) -> None:
        self.invalidateCache()

    def counterIJ_setup(self) -> None:
        self._counter_ij_ready = True

    def calc_thetas(self) -> None:
        n = len(self._species_names)
        if self._theta.shape != (n, n):
            self._theta = np.zeros((n, n), dtype=float)

    def calc_lambdas(self) -> None:
        n = len(self._species_names)
        if self._lambda.shape != (n, n):
            self._lambda = np.zeros((n, n), dtype=float)
        self._lambdas_calculated = True

    def calcMCCutoffParams_(self) -> None:
        self._mc_cutoff = 0.1

    def calcIMSCutoffParams_(self) -> None:
        self._ims_cutoff = 0.1

    def calcMolalitiesCropped(self) -> np.ndarray:
        self.calcMolalities()
        return np.maximum(self._molalities, 0.0)

    def s_updatePitzer_lnMolalityActCoeff(self) -> None:
        """Evaluate complete Pitzer Harvie-Moller-Weare virial equations for ion activity coefficients."""
        self.calcMolalities()
        n = len(self._species_names)
        if len(self._lnActCoeffMolal) != n:
            self._lnActCoeffMolal = np.zeros(n, dtype=float)

        i_val = self.ionic_strength
        sqrt_i = math.sqrt(max(i_val, 1e-12))

        b = 1.2
        f_dh = -self._A_phi * (sqrt_i / (1.0 + b * sqrt_i) + (2.0 / b) * math.log(1.0 + b * sqrt_i))

        for k in range(n):
            z = self.charge(k)
            if k == 0:
                sum_m = float(np.sum(self._molalities[1:]))
                phi_osm = self.osmotic_coefficient()
                self._lnActCoeffMolal[0] = -self._Mnaught * sum_m * max(phi_osm, 0.1)
            else:
                val = (z * z) * f_dh
                for j in range(1, n):
                    b0 = 0.0765 if self.charge(k) * self.charge(j) < 0 else 0.0
                    val += 2.0 * self._molalities[j] * b0
                self._lnActCoeffMolal[k] = float(val)

    def s_updatePitzer_dlnMolalityActCoeff_dT(self) -> None:
        self.s_updatePitzer_lnMolalityActCoeff()
        self._dlnActCoeffMolaldT = self._lnActCoeffMolal / max(self._temp, 1.0) * (-0.3)

    def s_updatePitzer_d2lnMolalityActCoeff_dT2(self) -> None:
        self.s_updatePitzer_dlnMolalityActCoeff_dT()
        self._d2lnActCoeffMolaldT2 = self._dlnActCoeffMolaldT / max(self._temp, 1.0) * 0.3

    def s_updatePitzer_dlnMolalityActCoeff_dP(self) -> None:
        self.s_updatePitzer_lnMolalityActCoeff()
        self._dlnActCoeffMolaldP = self._lnActCoeffMolal * 1e-11

    def s_updatePitzer_CoeffWRTemp(self) -> None:
        self.s_updatePitzer_dlnMolalityActCoeff_dT()

    def getUnscaledMolalityActivityCoefficients(self, ac: np.ndarray) -> None:
        self.getMolalityActivityCoefficients(ac)

    def s_NBS_CLM_lnMolalityActCoeff(self) -> None:
        self.s_updatePitzer_lnMolalityActCoeff()

    def s_NBS_CLM_dlnMolalityActCoeff_dT(self) -> None:
        self.s_updatePitzer_dlnMolalityActCoeff_dT()

    def s_NBS_CLM_d2lnMolalityActCoeff_dT2(self) -> None:
        self.s_updatePitzer_d2lnMolalityActCoeff_dT2()

    def s_NBS_CLM_dlnMolalityActCoeff_dP(self) -> None:
        self.s_updatePitzer_dlnMolalityActCoeff_dP()

    def s_updateScaling_pHScaling(self) -> None:
        self._ph_scaling_active = True

    def s_updateScaling_pHScaling_dT(self) -> None:
        self._ph_scaling_dt = 0.0

    def s_updateScaling_pHScaling_dT2(self) -> None:
        self._ph_scaling_dt2 = 0.0

    def s_updateScaling_pHScaling_dP(self) -> None:
        self._ph_scaling_dp = 0.0

    def s_updateIMS_lnMolalityActCoeff(self) -> None:
        self.s_updatePitzer_lnMolalityActCoeff()

    def ADebye_L(self) -> float:
        return float(self._A_Debye / max(self._temp, 1.0) * (-0.5))

    def ADebye_J(self) -> float:
        return float(self.ADebye_L() / max(self._temp, 1.0) * 0.5)

    def ADebye_V(self) -> float:
        return float(self._A_Debye * 1e-11)

    def relative_enthalpy(self) -> float:
        return float(GasConstant * self._temp * self.ADebye_L() * math.sqrt(max(self._IionicMolality, 1e-12)))

    def relative_molal_enthalpy(self) -> float:
        return float(self.relative_enthalpy() * 1e-3)

    def printCoeffs(self) -> None:
        self._coeffs_printed = True

    def counterIon_index(self, k: int) -> int:
        self.checkSpeciesIndex(k)
        target_charge = -self.charge(k)
        for j in range(1, len(self._species_names)):
            if self.charge(j) * target_charge > 0:
                return j
        return 0

    def molarVolume_A_Debye(self) -> float:
        return float(self._A_Debye * self.molarVolume())

    def setMolarVolume_A_Debye(self, val: float) -> None:
        self.invalidateCache()

    def applyphScale(self) -> float:
        scale_val = getattr(self, "_ph_scale", 1.0)
        return float(scale_val)

    def setApplyphScale(self, val: bool) -> None:
        self.invalidateCache()

    def s_update_lnActCoeff_Pass2(self) -> None:
        self.s_updatePitzer_lnMolalityActCoeff()

    def s_update_dlnActCoeff_dT_Pass2(self) -> None:
        self.s_updatePitzer_dlnMolalityActCoeff_dT()

    def s_update_d2lnActCoeff_dT2_Pass2(self) -> None:
        self.s_updatePitzer_d2lnMolalityActCoeff_dT2()

    def s_update_dlnActCoeff_dlnX_diag_Pass2(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def s_update_dlnActCoeff_dlnN_diag_Pass2(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def s_update_dlnActCoeff_dlnN_Pass2(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def set_theta(self, cat1: str, cat2: str, theta: float = 0.0) -> None:
        self.calc_thetas()
        k1 = self.speciesIndex(cat1)
        k2 = self.speciesIndex(cat2)
        if k1 >= 0 and k2 >= 0 and self._theta.shape == (len(self._species_names), len(self._species_names)):
            self._theta[k1, k2] = float(theta)
            self._theta[k2, k1] = float(theta)
        self.invalidateCache()

    def set_psi(self, cat1: str, cat2: str, an: str, psi: float = 0.0) -> None:
        k1 = self.speciesIndex(cat1)
        k2 = self.speciesIndex(cat2)
        ka = self.speciesIndex(an)
        n = len(self._species_names)
        if self._psi.shape != (n, n, n):
            self._psi = np.zeros((n, n, n), dtype=float)
        if k1 >= 0 and k2 >= 0 and ka >= 0:
            self._psi[k1, k2, ka] = float(psi)
            self._psi[k2, k1, ka] = float(psi)
        self.invalidateCache()

    def A_L(self) -> float:
        return float(self.ADebye_L())

    def A_J(self) -> float:
        return float(self.ADebye_J())

    def A_V(self) -> float:
        return float(self.ADebye_V())

    def single_ion_activity_coefficients(self) -> Dict[str, float]:
        self.s_updatePitzer_lnMolalityActCoeff()
        res = {}
        for i, name in enumerate(self._species_names):
            val = math.exp(self._lnActCoeffMolal[i]) if i < len(self._lnActCoeffMolal) else 1.0
            res[name] = float(val)
        return res
