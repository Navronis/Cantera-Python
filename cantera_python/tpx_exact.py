"""Source-faithful pure Python implementations of the pinned TPX fluids.

The equations and coefficient arrays in this module are direct translations of
``outputs/cantera-original/src/tpx``.  It intentionally has no cubic-EOS or
external-fluid fallback.
"""
from __future__ import annotations

from math import exp, log, sqrt

from .tpx import Substance, TPXError


class _Residual14(Substance):
    """Reynolds residual formulation shared by H2, CH4, N2 and O2."""
    A = ()
    G = ()
    D = ()
    F = ()
    R = 1.0
    Gamma = 1.0
    u0 = 0.0
    s0 = 0.0
    alpha = 0.0
    beta = 0.0
    Tt = 0.0
    Pt = 0.0
    alpha1 = 0.0
    Rot = 0.0
    _psat_kind = "power"

    def _C(self, i, rt, rt2):
        a = self.A; T = self.T
        if i == 0: return a[0]*T + a[1]*sqrt(T) + a[2] + (a[3] + a[4]*rt)*rt
        if i == 1: return a[5]*T + a[6] + rt*(a[7] + a[8]*rt)
        if i == 2: return a[9]*T + a[10] + a[11]*rt
        if i == 3: return a[12]
        if i == 4: return rt*(a[13] + a[14]*rt)
        if i == 5: return a[15]*rt
        if i == 6: return rt*(a[16] + a[17]*rt)
        if i == 7: return a[18]*rt2
        if i == 8: return rt2*(a[19] + a[20]*rt)
        if i == 9: return rt2*(a[21] + a[22]*rt2)
        if i == 10: return rt2*(a[23] + a[24]*rt)
        if i == 11: return rt2*(a[25] + a[26]*rt2)
        if i == 12: return rt2*(a[27] + a[28]*rt)
        if i == 13: return rt2*(a[29] + a[30]*rt + a[31]*rt2)
        return 0.0

    def _Cprime(self, i, rt, rt2, rt3):
        a = self.A
        if i == 0: return a[0] + .5*a[1]/sqrt(self.T) - (a[3] + 2*a[4]*rt)*rt2
        if i == 1: return a[5] - rt2*(a[7] + 2*a[8]*rt)
        if i == 2: return a[9] - a[11]*rt2
        if i == 3: return 0.0
        if i == 4: return -rt2*(a[13] + 2*a[14]*rt)
        if i == 5: return -a[15]*rt2
        if i == 6: return -rt2*(a[16] + 2*a[17]*rt)
        if i == 7: return -2*a[18]*rt3
        if i == 8: return -rt3*(2*a[19] + 3*a[20]*rt)
        if i == 9: return -rt3*(2*a[21] + 4*a[22]*rt2)
        if i == 10: return -rt3*(2*a[23] + 3*a[24]*rt)
        if i == 11: return -rt3*(2*a[25] + 4*a[26]*rt2)
        if i == 12: return -rt3*(2*a[27] + 3*a[28]*rt)
        if i == 13: return -rt3*(2*a[29] + 3*a[30]*rt + 4*a[31]*rt2)
        return 0.0

    def _W(self, n, er):
        if n == 0: return (1.0-er)/(2*self.Gamma)
        return (n*self._W(n-1, er) - .5*self.Rho**(2*n)*er)/self.Gamma

    def _I(self, i, er):
        return self.Rho**(i+1)/(i+1) if i < 8 else self._W(i-8, er)

    def _H(self, i, er):
        return self.Rho**(i+2) if i < 8 else self.Rho**(2*i-13)*er

    def Pp(self):
        rt=1/self.T; rt2=rt*rt; er=exp(-self.Gamma*self.Rho*self.Rho)
        return self.Rho*self.R*self.T + sum(self._C(i,rt,rt2)*self._H(i,er) for i in range(14))

    def MolWt(self): return self.M

    def _residual_us(self):
        rt=1/self.T; rt2=rt*rt; rt3=rt*rt2; er=exp(-self.Gamma*self.Rho*self.Rho)
        resu=sum((self._C(i,rt,rt2)-self.T*self._Cprime(i,rt,rt2,rt3))*self._I(i,er) for i in range(14))
        ress=sum(-self._Cprime(i,rt,rt2,rt3)*self._I(i,er) for i in range(14))
        return resu, ress

    def Tcrit(self): return self.Tc
    def Pcrit(self): return self.Pc
    def Vcrit(self): return 1.0/self.Roc
    def Tmin(self): return self.Tmn
    def Tmax(self): return self.Tmx


class _NitrogenOxygen(_Residual14):
    def up(self):
        ru, _ = self._residual_us(); T=self.T; rt=1/T; g=self.G
        val=ru + (((.25*g[6]*T+g[5]/3)*T+.5*g[4])*T+g[3])*T + g[2]*log(T) - (g[1]+.5*g[0]*rt)*rt + g[7]*self.beta/(exp(self.beta*rt)-1)+self.u0
        return val+self.m_energy_offset
    def sp(self):
        _,rs=self._residual_us(); T=self.T; rt=1/T; g=self.G
        z=self.beta*rt
        val=self.s0-self.R*log(self.Rho)+rs + (((g[6]/3)*T+.5*g[5])*T+g[4])*T+g[3]*log(T)-((g[0]*rt/3+.5*g[1])*rt+g[2])*rt+g[7]*(z+z/(exp(z)-1)-log(exp(z)-1))
        return val+self.m_entropy_offset
    def Psat(self):
        if not (self.Tmn <= self.T <= self.Tc): raise TPXError(f"{self.name()}: temperature out of range")
        z=0.0
        for i,c in enumerate(self.F[:8]): z += c*((self.Tc-self.T)**self.alpha if i==3 else self.T**(i-1))
        return exp(z + self.F[8]*log(self.T))
    def ldens(self):
        if not (self.Tmn <= self.T <= self.Tc): raise TPXError(f"{self.name()}: temperature out of range")
        return sum(c*(1-self.T/self.Tc)**(i/3) for i,c in enumerate(self.D))


class Nitrogen(_NitrogenOxygen):
    M=28.01348; Tmn=63.15; Tmx=2000.; Tc=126.200; Pc=3.4e6; Roc=314.03; R=2.96790515164171e2; Gamma=7.13602531283233e-6; alpha=1.95; beta=3353.40610; u0=150877.551; s0=214.9352518
    A=(1.75889959256970e-1,1.38197604384933e1,-3.14918412133921e2,4.40300150239380e3,-5.45358971644916e5,4.84413320182919e-4,-5.18964416491365e-2,6.57265859197103e-4,8.51299771713314e4,1.33459405162578e-8,3.83381319826746e-4,-8.35421151028455e-2,2.84874912286101e-7,-2.38296116270360e-7,-1.48321912935764e-4,5.62605853190540e-10,-2.98201050924595e-13,9.85319087685241e-11,-1.92002176056468e-14,-7.82250103373122e4,-5.51801778744598e5,-5.72781957607352e-1,3.25760529488327e2,-1.34659309828737e-6,-1.92036423064911e-5,-3.94564337674524e-12,-2.44388245328965e-9,-1.50970602460077e-18,1.25854885346038e-16,-8.34271144923969e-24,-1.17299202018417e-22,9.06544823455730e-22)
    F=(8.3944094440e3,-1.8785191705e3,-7.2822291650,1.0228509660e-2,5.5560638250e-4,-5.9445446620e-6,2.7154339320e-8,-4.8795359040e-11,5.0953608240e2)
    D=(3.1402991e2,4.4111015e2,9.4622994e2,-2.9067111e3,4.4785979e3,-2.2746914e3)
    G=(-2.18203473713518e5,1.01573580096247e4,-1.65504721657240e2,7.43175999190430e2,-5.14605623546025e-3,5.18347156760489e-6,-1.05922170493616e-9,2.98389393363817e2)
    def __init__(self): super().__init__(); self.m_name='nitrogen'; self.m_formula='N2'


class Oxygen(_NitrogenOxygen):
    M=31.9994; Tmn=54.34; Tmx=2000.; Tc=154.581; Pc=5.0429e6; Roc=436.15; R=2.59820853437877e2; Gamma=5.46895508389297e-6; alpha=1.91576; beta=2239.18105; u0=198884.2435; s0=668.542976
    A=(-4.26396872798684e-1,3.48334938784107e1,-5.77516910418738e2,2.40961751553325e4,-1.23332307855543e6,3.73585286319658e-4,-1.70178244046465e-1,-3.33226903068473e-4,8.61334799901291e3,-6.80394661057309e-7,7.09583347162704e-4,-5.73905688255053e-2,-1.92123080811409e-7,3.11764722329504e-8,-8.09463854745591e-6,-2.22562296356501e-11,9.18401045361994e-15,5.75758417511114e-12,-2.10752269644774e-15,3.62884761272184e3,-1.23317754317110e6,-5.03800414800672e-2,3.30686173177055e2,-5.26259633964252e-8,5.53075442383100e-6,-2.71042853363688e-13,-1.65732450675251e-9,-5.82711196409204e-20,4.42953322148281e-17,-2.95529679136244e-25,-1.92361786708846e-23,9.43758410350413e-23)
    F=(-5.581932039e2,-1.0966262185e2,-8.3456211630e-2,2.6603644330e-3,1.6875023830e-5,-2.1262477120e-7,9.5741096780e-10,-1.6617640450e-12,2.7545605710e1)
    D=(4.3615175e2,7.5897189e2,-4.2576866e2,2.3487106e3,-3.0474660e3,1.4850169e3)
    G=(-1.29442711174062e6,5.98231747005341e4,-8.97850772730944e2,6.55236176900400e2,-1.13131252131570e-2,3.4981070244228e-6,4.21065222886885e-9,2.67997030050139e2)
    def __init__(self): super().__init__(); self.m_name='oxygen'; self.m_formula='O2'


class Hydrogen(_Residual14):
    M=2.0159; Tmn=13.8; Tmx=5000.; Tc=32.938; Pc=1.2838e6; Roc=31.36; To=13.8; R=4124.299539; Gamma=1.008854772e-3; u0=3.9275114e5; s0=2.3900333e4; Tt=13.8; Pt=7042.09; alpha=1.5814454428; alpha1=.3479; T1=35.; T2=400.
    A=(11.504705193529,1055.427998826072,-12706.85949968568,72878.44527295619,-744878.0703363973,.2328994151810363,-16.35308393739296,3730.678064960389,629966.7723184814,.001210920358305697,1.753651095884817,-136.7022988058101,-.006869936641299885,.03644494201750974,-2.559784772600182,-.0004038855202905836,1.485396303520942e-6,.0004243613981060742,-2.307910113586888e-6,-608219.2173879582,-1961080.967486886,-578.6932854076408,27991.29504191752,-.2381566558300913,.8918796032452871,-6.985739539036644e-5,-.007339554179182899,-5.59703344028998e-9,8.842130160884514e-8,-2.655507264539047e-12,-4.544474518140164e-12,9.818775257001921e-11)
    D=(48.645813003,-34.779278180,407.76538192,-1171.9787304,1621.3924400,-1153.1096683,338.25492039)
    F=(3.05300134164,2.80810925813,-.655461216567,1.59514439374)
    G=(6193.4792,294.90437,-1540.1979,-4917.6101,68957.165,-222821.85,379900.59,-370942.16,213267.92,-71519.411,12971.743,-985.33014,10434.776,-391.44179,582.77696,654.09163,-187.28847)
    def __init__(self): super().__init__(); self.m_name='hydrogen'; self.m_formula='H2'
    def _icv(self,i,x,xlg): return x-1 if i==0 else x*xlg**i-i*self._icv(i-1,x,xlg)
    def up(self):
        ru,_=self._residual_us(); T=self.T; val=self.u0+ru+self.G[0]*(min(T,self.T1)-self.Tmn)
        if T>self.T1:
            x=min(T,self.T2)/self.T1
            val += sum(self.G[i]*self.T1*self._icv(i,x,log(x)) for i in range(12))
        if T>self.T2:
            x=T/self.T2; val += sum(self.G[i+12]*self.T2*self._icv(i,x,log(x)) for i in range(5))
        return val+self.m_energy_offset
    def sp(self):
        _,rs=self._residual_us(); T=self.T; val=self.s0-self.R*log(self.Rho)+rs+self.G[0]*log(min(T,self.T1)/self.Tmn)
        if T>self.T1:
            q=log(min(T,self.T2)/self.T1); val += sum(self.G[i]/(i+1)*q**(i+1) for i in range(12))
        if T>self.T2:
            q=log(T/self.T2); val += sum(self.G[i+12]/(i+1)*q**(i+1) for i in range(5))
        return val+self.m_entropy_offset
    def Psat(self):
        if not self.Tmn<=self.T<=self.Tc: raise TPXError('hydrogen: temperature out of range')
        x=(1-self.Tt/self.T)/(1-self.Tt/self.Tc); f=self.F[0]*x+self.F[1]*x*x+self.F[2]*x**3+self.F[3]*x*(1-x)**self.alpha
        return exp(f)*self.Pt
    def ldens(self):
        if not self.Tmn<=self.T<=self.Tc: raise TPXError('hydrogen: temperature out of range')
        x=1-self.T/self.Tc; return self.Roc+self.D[0]*x**self.alpha1+sum(self.D[i]*x**(1+(i-1)/3) for i in range(1,7))


class Methane(_Residual14):
    M=16.04996; Tmn=90.68; Tmx=1700.; Tc=190.555; Pc=4.5988e6; Roc=160.43; Tt=90.68; Pt=11743.5675; R=518.253475866; Gamma=3.72992471469e-5; alpha=1.5; alpha1=.36; Rot=451.562; beta=2009.152; u0=357696.0858; s0=-1918.035071
    A=(-7.25929210183,413.766054566,-6321.67316855,334015.577724,-16825337.9982,.0187884851902,-11.8673201223,2090.62618015,-407532.656958,-5.73917603241e-5,.0437711441593,-4.38766500673,1.13524630779e-5,-5.07028240949e-5,.0228002199522,9.25611329590e-9,1.33865662546e-10,-1.65439044196e-7,1.81030980110e-10,545753.645958,-36319228.1933,4.81463473761,156633.02262,7.89977010972e-5,.013999388121,-1.70656092212e-11,-4.55256623445e-5,-2.29314170748e-14,8.31548197665e-12,6.84673626259e-20,-4.70845544152e-17,5.21465091383e-16)
    D=(-.178860165,.04838475,-.01848987); F=(4.77748580,1.76065363,-.567888940,1.32786231); G=(1347.40610,135.512060,-29.3910458,2.12774600,2446.56600); Rot=451.562; beta=2009.152
    def __init__(self): super().__init__(); self.m_name='methane'; self.m_formula='CH4'
    def up(self):
        ru,_=self._residual_us(); t3=self.T**(1/3); rt=1/self.T
        return ru+self.T*(self.G[0]+.75*self.G[1]*t3+.6*self.G[2]*t3*t3+.5*self.G[3]*self.T)+self.G[4]*self.beta/(exp(self.beta*rt)-1)+self.u0+self.m_energy_offset
    def sp(self):
        _,rs=self._residual_us(); t3=self.T**(1/3); rt=1/self.T; z=self.beta*rt
        return self.s0-self.R*log(self.Rho)+rs+self.G[0]*log(self.T)+3*self.G[1]*t3+1.5*self.G[2]*t3*t3+self.G[3]*self.T+self.G[4]*(z+z/(exp(z)-1)-log(exp(z)-1))+self.m_entropy_offset
    def Psat(self):
        if not self.Tmn<=self.T<=self.Tc: raise TPXError('methane: temperature out of range')
        x=(1-self.Tt/self.T)/(1-self.Tt/self.Tc); return exp(self.F[0]*x+self.F[1]*x*x+self.F[2]*x**3+self.F[3]*x*(1-x)**self.alpha)*self.Pt
    def ldens(self):
        if not self.Tmn<=self.T<=self.Tc: raise TPXError('methane: temperature out of range')
        w=(self.Tc-self.T)/(self.Tc-self.Tt); s=self.D[0]*(1-w**(2/3))+self.D[1]*(1-w**(4/3))+self.D[2]*(1-w**2)
        return self.Roc+(self.Rot-self.Roc)*w**self.alpha1*exp(s)


class CarbonDioxide(Substance):
    M=44.01; Tmn=216.54; Tmx=1500.; Tc=304.21; Pc=7.38350e6; Roc=464.; To=216.54; R=188.918; Gamma=5e-6; u0=3.2174105e5; s0=2.1396056e3; Tp=250.; _A=(.22488558,-137.17965,-14430.214,-2963049.1,-206060390.,4.5554393e-5,.077042840,40.602371,4.0029509e-7,-3.9436077e-4,1.2115286e-10,1.0783386e-7,4.3962336e-11,-36505.545,1.9490511e7,-2.9186718e9,.024358627,-37.546530,11898.141); _F=(-6.541261,-.27914636,-3.4716202,-3.4989637,-19.770948,139.22839,-276.70389,-7051.0251); _D=(464.00009,679.38129,1477.6836,-3126.7676,3639.7656,-1343.7098); _G=(8726.361,184.0040,1.914025,-.001667825,7.305950e-7,-1.255290e-10)
    def __init__(self): super().__init__(); self.m_name='carbon-dioxide'; self.m_formula='CO2'
    def MolWt(self): return self.M
    def Tcrit(self): return self.Tc
    def Pcrit(self): return self.Pc
    def Vcrit(self): return 1/self.Roc
    def Tmin(self): return self.Tmn
    def Tmax(self): return self.Tmx
    def _C(self,j,rt,rt2,rt3,rt4):
        a=self._A; T=self.T
        return ([a[0]*T+a[1]+a[2]*rt+a[3]*rt2+a[4]*rt3,a[5]*T+a[6]+a[7]*rt,a[8]*T+a[9],a[10]*T+a[11],a[12],a[13]*rt2+a[14]*rt3+a[15]*rt4,a[16]*rt2+a[17]*rt3+a[18]*rt4][j]) if 0<=j<=6 else 0.
    def _Cp(self,j,rt2,rt3,rt4):
        a=self._A; T=self.T
        return ([a[0]-a[2]*rt2-2*a[3]*rt3-3*a[4]*rt4,a[5]-a[7]*rt2,a[8],a[10],0,-2*a[13]*rt3-3*a[14]*rt4-4*a[15]*T**-5,-2*a[16]*rt3-3*a[17]*rt4-4*a[18]*T**-5][j])
    def _I(self,j,e): return (self.Rho,self.Rho**2/2,self.Rho**3/3,self.Rho**4/4,self.Rho**5/5,(1-e)/(2*self.Gamma),(1-e*(self.Gamma*self.Rho**2+1))/(2*self.Gamma**2))[j]
    def _H(self,j,e): return self.Rho**(j+2) if j<5 else (self.Rho**3*e if j==5 else self.Rho**5*e)
    def Pp(self):
        rt=1/self.T; r2=rt*rt; r3=r2*rt; r4=r3*rt; e=exp(-self.Gamma*self.Rho**2); return self.Rho*self.R*self.T+sum(self._C(i,rt,r2,r3,r4)*self._H(i,e) for i in range(7))
    def up(self):
        rt=1/self.T; r2=rt*rt; r3=r2*rt; r4=r3*rt; e=exp(-self.Gamma*self.Rho**2); q=sum(self._G[i]*(self.T**i-self.To**i)/i for i in range(1,6))+self._G[0]*log(self.T/self.To)
        return self.u0+q+sum(self._I(i,e)*(self._C(i,rt,r2,r3,r4)-self.T*self._Cp(i,r2,r3,r4)) for i in range(7))+self.m_energy_offset
    def sp(self):
        rt=1/self.T; r2=rt*rt; r3=r2*rt; r4=r3*rt; e=exp(-self.Gamma*self.Rho**2); q=sum(self._G[i]*(self.T**(i-1)-self.To**(i-1))/(i-1) for i in range(2,6))+self._G[1]*log(self.T/self.To)-self._G[0]*(rt-1/self.To)
        return self.s0-self.R*log(self.Rho)+q-sum(self._Cp(i,r2,r3,r4)*self._I(i,e) for i in range(7))+self.m_entropy_offset
    def Psat(self):
        if not self.Tmn<=self.T<=self.Tc: raise TPXError('carbon-dioxide: temperature out of range')
        q=sum(c*(self.T/self.Tp-1)**i for i,c in enumerate(self._F)); return exp((self.Tc/self.T-1)*q)*self.Pc
    def ldens(self):
        if not self.Tmn<=self.T<=self.Tc: raise TPXError('carbon-dioxide: temperature out of range')
        return sum(c*(1-self.T/self.Tc)**(i/3) for i,c in enumerate(self._D))


class Heptane(Substance):
    M=100.20; Tmn=182.56; Tmx=1000.; Tc=537.68; Pc=2.6199e6; Roc=197.60; To=300.; R=82.99504; Gamma=9.611604e-6; u0=3.4058439e5; s0=1.1080254e3; Tp=400.; _A=(2.246032e-3,208.2990,5.085746e7,3.566396e9,1.622168e9,1.065237e-5,.5987922,7.736602,1.929386e5,5.291379e-9); _F=(-7.2298764,.38607475,-3.4216472,.46274432,-9.7926124,-42.058094,75.468678,317.58992); _D=(197.60405,894.51237,-1146.2908,1799.6947,-1725.0843,970.88329); _G=(119252.13,-772.31363,7.4463527,-.0030888167,0.,0.)
    def __init__(self): super().__init__(); self.m_name='heptane'; self.m_formula='C7H16'
    def MolWt(self): return self.M
    def Tcrit(self): return self.Tc
    def Pcrit(self): return self.Pc
    def Vcrit(self): return 1/self.Roc
    def Tmin(self): return self.Tmn
    def Tmax(self): return self.Tmx
    def _C(self,j,rt,r2,r3,r4):
        a=self._A; return [a[0]*self.R*self.T-a[1]-a[2]*r2+a[3]*r3-a[4]*r4,a[5]*self.R*self.T-a[6]-a[7]*rt,a[9]*(a[6]+a[7]*rt),a[8]*r2][j] if j<4 else 0.
    def _Cp(self,j,r2,r3,r4):
        a=self._A; return [a[0]*self.R+2*a[2]*r3-3*a[3]*r4+4*a[4]*self.T**-5,a[5]*self.R+a[7]*r2,a[9]*(-a[7]*r2),-2*a[8]*r3][j] if j<4 else 0.
    def _I(self,j,e): return (self.Rho,self.Rho**2/2,self.Rho**5/5,1/self.Gamma-(self.Gamma*self.Rho**2+2)*e/(2*self.Gamma))[j]
    def _H(self,j,e): return (self.Rho**(j+2) if j<2 else self.Rho**6 if j==2 else self.Rho**3*(1+self.Gamma*self.Rho**2)*e)
    def Pp(self):
        rt=1/self.T; r2=rt*rt; r3=r2*rt; r4=r3*rt; e=exp(-self.Gamma*self.Rho**2); return self.Rho*self.R*self.T+sum(self._C(i,rt,r2,r3,r4)*self._H(i,e) for i in range(4))
    def up(self):
        rt=1/self.T; r2=rt*rt; r3=r2*rt; r4=r3*rt; e=exp(-self.Gamma*self.Rho**2); return self.u0+sum(self._G[i]*(self.T**i-self.To**i)/i for i in range(1,6))+self._G[0]*log(self.T/self.To)+sum((self._C(i,rt,r2,r3,r4)-self.T*self._Cp(i,r2,r3,r4))*self._I(i,e) for i in range(4))+self.m_energy_offset
    def sp(self):
        rt=1/self.T; r2=rt*rt; r3=r2*rt; r4=r3*rt; e=exp(-self.Gamma*self.Rho**2); return self.s0-self.R*log(self.Rho)+sum(self._G[i]*(self.T**(i-1)-self.To**(i-1))/(i-1) for i in range(2,6))+self._G[1]*log(self.T/self.To)-self._G[0]*(rt-1/self.To)-sum(self._Cp(i,r2,r3,r4)*self._I(i,e) for i in range(4))+self.m_entropy_offset
    def Psat(self):
        if not self.Tmn<=self.T<=self.Tc: raise TPXError('heptane: temperature out of range')
        q=sum(c*(self.T/self.Tp-1)**i for i,c in enumerate(self._F)); return exp((self.Tc/self.T-1)*q)*self.Pc
    def ldens(self):
        if not self.Tmn<=self.T<=self.Tc: raise TPXError('heptane: temperature out of range')
        return sum(c*(1-self.T/self.Tc)**(i/3) for i,c in enumerate(self._D))


class HFC134a(Substance):
    M=102.032; Tmn=170.; Tmx=455.; Tc=374.18; Pc=4056290.; Roc=508.; R=81.48885644
    a=(.05586817,.498223,.02458698,.0008570145,.0004788584,-1.800808,.2671641,-.04781652,.01423987,.3324062,-.007485907,.0001017263,-.5184567,-.08692288,.2057144,-.005000457,.0004603262,-.003497836,.006995038,-.01452184,-.0001285458)
    t=(-.5,0,0,0,1.5,1.5,2,2,1,3,5,1,5,5,6,10,10,10,18,22,50); d=(2,1,3,6,6,1,1,2,5,2,2,4,1,4,1,2,4,1,5,3,10); b=(-1.019535,9.047135,-1.629789,-9.723916,-3.927170)
    def __init__(self): super().__init__(); self.m_name='HFC-134a'; self.m_formula='C2H2F4'
    def MolWt(self): return self.M
    def Tcrit(self): return 374.21
    def Pcrit(self): return 4059280.
    def Vcrit(self): return 1/511.95
    def Tmin(self): return self.Tmn
    def Tmax(self): return self.Tmx
    def _terms(self):
        tau=self.Tc/self.T; delta=self.Rho/self.Roc; phi0=self.b[0]+self.b[1]*tau+self.b[2]*log(tau)+log(delta)+self.b[3]/sqrt(tau)+self.b[4]*tau**-.75
        phir=0.0
        for lo,hi,k in ((0,8,0),(8,11,1),(11,17,2),(17,20,3),(20,21,4)):
            damping=1.0 if k==0 else exp(-delta**k)
            phir += damping*sum(self.a[i]*tau**self.t[i]*delta**self.d[i] for i in range(lo,hi))
        return tau,delta,phi0,phir
    def fp(self):
        tau,delta,p0,pr=self._terms(); return self.R*self.T*(p0+pr)
    def up(self):
        tau,delta,_,_=self._terms(); pt=self.b[1]*tau+self.b[2]-.5*self.b[3]*tau**-.5-.75*self.b[4]*tau**-.75
        pr=0.0
        for lo,hi,k in ((0,8,0),(8,11,1),(11,17,2),(17,20,3),(20,21,4)):
            damping=1.0 if k==0 else exp(-delta**k)
            pr += damping*sum(self.a[i]*self.t[i]*tau**self.t[i]*delta**self.d[i] for i in range(lo,hi))
        return self.R*self.T*(pr+pt)+self.m_energy_offset
    def sp(self):
        # HFC134a's pinned header defines entropy from u-f/T, including the
        # configured reference offsets exactly this way.
        return ((self.up()-self.m_energy_offset)-self.fp())/self.T + self.m_entropy_offset
    def Pp(self):
        tau,delta,_,_=self._terms(); q=0.
        for i in range(21):
            k=0 if i<8 else 1 if i<11 else 2 if i<17 else 3 if i<20 else 4
            damping=1.0 if k==0 else exp(-delta**k)
            q += self.a[i]*tau**self.t[i]*(self.d[i]-k*delta**k)*delta**(self.d[i]-1)*damping
        return self.R*self.T*delta*delta*self.Roc*(q+1/delta)
    def Psat(self):
        if not self.Tmn<=self.T<=self.Tc: raise TPXError('HFC134a: temperature out of range')
        x=self.T/self.Tc; q=1-x; return self.Pc*exp((-7.686556*q+2.311791*q**1.5-2.039554*q*q-3.583758*q**4)/x)
    def ldens(self):
        if not self.Tmn<=self.T<=self.Tc: raise TPXError('HFC134a: temperature out of range')
        q=1-self.T/self.Tc; return 518.2+884.13*q**(1/3)+485.84*q**(2/3)+193.29*q**(10/3)
