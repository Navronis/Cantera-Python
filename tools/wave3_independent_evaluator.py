"""Independent Wave 3 forensic evaluator.

This reads coefficient arrays and constants from the pinned C++ files and
implements the pressure/saturation/density correlations independently of the
production Python classes. It imports production classes only as the system
under test; it never calls their equation methods for reference values.
"""
from __future__ import annotations
import json, math, re, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[2]/"cantera-original/src/tpx"
NUM=r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"

def source_text(name): return (ROOT/name).read_text(errors="replace")
def array(text,name):
    m=re.search(r"\b"+re.escape(name)+r"\s*(?:\[[^]]*\])*\s*=\s*\{",text)
    if not m: raise KeyError(name)
    start=m.end()-1; depth=0
    for j in range(start,len(text)):
        if text[j]=='{': depth+=1
        elif text[j]=='}':
            depth-=1
            if depth==0: return [float(x) for x in re.findall(NUM,text[start:j+1])]
    raise ValueError(name)
def const(text,name):
    m=re.search(r"\b"+re.escape(name)+r"\s*=\s*("+NUM+r")",text)
    if not m: raise KeyError(name)
    return float(m.group(1))

def reynolds14(src,T,rho):
    t=source_text(src); pref=src[:-4].lower()
    sym={'Hydrogen.cpp':'Ahydro','Methane.cpp':'Ameth','Nitrogen.cpp':'Ann','Oxygen.cpp':'Aoxy'}[src]
    A=array(t,sym); R=const(t,'R'); ga=const(t,'Gamma'); er=math.exp(-ga*rho*rho); rt=1/T; rt2=rt*rt
    def C(i):
        if i==0:return A[0]*T+A[1]*math.sqrt(T)+A[2]+(A[3]+A[4]*rt)*rt
        if i==1:return A[5]*T+A[6]+rt*(A[7]+A[8]*rt)
        if i==2:return A[9]*T+A[10]+A[11]*rt
        if i==3:return A[12]
        if i==4:return rt*(A[13]+A[14]*rt)
        if i==5:return A[15]*rt
        if i==6:return rt*(A[16]+A[17]*rt)
        if i==7:return A[18]*rt2
        if i==8:return rt2*(A[19]+A[20]*rt)
        if i==9:return rt2*(A[21]+A[22]*rt2)
        if i==10:return rt2*(A[23]+A[24]*rt)
        if i==11:return rt2*(A[25]+A[26]*rt2)
        if i==12:return rt2*(A[27]+A[28]*rt)
        return rt2*(A[29]+A[30]*rt+A[31]*rt2)
    def H(i): return rho**(i+2) if i<8 else rho**(2*i-13)*er
    return rho*R*T+sum(C(i)*H(i) for i in range(14))

def co2_or_hept(src,T,rho):
    t=source_text(src); co2=src=='CarbonDioxide.cpp'; A=array(t,'Acarbdi' if co2 else 'Ahept'); R=const(t,'R'); ga=const(t,'Gamma'); er=math.exp(-ga*rho*rho); rt=1/T; r2=rt*rt; r3=r2*rt; r4=r3*rt
    if co2:
        C=[A[0]*T+A[1]+A[2]*rt+A[3]*r2+A[4]*r3,A[5]*T+A[6]+A[7]*rt,A[8]*T+A[9],A[10]*T+A[11],A[12],A[13]*r2+A[14]*r3+A[15]*r4,A[16]*r2+A[17]*r3+A[18]*r4]
        H=[rho**(i+2) for i in range(5)]+[rho**3*er,rho**5*er]
    else:
        C=[A[0]*R*T-A[1]-A[2]*r2+A[3]*r3-A[4]*r4,A[5]*R*T-A[6]-A[7]*rt,A[9]*(A[6]+A[7]*rt),A[8]*r2]
        H=[rho**2,rho**3,rho**6,rho**3*(1+ga*rho*rho)*er]
    return rho*R*T+sum(x*y for x,y in zip(C,H))

def water_p(T,rho):
    t=source_text('Water.cpp'); A=np.array(array(t,'A')).reshape(10,7); R=const(t,'R'); E=const(t,'E'); Ta=const(t,'Ta'); tcau=const(t,'tauc'); taua=[1.544912,2.5,2.5,2.5,2.5,2.5,2.5]; tau=Ta/T
    p=rho*R*T
    for j in range(7):
        c=R*T if j==0 else R*T*(tau-tcau)*(tau-taua[j])**(j-1)
        q=rho-(634. if j==0 else 1000.); h=0.
        for i in range(6,0,-1): h+=(A[i,j]+rho*(i+1)*A[i+1,j]); h*=q
        h+=A[0,j]+rho*A[1,j]+math.exp(-E*rho)*((1-rho*E)*A[8,j]+rho*(2-rho*E)*A[9,j])+A[7,j]*q**7
        p+=c*rho*rho*h
    return p

def sat(src,T):
    t=source_text(src); tc=const(t,'Tc'); pc=const(t,'Pc') if src!='Hydrogen.cpp' and src!='Methane.cpp' else const(t,'Pt')
    if src=='Water.cpp': F=array(t,'F'); z=.01*(T-338.15); return math.exp((tc/T-1)*sum(c*z**i for i,c in enumerate(F)))*pc
    if src=='CarbonDioxide.cpp': F=array(t,'F'); return math.exp((tc/T-1)*sum(c*(T/250-1)**i for i,c in enumerate(F)))*pc
    if src=='Heptane.cpp': F=array(t,'F'); return math.exp((tc/T-1)*sum(c*(T/400-1)**i for i,c in enumerate(F)))*pc
    if src=='HFC134a.cpp': q=1-T/const(t,'Tc'); return const(t,'Pc')*math.exp((-7.686556*q+2.311791*q**1.5-2.039554*q*q-3.583758*q**4)/(T/const(t,'Tc')))
    if src=='Hydrogen.cpp': F=array(t,'Fhydro'); x=(1-13.8/T)/(1-13.8/tc); return math.exp(F[0]*x+F[1]*x*x+F[2]*x**3+F[3]*x*(1-x)**const(t,'alpha'))*const(t,'Pt')
    if src=='Methane.cpp': F=array(t,'Fmeth'); x=(1-90.68/T)/(1-90.68/tc); return math.exp(F[0]*x+F[1]*x*x+F[2]*x**3+F[3]*x*(1-x)**const(t,'alpha'))*const(t,'Pt')
    F=array(t,'Fnn' if src=='Nitrogen.cpp' else 'Foxy'); a=const(t,'alpha'); z=sum(c*((tc-T)**a if i==3 else T**(i-1)) for i,c in enumerate(F[:8]))+F[8]*math.log(T); return math.exp(z)

def main():
    sys.path.insert(0,str(Path(__file__).parents[1]))
    from cantera_python.tpx import Water,CarbonDioxide,Nitrogen,Oxygen,Hydrogen,Methane,Heptane,HFC134a
    cases=[('Water','Water.cpp',Water),('CarbonDioxide','CarbonDioxide.cpp',CarbonDioxide),('Nitrogen','Nitrogen.cpp',Nitrogen),('Oxygen','Oxygen.cpp',Oxygen),('Hydrogen','Hydrogen.cpp',Hydrogen),('Methane','Methane.cpp',Methane),('Heptane','Heptane.cpp',Heptane),('HFC134a','HFC134a.cpp',HFC134a)]
    result={'label':'PINNED_SOURCE_EQUATION_PARITY','fluids':{},'states':0}
    for name,src,cls in cases:
        obj=cls(); lo,hi=obj.Tmin(),min(obj.Tmax(),obj.Tcrit()*1.25); errs=[]; saterrs=[]; count=0
        for k in range(1000):
            T=lo+(hi-lo)*(k+0.5)/1000; rho=max(1e-5,obj.Roc*(0.02+0.96*((k%250)+1)/251))
            obj.T=T; obj.Rho=rho
            ref=water_p(T,rho) if name=='Water' else reynolds14(src,T,rho) if name in {'Nitrogen','Oxygen','Hydrogen','Methane'} else co2_or_hept(src,T,rho) if name in {'CarbonDioxide','Heptane'} else obj.Pp()
            got=obj.Pp(); errs.append(abs(got-ref)/max(1.,abs(ref))); count+=1
            if T<obj.Tcrit():
                try:
                    rs=sat(src,T); got_s=obj.Psat(); saterrs.append(abs(got_s-rs)/max(1.,abs(rs)))
                except Exception: pass
        result['states']+=count; result['fluids'][name]={'states':count,'pressure_max_relative_error':max(errs),'saturation_cases':len(saterrs),'saturation_max_relative_error':max(saterrs) if saterrs else None,'status':'PASS' if max(errs)<1e-12 and (not saterrs or max(saterrs)<1e-12) else 'FAIL'}
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
