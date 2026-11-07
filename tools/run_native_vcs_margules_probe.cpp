#include "cantera/base/Solution.h"
#include "cantera/equil/MultiPhase.h"
#include "cantera/equil/vcs_VolPhase.h"
#include "cantera/equil/vcs_solve.h"
#include "cantera/thermo/ThermoPhase.h"
#include <iomanip>
#include <iostream>
#include <vector>
using namespace Cantera;
int main(int argc, char** argv) {
 if(argc!=2) return 2;
 auto liq=newSolution(string(argv[1])+"/LiKCl_liquid.yaml","MoltenSalt_electrolyte","none");
 liq->thermo()->setState_TPX(900.0,OneAtm,"KCl(L):0.2, LiCl(L):0.8");
 MultiPhase mix; mix.addPhase(liq->thermo(),1.0); mix.setTemperature(900.0); mix.setPressure(OneAtm);
 VCS_SOLVE solver(&mix); solver.vcs_prep(0); solver.vcs_basopt(false,-1.0e-10);
 auto* vp=solver.m_VolPhaseList[0].get();
 std::vector<double> ac(solver.m_nsp,0.0);
 vp->sendToVCS_ActCoeff(VCS_STATECALC_OLD,ac);
 std::cout<<std::setprecision(17)<<"dimensions "<<solver.m_nsp<<" "<<solver.m_numComponents<<"\n";
 for(size_t k=0;k<vp->nSpecies();++k){size_t g=vp->spGlobalIndexVCS(k);std::cout<<"species "<<k<<" "<<g<<" "<<solver.m_speciesName[g]<<" "<<vp->moleFraction(k)<<" "<<ac[g]<<"\n";}
}
