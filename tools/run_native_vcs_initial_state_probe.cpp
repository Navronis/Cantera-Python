#include "cantera/base/Solution.h"
#include "cantera/equil/MultiPhase.h"
#include "cantera/equil/vcs_VolPhase.h"
#include "cantera/equil/vcs_solve.h"
#include "cantera/thermo/ThermoPhase.h"
#include <iomanip>
#include <iostream>
using namespace Cantera;
int main(int argc,char** argv){
 if(argc!=2)return 2; string data=argv[1];
 auto gas=newSolution(data+"/gri30.yaml","gri30","none");
 auto solid=newSolution(data+"/graphite.yaml","graphite","none");
 gas->thermo()->setState_TPX(1200.0,OneAtm,"CH4:1,O2:2,N2:7.52");
 solid->thermo()->setState_TP(1200.0,OneAtm);
 MultiPhase mix; mix.addPhase(gas->thermo(),1.0);mix.addPhase(solid->thermo(),0.2);
 mix.setTemperature(1200.0);mix.setPressure(OneAtm);
 VCS_SOLVE solver(&mix);solver.vcs_prep(0);solver.vcs_basopt(false,-1.0e-10);
 solver.vcs_evalSS_TP(0,0,1200.0,OneAtm);
 solver.vcs_dfe(VCS_STATECALC_OLD,0,0,solver.m_nsp);
 solver.vcs_deltag(0,false,VCS_STATECALC_OLD,false);
 std::cout<<std::setprecision(17)<<"dims "<<solver.m_nsp<<" "<<solver.m_numComponents<<" "<<solver.m_numRxnTot<<" "<<solver.m_numPhases<<"\n";
 for(size_t k=0;k<solver.m_nsp;k++)std::cout<<"sp "<<k<<" "<<solver.m_speciesName[k]<<" "<<solver.m_SSfeSpecies[k]<<" "<<solver.m_lnMnaughtSpecies[k]<<" "<<solver.m_molNumSpecies_old[k]<<" "<<solver.m_speciesStatus[k]<<" "<<solver.m_chargeSpecies[k]<<" "<<solver.m_phaseID[k]<<" "<<solver.m_speciesUnknownType[k]<<" "<<solver.m_actCoeffSpecies_old[k]<<" "<<solver.m_feSpecies_old[k]<<"\n";
 for(size_t i=0;i<solver.m_numPhases;i++)std::cout<<"phase "<<i<<" "<<solver.m_tPhaseMoles_old[i]<<" "<<solver.m_phasePhi[i]<<"\n";
 for(size_t i=0;i<solver.m_numRxnTot;i++)std::cout<<"dg "<<i<<" "<<solver.m_deltaGRxn_old[i]<<"\n";
}
