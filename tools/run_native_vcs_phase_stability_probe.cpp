// Compare the pinned native VCS dead-phase stability iteration.
#include "cantera/base/Solution.h"
#include "cantera/equil/MultiPhase.h"
#include "cantera/equil/vcs_VolPhase.h"
#include "cantera/equil/vcs_solve.h"
#include "cantera/thermo/ThermoPhase.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace Cantera;
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const string dataDir = argv[1];
    auto gas = newSolution(dataDir + "/gri30.yaml", "gri30", "none");
    auto graphite = newSolution(dataDir + "/graphite.yaml", "graphite", "none");
    gas->thermo()->setState_TPX(1200.0, OneAtm, "CH4:1, O2:2, N2:7.52");
    graphite->thermo()->setState_TP(1200.0, OneAtm);
    MultiPhase mixture;
    mixture.addPhase(gas->thermo(), 1.0);
    mixture.addPhase(graphite->thermo(), 0.2);
    mixture.setTemperature(1200.0);
    mixture.setPressure(OneAtm);
    VCS_SOLVE solver(&mixture);
    solver.vcs_prep(0);
    solver.vcs_basopt(false, -1.0e-10);
    auto* phase = solver.m_VolPhaseList[0].get();
    const size_t nlocal = phase->nSpecies();
    std::vector<size_t> creationRxn(nlocal, npos);
    std::vector<double> creation(phase->moleFractions().begin(), phase->moleFractions().end());
    phase->setCreationMoleNumbers(creation, creationRxn);
    std::fill(solver.m_feSpecies_old.begin(), solver.m_feSpecies_old.end(), 0.0);
    for (size_t k = 0; k < solver.m_nsp; ++k) {
        solver.m_feSpecies_old[k] = 0.025 * static_cast<double>(k);
    }
    std::fill(solver.m_deltaGRxn_old.begin(), solver.m_deltaGRxn_old.end(), 0.1);
    for (size_t k = 0; k < solver.m_nsp; ++k) {
        if (solver.m_phaseID[k] == 0) solver.m_molNumSpecies_old[k] = 0.0;
    }
    phase->setTotalMoles(0.0);
    std::cout << std::setprecision(17);
    std::cout << "setup " << solver.m_numComponents << " " << solver.m_nsp << " "
              << solver.m_numRxnTot << " " << nlocal << "\n";
    for (size_t k = 0; k < solver.m_nsp; ++k) {
        std::cout << "sp " << k << " " << solver.m_speciesName[k] << " "
                  << solver.m_phaseID[k] << " " << solver.m_feSpecies_old[k] << " "
                  << solver.m_molNumSpecies_old[k] << "\n";
    }
    for (size_t j = 0; j < solver.m_numRxnTot; ++j) {
        std::cout << "dg " << j << " " << solver.m_deltaGRxn_old[j] << "\n";
        std::cout << "rxn_species " << j << " " << solver.m_indexRxnToSpecies[j] << "\n";
    }
    for (size_t i = 0; i < solver.m_numComponents; ++i) {
        for (size_t j = 0; j < solver.m_numRxnTot; ++j) {
            std::cout << "nu " << i << " " << j << " "
                      << solver.m_stoichCoeffRxnMatrix(i,j) << "\n";
        }
    }
    for (size_t k = 0; k < nlocal; ++k) {
        std::cout << "local " << k << " " << phase->spGlobalIndexVCS(k) << " "
                  << creation[k] << " " << creationRxn[k] << "\n";
    }
    const double value = solver.vcs_phaseStabilityTest(0);
    std::cout << "result " << value << "\n";
    for (size_t k = 0; k < nlocal; ++k) {
        std::cout << "out " << k << " " << phase->moleFraction(k) << " "
                  << phase->creationMoleNumbers(creationRxn)[k] << " "
                  << solver.m_actCoeffSpecies_new[phase->spGlobalIndexVCS(k)] << "\n";
    }
}
