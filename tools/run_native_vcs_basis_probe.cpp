// Read-only probe of the pinned VCS basis and formation coefficients.
// Build against the exact pinned Cantera shared library, never the Python port.
#include "cantera/base/Solution.h"
#include "cantera/equil/MultiPhase.h"
#include "cantera/equil/vcs_VolPhase.h"
#include "cantera/equil/vcs_solve.h"
#include "cantera/thermo/ThermoPhase.h"

#include <iomanip>
#include <iostream>
#include <algorithm>

using namespace Cantera;

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "usage: run_native_vcs_basis_probe DATA_DIR\n";
        return 2;
    }
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

    std::cout << "components " << solver.m_numComponents << "\n";
    std::cout << "species " << solver.m_nsp << "\n";
    std::cout << "elements " << solver.m_nelem << "\n";
    std::cout << std::setprecision(17);
    for (size_t e = 0; e < solver.m_nelem; ++e) {
        std::cout << "element " << e << " " << solver.m_elementName[e] << "\n";
    }
    for (size_t k = 0; k < solver.m_nsp; ++k) {
        std::cout << "species " << k << " " << solver.m_speciesName[k] << "\n";
    }
    for (size_t rxn = 0; rxn < solver.m_numRxnTot; ++rxn) {
        for (size_t comp = 0; comp < solver.m_numComponents; ++comp) {
            std::cout << "stoich " << solver.m_speciesName[comp] << " "
                      << solver.m_speciesName[rxn + solver.m_numComponents] << " "
                      << solver.m_stoichCoeffRxnMatrix(comp, rxn) << "\n";
        }
    }

    // Exercise the source method's major/minor filters and zero-component
    // restriction with deterministic dimensionless species free energies.
    std::fill(solver.m_feSpecies_old.begin(), solver.m_feSpecies_old.end(), 0.0);
    std::fill(solver.m_deltaGRxn_old.begin(), solver.m_deltaGRxn_old.end(), 99.0);
    std::fill(solver.m_speciesStatus.begin(), solver.m_speciesStatus.end(),
              VCS_SPECIES_MAJOR);
    solver.m_feSpecies_old[0] = 5.0;
    solver.m_feSpecies_old[1] = 2.0;
    solver.m_feSpecies_old[solver.m_indexRxnToSpecies[0]] = -10.0;
    solver.m_feSpecies_old[solver.m_indexRxnToSpecies[1]] = -1.0;
    solver.m_speciesStatus[solver.m_indexRxnToSpecies[1]] = VCS_SPECIES_MINOR;
    solver.m_molNumSpecies_old[0] = 0.0;
    solver.m_molNumSpecies_old[1] = 1.0;
    for (int filter : {-1, 0, 1}) {
        std::fill(solver.m_deltaGRxn_old.begin(), solver.m_deltaGRxn_old.end(), 99.0);
        solver.vcs_deltag(filter, false, VCS_STATECALC_OLD, false);
        std::cout << "deltag " << filter << " 0 "
                  << solver.m_deltaGRxn_old[0] << "\n";
        std::cout << "deltag " << filter << " 1 "
                  << solver.m_deltaGRxn_old[1] << "\n";
    }

    // Compare the native VCS species chemical-potential update for the
    // same prepared GRI-Mech/graphite problem. Refresh phase totals after the
    // deterministic mole edits above so vcs_dfe sees a consistent old state.
    solver.vcs_tmoles();
    for (size_t i = 0; i < solver.m_numPhases; ++i) {
        solver.m_VolPhaseList[i]->setMolesOutOfDate(VCS_STATECALC_OLD);
    }
    std::fill(solver.m_feSpecies_old.begin(), solver.m_feSpecies_old.end(), -900.0);
    solver.vcs_dfe(VCS_STATECALC_OLD, 0, 0, solver.m_nsp);
    for (size_t k = 0; k < solver.m_nsp; ++k) {
        std::cout << "dfe_input " << k << " " << solver.m_SSfeSpecies[k] << " "
                  << solver.m_lnMnaughtSpecies[k] << " "
                  << solver.m_molNumSpecies_old[k] << " "
                  << solver.m_actCoeffSpecies_old[k] << " "
                  << solver.m_chargeSpecies[k] << " " << solver.m_phaseID[k] << " "
                  << solver.m_speciesStatus[k] << " "
                  << solver.m_speciesUnknownType[k] << " "
                  << (solver.m_SSPhase[k] ? 1 : 0) << "\n";
    }
    for (size_t i = 0; i < solver.m_numPhases; ++i) {
        std::cout << "dfe_phase " << i << " " << solver.m_tPhaseMoles_old[i]
                  << " " << solver.m_phasePhi[i] << "\n";
    }
    for (size_t i = 0; i < solver.m_numRxnRdc; ++i) {
        std::cout << "dfe_reaction " << i << " "
                  << solver.m_indexRxnToSpecies[i] << "\n";
    }
    std::cout << "dfe_dimensions " << solver.m_numComponents << " "
              << solver.m_numRxnRdc << " " << solver.m_nsp << "\n";
    for (int filter : {-1, 0, 1}) {
        std::fill(solver.m_feSpecies_old.begin(), solver.m_feSpecies_old.end(), -900.0);
        solver.vcs_dfe(VCS_STATECALC_OLD, filter, 0, solver.m_nsp);
        for (size_t k = 0; k < solver.m_nsp; ++k) {
            std::cout << "dfe_result " << filter << " " << k << " "
                      << solver.m_feSpecies_old[k] << "\n";
        }
    }
    for (size_t iph = 0; iph < solver.m_numPhases; ++iph) {
        auto* phase = solver.m_VolPhaseList[iph].get();
        std::cout << "dfe_phase_state " << iph << " " << phase->totalMoles()
                  << " " << phase->electricPotential() << " " << phase->exists() << "\n";
        for (size_t local = 0; local < phase->nSpecies(); ++local) {
            size_t global = phase->spGlobalIndexVCS(local);
            std::cout << "dfe_phase_species " << iph << " " << local << " "
                      << global << " " << solver.m_speciesName[global] << " "
                      << phase->moleFraction(local) << " "
                      << solver.m_actCoeffSpecies_old[global] << "\n";
        }
    }
}
