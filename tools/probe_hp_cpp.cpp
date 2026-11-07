#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "cantera/core.h"

#define private public
#define protected public
#include "cantera/transport/HighPressureGasTransport.h"

using namespace Cantera;

int main() {
    std::string path = "/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-original/test/data/methane_co2.yaml";
    auto sol = newSolution(path, "methane_co2", "high-pressure");
    auto gas = sol->thermo();
    gas->setState_TPX(350.0, 101325.0, "CH4:0.755, CO2:0.245");
    auto tr = std::dynamic_pointer_cast<HighPressureGasTransport>(sol->transport());

    std::cout << std::setprecision(16);
    tr->computeMixtureParameters();
    std::cout << "Tc_mix: " << tr->m_Tc_mix << std::endl;
    std::cout << "Pc_mix: " << tr->m_Pc_mix << std::endl;
    std::cout << "Tr_mix: " << tr->m_Tr_mix << std::endl;
    std::cout << "Pr_mix: " << tr->m_Pr_mix << std::endl;
    std::cout << "MW_mix: " << tr->m_MW_mix << std::endl;
    std::cout << "FP_mix_o: " << tr->m_FP_mix_o << std::endl;
    std::cout << "FQ_mix_o: " << tr->m_FQ_mix_o << std::endl;
    std::cout << "P_vap_mix: " << tr->m_P_vap_mix << std::endl;
    for (size_t i = 0; i < gas->nSpecies(); i++) {
        std::cout << "Species " << i << ":" << std::endl;
        std::cout << "  Tcrit: " << tr->Tcrit_i(i) << std::endl;
        std::cout << "  Pcrit: " << tr->Pcrit_i(i) << std::endl;
        std::cout << "  Vcrit: " << tr->Vcrit_i(i) << std::endl;
        std::cout << "  Zcrit: " << tr->Zcrit_i(i) << std::endl;
        std::cout << "  w_ac: " << tr->m_w_ac[i] << std::endl;
        std::cout << "  mw: " << tr->m_mw[i] << std::endl;
        std::cout << "  dipole: " << tr->m_dipole(i, i) << std::endl;
    }
    double Z_1 = tr->lowPressureNondimensionalViscosity(tr->m_Tr_mix, tr->m_FP_mix_o, tr->m_FQ_mix_o);
    std::cout << "Z_1: " << Z_1 << std::endl;
    double Z_2 = tr->highPressureNondimensionalViscosity(tr->m_Tr_mix, tr->m_Pr_mix, tr->m_FP_mix_o, tr->m_FQ_mix_o, tr->m_P_vap_mix, tr->m_Pc_mix);
    std::cout << "Z_2: " << Z_2 << std::endl;
    double numerator = GasConstant*tr->m_Tc_mix*pow(Avogadro,2.0);
    double denominator = pow(tr->m_MW_mix,3.0)*pow(tr->m_Pc_mix,4.0);
    double xi = pow(numerator / denominator, 1.0/6.0);
    std::cout << "xi: " << xi << std::endl;
    std::cout << "Viscosity: " << tr->viscosity() << std::endl;
    std::cout << "ThermalConductivity: " << tr->thermalConductivity() << std::endl;

    auto chungSol = newSolution(path, "methane_co2", "high-pressure-Chung");
    auto chungGas = chungSol->thermo();
    chungGas->setState_TPX(350.0, 101325.0, "CH4:0.755, CO2:0.245");
    auto chungTr = std::dynamic_pointer_cast<ChungHighPressureGasTransport>(chungSol->transport());
    std::cout << "Chung viscosity: " << chungTr->viscosity() << std::endl;
    std::cout << "Chung thermalConductivity: " << chungTr->thermalConductivity() << std::endl;
    return 0;
}
