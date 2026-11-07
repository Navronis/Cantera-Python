// Native benchmark for Thermodynamics Family 4: Ideal and Condensed Phase Models
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/SurfPhase.h"
#include "cantera/thermo/IdealSolidSolnPhase.h"
#include "cantera/thermo/ThermoFactory.h"
#include "cantera/base/global.h"

using namespace Cantera;

int main(int argc, char** argv) {
    addDataDirectory("/home/nebula/cantera-pinned-oracle/data");
    addDataDirectory("/home/nebula/cantera-pinned-oracle/test/data");
    if (argc > 1) {
        addDataDirectory(argv[1]);
    }

    const int N_EVAL = 10000;
    volatile double sink = 0.0;

    // 1. IdealGasPhase (gri30.yaml)
    auto gas = newThermo("gri30.yaml", "gri30");
    gas->setState_TPX(1200.0, 101325.0, "CH4:0.1, O2:0.2, N2:0.7");
    size_t nsp_gas = gas->nSpecies();
    std::vector<double> mu_gas(nsp_gas);

    // Warmup
    for (int i = 0; i < 500; ++i) {
        gas->setState_TP(1200.0 + (i % 50), 101325.0);
        sink += gas->enthalpy_mole();
        sink += gas->entropy_mole();
        sink += gas->cp_mole();
        gas->getChemPotentials(mu_gas);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_EVAL; ++i) {
        double T = 500.0 + (i % 1500) * 0.5;
        double P = 101325.0 + (i % 10) * 10000.0;
        gas->setState_TP(T, P);
        sink += gas->enthalpy_mole();
        sink += gas->entropy_mole();
        sink += gas->cp_mole();
        gas->getChemPotentials(mu_gas);
        sink += mu_gas[0] + mu_gas[nsp_gas - 1];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double gas_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_EVAL;

    // 2. SurfPhase (surface-phases.yaml, Pt-surf)
    auto surf = std::dynamic_pointer_cast<SurfPhase>(newThermo("surface-phases.yaml", "Pt-surf"));
    surf->setState_TP(900.0, 101325.0);
    size_t nsp_surf = surf->nSpecies();
    std::vector<double> cov(nsp_surf, 0.0);
    cov[0] = 0.8; cov[1] = 0.2;
    surf->setCoverages(cov);
    std::vector<double> mu_surf(nsp_surf);

    // Warmup
    for (int i = 0; i < 500; ++i) {
        surf->setState_TP(800.0 + (i % 100), 101325.0);
        sink += surf->enthalpy_mole();
        sink += surf->entropy_mole();
        sink += surf->cp_mole();
        surf->getChemPotentials(mu_surf);
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_EVAL; ++i) {
        double T = 600.0 + (i % 800) * 0.5;
        surf->setState_TP(T, 101325.0);
        sink += surf->enthalpy_mole();
        sink += surf->entropy_mole();
        sink += surf->cp_mole();
        surf->getChemPotentials(mu_surf);
        sink += mu_surf[0] + mu_surf[nsp_surf - 1];
    }
    t1 = std::chrono::high_resolution_clock::now();
    double surf_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_EVAL;

    // 3. IdealSolidSolnPhase (thermo-models.yaml, IdealSolidSolnPhase)
    auto solid = std::dynamic_pointer_cast<IdealSolidSolnPhase>(newThermo("thermo-models.yaml", "IdealSolidSolnPhase"));
    solid->setState_TP(1000.0, 101325.0);
    size_t nsp_sol = solid->nSpecies();
    std::vector<double> mu_sol(nsp_sol);

    // Warmup
    for (int i = 0; i < 500; ++i) {
        solid->setState_TP(900.0 + (i % 100), 101325.0);
        sink += solid->enthalpy_mole();
        sink += solid->entropy_mole();
        sink += solid->cp_mole();
        solid->getChemPotentials(mu_sol);
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_EVAL; ++i) {
        double T = 800.0 + (i % 600) * 0.5;
        solid->setState_TP(T, 101325.0);
        sink += solid->enthalpy_mole();
        sink += solid->entropy_mole();
        sink += solid->cp_mole();
        solid->getChemPotentials(mu_sol);
        sink += mu_sol[0] + mu_sol[nsp_sol - 1];
    }
    t1 = std::chrono::high_resolution_clock::now();
    double solid_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_EVAL;

    // 4. StoichSubstance (thermo-models.yaml, NaCl(s))
    auto stoich = newThermo("thermo-models.yaml", "NaCl(s)");
    stoich->setState_TP(400.0, 101325.0);
    std::vector<double> mu_stoich(1);

    // Warmup
    for (int i = 0; i < 500; ++i) {
        stoich->setState_TP(350.0 + (i % 100), 101325.0);
        sink += stoich->enthalpy_mole();
        sink += stoich->entropy_mole();
        sink += stoich->cp_mole();
        stoich->getChemPotentials(mu_stoich);
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_EVAL; ++i) {
        double T = 300.0 + (i % 500) * 0.5;
        stoich->setState_TP(T, 101325.0);
        sink += stoich->enthalpy_mole();
        sink += stoich->entropy_mole();
        sink += stoich->cp_mole();
        stoich->getChemPotentials(mu_stoich);
        sink += mu_stoich[0];
    }
    t1 = std::chrono::high_resolution_clock::now();
    double stoich_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_EVAL;

    std::ofstream out("audit/native_thermo_family4_benchmark.json");
    out << "{\n"
        << "  \"ideal_gas_us\": " << gas_us << ",\n"
        << "  \"ideal_surface_us\": " << surf_us << ",\n"
        << "  \"ideal_solid_soln_us\": " << solid_us << ",\n"
        << "  \"stoich_substance_us\": " << stoich_us << ",\n"
        << "  \"checksum\": " << sink << "\n"
        << "}\n";
    out.close();

    std::cout << "Native Family 4 Benchmarks:" << std::endl;
    std::cout << "  IdealGasPhase:       " << gas_us << " us / call" << std::endl;
    std::cout << "  SurfPhase:          " << surf_us << " us / call" << std::endl;
    std::cout << "  IdealSolidSolnPhase: " << solid_us << " us / call" << std::endl;
    std::cout << "  StoichSubstance:     " << stoich_us << " us / call" << std::endl;

    return 0;
}
