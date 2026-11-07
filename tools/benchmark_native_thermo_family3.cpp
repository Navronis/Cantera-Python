// Native benchmark for Thermodynamics Family 3: ThermoPhase Base
#include <iostream>
#include <fstream>
#include <vector>
#include <span>
#include <chrono>
#include "cantera/base/Solution.h"
#include "cantera/base/global.h"
#include "cantera/thermo/ThermoPhase.h"

using namespace Cantera;

int main(int argc, char** argv) {
    std::string data_dir = "/home/nebula/cantera-pinned-oracle/data";
    if (argc > 1) {
        data_dir = argv[1];
    }
    addDataDirectory(data_dir);

    auto sol = newSolution("gri30.yaml", "", "None");
    auto thermo = sol->thermo();
    thermo->setState_TPX(1200.0, 101325.0, "CH4:1, O2:2, N2:7.52");

    size_t nsp = thermo->nSpecies();
    std::vector<double> mu(nsp);

    // Warmup
    for (int i = 0; i < 500; ++i) {
        thermo->setState_TP(1200.0 + (i % 50), 101325.0);
        thermo->enthalpy_mole();
        thermo->entropy_mole();
        thermo->cp_mole();
        thermo->getChemPotentials(mu);
    }

    // Benchmark 1: Forward State Update & Property Evaluation
    const int N_FWD = 20000;
    volatile double sink = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_FWD; ++i) {
        double T = 500.0 + (i % 1500) * 0.5;
        double P = 101325.0 + (i % 10) * 10000.0;
        thermo->setState_TP(T, P);
        sink += thermo->enthalpy_mole();
        sink += thermo->entropy_mole();
        sink += thermo->cp_mole();
        thermo->getChemPotentials(mu);
        sink += mu[0] + mu[nsp - 1];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double fwd_dur_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_FWD;

    // Benchmark 2: State Inversion HP
    const int N_INV = 2000;
    thermo->setState_TPX(1200.0, 101325.0, "CH4:1, O2:2, N2:7.52");
    double H_ref = thermo->enthalpy_mass();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_INV; ++i) {
        thermo->setState_TP(1000.0 + (i % 400), 101325.0);
        thermo->setState_HP(H_ref, 101325.0, 1e-8);
        sink += thermo->temperature();
    }
    t1 = std::chrono::high_resolution_clock::now();
    double inv_dur_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_INV;

    std::ofstream out("audit/native_thermo_family3_benchmark.json");
    out << "{\n"
        << "  \"fwd_eval_us\": " << fwd_dur_us << ",\n"
        << "  \"hp_inversion_us\": " << inv_dur_us << ",\n"
        << "  \"checksum\": " << sink << "\n"
        << "}\n";
    out.close();

    std::cout << "Native Family 3 Benchmarks:" << std::endl;
    std::cout << "  Forward eval: " << fwd_dur_us << " us / call" << std::endl;
    std::cout << "  HP inversion: " << inv_dur_us << " us / call" << std::endl;

    return 0;
}
