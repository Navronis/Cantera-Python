// Native benchmark for Thermodynamics Family 2: SpeciesThermo polynomial evaluation
// Rigorous benchmark with non-removable checksum sink and varying temperatures.
#include <iostream>
#include <fstream>
#include <vector>
#include <span>
#include <chrono>
#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/SpeciesThermoInterpType.h"
#include "cantera/thermo/MultiSpeciesThermo.h"

using namespace Cantera;

int main() {
    try {
        auto sol = newSolution("gri30.yaml", "gri30", "none");
        auto thermo = sol->thermo();
        auto& spthermo = thermo->speciesThermo();
        size_t n_sp = thermo->nSpecies();

        double t_base = 300.0;
        int N_evals = 100000;
        int N_batch = 10000;

        // 1. Benchmark scalar update_single (species 0)
        volatile double checksum_scalar = 0.0;
        // Warmup
        for (int i = 0; i < 1000; ++i) {
            double T = t_base + (i % 1000) * 0.5;
            double cp_R = 0.0, h_RT = 0.0, s_R = 0.0;
            spthermo.update_single(0, T, cp_R, h_RT, s_R);
            checksum_scalar += (cp_R + h_RT + s_R);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_evals; ++i) {
            double T = t_base + (i % 1000) * 0.5;
            double cp_R = 0.0, h_RT = 0.0, s_R = 0.0;
            spthermo.update_single(0, T, cp_R, h_RT, s_R);
            checksum_scalar += (cp_R + h_RT + s_R);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt_single_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_evals;

        // 2. Benchmark batch update across all 53 species
        volatile double checksum_batch = 0.0;
        std::vector<double> cp_R(n_sp, 0.0);
        std::vector<double> h_RT(n_sp, 0.0);
        std::vector<double> s_R(n_sp, 0.0);
        std::span<double> cp_span(cp_R);
        std::span<double> h_span(h_RT);
        std::span<double> s_span(s_R);

        // Warmup
        for (int i = 0; i < 500; ++i) {
            double T = t_base + (i % 1000) * 0.5;
            spthermo.update(T, cp_span, h_span, s_span);
            checksum_batch += (cp_R[0] + h_RT[0] + s_R[0]);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_batch; ++i) {
            double T = t_base + (i % 1000) * 0.5;
            spthermo.update(T, cp_span, h_span, s_span);
            checksum_batch += (cp_R[0] + h_RT[0] + s_R[0]);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double dt_batch_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_batch;

        std::cout << "{\n";
        std::cout << "  \"single_eval_us\": " << dt_single_us << ",\n";
        std::cout << "  \"batch_eval_us\": " << dt_batch_us << ",\n";
        std::cout << "  \"checksum_scalar\": " << checksum_scalar << ",\n";
        std::cout << "  \"checksum_batch\": " << checksum_batch << "\n";
        std::cout << "}\n";

        std::ofstream out("audit/native_thermo_family2_benchmark.json");
        out << "{\n";
        out << "  \"single_eval_us\": " << dt_single_us << ",\n";
        out << "  \"batch_eval_us\": " << dt_batch_us << ",\n";
        out << "  \"checksum_scalar\": " << checksum_scalar << ",\n";
        out << "  \"checksum_batch\": " << checksum_batch << "\n";
        out << "}\n";
        out.close();

        return 0;
    } catch (std::exception& err) {
        std::cerr << "Error: " << err.what() << std::endl;
        return 1;
    }
}
