#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/ThermoFactory.h"
#include "cantera/thermo/PureFluidPhase.h"
#include "cantera/thermo/WaterProps.h"
#include "cantera/thermo/WaterPropsIAPWS.h"
#include "cantera/base/global.h"

using namespace Cantera;

int main() {
    addDataDirectory("/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-original/test/data");
    addDataDirectory("/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-original/data");

    const int N = 5000;
    std::cout << "Starting native Cantera Family 5 benchmark (" << N << " iterations)...\n";

    // 1. Peng-Robinson
    auto pr = newThermo("co2_PR_example.yaml", "CO2-PR");
    auto start_pr = std::chrono::high_resolution_clock::now();
    double dummy_pr = 0.0;
    for (int i = 0; i < N; ++i) {
        double T = 300.0 + (i % 200) * 1.0;
        double P = 1.0e5 + (i % 50) * 1.0e5;
        pr->setState_TP(T, P);
        dummy_pr += pr->enthalpy_mole() + pr->cp_mole() + pr->soundSpeed();
    }
    auto end_pr = std::chrono::high_resolution_clock::now();
    double time_pr = std::chrono::duration<double>(end_pr - start_pr).count();

    // 2. Redlich-Kwong
    auto rk = newThermo("co2_RK_example.yaml", "carbondioxide");
    auto start_rk = std::chrono::high_resolution_clock::now();
    double dummy_rk = 0.0;
    for (int i = 0; i < N; ++i) {
        double T = 300.0 + (i % 200) * 1.0;
        double P = 1.0e5 + (i % 50) * 1.0e5;
        rk->setState_TP(T, P);
        dummy_rk += rk->enthalpy_mole() + rk->cp_mole() + rk->soundSpeed();
    }
    auto end_rk = std::chrono::high_resolution_clock::now();
    double time_rk = std::chrono::duration<double>(end_rk - start_rk).count();

    // 3. PureFluidPhase (liquidvapor.yaml, water)
    auto water = newThermo("liquidvapor.yaml", "water");
    auto start_water = std::chrono::high_resolution_clock::now();
    double dummy_water = 0.0;
    for (int i = 0; i < N; ++i) {
        double T = 300.0 + (i % 150) * 1.0;
        double P = 1.0e5 + (i % 50) * 1.0e5;
        water->setState_TP(T, P);
        dummy_water += water->density() + water->enthalpy_mass() + water->entropy_mass();
    }
    auto end_water = std::chrono::high_resolution_clock::now();
    double time_water = std::chrono::duration<double>(end_water - start_water).count();

    // 4. WaterProps IAPWS-95 density
    WaterProps wp;
    auto start_wp = std::chrono::high_resolution_clock::now();
    double dummy_wp = 0.0;
    for (int i = 0; i < N; ++i) {
        double T = 300.0 + (i % 150) * 1.0;
        double P = 1.0e5 + (i % 50) * 1.0e5;
        dummy_wp += wp.density_IAPWS(T, P);
    }
    auto end_wp = std::chrono::high_resolution_clock::now();
    double time_wp = std::chrono::duration<double>(end_wp - start_wp).count();

    double total_native = time_pr + time_rk + time_water + time_wp;

    std::cout << "Native benchmark complete:\n";
    std::cout << "  Peng-Robinson: " << time_pr << " s (" << (time_pr / N * 1e6) << " us/call)\n";
    std::cout << "  Redlich-Kwong: " << time_rk << " s (" << (time_rk / N * 1e6) << " us/call)\n";
    std::cout << "  PureFluidPhase: " << time_water << " s (" << (time_water / N * 1e6) << " us/call)\n";
    std::cout << "  WaterProps IAPWS: " << time_wp << " s (" << (time_wp / N * 1e6) << " us/call)\n";
    std::cout << "  Total time: " << total_native << " s\n";

    std::ofstream out("audit/native_thermo_family5_benchmark.json");
    out << "{\n";
    out << "  \"iterations\": " << N << ",\n";
    out << "  \"workloads\": {\n";
    out << "    \"peng_robinson\": {\"time_sec\": " << time_pr << ", \"us_per_call\": " << (time_pr / N * 1e6) << ", \"dummy\": " << dummy_pr << "},\n";
    out << "    \"redlich_kwong\": {\"time_sec\": " << time_rk << ", \"us_per_call\": " << (time_rk / N * 1e6) << ", \"dummy\": " << dummy_rk << "},\n";
    out << "    \"pure_fluid\": {\"time_sec\": " << time_water << ", \"us_per_call\": " << (time_water / N * 1e6) << ", \"dummy\": " << dummy_water << "},\n";
    out << "    \"water_props\": {\"time_sec\": " << time_wp << ", \"us_per_call\": " << (time_wp / N * 1e6) << ", \"dummy\": " << dummy_wp << "}\n";
    out << "  },\n";
    out << "  \"total_time_sec\": " << total_native << "\n";
    out << "}\n";

    return 0;
}
