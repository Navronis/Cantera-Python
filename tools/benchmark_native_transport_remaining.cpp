#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <span>
#include <iomanip>
#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/ThermoFactory.h"
#include "cantera/transport/TransportFactory.h"
#include "cantera/base/global.h"

using namespace Cantera;
using namespace std;
using namespace std::chrono;

int main() {
    int iterations = 10000;
    cout << "Benchmarking Native Transport Remaining Models (" << iterations << " iterations each)..." << endl;

    // 1. WaterTransport
    auto phase_water = newThermo("thermo-models.yaml", "liquid-water");
    auto tran_water = newTransport(phase_water, "default");
    phase_water->setState_TP(300.0, 101325.0);

    auto t0 = high_resolution_clock::now();
    double sum_mu = 0.0, sum_lam = 0.0;
    for (int i = 0; i < iterations; i++) {
        sum_mu += tran_water->viscosity();
        sum_lam += tran_water->thermalConductivity();
    }
    auto t1 = high_resolution_clock::now();
    double time_water_sec = duration_cast<microseconds>(t1 - t0).count() / 1e6;
    cout << "WaterTransport: " << time_water_sec << " s (" << (time_water_sec / iterations * 1e6) << " us/call)" << endl;

    // 2. UnityLewisTransport
    auto sol_ul = newSolution("gri30.yaml", "gri30", "unity-Lewis-number");
    sol_ul->thermo()->setState_TPX(1200.0, 101325.0, "CH4:0.095, O2:0.19, N2:0.715");
    auto tran_ul = sol_ul->transport();
    size_t nsp_ul = sol_ul->thermo()->nSpecies();
    vector<double> d_ul(nsp_ul);

    t0 = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        sum_lam += tran_ul->thermalConductivity();
        tran_ul->getMixDiffCoeffs(span<double>(d_ul));
    }
    t1 = high_resolution_clock::now();
    double time_ul_sec = duration_cast<microseconds>(t1 - t0).count() / 1e6;
    cout << "UnityLewisTransport: " << time_ul_sec << " s (" << (time_ul_sec / iterations * 1e6) << " us/call)" << endl;

    // 3. IonGasTransport
    auto sol_ion = newSolution("ch4_ion.yaml", "gas", "ionized-gas");
    sol_ion->thermo()->setState_TPX(1500.0, 101325.0, "CO:0.05, CO2:0.05, H2O:0.15, N2:0.74, E:0.005, H3O+:0.005");
    auto tran_ion = sol_ion->transport();
    size_t nsp_ion = sol_ion->thermo()->nSpecies();
    vector<double> mobi_ion(nsp_ion);
    vector<double> d_ion(nsp_ion);

    t0 = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        sum_mu += tran_ion->viscosity();
        tran_ion->getMobilities(span<double>(mobi_ion));
        tran_ion->getMixDiffCoeffs(span<double>(d_ion));
    }
    t1 = high_resolution_clock::now();
    double time_ion_sec = duration_cast<microseconds>(t1 - t0).count() / 1e6;
    cout << "IonGasTransport: " << time_ion_sec << " s (" << (time_ion_sec / iterations * 1e6) << " us/call)" << endl;

    // Write benchmark JSON
    ofstream outfile("audit/native_transport_remaining_benchmark.json");
    outfile << "{\n";
    outfile << "  \"iterations\": " << iterations << ",\n";
    outfile << "  \"water_transport_seconds\": " << time_water_sec << ",\n";
    outfile << "  \"water_transport_us_per_call\": " << (time_water_sec / iterations * 1e6) << ",\n";
    outfile << "  \"unity_lewis_seconds\": " << time_ul_sec << ",\n";
    outfile << "  \"unity_lewis_us_per_call\": " << (time_ul_sec / iterations * 1e6) << ",\n";
    outfile << "  \"ion_gas_seconds\": " << time_ion_sec << ",\n";
    outfile << "  \"ion_gas_us_per_call\": " << (time_ion_sec / iterations * 1e6) << "\n";
    outfile << "}\n";

    return 0;
}
