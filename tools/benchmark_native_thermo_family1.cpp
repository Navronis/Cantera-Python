#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"

using namespace Cantera;
using namespace std;
using namespace std::chrono;

int main() {
    const int iterations = 100000;
    cout << "Benchmarking Native Cantera Family 1 Phase Core (" << iterations << " iterations)..." << endl;

    auto sol = newSolution("gri30.yaml");
    auto p = sol->thermo();

    // 1. setState_TD benchmark
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        double T = 300.0 + (i % 1000);
        double rho = 0.1 + (i % 50) * 0.01;
        p->setState_TD(T, rho);
    }
    auto t1 = high_resolution_clock::now();
    double time_state_sec = duration_cast<duration<double>>(t1 - t0).count();
    double us_state = time_state_sec / iterations * 1e6;
    cout << "setState_TD: " << time_state_sec << " s (" << us_state << " us/call)" << endl;

    // 2. setMoleFractions benchmark
    vector<double> x(p->nSpecies(), 0.0);
    x[0] = 0.1; x[1] = 0.2; x[2] = 0.7;
    t0 = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        x[0] = 0.1 + (i % 10) * 0.01;
        p->setMoleFractions(span<const double>(x));
    }
    t1 = high_resolution_clock::now();
    double time_mf_sec = duration_cast<duration<double>>(t1 - t0).count();
    double us_mf = time_mf_sec / iterations * 1e6;
    cout << "setMoleFractions: " << time_mf_sec << " s (" << us_mf << " us/call)" << endl;

    // 3. saveState / restoreState benchmark
    vector<double> state(p->stateSize());
    t0 = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        p->saveState(span<double>(state));
        p->restoreState(span<const double>(state));
    }
    t1 = high_resolution_clock::now();
    double time_sr_sec = duration_cast<duration<double>>(t1 - t0).count();
    double us_sr = time_sr_sec / iterations * 1e6;
    cout << "saveState/restoreState: " << time_sr_sec << " s (" << us_sr << " us/call)" << endl;

    ofstream out("audit/native_thermo_family1_benchmark.json");
    out << "{\n";
    out << "  \"iterations\": " << iterations << ",\n";
    out << "  \"setState_TD_us_per_call\": " << us_state << ",\n";
    out << "  \"setMoleFractions_us_per_call\": " << us_mf << ",\n";
    out << "  \"saveRestoreState_us_per_call\": " << us_sr << "\n";
    out << "}\n";

    return 0;
}
