#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include "cantera/base/Solution.h"
#include "cantera/kinetics/Kinetics.h"
#include "cantera/thermo/ThermoPhase.h"

using namespace Cantera;
using namespace std;
using namespace std::chrono;

double benchmarkMech(const string& file, const string& phase, const string& comp, double T, double P, int iters) {
    shared_ptr<Solution> sol;
    if (phase.empty()) sol = newSolution(file);
    else sol = newSolution(file, phase);

    auto thermo = sol->thermo();
    auto kin = sol->kinetics();
    thermo->setState_TPX(T, P, comp);

    size_t nRxn = kin->nReactions();
    size_t nSp = kin->nTotalSpecies();
    vector<double> wdot(nSp);
    vector<double> ropnet(nRxn);

    // Warm-up
    for (int i = 0; i < 50; ++i) {
        kin->getNetRatesOfProgress(ropnet);
        kin->getNetProductionRates(wdot);
    }

    vector<double> times;
    times.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        auto t0 = high_resolution_clock::now();
        kin->getNetRatesOfProgress(ropnet);
        kin->getNetProductionRates(wdot);
        auto t1 = high_resolution_clock::now();
        times.push_back(duration_cast<nanoseconds>(t1 - t0).count() / 1e6);
    }

    sort(times.begin(), times.end());
    return times[times.size() / 2]; // median ms
}

int main() {
    cout << "=== Native C++ Kinetics Family 2 Benchmark ===" << endl;
    cout << "gri30_median_ms: " << benchmarkMech("gri30.yaml", "", "CH4:0.095, O2:0.19, N2:0.715", 1500.0, 101325.0, 500) << endl;
    cout << "pdep_median_ms: " << benchmarkMech("pdep-test.yaml", "", "H:0.1, R1A:0.1, R1B:0.1, P1:0.1, R2:0.1, P2A:0.1, P2B:0.1, R3:0.1, P3A:0.1, P3B:0.1", 1200.0, 101325.0, 500) << endl;
    cout << "sri_median_ms: " << benchmarkMech("sri-falloff.yaml", "", "H:0.1, R1A:0.2, R1B:0.2, P1:0.1, R2:0.1, P2A:0.1, P2B:0.1, R3:0.1", 1000.0, 101325.0, 500) << endl;
    cout << "burke_median_ms: " << benchmarkMech("linearBurke-test.yaml", "", "H:0.1, OH:0.1, H2O:0.3, O2:0.2, HO2:0.1, H2O2:0.2", 1500.0, 101325.0, 500) << endl;
    return 0;
}
