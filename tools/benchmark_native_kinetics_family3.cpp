#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <span>

#include "cantera/base/Interface.h"
#include "cantera/kinetics/InterfaceKinetics.h"
#include "cantera/thermo/SurfPhase.h"

using namespace Cantera;
using namespace std;
using namespace std::chrono;

double benchmarkInterface(const string& file, const string& phase, const string& gas_phase,
                          const string& gas_comp, const string& surf_cov,
                          double T, double P, int iters) {
    auto iface = newInterface(file, phase);
    auto thermo = iface->thermo();
    auto kin = iface->kinetics();

    if (!gas_phase.empty()) {
        auto gas = iface->adjacent(gas_phase);
        if (gas) {
            gas->thermo()->setState_TPX(T, P, gas_comp);
        }
    }
    thermo->setState_TP(T, P);
    thermo->setCoveragesByName(surf_cov);

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
    cout << "=== Native C++ Kinetics Family 3 Benchmark ===" << endl;
    cout << "ptcombust_median_ms: " << benchmarkInterface("ptcombust.yaml", "Pt_surf", "gas", "CH4:0.095, O2:0.19, AR:0.715", "PT(S):0.6, C(S):0.1, CO(S):0.1, CO2(S):0.1, H2O(S):0.1", 1000.0, 101325.0, 500) << endl;
    cout << "ptcombust_motzwise_median_ms: " << benchmarkInterface("ptcombust-motzwise.yaml", "Pt_surf", "gas", "CH4:0.095, O2:0.19, AR:0.715", "PT(S):0.6, H(S):0.2, O(S):0.2", 1000.0, 101325.0, 500) << endl;
    cout << "blowers_masel_median_ms: " << benchmarkInterface("blowers-masel.yaml", "Pt_surf", "gas", "H2:0.4, O2:0.2, H2O:0.1, AR:0.3", "PT(S):0.4, H(S):0.2, O(S):0.2, OH(S):0.1, H2O(S):0.1", 1000.0, 101325.0, 500) << endl;
    cout << "sofc_median_ms: " << benchmarkInterface("sofc.yaml", "metal_surface", "gas", "H2:0.95, H2O:0.05", "(m):0.5, H(m):0.5", 1073.15, 101325.0, 500) << endl;
    return 0;
}
