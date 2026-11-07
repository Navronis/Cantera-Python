/**
 * Benchmark Native Cantera Flow1D / StFlow Performance
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <memory>

#include "cantera/oneD/Flow1D.h"
#include "cantera/oneD/Boundary1D.h"
#include "cantera/oneD/OneDim.h"
#include "cantera/base/Solution.h"

using namespace Cantera;
using namespace std;
using namespace std::chrono;

struct BenchmarkResult {
    string name;
    string domain_type;
    int points;
    int reps;
    double median_us;
    double mean_us;
    double min_us;
    double max_us;
};

BenchmarkResult run_benchmark(
    const string& name,
    const string& domain_type,
    bool energy_enabled,
    bool radiation_enabled,
    bool soret_enabled,
    int points,
    int reps
) {
    auto sol = newSolution("h2o2.yaml", "", "mixture-averaged");
    auto inlet = make_shared<Inlet1D>(sol, "inlet");
    auto flow = make_shared<Flow1D>(sol, "flame", points);
    auto outlet = make_shared<Outlet1D>(sol, "outlet");

    vector<double> grid(points);
    for (int j = 0; j < points; j++) {
        grid[j] = 0.02 * j / (points - 1);
    }
    flow->setupGrid(grid);
    flow->setPressure(101325.0);

    if (domain_type == "free-flow") {
        flow->setFreeFlow();
    } else if (domain_type == "axisymmetric-flow") {
        flow->setAxisymmetricFlow();
    } else if (domain_type == "unstrained-flow") {
        flow->setUnstrainedFlow();
    }

    if (energy_enabled) {
        flow->solveEnergyEqn();
    } else {
        flow->fixTemperature();
    }

    flow->enableRadiation(radiation_enabled);
    if (radiation_enabled) {
        flow->setBoundaryEmissivities(0.1, 0.1);
    }
    flow->enableSoret(soret_enabled);

    vector<shared_ptr<Domain1D>> domains = {inlet, flow, outlet};
    OneDim sim(domains);
    sim.init();

    vector<double> x_global(sim.size(), 0.0);
    vector<double> r_global(sim.size(), 0.0);
    vector<int> mask_global(sim.size(), 0);

    size_t flow_loc = flow->loc();
    size_t nc = flow->nComponents();
    size_t nsp = sol->thermo()->nSpecies();

    // Populate realistic state profile
    for (int j = 0; j < points; j++) {
        double frac = double(j) / double(points - 1);
        double T = 300.0 + (2200.0 - 300.0) * frac;
        double u = 0.2 + (2.0 - 0.2) * frac;
        double V = 0.0;
        double L = 0.0;
        double Uo = 0.0;

        size_t base = flow_loc + j * nc;
        x_global[base + 0] = u;
        x_global[base + 1] = V;
        x_global[base + 2] = T;
        x_global[base + 3] = L;
        x_global[base + 4] = 0.0; // E
        x_global[base + 5] = Uo;

        // Species: simple consumption profile H2 + O2 -> H2O
        for (size_t k = 0; k < nsp; k++) {
            string sp = sol->thermo()->speciesName(k);
            double Yk = 0.01;
            if (sp == "H2") Yk = 0.028 * (1.0 - 0.9 * frac);
            else if (sp == "O2") Yk = 0.226 * (1.0 - 0.9 * frac);
            else if (sp == "H2O") Yk = 0.15 * frac;
            else if (sp == "N2") Yk = 0.73;
            else if (sp == "AR") Yk = 0.70 * (1.0 - 0.5 * frac);
            x_global[base + 6 + k] = Yk;
        }
    }

    // Warmup
    for (int r = 0; r < 50; r++) {
        flow->eval(npos, x_global, r_global, mask_global, 0.0);
    }

    vector<double> durations;
    durations.reserve(reps);

    for (int r = 0; r < reps; r++) {
        auto t0 = high_resolution_clock::now();
        flow->eval(npos, x_global, r_global, mask_global, 0.0);
        auto t1 = high_resolution_clock::now();
        double dur = duration<double, micro>(t1 - t0).count();
        durations.push_back(dur);
    }

    sort(durations.begin(), durations.end());
    double median_us = durations[durations.size() / 2];
    double mean_us = accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
    double min_us = durations.front();
    double max_us = durations.back();

    cout << "  [" << name << "] Median: " << fixed << setprecision(2)
         << median_us << " us, Mean: " << mean_us << " us" << endl;

    return {name, domain_type, points, reps, median_us, mean_us, min_us, max_us};
}

int main() {
    cout << "Starting Native Flow1D benchmarks..." << endl;
    vector<BenchmarkResult> results;

    results.push_back(run_benchmark("FreeFlow_10pts", "free-flow", true, false, false, 10, 5000));
    results.push_back(run_benchmark("AxisymmetricFlow_10pts_Rad", "axisymmetric-flow", true, true, false, 10, 5000));
    results.push_back(run_benchmark("UnstrainedFlow_10pts", "unstrained-flow", true, false, false, 10, 5000));
    results.push_back(run_benchmark("FreeFlow_25pts", "free-flow", true, false, false, 25, 2000));

    ofstream out("audit/flow1d_native_benchmark_results.json");
    out << "{\n  \"benchmarks\": [\n";
    for (size_t i = 0; i < results.size(); i++) {
        const auto& b = results[i];
        out << "    {\n"
            << "      \"name\": \"" << b.name << "\",\n"
            << "      \"domain_type\": \"" << b.domain_type << "\",\n"
            << "      \"points\": " << b.points << ",\n"
            << "      \"reps\": " << b.reps << ",\n"
            << "      \"median_us\": " << fixed << setprecision(3) << b.median_us << ",\n"
            << "      \"mean_us\": " << b.mean_us << ",\n"
            << "      \"min_us\": " << b.min_us << ",\n"
            << "      \"max_us\": " << b.max_us << "\n"
            << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    out.close();

    cout << "Wrote native benchmark results to audit/flow1d_native_benchmark_results.json" << endl;
    return 0;
}
