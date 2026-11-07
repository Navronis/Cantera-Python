/**
 * Pinned native Cantera Boundary1D performance benchmark.
 * Measures execution timing over 10^5 evaluations for canonical boundary conditions.
 */

#include <iostream>
#include <vector>
#include <span>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <memory>

#include "cantera/base/Solution.h"
#include "cantera/oneD/Boundary1D.h"
#include "cantera/oneD/Flow1D.h"
#include "cantera/oneD/DomainFactory.h"
#include "cantera/oneD/OneDim.h"

using namespace Cantera;
using namespace std;
using namespace std::chrono;

struct BenchResult {
    string name;
    int reps;
    double mean_us;
    double total_ms;
};

int main()
{
    int reps = 100000;
    vector<BenchResult> results;

    // Load solutions
    auto sol_gas = newSolution("h2o2.yaml");
    sol_gas->thermo()->setState_TPX(1000.0, 101325.0, "H2:2.0, O2:1.0, AR:0.5");

    auto sol_pt_gas = newSolution("ptcombust.yaml", "gas");
    sol_pt_gas->thermo()->setState_TPX(1200.0, 101325.0, "CH4:0.1, O2:0.2, N2:0.7");
    auto sol_surf = newSolution("ptcombust.yaml", "Pt_surf", "none", {sol_pt_gas});

    // 1. Inlet1D
    {
        auto inlet = newDomain<Inlet1D>("inlet", sol_gas);
        auto flow = newFlow1D("unstrained-flow", sol_gas);
        OneDim container;
        container.addDomain(inlet);
        container.addDomain(flow);
        vector<double> z = {0.0, 0.02, 0.04, 0.06, 0.08, 0.10};
        flow->setupGrid(z);
        container.resize();
        container.init();
        container.resize();

        vector<double> x(container.size(), 1.0);
        size_t nc_gas = flow->nComponents();
        for (size_t pt = flow->loc(); pt < flow->loc() + flow->size(); pt += nc_gas) {
            x[pt + c_offset_T] = 1000.0;
        }
        vector<double> r(container.size(), 0.0);
        vector<int> mask(container.size(), 0);

        // Warmup
        for (int i = 0; i < 1000; ++i) {
            inlet->eval(0, std::span(x), std::span(r), std::span(mask), 0.0);
        }

        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            inlet->eval(0, std::span(x), std::span(r), std::span(mask), 0.0);
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double mean_us = (total_ms * 1000.0) / reps;
        results.push_back({"Inlet1D", reps, mean_us, total_ms});
    }

    // 2. Outlet1D
    {
        auto flow = newFlow1D("unstrained-flow", sol_gas);
        auto outlet = newDomain<Outlet1D>("outlet", sol_gas);
        OneDim container;
        container.addDomain(flow);
        container.addDomain(outlet);
        vector<double> z = {0.0, 0.02, 0.04, 0.06, 0.08, 0.10};
        flow->setupGrid(z);
        container.resize();
        container.init();
        container.resize();

        vector<double> x(container.size(), 1.0);
        size_t nc_gas = flow->nComponents();
        for (size_t pt = flow->loc(); pt < flow->loc() + flow->size(); pt += nc_gas) {
            x[pt + c_offset_T] = 1000.0;
        }
        vector<double> r(container.size(), 0.0);
        vector<int> mask(container.size(), 0);

        for (int i = 0; i < 1000; ++i) {
            outlet->eval(flow->nPoints() - 1, std::span(x), std::span(r), std::span(mask), 0.0);
        }

        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            outlet->eval(flow->nPoints() - 1, std::span(x), std::span(r), std::span(mask), 0.0);
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double mean_us = (total_ms * 1000.0) / reps;
        results.push_back({"Outlet1D", reps, mean_us, total_ms});
    }

    // 3. OutletRes1D
    {
        auto flow = newFlow1D("unstrained-flow", sol_gas);
        auto outlet_res = newDomain<OutletRes1D>("outlet-reservoir", sol_gas);
        OneDim container;
        container.addDomain(flow);
        container.addDomain(outlet_res);
        vector<double> z = {0.0, 0.02, 0.04, 0.06, 0.08, 0.10};
        flow->setupGrid(z);
        container.resize();
        outlet_res->setMoleFractions("H2:0.5, O2:0.5");
        container.init();
        container.resize();

        vector<double> x(container.size(), 1.0);
        size_t nc_gas = flow->nComponents();
        for (size_t pt = flow->loc(); pt < flow->loc() + flow->size(); pt += nc_gas) {
            x[pt + c_offset_T] = 1000.0;
        }
        vector<double> r(container.size(), 0.0);
        vector<int> mask(container.size(), 0);

        for (int i = 0; i < 1000; ++i) {
            outlet_res->eval(flow->nPoints() - 1, std::span(x), std::span(r), std::span(mask), 0.0);
        }

        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            outlet_res->eval(flow->nPoints() - 1, std::span(x), std::span(r), std::span(mask), 0.0);
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double mean_us = (total_ms * 1000.0) / reps;
        results.push_back({"OutletRes1D", reps, mean_us, total_ms});
    }

    // 4. Symm1D
    {
        auto symm = newDomain<Symm1D>("symmetry-plane", sol_gas);
        auto flow = newFlow1D("unstrained-flow", sol_gas);
        OneDim container;
        container.addDomain(symm);
        container.addDomain(flow);
        vector<double> z = {0.0, 0.02, 0.04, 0.06, 0.08, 0.10};
        flow->setupGrid(z);
        container.resize();
        container.init();
        container.resize();

        vector<double> x(container.size(), 1.0);
        size_t nc_gas = flow->nComponents();
        for (size_t pt = flow->loc(); pt < flow->loc() + flow->size(); pt += nc_gas) {
            x[pt + c_offset_T] = 1000.0;
        }
        vector<double> r(container.size(), 0.0);
        vector<int> mask(container.size(), 0);

        for (int i = 0; i < 1000; ++i) {
            symm->eval(0, std::span(x), std::span(r), std::span(mask), 0.0);
        }

        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            symm->eval(0, std::span(x), std::span(r), std::span(mask), 0.0);
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double mean_us = (total_ms * 1000.0) / reps;
        results.push_back({"Symm1D", reps, mean_us, total_ms});
    }

    // 5. Surf1D
    {
        auto flow = newFlow1D("unstrained-flow", sol_gas);
        auto surf = newDomain<Surf1D>("surface", sol_gas);
        OneDim container;
        container.addDomain(flow);
        container.addDomain(surf);
        vector<double> z = {0.0, 0.02, 0.04, 0.06, 0.08, 0.10};
        flow->setupGrid(z);
        container.resize();
        surf->setTemperature(1000.0);
        container.init();
        container.resize();

        vector<double> x(container.size(), 1.0);
        size_t nc_gas = flow->nComponents();
        for (size_t pt = flow->loc(); pt < flow->loc() + flow->size(); pt += nc_gas) {
            x[pt + c_offset_T] = 1000.0;
        }
        vector<double> r(container.size(), 0.0);
        vector<int> mask(container.size(), 0);

        for (int i = 0; i < 1000; ++i) {
            surf->eval(flow->nPoints() - 1, std::span(x), std::span(r), std::span(mask), 0.0);
        }

        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            surf->eval(flow->nPoints() - 1, std::span(x), std::span(r), std::span(mask), 0.0);
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double mean_us = (total_ms * 1000.0) / reps;
        results.push_back({"Surf1D", reps, mean_us, total_ms});
    }

    // 6. ReactingSurf1D
    {
        auto flow = newFlow1D("unstrained-flow", sol_pt_gas);
        auto rsurf = newDomain<ReactingSurf1D>("reacting-surface", sol_surf);
        OneDim container;
        container.addDomain(flow);
        container.addDomain(rsurf);
        vector<double> z = {0.0, 0.02, 0.04, 0.06, 0.08, 0.10};
        flow->setupGrid(z);
        container.resize();
        rsurf->setTemperature(1200.0);
        container.init();
        container.resize();

        vector<double> x(container.size(), 1.0);
        size_t nc_gas = flow->nComponents();
        size_t nsp_gas = sol_pt_gas->thermo()->nSpecies();
        for (size_t pt = 0; pt < flow->nPoints() * nc_gas; pt += nc_gas) {
            x[pt + c_offset_U] = 0.5;
            x[pt + c_offset_V] = 0.0;
            x[pt + c_offset_T] = 1200.0;
            x[pt + c_offset_L] = 0.0;
            for (size_t k = 0; k < nsp_gas; k++) {
                x[pt + c_offset_Y + k] = 1.0 / nsp_gas;
            }
        }
        for (size_t n = 0; n < rsurf->nSpecies(); ++n) {
            x[rsurf->loc() + n] = 1.0 / rsurf->nSpecies();
        }
        vector<double> r(container.size(), 0.0);
        vector<int> mask(container.size(), 0);

        for (int i = 0; i < 1000; ++i) {
            rsurf->eval(npos, std::span(x), std::span(r), std::span(mask), 0.0);
        }

        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            rsurf->eval(npos, std::span(x), std::span(r), std::span(mask), 0.0);
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double mean_us = (total_ms * 1000.0) / reps;
        results.push_back({"ReactingSurf1D", reps, mean_us, total_ms});
    }

    // Output JSON
    ofstream ofs("audit/boundary1d_native_benchmark_results.json");
    ofs << "{\n  \"benchmarks\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        ofs << "    {\n";
        ofs << "      \"name\": \"" << results[i].name << "\",\n";
        ofs << "      \"reps\": " << results[i].reps << ",\n";
        ofs << "      \"mean_us\": " << fixed << setprecision(4) << results[i].mean_us << ",\n";
        ofs << "      \"total_ms\": " << fixed << setprecision(3) << results[i].total_ms << "\n";
        ofs << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n}\n";

    cout << "NAME,REPS,MEAN_US,TOTAL_MS" << endl;
    for (const auto& r : results) {
        cout << r.name << "," << r.reps << "," << fixed << setprecision(4) << r.mean_us << "," << setprecision(2) << r.total_ms << endl;
    }

    return 0;
}
