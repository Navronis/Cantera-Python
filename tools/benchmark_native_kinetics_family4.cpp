#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>

#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/kinetics/Kinetics.h"
#include "cantera/kinetics/ReactionPath.h"

using namespace Cantera;
using namespace std;
using namespace std::chrono;

int main() {
    ofstream out("audit/native_kinetics_family4_benchmark.json");
    if (!out.is_open()) {
        cerr << "Failed to open output file" << endl;
        return 1;
    }

    out << "{\n  \"pinned_commit\": \"726522be4e2a13454d8415b7ef799d621f665cf3\",\n  \"benchmarks\": [\n";

    // Benchmark 1: gri30 C (50 iterations)
    {
        auto sol = newSolution("gri30.yaml");
        sol->thermo()->setState_TPX(1500.0, 101325.0, "CH4:0.1, O2:0.2, N2:0.7");
        auto diag = newReactionPathDiagram(sol->kinetics(), "C");
        diag->build(); // warm up

        int iters = 50;
        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            diag->build();
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double avg_ms = total_ms / iters;

        out << "    {\n      \"name\": \"gri30_C_build\",\n      \"iterations\": " << iters
            << ",\n      \"total_ms\": " << setprecision(8) << total_ms
            << ",\n      \"avg_ms\": " << setprecision(8) << avg_ms << "\n    },\n";
        cout << "Native gri30 C: " << avg_ms << " ms/iter" << endl;
    }

    // Benchmark 2: h2o2 H (500 iterations)
    {
        auto sol = newSolution("h2o2.yaml");
        sol->thermo()->setState_TPX(1000.0, 101325.0, "H2:2.0, O2:1.0, AR:7.0");
        auto diag = newReactionPathDiagram(sol->kinetics(), "H");
        diag->build(); // warm up

        int iters = 500;
        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            diag->build();
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double avg_ms = total_ms / iters;

        out << "    {\n      \"name\": \"h2o2_H_build\",\n      \"iterations\": " << iters
            << ",\n      \"total_ms\": " << setprecision(8) << total_ms
            << ",\n      \"avg_ms\": " << setprecision(8) << avg_ms << "\n    },\n";
        cout << "Native h2o2 H: " << avg_ms << " ms/iter" << endl;
    }

    // Benchmark 3: ptcombust C (100 iterations)
    {
        auto sol = newSolution("ptcombust.yaml", "gas");
        sol->thermo()->setState_TPX(1000.0, 101325.0, "CH4:0.1, O2:0.2, AR:0.7");
        auto diag = newReactionPathDiagram(sol->kinetics(), "C");
        diag->build(); // warm up

        int iters = 100;
        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            diag->build();
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double avg_ms = total_ms / iters;

        out << "    {\n      \"name\": \"ptcombust_C_build\",\n      \"iterations\": " << iters
            << ",\n      \"total_ms\": " << setprecision(8) << total_ms
            << ",\n      \"avg_ms\": " << setprecision(8) << avg_ms << "\n    }\n";
        cout << "Native ptcombust C: " << avg_ms << " ms/iter" << endl;
    }

    out << "  ]\n}\n";
    cout << "Finished native benchmarks." << endl;
    return 0;
}
