#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <span>

#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/kinetics/Kinetics.h"
#include "cantera/kinetics/TwoTempPlasmaRate.h"

using namespace Cantera;
using namespace std;
using namespace std::chrono;

int main() {
    ofstream out("audit/native_kinetics_family5_benchmark.json");
    if (!out.is_open()) {
        cerr << "Failed to open output file" << endl;
        return 1;
    }

    out << "{\n  \"pinned_commit\": \"726522be4e2a13454d8415b7ef799d621f665cf3\",\n  \"benchmarks\": [\n";

    // Benchmark 1: TwoTempPlasmaRate evaluation (100,000 iterations)
    {
        TwoTempPlasmaRate ttRate;
        AnyMap node;
        node["rate-constant"] = AnyMap();
        node["rate-constant"]["A"] = 1.0e13;
        node["rate-constant"]["b"] = 0.5;
        node["rate-constant"]["Ea-gas"] = 1.0e7;
        node["rate-constant"]["Ea-electron"] = 2.0e7;
        node["rate-constant"]["b-gas"] = 0.1;
        ttRate.setParameters(node, UnitStack({}));

        TwoTempPlasmaData data;
        data.update(1200.0, 25000.0);
        ttRate.evalFromStruct(data); // warm up

        int iters = 100000;
        double dummy = 0.0;
        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            dummy += ttRate.evalFromStruct(data);
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double avg_ms = total_ms / iters;

        out << "    {\n      \"name\": \"twotemp_plasma_eval\",\n      \"iterations\": " << iters
            << ",\n      \"total_ms\": " << setprecision(8) << total_ms
            << ",\n      \"avg_ms\": " << setprecision(8) << avg_ms << "\n    },\n";
        cout << "Native twotemp_plasma_eval: " << avg_ms << " ms/iter (dummy=" << dummy << ")" << endl;
    }

    // Benchmark 2: Oxygen plasma rates (TwoTemp + ElectronCollision) (10,000 iterations)
    {
        auto sol = newSolution("oxygen-plasma.yaml", "", "none");
        sol->thermo()->setState_TP(1200.0, 101325.0);
        auto kin = sol->kinetics();
        vector<double> kf(kin->nReactions(), 0.0);
        kin->getFwdRateConstants(span<double>(kf)); // warm up

        int iters = 10000;
        double dummy = 0.0;
        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            kin->getFwdRateConstants(span<double>(kf));
            dummy += kf[0] + kf[1];
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double avg_ms = total_ms / iters;

        out << "    {\n      \"name\": \"oxygen_plasma_rates\",\n      \"iterations\": " << iters
            << ",\n      \"total_ms\": " << setprecision(8) << total_ms
            << ",\n      \"avg_ms\": " << setprecision(8) << avg_ms << "\n    },\n";
        cout << "Native oxygen_plasma_rates: " << avg_ms << " ms/iter (dummy=" << dummy << ")" << endl;
    }

    // Benchmark 3: Vibrational relaxation rates (10,000 iterations)
    {
        auto sol = newSolution("vibrational-relaxation.yaml", "", "none");
        sol->thermo()->setState_TP(1500.0, 101325.0);
        auto kin = sol->kinetics();
        vector<double> kf(kin->nReactions(), 0.0);
        kin->getFwdRateConstants(span<double>(kf)); // warm up

        int iters = 10000;
        double dummy = 0.0;
        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            kin->getFwdRateConstants(span<double>(kf));
            dummy += kf[0];
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double avg_ms = total_ms / iters;

        out << "    {\n      \"name\": \"vibrational_relaxation_rates\",\n      \"iterations\": " << iters
            << ",\n      \"total_ms\": " << setprecision(8) << total_ms
            << ",\n      \"avg_ms\": " << setprecision(8) << avg_ms << "\n    },\n";
        cout << "Native vibrational_relaxation_rates: " << avg_ms << " ms/iter (dummy=" << dummy << ")" << endl;
    }

    // Benchmark 4: Kinetics getNetProductionRates_ddT (1,000 iterations)
    {
        auto sol = newSolution("h2o2.yaml", "", "none");
        sol->thermo()->setState_TPX(1200.0, 506625.0, "H2:0.5, O2:0.5");
        auto kin = sol->kinetics();
        vector<double> dwdot(kin->nTotalSpecies(), 0.0);
        kin->getNetProductionRates_ddT(span<double>(dwdot)); // warm up

        int iters = 1000;
        double dummy = 0.0;
        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            kin->getNetProductionRates_ddT(span<double>(dwdot));
            dummy += dwdot[0];
        }
        auto t1 = high_resolution_clock::now();
        double total_ms = duration<double, milli>(t1 - t0).count();
        double avg_ms = total_ms / iters;

        out << "    {\n      \"name\": \"kinetics_derivatives_ddT\",\n      \"iterations\": " << iters
            << ",\n      \"total_ms\": " << setprecision(8) << total_ms
            << ",\n      \"avg_ms\": " << setprecision(8) << avg_ms << "\n    }\n";
        cout << "Native kinetics_derivatives_ddT: " << avg_ms << " ms/iter (dummy=" << dummy << ")" << endl;
    }

    out << "  ]\n}\n";
    cout << "Finished native benchmarks." << endl;
    return 0;
}
