#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/MultiSpeciesThermo.h"
#include "cantera/thermo/SpeciesThermoInterpType.h"

using namespace Cantera;
using namespace std;

int main() {
    ofstream out("audit/native_thermo_family2_oracle_results.json");
    out << "{\n  \"pinned_commit\": \"726522be4e2a13454d8415b7ef799d621f665cf3\",\n  \"species_evaluations\": [\n";

    auto sol = newSolution("gri30.yaml");
    auto p = sol->thermo();

    vector<string> test_species = {"CH4", "O2", "CO2", "H2O", "H2", "OH"};
    vector<double> temperatures = {300.0, 500.0, 1000.0, 1200.0, 1500.0, 2000.0, 2500.0};

    bool first_spec = true;
    for (const auto& sp_name : test_species) {
        size_t k = p->speciesIndex(sp_name);
        if (!first_spec) out << ",\n";
        first_spec = false;

        out << "    {\n";
        out << "      \"species\": \"" << sp_name << "\",\n";
        out << "      \"points\": [\n";

        for (size_t t_idx = 0; t_idx < temperatures.size(); t_idx++) {
            double T = temperatures[t_idx];
            p->setState_TP(T, 101325.0);

            vector<double> cp_R(p->nSpecies(), 0.0);
            vector<double> h_RT(p->nSpecies(), 0.0);
            vector<double> s_R(p->nSpecies(), 0.0);

            p->speciesThermo().update(T, span<double>(cp_R), span<double>(h_RT), span<double>(s_R));

            if (t_idx > 0) out << ",\n";
            out << "        {\n";
            out << "          \"temperature\": " << T << ",\n";
            out << "          \"cp_R\": " << setprecision(17) << cp_R[k] << ",\n";
            out << "          \"h_RT\": " << setprecision(17) << h_RT[k] << ",\n";
            out << "          \"s_R\": " << setprecision(17) << s_R[k] << "\n";
            out << "        }";
        }
        out << "\n      ]\n    }";
    }

    out << "\n  ]\n}\n";
    cout << "Wrote native SpeciesThermo oracle results to audit/native_thermo_family2_oracle_results.json" << endl;
    return 0;
}
