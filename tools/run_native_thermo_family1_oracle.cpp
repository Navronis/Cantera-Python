#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/base/global.h"

using namespace Cantera;
using namespace std;

int main() {
    ofstream out("audit/native_thermo_family1_oracle_results.json");
    out << "{\n  \"pinned_commit\": \"726522be4e2a13454d8415b7ef799d621f665cf3\",\n  \"phases\": [\n";

    vector<string> files = {"gri30.yaml", "air.yaml", "h2o2.yaml"};
    for (size_t f_idx = 0; f_idx < files.size(); f_idx++) {
        string fname = files[f_idx];
        auto sol = newSolution(fname);
        auto p = sol->thermo();
        p->setState_TD(1200.0, 0.35);

        out << "    {\n";
        out << "      \"file\": \"" << fname << "\",\n";
        out << "      \"nElements\": " << p->nElements() << ",\n";
        out << "      \"nSpecies\": " << p->nSpecies() << ",\n";
        out << "      \"temperature\": " << setprecision(17) << p->temperature() << ",\n";
        out << "      \"density\": " << setprecision(17) << p->density() << ",\n";
        out << "      \"molarDensity\": " << setprecision(17) << p->molarDensity() << ",\n";
        out << "      \"meanMolecularWeight\": " << setprecision(17) << p->meanMolecularWeight() << ",\n";

        // Element names and atomic weights
        out << "      \"elementNames\": [";
        for (size_t m = 0; m < p->nElements(); m++) {
            if (m > 0) out << ", ";
            out << "\"" << p->elementName(m) << "\"";
        }
        out << "],\n";

        out << "      \"atomicWeights\": [";
        for (size_t m = 0; m < p->nElements(); m++) {
            if (m > 0) out << ", ";
            out << setprecision(17) << p->atomicWeight(m);
        }
        out << "],\n";

        // Sample nAtoms
        out << "      \"nAtoms_sample\": [";
        size_t count = 0;
        for (size_t k = 0; k < min((size_t)5, p->nSpecies()); k++) {
            for (size_t m = 0; m < p->nElements(); m++) {
                if (count > 0) out << ", ";
                out << p->nAtoms(k, m);
                count++;
            }
        }
        out << "]\n";
        out << "    }" << (f_idx + 1 < files.size() ? ",\n" : "\n");
    }

    out << "  ]\n}\n";
    cout << "Executed native thermo family 1 oracle across " << files.size() << " mechanisms." << endl;
    return 0;
}
