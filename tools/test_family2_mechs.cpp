#include <iostream>
#include <string>
#include <vector>
#include "cantera/base/Solution.h"
#include "cantera/kinetics/Kinetics.h"
#include "cantera/thermo/ThermoPhase.h"

using namespace Cantera;
using namespace std;

int main() {
    vector<pair<string, string>> mechs = {
        {"gri30.yaml", ""},
        {"sri-falloff.yaml", ""},
        {"tsang-falloff.yaml", ""},
        {"chemically-activated-reaction.yaml", ""},
        {"pdep-test.yaml", ""},
        {"linearBurke-test.yaml", ""},
        {"blowers-masel.yaml", "gas"}
    };

    for (const auto& [file, phase] : mechs) {
        try {
            shared_ptr<Solution> sol;
            if (phase.empty()) {
                sol = newSolution(file);
            } else {
                sol = newSolution(file, phase);
            }
            cout << "SUCCESS: " << file << " (species=" << sol->thermo()->nSpecies()
                 << ", rxns=" << sol->kinetics()->nReactions() << ")" << endl;
        } catch (const exception& e) {
            cout << "FAILED: " << file << " -> " << e.what() << endl;
        }
    }
    return 0;
}
