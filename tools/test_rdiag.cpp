#include <iostream>
#include <memory>
#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/kinetics/ReactionPath.h"

using namespace Cantera;
using namespace std;

int main() {
    auto sol = newSolution("h2o2.yaml");
    sol->thermo()->setState_TPX(1100.0, 10.0 * OneAtm, "H:0.1, HO2:0.1, AR:6.0");
    auto diag = newReactionPathDiagram(sol->kinetics(), "H");
    diag->build();
    cout << "maxFlow: " << diag->maxFlow() << endl;
    cout << "nNodes: " << diag->nNodes() << endl;
    cout << "nPaths: " << diag->nPaths() << endl;
    cout << "species count: " << diag->species().size() << endl;
    cout << "reactions count: " << diag->reactions().size() << endl;
    cout << "data:\n" << diag->getData() << endl;
    cout << "dot:\n" << diag->getDot() << endl;
    cout << "log:\n" << diag->getLog() << endl;
    return 0;
}
