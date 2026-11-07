#include <iostream>
#include "cantera/base/Interface.h"
#include "cantera/thermo/SurfPhase.h"
#include "cantera/kinetics/InterfaceKinetics.h"
#include "cantera/kinetics/EdgeKinetics.h"
using namespace Cantera;
using namespace std;

int main() {
    try {
        auto metal = newSolution("sofc.yaml", "metal");
        auto metal_surface = newInterface("sofc.yaml", "metal_surface", {metal});
        auto oxide_bulk = newSolution("sofc.yaml", "oxide_bulk");
        auto oxide_surface = newInterface("sofc.yaml", "oxide_surface", {oxide_bulk});
        auto tpb = newInterface("sofc.yaml", "tpb", {metal, metal_surface, oxide_surface});
        cout << "tpb (edge): rxns=" << tpb->kinetics()->nReactions()
             << " kineticsType=" << tpb->kinetics()->kineticsType()
             << " nPhases=" << tpb->kinetics()->nPhases() << endl;
    } catch (const exception& e) {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
    return 0;
}
