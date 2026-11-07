#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <span>
#include "cantera/base/Interface.h"
#include "cantera/thermo/SurfPhase.h"
#include "cantera/kinetics/InterfaceKinetics.h"

using namespace Cantera;
using namespace std;

int main() {
    try {
        cout << "1. ptcombust.yaml (Pt_surf)..." << endl;
        auto s1 = newInterface("ptcombust.yaml", "Pt_surf");
        cout << "   s1: " << s1->thermo()->name() << ", nRxn: " << s1->kinetics()->nReactions() << endl;

        cout << "2. ptcombust-motzwise.yaml (Pt_surf)..." << endl;
        auto s2 = newInterface("ptcombust-motzwise.yaml", "Pt_surf");
        cout << "   s2: " << s2->thermo()->name() << ", nRxn: " << s2->kinetics()->nReactions() << endl;

        cout << "3. sticking_coeff_check.yaml (Pt_surf)..." << endl;
        auto s3 = newInterface("sticking_coeff_check.yaml", "Pt_surf");
        cout << "   s3: " << s3->thermo()->name() << ", nRxn: " << s3->kinetics()->nReactions() << endl;

        cout << "4. blowers-masel.yaml (Pt_surf)..." << endl;
        auto s4 = newInterface("blowers-masel.yaml", "Pt_surf");
        cout << "   s4: " << s4->thermo()->name() << ", nRxn: " << s4->kinetics()->nReactions() << endl;

        cout << "5. sofc.yaml (metal_surface)..." << endl;
        auto s5 = newInterface("sofc.yaml", "metal_surface");
        cout << "   s5: " << s5->thermo()->name() << ", nRxn: " << s5->kinetics()->nReactions() << endl;

        cout << "6. sofc.yaml (oxide_surface)..." << endl;
        auto s6 = newInterface("sofc.yaml", "oxide_surface");
        cout << "   s6: " << s6->thermo()->name() << ", nRxn: " << s6->kinetics()->nReactions() << endl;

        cout << "All 6 surface mechanisms loaded cleanly!" << endl;
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}
