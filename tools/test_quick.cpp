#include <iostream>
#include "cantera/kinetics/ReactionRateFactory.h"
#include "cantera/kinetics/ReactionRate.h"

using namespace Cantera;
using namespace std;

int main() {
    vector<string> rtypes = {
        "Arrhenius", "elementary", "three-body", "Troe", "Lindemann",
        "falloff", "Blowers-Masel", "interface-Arrhenius", "sticking-Arrhenius"
    };
    for (const auto& t : rtypes) {
        auto rate = newReactionRate(t);
        cout << t << " -> " << rate->type() << endl;
    }
    return 0;
}
