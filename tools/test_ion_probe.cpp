#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/transport/IonGasTransport.h"
#include "cantera/base/global.h"
#include "cantera/base/utilities.h"
#include "cantera/numerics/polyfit.h"

using namespace Cantera;
using namespace std;

int main() {
    auto sol = newSolution("ch4_ion.yaml", "gas", "ionized-gas");
    sol->thermo()->setState_TPX(300.0, 101325.0, "CH4:0.095, O2:0.19, N2:0.715");
    auto tran = dynamic_pointer_cast<IonGasTransport>(sol->transport());

    size_t k_o2m = sol->thermo()->speciesIndex("O2-");
    size_t k_o2 = sol->thermo()->speciesIndex("O2");

    double mw_o2m = sol->thermo()->molecularWeight(k_o2m);
    double mw_o2 = sol->thermo()->molecularWeight(k_o2);
    double red_mass = mw_o2m * mw_o2 / (Avogadro * (mw_o2m + mw_o2));

    cout << "mw_o2m: " << mw_o2m << " mw_o2: " << mw_o2 << endl;
    cout << "red_mass: " << red_mass << endl;

    // Prager polyfit data
    const vector<double> temp{300.0, 400.0, 500.0, 600.0, 800.0, 1000.0,
                         1200.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0};
    const vector<double> om11_O2{120.0, 107.0, 98.1, 92.1, 83.0, 77.0,
                            72.6, 67.9, 62.7, 59.3, 56.7, 53.8};
    vector<double> c_prager(6);
    polyfit(5, temp, om11_O2, {}, c_prager);
    for (size_t i = 0; i < 6; i++) {
        cout << "c_prager[" << i << "] = " << setprecision(17) << c_prager[i] << endl;
    }

    double om11 = poly5(300.0, c_prager) / 1e20;
    cout << "poly5(300): " << poly5(300.0, c_prager) << " om11: " << om11 << endl;

    double diffcoeff = 3.0/16.0 * sqrt(2.0 * Pi/red_mass) * pow(Boltzmann * 300.0, 1.5) / om11;
    cout << "diffcoeff: " << diffcoeff << endl;

    DenseMatrix bdiff(sol->thermo()->nSpecies(), sol->thermo()->nSpecies());
    tran->getBinaryDiffCoeffs(sol->thermo()->nSpecies(), bdiff.data());
    cout << "native getBinaryDiffCoeffs(O2-, O2): " << bdiff(k_o2m, k_o2) << endl;

    return 0;
}
