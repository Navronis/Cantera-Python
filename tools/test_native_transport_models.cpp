#include <iostream>
#include <iomanip>
#include <memory>
#include "cantera/base/Solution.h"
#include "cantera/transport/TransportFactory.h"
#include "cantera/transport/WaterTransport.h"
#include "cantera/transport/UnityLewisTransport.h"
#include "cantera/transport/IonGasTransport.h"
#include "cantera/thermo/ThermoFactory.h"

using namespace Cantera;
using namespace std;

int main() {
    cout << "Testing WaterTransport:" << endl;
    auto phase = newThermo("thermo-models.yaml", "liquid-water");
    auto tran = newTransport(phase, "default");
    cout << "Model: " << tran->transportModel() << endl;

    double T_test[] = {25, 100, 100, 250, 250, 350, 400};
    double P_test[] = {1e5, 5e5, 1e7, 5e6, 5e7, 1.75e7, 1.5e7};
    for (int i = 0; i < 7; i++) {
        phase->setState_TP(T_test[i] + 273.15, P_test[i]);
        cout << "T=" << T_test[i] << " C, P=" << P_test[i] 
             << " Pa | mu=" << setprecision(10) << tran->viscosity() 
             << " | lambda=" << setprecision(10) << tran->thermalConductivity() 
             << " | rho=" << phase->density()
             << " | kappa=" << phase->isothermalCompressibility()
             << " | alpha=" << phase->thermalExpansionCoeff() << endl;
    }

    cout << "\nTesting UnityLewisTransport:" << endl;
    auto sol_gri = newSolution("gri30.yaml", "gri30", "unity-Lewis-number");
    cout << "Model: " << sol_gri->transport()->transportModel() << endl;
    sol_gri->thermo()->setState_TPX(1500.0, 101325.0, "CH4:0.1, O2:0.2, N2:0.7");
    size_t nsp = sol_gri->thermo()->nSpecies();
    vector<double> d(nsp, 0.0);
    vector<double> dt(nsp, 0.0);
    sol_gri->transport()->getMixDiffCoeffs(std::span<double>(d));
    sol_gri->transport()->getThermalDiffCoeffs(std::span<double>(dt));
    cout << "mu=" << sol_gri->transport()->viscosity() << ", lambda=" << sol_gri->transport()->thermalConductivity() << endl;
    cout << "d[0]=" << d[0] << ", d[1]=" << d[1] << ", dt[0]=" << dt[0] << endl;

    cout << "\nTesting IonGasTransport:" << endl;
    auto sol_ion = newSolution("ch4_ion.yaml", "gas", "ionized-gas");
    cout << "Model: " << sol_ion->transport()->transportModel() << endl;
    sol_ion->thermo()->setState_TPX(1200.0, 101325.0, "CH4:0.09, O2:0.18, N2:0.7, E:0.01, H3O+:0.01, HCO+:0.01");
    size_t nsp_ion = sol_ion->thermo()->nSpecies();
    vector<double> mobi(nsp_ion, 0.0);
    vector<double> d_ion(nsp_ion, 0.0);
    sol_ion->transport()->getMobilities(std::span<double>(mobi));
    sol_ion->transport()->getMixDiffCoeffs(std::span<double>(d_ion));
    cout << "viscosity=" << sol_ion->transport()->viscosity() << ", lambda=" << sol_ion->transport()->thermalConductivity() << endl;
    cout << "elecCond=" << sol_ion->transport()->electricalConductivity() << endl;
    for (size_t k = 0; k < nsp_ion; k++) {
        if (sol_ion->thermo()->charge(k) != 0 || sol_ion->thermo()->speciesName(k) == "E") {
            cout << "  species " << sol_ion->thermo()->speciesName(k) << ": mobi=" << mobi[k] << ", diff=" << d_ion[k] << endl;
        }
    }

    cout << "\nTesting TransportFactory:" << endl;
    TransportFactory* fac = TransportFactory::factory();
    auto tr_mix = fac->newTransport("mixture-averaged", sol_gri->thermo());
    cout << "fac newTransport('mixture-averaged'): " << tr_mix->transportModel() << endl;
    delete tr_mix;

    return 0;
}
