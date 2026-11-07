
#include <iostream>
#include <iomanip>
#include <vector>
#include <cantera/core.h>
#include <cantera/transport/TransportFactory.h>
#include <cantera/transport/MultiTransport.h>

using namespace Cantera;

int main() {
    auto soln = newSolution("gri30.yaml", "", "none");
    auto thermo = soln->thermo();
    auto multi = std::dynamic_pointer_cast<MultiTransport>(newTransport(thermo, "multicomponent"));
    
    std::vector<double> X0 = {
        0.269205, 0.000107082, 1.36377e-09, 4.35475e-10, 4.34036e-06, 0.192249,
        3.59356e-13, 2.78061e-12, 4.7406e-18, 4.12955e-17, 2.58549e-14, 8.96502e-16,
        6.09056e-11, 7.56752e-09, 0.192253, 0.0385036, 1.49596e-08, 2.22378e-08,
        1.43096e-13, 1.45312e-15, 1.96948e-12, 8.41937e-19, 3.18852e-13, 7.93625e-18,
        3.20653e-15, 1.15149e-19, 1.61189e-18, 1.4719e-15, 5.24728e-13, 6.90582e-17,
        6.37248e-12, 5.93728e-11, 2.71219e-09, 2.66645e-06, 6.57142e-11, 9.52453e-08,
        1.26006e-14, 3.49802e-12, 1.19232e-11, 7.17782e-13, 1.85347e-07, 8.25325e-14,
        5.00914e-20, 1.54407e-16, 3.07176e-11, 4.93198e-08, 4.84792e-12, 0.307675,
        0.0, 6.21649e-29, 8.42393e-28, 6.77865e-18, 2.19225e-16
    };
    thermo->setState_TPX(1500.0, 1e5, X0);
    
    std::cout << std::scientific << std::setprecision(6);
    for (size_t i = 0; i < 10; i++) {
        double T = 400.0 + 100.0 * i;
        thermo->setState_TP(T, 1e5);
        std::cout << "T=" << T << " native_cond=" << multi->thermalConductivity() << std::endl;
    }
    return 0;
}
