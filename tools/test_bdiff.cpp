#include <iostream>
#include <iomanip>
#include <vector>
#include <cantera/core.h>
#include <cantera/transport/TransportFactory.h>
#include <cantera/transport/MultiTransport.h>

using namespace Cantera;

int main() {
    auto soln = newSolution("h2o2.yaml", "", "none");
    auto thermo = soln->thermo();
    auto multi = std::dynamic_pointer_cast<MultiTransport>(newTransport(thermo, "multicomponent"));
    thermo->setState_TPX(2389.681331101101, 155127.35933289427, "H2:0.25, O2:0.75");
    
    Array2D bdiff1(10, 10, 0.0);
    multi->getBinaryDiffCoeffs(10, bdiff1.data());
    std::cout << std::setprecision(16);
    std::cout << "Before other calls, bdiff(0, 0): " << bdiff1(0, 0) << std::endl;

    double visc = multi->viscosity();
    double cond = multi->thermalConductivity();
    std::vector<double> dt(10, 0.0);
    multi->getThermalDiffCoeffs(dt);

    Array2D bdiff2(10, 10, 0.0);
    multi->getBinaryDiffCoeffs(10, bdiff2.data());
    std::cout << "After other calls,  bdiff(0, 0): " << bdiff2(0, 0) << std::endl;
    return 0;
}
