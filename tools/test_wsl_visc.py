import subprocess

cpp = r"""#include <iostream>
#include <iomanip>
#include <cantera/core.h>
#include <cantera/transport/TransportFactory.h>
#include <cantera/transport/MultiTransport.h>

int main() {
    auto sol = Cantera::newSolution("gri30.yaml", "", "multicomponent");
    auto thermo = sol->thermo();
    std::cout << "C++ Tmin: " << thermo->minTemp() << " Tmax: " << thermo->maxTemp() << std::endl;
    thermo->setState_TPX(1200.0, Cantera::OneAtm, "CH4:0.1, O2:0.2, N2:0.7");
    auto multi = std::dynamic_pointer_cast<Cantera::MultiTransport>(sol->transport());
    size_t nsp = thermo->nSpecies();
    std::vector<double> dt(nsp);
    multi->getThermalDiffCoeffs(dt);
    std::cout << "C++ ThermalDiff for first 5 species:" << std::endl;
    for (size_t k = 0; k < 5; ++k) {
        std::cout << thermo->speciesName(k) << ": " << std::setprecision(16) << dt[k] << std::endl;
    }

    std::vector<double> grad_T = {100.0};
    std::vector<double> grad_X(nsp, 0.0);
    grad_X[0] = 0.1;
    grad_X[1] = -0.1;
    std::vector<double> fluxes(nsp, 0.0);
    multi->getSpeciesFluxes(1, grad_T, nsp, grad_X, nsp, fluxes);
    std::cout << "C++ Fluxes for first 5 species:" << std::endl;
    for (size_t k = 0; k < 5; ++k) {
        std::cout << thermo->speciesName(k) << ": " << std::setprecision(16) << fluxes[k] << std::endl;
    }
    return 0;
}"""

with open("/tmp/test_visc.cpp", "w") as f:
    f.write(cpp)

subprocess.run(["g++", "-std=c++20", "-O2", "/tmp/test_visc.cpp",
    "-I/home/nebula/cantera-pinned-oracle/include",
    "-L/home/nebula/cantera-pinned-oracle/build/lib",
    "-lcantera_shared", "-Wl,-rpath,/home/nebula/cantera-pinned-oracle/build/lib",
    "-o", "/tmp/test_visc"], check=True)

res = subprocess.run(["/tmp/test_visc"], capture_output=True, text=True)
print(res.stdout)
